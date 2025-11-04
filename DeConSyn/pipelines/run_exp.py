import argparse
import asyncio
from pathlib import Path

import spade
from joblib import parallel_config
from loguru import logger
import pandas as pd

from DeConSyn.data.data_loader import DatasetLoader
from DeConSyn.data.data_transformer import DataTransformer
from DeConSyn.logging.logger import init_logging
from DeConSyn.pipelines.config import load_config
from DeConSyn.training_framework.agent.node_agent import NodeData, NodeAgent, NodeConfig
from DeConSyn.utils.graph import Graph
from DeConSyn.utils.seed import set_global_seed

XMPP_DOMAIN = "localhost"


def partition_for(i: int, splits: list, train: pd.DataFrame, test: pd.DataFrame) -> NodeData:
    part_train = splits[i]
    return NodeData(part_train=part_train, full_train=train, full_test=test)

async def _shutdown_agents(agents: list[NodeAgent]) -> None:
    for a in agents:
        try:
            a.presence.set_unavailable()
        except Exception:
            pass

    await asyncio.gather(*[a.stop() for a in agents], return_exceptions=True)

    await asyncio.sleep(0.3)

    try:
        await spade.quit_spade()
    except Exception:
        pass

    try:
        logger.remove()
    except Exception:
        pass

async def run(config):
    params = config.get("deconsyn_params")
    log_level = params.get("log_level", "INFO").upper()
    n = params.get("n")
    epochs = params.get("epochs")
    iterations = params.get("iterations")
    topology = params.get("topology")
    gen_model_type = params.get("gen_model_type")
    seed = config.get("seed", 42)


    run_id = init_logging(level=log_level,
                            epochs=epochs,
                            iterations=iterations,
                            topology=topology,
                            model_type=gen_model_type)
    set_global_seed(seed)

    data_root = params.get("data_root")
    categorical_columns = params.get("categorical_columns")
    target = params.get("target")
    npy_path = data_root + "/npy"
    loader = DatasetLoader(data_root, categorical_columns, target)
    train = loader.get_train()
    test = loader.get_test()
    DataTransformer.save_full_npy(train, Path(npy_path), categorical_columns, target, "_train")
    DataTransformer.save_full_npy(test, Path(npy_path), categorical_columns, target, "_test")
    splits = loader.split_iid(n, seed=seed)
    test_splits = loader.split_test_iid(n, seed=seed)
    logger.info(f"Data loaded from {data_root} and partitioned for {n} agents.")

    for i in range(n):
        part = splits[i]
        DataTransformer.save_split_npy(
            part,
            Path(npy_path) / "splits" / str(n),
            i,
            categorical_columns,
            target
        )
        DataTransformer.save_test_split_npy(
            test_splits[i],
            Path(npy_path) / "splits" / str(n),
            i,
            categorical_columns,
            target
        )
        logger.info(f"Agent {i} partition: {part.shape}, head:\n{part.head(3)}")

    k = params.get("k", 4)
    p = params.get("p", 0.1)
    alpha = params.get("alpha", 1.0)
    if topology.lower() == "ring":
        neighbors_map = Graph.ring(n)
    elif topology.lower() == "full":
        neighbors_map = Graph.full(n)
    elif topology.lower() == "small-world":
        neighbors_map = Graph.small_world(n, k=k, p=p, seed=seed)
    else:
        raise ValueError("Unsupported topology. Use 'ring', 'full' or 'small-world'.")

    agents: list[NodeAgent] = []
    try:
        for i in range(n):
            cfg = NodeConfig(
                jid=f"agent{i}@{XMPP_DOMAIN}",
                id=i,
                password="password",
                epochs=epochs,
                max_iterations=iterations,
                alpha=alpha,
                run_id=run_id,
                model_type=gen_model_type,
                real_data_path=npy_path + f"/splits/{n}/split_{i}",
                target=target,
                cat_encoder=loader.get_cat_oe(),
                num_encoder=loader.get_num_transformer(),
                y_encoder=loader.get_y_oe(),
                data_transformer=loader.get_data_transformer(),
                config=config
            )
            data = partition_for(i, splits, train, test)
            agent = NodeAgent(cfg=cfg, data=data, neighbors=neighbors_map[cfg.jid])
            agents.append(agent)

        with parallel_config(n_jobs=1, prefer=None):
            await asyncio.gather(*[a.start(auto_register=True) for a in agents])
            logger.info(f"{n} agents started (epochs={epochs}, iters={iterations}, alpha={alpha}).")

            while True:
                await asyncio.sleep(2)
                if all(a.is_final for a in agents):
                    for a in agents:
                        a.fsm_done.set()
                    break

            await asyncio.gather(*[a.fsm_done.wait() for a in agents])
            logger.info("All FSMs finished — stopping agents...")

            await asyncio.gather(*[a.stop() for a in agents], return_exceptions=True)

            await asyncio.gather(*[spade.wait_until_finished(a) for a in agents], return_exceptions=True)

            logger.info("Agents stopped.")
    finally:
        await _shutdown_agents(agents)
        logger.info("Agents stopped cleanly.")

async def main():
    parser = argparse.ArgumentParser(description="Evaluate experiments in a given directory.")
    parser.add_argument("--config", type=str, required=True, help="Path to the evaluation configuration file.")
    args = parser.parse_args()
    config = load_config(args.config)
    await run(config)

if __name__ == "__main__":
    asyncio.run(main())