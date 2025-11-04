from pathlib import Path

import networkx as nx
from matplotlib import pyplot as plt


def agent_jid(i: int) -> str:
    """
    Generate a jid for an agent given its index.
    Parameters
    ----------
    i: int
        The index of the agent.

    Returns
    -------
    str
        The jid of the agent.
    """
    return f"agent{i}@localhost"

class Graph:
    @staticmethod
    def ring(n: int) -> dict[str, list[str]]:
        """
        Create a ring topology among the given agents.
        Parameters
        ----------
        n : int
            The number of agents in the ring.

        Returns
        -------
        dict[str, list[str]]
            A dictionary mapping each agent to its neighbor list, containing the jid's of the neighbors.
        """
        topology = {}
        for i in range(n):
            current = agent_jid(i)
            left_neighbor = agent_jid((i - 1) % n)
            right_neighbor = agent_jid((i + 1) % n)
            topology[current] = [left_neighbor, right_neighbor]
        return topology

    @staticmethod
    def full(n: int) -> dict[str, list[str]]:
        """
        Create a fully connected topology among the given agents.
        Parameters
        ----------
        n : int
            The number of agents in the fully connected graph.

        Returns
        -------
        dict[str, list[str]]
            A dictionary mapping each agent to its neighbor list, containing the jid's of the neighbors.
        """
        topology = {}
        for i in range(n):
            current = agent_jid(i)
            neighbors = [agent_jid(j) for j in range(n) if j != i]
            topology[current] = neighbors
        return topology

    @staticmethod
    def small_world(n: int, k: int = 4, p: float = 0.1, seed: int = 42) -> dict[str, list[str]]:
        """
        Create a small-world topology using NetworkX Watts-Strogatz generator.

        Parameters
        ----------
        n : int
            Number of agents (nodes).
        k : int
            Each agent is initially connected to k/2 neighbors on each side in the ring.
        p : float
            Rewiring probability (0 <= p <= 1).

        Returns
        -------
        dict[str, list[str]]
            Mapping agent jid -> list of neighbor jids
        """
        G = nx.watts_strogatz_graph(n, k, p, seed=seed)
        pos = nx.circular_layout(G)

        # plt.figure(figsize=(6, 6))
        # nx.draw(
        #     G, pos,
        #     with_labels=True,
        #     node_size=600,
        #     node_color="lightblue",
        #     font_size=8,
        #     font_weight="bold",
        #     edge_color="gray"
        # )
        # plt.title(f"Small-World Graph (n={n}, k={k}, p={p})")
        # plt.show()

        topology = {agent_jid(i): [agent_jid(j) for j in G.neighbors(i)] for i in G.nodes()}
        return topology

if __name__ == "__main__":
    n= 10
    k = 4
    p = 0.1
    seed = 42

    G_ring = nx.cycle_graph(n)
    G_full = nx.complete_graph(n)
    G_sw = nx.watts_strogatz_graph(n, k, p, seed=seed)

    pos = nx.circular_layout(G_ring)

    plt.figure(figsize=(12, 4), dpi=200)

    ax1 = plt.subplot(1, 3, 1)
    nx.draw(G_ring, pos=pos, with_labels=False, node_size=500, width=1.5)
    ax1.set_title(f"Ring (n={n})")
    ax1.set_axis_off()

    ax2 = plt.subplot(1, 3, 2)
    nx.draw(G_sw, pos=pos, with_labels=False, node_size=500, width=1.5)
    ax2.set_title(f"Small-World (n={n}, k={k}, p={p})")
    ax2.set_axis_off()

    ax3 = plt.subplot(1, 3, 3)
    nx.draw(G_full, pos=pos, with_labels=False, node_size=500, width=1.0)
    ax3.set_title(f"Fully Connected (n={n})")
    ax3.set_axis_off()

    plt.tight_layout()
    plt_path = Path("../../plots")
    plt_path.mkdir(parents=True, exist_ok=True)
    plt.savefig("../../plots/topologies_side_by_side.png", bbox_inches="tight")