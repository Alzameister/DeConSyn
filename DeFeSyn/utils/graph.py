import pandas as pd

from DeFeSyn.spade.NodeAgent import NodeAgent, NodeConfig, NodeData


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




if __name__ == "__main__":
    nr_agents = 4

    topology = Graph.ring(nr_agents)
    for agent, neighbors in topology.items():
        print(f"Agent {agent} neighbors: {neighbors}")