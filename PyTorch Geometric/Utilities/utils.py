from pathlib import Path

import networkx as nx
import numpy as np
from matplotlib import colormaps
from matplotlib import pyplot as plt


def find_MIDS(G):
    # Minimum Independent Dominating Set == Smallest Maximal Independent Set
    # Maximal Independent Set == Maximal Clique of the complement
    # ===
    # Minimum Independent Dominating Set == Smallest Maximal Clique of the complement
    Gc = nx.complement(G)
    min_size = len(Gc)
    min_cliques = []
    for nodes in nx.find_cliques(Gc):
        size = len(nodes)
        if size < min_size:
            min_size = size
            min_cliques = [nodes]
        elif size == min_size:
            min_cliques.append(nodes)
    return min_cliques


def check_MIDS(A, candidate, target_value):
    """
    Args:
        - A: adjacency matrix
        - candidate: node labels that are candidate for MIDS (data.y)
        - target_value: known size of the MIDS
    """
    # TODO: This function needs to be adjusted.
    #   - Instead of adjacency matrix, we may pass the edgelist and convert it here

    n = len(candidate)

    # Candidate set is not minimal
    if sum(candidate) > target_value:
        return False

    # Candidate set is not dominating.
    if not all((A + np.eye(n)) @ candidate >= 1):
        return False

    # Candidate set is not independent.
    for i in range(n):
        for j in range(i + 1, n):
            if candidate[i] == 1 and candidate[j] == 1 and A[i, j] == 1:
                return False

    if sum(candidate) < target_value:
        print("Somehow we found an even smaller MIDS.")

    return True


def load_graph(graph_file):
    G_init = nx.read_edgelist(graph_file, nodetype=int)
    G = nx.Graph()
    G.add_nodes_from(sorted(G_init.nodes))
    G.add_edges_from(G_init.edges)
    return G


def test_find_MIDS():
    def to_vector(nodes, num_nodes):
        # Encode found cliques as support vectors.
        mids = np.zeros(num_nodes)
        mids[nodes] = 1
        return mids

    graph_files_dir = Path("/home/marko/PROJECTS/MIDS_collection/PyTorch Geometric/Dataset/raw/")
    for graph_file in graph_files_dir.glob("*.txt"):
        # print(graph_file.stem)
        # if graph_file.stem != "G8,1437":
        #     continue
        G = load_graph(graph_file)
        # nx.draw(G, with_labels=True)
        # plt.show()
        A = nx.to_numpy_array(G)
        possible_MIDS = find_MIDS(G)
        for MIDS in possible_MIDS:
            np_MIDS = to_vector(MIDS, G.number_of_nodes())
            print(MIDS, np_MIDS)
            if not check_MIDS(A, np_MIDS, len(MIDS)):
                pos=nx.spring_layout(G)
                plt.figure()
                nx.draw(G, with_labels=True, node_color=np_MIDS, cmap=colormaps["bwr"], pos=pos)
                plt.figure()
                Gc = nx.complement(G)
                nx.draw(Gc, with_labels=True, node_color=np_MIDS, cmap=colormaps["bwr"], pos=pos)
                plt.show()


def test_find_cliques():
    def to_vector(nodes, num_nodes):
        # Encode found cliques as support vectors.
        clique = np.zeros(num_nodes)
        clique[nodes] = 1
        return clique

    graph_file = Path("/home/marko/PROJECTS/MIDS_collection/PyTorch Geometric/Dataset/raw/8_nodes/G8,1437.txt")
    G = nx.read_edgelist(graph_file, nodetype=int)
    nx.draw(G, with_labels=True)
    plt.show()
    Gc = nx.complement(G)
    for nodes in nx.find_cliques(Gc):
        np_nodes = to_vector(nodes, Gc.number_of_nodes())
        print(nodes, np_nodes)
        nx.draw(Gc, with_labels=True, node_color=np_nodes, cmap=colormaps["bwr"])
        plt.show()


if __name__ == "__main__":
    test_find_MIDS()
