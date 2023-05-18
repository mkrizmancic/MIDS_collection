import torch
import networkx as nx
from matplotlib import pyplot as plt


def find_MIDS(G):
    # Minimum Independent Dominating Set == Smallest Maximal Independent Set
    # Maximal Independent Set == Maximal Clique of the complement
    # ===
    # Minimum Independent Dominating Set == Smallest Maximal Clique of the complement
    Gc = nx.complement(G)
    min_size = len(Gc)
    min_cliques = []
    for nodes in nx.find_cliques_recursive(Gc):
        size = len(nodes)
        if size < min_size:
            min_size = size
            min_cliques = [nodes]
        elif size == min_size:
            min_cliques.append(nodes)
    return min_cliques

def test_find_MIDS():
    graph_file = '/home/marko/PROJECTS/MIDS_collection/PyTorch Geometric/Dataset/raw/5_nodes/G5,5.txt'
    G = nx.read_edgelist(graph_file, nodetype=int)
    nx.draw(G, with_labels=True)
    plt.show()
    
    print(find_MIDS(G))