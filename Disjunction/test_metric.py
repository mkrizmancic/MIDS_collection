import itertools

import codetiming
import networkx as nx
import numpy as np
from matplotlib import colormaps
from matplotlib import pyplot as plt
from my_graphs_dataset.loader import GraphDataset
from tqdm.contrib.concurrent import process_map


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


def disjunction_value(G):
    A = nx.to_numpy_array(G)
    nodes = list(G.nodes)
    n = A.shape[0]
    s = np.linalg.lstsq(A + np.eye(n), np.ones(n), rcond=None)[0]
    s = np.round(s, 2)
    return {nodes[i]: s[i] for i in range(n)}


def disjunction_vector(G):
    A = nx.to_numpy_array(G)
    n = A.shape[0]
    s = np.linalg.lstsq(A + np.eye(n), np.ones(n), rcond=None)[0]
    s = np.round(s, 2)
    return s


def to_vector(nodes, num_nodes):
    # Encode found cliques as support vectors.
    mids = np.zeros(num_nodes)
    mids[nodes] = 1
    return mids


def group_by_value(data):
    """Groups a dictionary by value, sorted by value descending.

    Args:
        data: A dictionary where keys are hashable and values are comparable.

    Returns:
        A list of dictionaries where each dictionary contains entries from the
        original dictionary that share the same value, sorted by value descending.
    """
    value_to_entries = {}
    for key, value in data.items():
        # Since value comparison might be imprecise for floats, round them to a
        # set number of decimal places before comparison. Here, we round to 6.
        rounded_value = round(value, 6)
        value_to_entries.setdefault(rounded_value, []).append(key)
    # Sort the values in descending order
    sorted_values = sorted(value_to_entries.keys(), reverse=True)
    return [value_to_entries[value] for value in sorted_values]


def worker(graph):
    graph = graph.strip()
    G = GraphDataset.parse_graph6(graph)
    # nx.draw(G, with_labels=True)
    # plt.show()
    possible_MIDS = find_MIDS(G)
    disj_val = disjunction_value(G)
    len_MIDS = len(possible_MIDS[0])
    possible_MIDS = [set(MIDS) for MIDS in possible_MIDS]

    grouped = group_by_value(disj_val)  # Group nodes by disjunction value
    must_have_nodes = []  # Top nodes with highest disjunction value. Total number of nodes is <= len_MIDS.
    remaining_nodes = []  # If number of nodes in the must-have group is < len_MIDS, we must add all the nodes from
    # the next group because any combination could be valid.
    for group in grouped:
        if len(group) + len(must_have_nodes) <= len_MIDS:
            must_have_nodes.extend(group)
        else:
            remaining_nodes.extend(group)
        if len(must_have_nodes) + len(remaining_nodes) >= len_MIDS:
            break

    num_needed_nodes = len_MIDS - len(must_have_nodes)
    disj_combinations = []
    for additional in itertools.combinations(remaining_nodes, num_needed_nodes):
        disj_combinations.append(must_have_nodes + list(additional))

    for comb in disj_combinations:
        if set(comb) in possible_MIDS:
            return False

    return graph


def test_disjunction_metric():
    display_results = False
    loader = GraphDataset(selection=[])
    all_results = []

    for graphs in loader.graphs(raw=True, batch_size="auto"):
        result = process_map(worker, graphs, chunksize=1000)
        all_results.extend(result)

    wrong = 0
    for res in all_results:
        if res:
            wrong += 1

            if display_results:
                G = GraphDataset.parse_graph6(res)
                possible_MIDS = find_MIDS(G)
                disjunction_vec = disjunction_vector(G)

                print(f"Graph {res}:")
                for i, MIDS in enumerate(possible_MIDS):
                    np_MIDS = to_vector(MIDS, G.number_of_nodes())
                    print(f"  {i}: {MIDS=}, d_vec={disjunction_vec}")

                    pos = nx.spring_layout(G)
                    plt.figure()
                    nx.draw(G, with_labels=True, node_color=np_MIDS, cmap=colormaps["bwr"], pos=pos)
                    plt.show()

    total = len(all_results)
    correct = total - wrong
    print(f"Correct: {correct}/{total} ({correct/total*100:.2f}%)\nWrong: {wrong}")


if __name__ == "__main__":
    with codetiming.Timer():
        test_disjunction_metric()
