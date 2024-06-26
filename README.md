# MIDS_collection
Collection of algorithms and scripts for finding Minimum Independent Dominating Sets (MIDS) written in MATLAB and Python.

DATASET: https://github.com/mkrizmancic/my_graphs_dataset

## Bron-Kerbosch
For any graph it holds:
- Minimum Independent Dominating Set == Maximal Independent Set of smallest size
- Maximal Independent Set == Maximal Clique of graph's complement

This method written in MATLAB uses [Bron-Kerbosch](https://en.wikipedia.org/wiki/Bron%E2%80%93Kerbosch_algorithm) algorithm to find all Maximal Independent Sets in
a given graph by finding Maximal Cliques of its complement. MIDS is then found by finding the smallest IDS.

**Usage:** Select the desired graph in the first section of `main.m` and run it. The script prints out the number of found MIDS, their size, and its run-time.
Graphs with marked MIDS are displayed in figures.

## Disjunction
Let **d** be a support vector of a set of a potential MIDS  (_i_-th element is 1 if _i_-th node is in the set, 0 otherwise). Then, the following holds: If (**A** + **I**) * **d** = **1** (vector of ones), then **d** is MIDS. (This is a weak condition, the inverse does not hold.)

It might be possible to find MIDS by finding **d** such that **d** = (**A** + **I**)^-1 * **1** using a pseudo-inverse. We call this vector disjunction vector.

**Usage:** You can explore selected graphs in MATLAB by running `MIDS_check.mlx` or test the hypothesis on a larger number of graphs by running `test_metric.py`.

## MILP_Gurobi
It turns out, the problem of finding MIDS can be written as Integer Linear Program and solved very "quickly" with commercial solvers such as [Gurobi](https://www.gurobi.com/).
Of course, since this is an NP-hard problem, the solutions are quick only for smaller graphs. Still, it is useful a simple benchmark.

On Lenovo Legion Y-540 with 16 GB of RAM and Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz, Gurobi can solve random graphs with 120 nodes in under a minute.

**Usage:** If you want to find MIDS on a specific graph, use `milpMIDS.py`. If you want to test the speed on a larger number of random graphs, use `speed_test.py`.

## PyTorch Geometric
This is a new idea that MIDS could be found by training a neural network to predict MIDS. The idea is to use a graph neural network to predict the probability of each node being in MIDS.

## Token-based
This is a new idea that MIDS could be found by simulating communication between the nodes in the graph.

The procedure is as follows. Nodes that are in a potential MIDS send a message (token) to each of their neighbors and we record the number of received messages.
Our hypothesis was that if we the sum up all tokens and divide that number with the number of nodes which did not send messages, the result will always be smaller
if those node were really in MIDS than any other non-minimum IDS. Take a look at [this example](Token-based/token_mids.pdf) (in Croatian).

When doing tests on random graphs with 5-9 nodes, this showed to be true in 90 % of the cases.

The code is mainly focused on testing out the hypothesis. Searching for IDS and MIDS is brute-force.

**Usage:** If you want to test the hypothesis on a specific graph, use `test.m`. If you want to test the hypothesis on a larger number of random graphs, use `statistics_test.m`.

