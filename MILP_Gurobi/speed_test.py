"""
Use this script to test the speed of the MIDS optimization algorithm implemented
as MILP in Gurobi. You can set the maximum time to find a solution for each
graph size, the step size to increase the number of nodes in the graph, and the
number of times to repeat the search for each graph size.

The script generates a random graph using the Watts-Strogatz model and tries to
find a solution for the MIDS problem. The script stops when the maximum time is
reached or the algorithm fails to find a solution.
"""

import math
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from milpMIDS import optimize


def main():
    # Set test parameters.
    max_time = 5  # Maximum time to find a solution for each graph size.
    step = 5  # Step size to increase the number of nodes in the graph.
    repeat = 5  # Number of times to repeat the search for each graph size.

    # Prepare variables.
    n = 0
    sol = True
    elps = 0
    avg = 0
    x = []
    y = []

    while sol and elps <= max_time:
        n += step

        print(f"Searching MIDS for {n=}...", end="")
        for _ in range(repeat):
            G = nx.connected_watts_strogatz_graph(n, max(int(math.sqrt(n)), 2), 0.5)
            sol, elps, _ = optimize(G, "MIDS", outputFlag=0)
            if sol:
                avg += elps
            else:
                print("FAILED")
                break
        else:
            elps = avg / repeat
            print("SUCCESS")

        x.append(n)
        y.append(elps)

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    plt.stem(x, y, basefmt=" ")
    plt.title("Linear scale")
    plt.xlabel("Number of nodes")
    plt.ylabel("Average solve time (s)")
    plt.xticks(x, rotation=90)
    plt.yticks(np.arange(0, plt.ylim()[1], step=0.2))
    plt.grid(axis="y")

    ax = fig.add_subplot(1, 2, 2)
    plt.stem(x, y)
    ax.set_yscale("log")
    plt.title("Logarithmic scale")
    plt.xlabel("Number of nodes")
    plt.ylabel("Average solve time (s)")
    plt.xticks(x, rotation=90)
    plt.grid(axis="y")

    plt.show()


if __name__ == "__main__":
    main()
