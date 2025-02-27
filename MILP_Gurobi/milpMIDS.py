"""
Use this script to find solutions for Minimum Dominating Set (MDS), Maximum
Independent Set (MIS), and Minimum Independent Dominating Set (MIDS) problems
for individual graphs. The script uses the Gurobi optimizer to solve the
problems.

The script can also be used to compare the results of directly optimizing the
number of elements in the support vector of the solution set (D) and the novel
proposed objective function (J) based on the adjacency matrix of the graph.

Function 'optimize' can be used in a loop to find solutions for multiple graphs.
"""
import time

import gurobipy as gp
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def main():
    ## Select one of the existing NetworkX graphs or import it from saved ones.
    # G = nx.grid_graph(dim=[5, 5])
    # G = nx.cycle_graph(100)
    # G = nx.star_graph(5)
    G = nx.petersen_graph()
    # G = nx.dodecahedral_graph()
    # G = nx.connected_watts_strogatz_graph(100, max(int(math.sqrt(100)), 2), 0.5)
    # from saved_graphs import G

    ## Set the problem to solve: MDS, MIS, MIDS
    problem = "MIDS"

    ## Set the goal to optimize:
    ## - D: minimize/maximize the number of elements in the support vector of the solution set
    ## - J: a theoretical proposed objective function

    solution, elapsed, dets = optimize(G, problem, goal="D")

    # Print and draw the solution.
    print("=================================================")
    if solution:
        print("Found MIDS solution with {} nodes in {} seconds.".format(len(solution), elapsed))
        print(dets)
        color_map = ["red" if x in solution else "darkgray" for x in G.nodes]
        nx.draw(
            G, pos=nx.kamada_kawai_layout(G), with_labels=True, font_weight="bold", node_color=color_map, node_size=1000
        )
        plt.show()
    else:
        print("No solution found.")


def optimize(G, problem, goal="D", outputFlag=1, single_cpu=False):
    A = nx.to_numpy_array(G)
    n = np.size(A, 1)
    A_ = A + np.eye(n)

    # Set up environment.
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", outputFlag)
    env.start()

    # Create a new model.
    model = gp.Model("MIDS", env=env)
    model.setParam("presolve", 0)
    if single_cpu:
        model.setParam('Threads', 1)

    # Add variables to the model.
    D = model.addVars(G.nodes, vtype=gp.GRB.BINARY)

    # Add constraints.
    if problem == "MDS" or problem == "MIDS":
        # Constraint for domination: (A+I)*D >= 1 for all nodes
        con1 = model.addConstrs(
            (sum([A_[i, j] * D[k] for j, k in enumerate(G.nodes)]) >= 1 for i in range(n)), name="Con1" # type: ignore
        )
    if problem == "MIS" or problem == "MIDS":
        # Constraint for independence: D[i] + D[j] <= 1 for all edges (i, j)
        con2 = model.addConstrs((D[edge[0]] + D[edge[1]] <= 1 for edge in G.edges), name="Con2")

    # Set the objective
    if problem == "MDS" or problem == "MIDS":
        if goal == "D":
            model.setObjective(D.sum(), gp.GRB.MINIMIZE)
        elif goal == "J":
            model.setObjective(
                sum(sum(A_[i, j] * D[k] for j, k in enumerate(G.nodes)) for i in range(n)), gp.GRB.MINIMIZE
            )
    else:
        model.setObjective(D.sum(), gp.GRB.MAXIMIZE)

    # Start the optimization.
    start = time.perf_counter()
    model.optimize()
    end = time.perf_counter()

    solution = []
    if model.getAttr("SolCount") >= 1:
        for vertex in G.nodes:
            if D[vertex].X > 0.5:
                solution.append(vertex)

    details = dict(
        goal=goal,
        goal_value=model.getAttr("ObjVal"),
        lenD=len(solution),
        valJ=sum(sum(A_[i, j] * D[k].X for j, k in enumerate(G.nodes)) for i in range(n)),
        #    A=A,
        d=solution,
    )

    return solution, (end - start) * 1000, details


if __name__ == "__main__":
    main()
