import time

import gurobipy as gp
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def main():
    ## Select one of the existing NetworkX graphs or import it from saved ones.
    # G = nx.grid_graph(dim=[10, 10])
    # G = nx.cycle_graph(100)
    # G = nx.star_graph(5)
    G = nx.petersen_graph()
    # G = nx.dodecahedral_graph()
    # G = nx.connected_watts_strogatz_graph(100, 10, 0.5)
    # from saved_graphs import G
    goal = 'MIDS'

    solution, elapsed = find_MIDS(G, goal)

    # Print and draw the solution.
    print("====================")
    if solution:
        print("Found MIDS solution with {} nodes in {} seconds.".format(len(solution), elapsed))
        color_map = ['red' if x in solution else 'darkgray' for x in G.nodes]
        nx.draw(G, with_labels=True, font_weight='bold', node_color=color_map, node_size=1000)
        plt.show()
    else:
        print("No solution found.")


def find_MIDS(G, goal, outputFlag=1):
    A = nx.to_numpy_array(G)
    n = np.size(A, 1)
    A_ = A + np.eye(n)

    # Set up environment.
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", outputFlag)
    env.start()

    # Create a new model.
    model = gp.Model('MIDS', env=env)
    model.setParam('presolve', 0)

    # Add variables to the model.
    V = model.addVars(G.nodes, vtype=gp.GRB.BINARY)

    # Add constraints.
    if goal == 'MDS' or goal == 'MIDS':
        con1 = model.addConstrs((sum([A_[i, j] * V[k] for j, k in enumerate(G.nodes)]) >= 1 for i in range(n)), name='Con1')
    if goal == 'MIS' or goal == 'MIDS':
        con2 = model.addConstrs((V[edge[0]] + V[edge[1]] <= 1 for edge in G.edges), name='Con2')

    # Set the objective
    if goal == 'MDS' or goal == 'MIDS':
        model.setObjective(V.sum(), gp.GRB.MINIMIZE)
    else:
        model.setObjective(V.sum(), gp.GRB.MAXIMIZE)

    # Start the optimization.
    start = time.time()
    model.optimize()
    end = time.time()

    solution = []
    if model.getAttr('SolCount') >= 1:
        for vertex in G.nodes:
            if V[vertex].X:
                solution.append(vertex)

    return solution, end - start


if __name__ == '__main__':
    main()