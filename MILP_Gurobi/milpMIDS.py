import time
import math

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
    # G = nx.connected_watts_strogatz_graph(100, max(int(math.sqrt(100)), 2), 0.5)
    # from saved_graphs import G
    problem = 'MIDS'

    solution, elapsed, dets = optimize(G, problem)

    # Print and draw the solution.
    print("====================")
    if solution:
        print("Found MIDS solution with {} nodes in {} seconds.".format(len(solution), elapsed))
        print(dets)
        color_map = ['red' if x in solution else 'darkgray' for x in G.nodes]
        nx.draw(G, pos=nx.kamada_kawai_layout(G), with_labels=True,
                font_weight='bold', node_color=color_map, node_size=1000)
        plt.show()
    else:
        print("No solution found.")


def optimize(G, problem, goal='D', outputFlag=1):
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
    D = model.addVars(G.nodes, vtype=gp.GRB.BINARY)

    # Add constraints.
    if problem == 'MDS' or problem == 'MIDS':
        con1 = model.addConstrs((sum([A_[i, j] * D[k] for j, k in enumerate(G.nodes)]) >= 1 for i in range(n)), name='Con1')
    if problem == 'MIS' or problem == 'MIDS':
        con2 = model.addConstrs((D[edge[0]] + D[edge[1]] <= 1 for edge in G.edges), name='Con2')

    # Set the objective
    if problem == 'MDS' or problem == 'MIDS':
        if goal == 'D':
            model.setObjective(D.sum(), gp.GRB.MINIMIZE)
        elif goal == 'J':
            model.setObjective(sum(sum(A_[i, j] * D[k] for j, k in enumerate(G.nodes)) for i in range(n)), gp.GRB.MINIMIZE)
    else:
        model.setObjective(D.sum(), gp.GRB.MAXIMIZE)

    # Start the optimization.
    start = time.time()
    model.optimize()
    end = time.time()

    solution = []
    if model.getAttr('SolCount') >= 1:
        for vertex in G.nodes:
            if D[vertex].X > 0.5:
                solution.append(vertex)

    details = dict(goal=goal,
                   goal_value=model.getAttr('ObjVal'),
                   lenD=len(solution),
                   valJ=sum(sum(A_[i, j] * D[k].X for j, k in enumerate(G.nodes)) for i in range(n)))

    return solution, end - start, details


if __name__ == '__main__':
    main()
