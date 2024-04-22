import math

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from milpMIDS import optimize


def main():
    repeat = 100
    num_nodes = 6
    count = 0
    avg_dur = [0, 0]

    for i in range(repeat):
        print(f"Running experiment {i}...", end=' ')
        G = nx.connected_watts_strogatz_graph(num_nodes, max(int(math.sqrt(num_nodes)), 2), 0.5)
        solD, elpsD, detD = optimize(G, 'MIDS', goal='D', outputFlag=0)
        solJ, elpsJ, detJ = optimize(G, 'MIDS', goal='J', outputFlag=0)
        if solD and solJ and len(solD) == len(solJ):
        # if solD and solJ and detD['goal_value'] <= detJ['goal_value']:
            print('SUCCESS')
            count += 1
        else:
            print('FAILED')
            print(f"\t Goal 'D' solution: {len(solD)}")
            print(f"\t Goal 'J' solution: {len(solJ)}")
            print(f"\t {detD}")
            print(f"\t {detJ}")
            print(nx.to_numpy_array(G))

        avg_dur[0] += elpsD
        avg_dur[1] += elpsJ

    for i in range(len(avg_dur)):
        avg_dur[i] /= count

    print(f"Results are the same in {count}/{repeat} experiments.")
    print(f"Average durations are: {avg_dur}")


if __name__ == '__main__':
    main()
