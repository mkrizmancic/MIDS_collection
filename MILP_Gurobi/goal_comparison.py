import math

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from milpMIDS import optimize


def main():
    repeat = 20
    num_nodes = 30
    count = 0
    avg_dur = [0, 0]

    for i in range(repeat):
        print(f"Running experiment {i}...", end=' ')
        G = nx.connected_watts_strogatz_graph(num_nodes, max(int(math.sqrt(num_nodes)), 2), 0.5)
        sol1, elps0, det0 = optimize(G, 'MIDS', goal='D', outputFlag=0)
        sol2, elps1, det1 = optimize(G, 'MIDS', goal='J', outputFlag=0)
        if sol1 and sol2 and len(sol1) == len(sol2):
            print('SUCCESS')
            count += 1
        else:
            print('FAILED')
            print(f"\t Goal 'D' solution: {len(sol1)}")
            print(f"\t Goal 'J' solution: {len(sol2)}")
            print(f"\t {det0}")
            print(f"\t {det1}")

        avg_dur[0] += elps0
        avg_dur[1] += elps1

    for i in range(len(avg_dur)):
        avg_dur[i] /= count

    print(f"Results are the same in {count}/{repeat} experiments.")
    print(f"Average durations are: {avg_dur}")


if __name__ == '__main__':
    main()
