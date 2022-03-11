#!/usr/bin/env python3
import sys, os
import numpy as np
import scipy as sp
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

#file_location = Path(os.path.join('%s/data/interesting_graphs.txt' % (os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))))
#if file_location.exists():
#	os.remove(file_location)
#f = open(file_location,"w+")

funcs = []
funcs.append(lambda n,p,m : nx.random_regular_graph(3,n))
funcs.append(lambda n,p,m : nx.newman_watts_strogatz_graph(n, 4, p))
funcs.append(lambda n,p,m : nx.erdos_renyi_graph(n, p))
funcs.append(lambda n,p,m : nx.gnm_random_graph(n, m))
funcs.append(lambda n,p,m : nx.binomial_graph(n, p))

n = 30 #Num of nodes
m = 100 #Num of edges
p = 0.1 #probability of creating an edge

for fx in funcs:
    c = False
    while not c:
        G = fx(n,p,m)
        c = nx.is_connected(G)
        print("G connectivity:", c ,"G Nodes:",G.nodes, "G Edges:", G.edges)
    nx.draw(G, with_labels=True, font_weight='bold',node_color="darkgray", node_size=1000)
    plt.show()
        
    #f.write("G Nodes: %s, \n G Edges: %s \n \n" %(G.nodes, G.edges))
#f.close()
