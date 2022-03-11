function [A, G] = tree_graph(d)
%TREE_GRAPH Create a tree graph with depth d.
%   Return the adjacency matrix and graph object.

n = 2^(d-1) - 1;

edges = [1:n, 1:n; (1:n) * 2, (1:n) * 2 + 1].';
G = graph(table(edges, 'VariableNames',{'EndNodes'}));
A = adjacency(G);

end

