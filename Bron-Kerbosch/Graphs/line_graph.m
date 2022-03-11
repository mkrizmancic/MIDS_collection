function [A, G] = line_graph(n)
%LINE_GRAPH Create a line graph with n nodes.
%   Return the adjacency matrix and graph object.

A = diag(ones(n-1, 1), 1) + diag(ones(n-1, 1), -1);
G = graph(A);

end

