function [A, G] = circle_graph(n)
%CIRCLE_GRAPH Create a circle graph with n nodes.
%   Return the adjacency matrix and graph object.

A = diag(ones(n-1, 1), 1) + diag(ones(n-1, 1), -1);
A(1,n) = 1;
A(n,1) = 1;
G = graph(A);

end

