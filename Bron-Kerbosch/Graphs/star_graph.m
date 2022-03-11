function [A, G] = star_graph(n)
%STAR_GRAPH Create a star graph with n nodes.
%   Return the adjacency matrix and graph object.

A = zeros(n);
A(1, 2:n) = ones(1, n-1);
A(2:n, 1) = ones(n-1, 1);
G = graph(A);

end

