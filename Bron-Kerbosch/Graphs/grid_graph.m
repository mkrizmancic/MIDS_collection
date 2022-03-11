function [A,G] = grid_graph(a,b)
%GRID_GRAPH Create a grid graph with size axb.
%   Return the adjacency matrix and graph object.
%   https://stackoverflow.com/questions/16329403/how-can-you-make-an-adjacency-matrix-which-would-emulate-a-2d-grid

    n = a * b;
    A = zeros(n,n);
    for r=0:a-1
        for c=0:b-1
            i = r*b + c;
            % Two inner diagonals
            if (c > 0)
                A(i,i+1) = 1;
                A(i+1,i) = 1;
            end
            % Two outer diagonals
            if (r > 0)
                A(i-b+1,i+1) = 1;
                A(i+1,i-b+1) = 1;
            end
        end
    end
    
G = graph(A);

end

