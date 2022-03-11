function [m, W] = makeTransition(G, nodes)

    W = zeros(G.numnodes, G.numedges);
    tau = zeros(G.numedges, 1);
    A = adjacency(G);
    k = 1;
    
    % Create some form of incidence matrix. Not sure how I came up with this.
    for n=nodes
        for i=1:size(A, 1)
            if A(n, i) == 1
%                 W(n, k) = -1;
                W(i, k) = 1;
                tau(k) = 1;
                k = k + 1;
            end
        end
    end
    
    % Vector m is equivalent to number messages each node received.
    m = W * tau;
end