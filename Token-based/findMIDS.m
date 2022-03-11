function [MIDS] = findMIDS(G)

    MIDS = [];
    n = G.numnodes;
    
    % Make a power set.
    x = 1:n;
    for nn = 1:n
        combnz{nn} = combnk(x,nn);
    end

    for i=1:n
        for j=1:size(combnz{i}, 1)
            test_nodes = combnz{i}(j,:);
            if isIDS(G, test_nodes)
                MIDS = test_nodes;
                return
            end
        end
    end

end