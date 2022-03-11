function [IDS] = findEveryIDS(G)

    IDS = {};
    n = G.numnodes;
    
    % Make a power set.
    x = 1:n;
    for nn = 1:n
        combnz{nn} = combnk(x,nn);
    end

    k = 1;
    for i=1:n
        for j=1:size(combnz{i}, 1)
            test_nodes = combnz{i}(j,:);
            if isIDS(G, test_nodes)
                IDS{k, 1} = test_nodes;
                k = k + 1;
            end
        end
    end
    
end