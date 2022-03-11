function [r, dominated] = isIDS(G, test_nodes)

    dominated = [];
    other_nodes = setdiff(1:G.numnodes, test_nodes);

    for n=test_nodes
        dominated = union(dominated, neighbors(G, n));
    end

    r = isempty(setdiff(other_nodes,dominated));
    r = r && isempty(setdiff(dominated, other_nodes));
end