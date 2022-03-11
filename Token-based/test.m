% rng(1)


n = 5;
G = randSimpleGraph(n);
p = plot(G);

% Find MIDS using brute force search.
MIDS = findMIDS(G);
highlight(p,MIDS,'NodeColor','r')

% Find all IDSs.
IDSs = findEveryIDS(G);

max_token_mids = 0;
min_token_other = 1000;

for i=1:size(IDSs, 1)
    m = makeTransition(G, IDSs{i});  % Simulate that IDS nodes are sending a message to their neighbor.
    avg = sum(m) / (G.numnodes - length(IDSs{i}));  % This is our hypotethical measure of how many messages are received.
    
    if (length(IDSs{i}) == length(MIDS))
        max_token_mids = max(max_token_mids, avg);
    else
        min_token_other = min(min_token_other, avg);
    end
    
    
    IDS_field = [sprintf('IDS: ['), sprintf('%g, ', IDSs{i}(1:end-1)), sprintf('%g]', IDSs{i}(end))];
    token_field = [sprintf('Tokens: ['), sprintf('%g, ', m(1:end-1)), sprintf('%g]', m(end))];
    avg_field = sprintf('     Avg: %5.2f', avg);
    fprintf('%-20s %s %s\n', IDS_field, token_field, avg_field);
end

if (max_token_mids < min_token_other)
    fprintf('Maximum average of MIDS is smaller than the minimum average of other IDS.\n\n');
else
    fprintf('Maximum average of MIDS is NOT smaller than the minimum average of other IDS.\n\n');
end