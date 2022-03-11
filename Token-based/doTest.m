% Same as test.m but in form of a function.

function [outcome, G, text] = doTest(G)

% Find MIDS using brute force search.
MIDS = findMIDS(G);

% Find all IDSs.
IDSs = findEveryIDS(G);

max_token_mids = 0;
min_token_other = 1000;

m = makeTransition(G, MIDS);
avg = sum(m) / (G.numnodes - length(MIDS));

IDS_field = [sprintf('MIDS: ['), sprintf('%g, ', MIDS(1:end-1)), sprintf('%g]', MIDS(end))];
token_field = [sprintf('Tokens: ['), sprintf('%g, ', m(1:end-1)), sprintf('%g]', m(end))];
avg_field = sprintf('     Avg: %5.2f', avg);
text = sprintf('%-20s %s %s\n', IDS_field, token_field, avg_field);

for i=1:size(IDSs, 1)
    m = makeTransition(G, IDSs{i});
    avg = sum(m) / (G.numnodes - length(IDSs{i}));
    
    if (length(IDSs{i}) == length(MIDS))
        max_token_mids = max(max_token_mids, avg);
    else
        min_token_other = min(min_token_other, avg);
    end
    
    IDS_field = [sprintf('IDS: ['), sprintf('%g, ', IDSs{i}(1:end-1)), sprintf('%g]', IDSs{i}(end))];
    token_field = [sprintf('Tokens: ['), sprintf('%g, ', m(1:end-1)), sprintf('%g]', m(end))];
    avg_field = sprintf('     Avg: %5.2f', avg);
    text = [text, sprintf('%-20s %s %s\n', IDS_field, token_field, avg_field)];
end

if (max_token_mids < min_token_other) || (size(IDSs, 1) == 1)
%     fprintf('Maximum average of MIDS is smaller than the minimum average of other IDS.\n\n');
    outcome = 1;
else
%     fprintf('Maximum average of MIDS is NOT smaller than the minimum average of other IDS.\n\n');
    outcome = 0;
end

end