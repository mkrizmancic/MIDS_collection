% Try the hypothesis a bunch of times and display results.
% If a test fails, display details.

num_trials = 100;
num_nodes  = 6;
num_passes = 0;

for i=1:num_trials
    G = randSimpleGraph(num_nodes);
    [out, G, msg] = doTest(G);
    
    num_passes = num_passes + out;
    if (out == 0)
        fprintf('%s\n', msg);
        figure;
        plot(G)
    end
end

fprintf('The test passed in %d/%d [%.2f %%] cases.\n\n', num_passes, num_trials, num_passes/num_trials*100) 
    