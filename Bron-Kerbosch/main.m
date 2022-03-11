%% Load the graph or generate graph.
% load dogtail.mat

% load petersen.mat         % n=10 https://www.distanceregular.org/graphs/petersen.html
% load 4cube.mat            % n=16 https://www.distanceregular.org/graphs/4cube.html
% load dodecahedron.mat     % n=20 https://www.distanceregular.org/graphs/dodecahedron.html
% load desargues.mat        % n=20 https://www.distanceregular.org/graphs/desargues.html
% load crown12.mat          % n=24 https://www.distanceregular.org/graphs/k12.12-i.html

% load matea_tree.mat
% load matea_montyhall.mat
% load matea_graf9.mat  % 30 nodes, takes a long time

% [A, G] = circle_graph(10);
% [A, G] = line_graph(5);
% [A, G] = star_graph(5);
% [A, G] = tree_graph(4);
[A, G] = grid_graph(5,5);

%% Visualize and inspect the current graph.
% figure(1)
% plot(G)

%% Find MIDS via smallest maximal independent set using Bron-Kerbosch method.
tic;
IDS = BK_MaxIS(A);
elapsed = toc;

i_g = min(sum(IDS, 1));  % Independent domination number
MIDS = IDS(:, sum(IDS, 1) == i_g);  % Minimum independent dominating set as a vector of 1s and 0s
n_MIDS = size(MIDS, 2);

fprintf("Found %d MID sets with %d nodes in %f seconds.\n", n_MIDS, i_g, elapsed);


%% Visualize MIDS.

% If there are more than 20 MID sets, group them for better readability.
groups_of_20 = ceil(n_MIDS / 20);
remainder = mod(n_MIDS, 20);

for j=1:groups_of_20
    % Create subplots.
    figure(j+1)
    if (j ~= groups_of_20 || remainder == 0)
        group_size = 20;
    else
        group_size = remainder;
    end
    s = numSubplots(group_size);
    t = tiledlayout(s(1), s(2));
    title(t, sprintf("MIDS %d to %d", (j-1)*20+1, min(j*20, (groups_of_20-1)*20+remainder)))
    
    % Plot MIDS in this group.
    for i=1:group_size
        nexttile
        p = plot(G);
        highlight(p, find(MIDS(:, (j-1)*20 + i)), 'NodeColor','r');
    end
end

