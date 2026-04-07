% Set image folder and create if it doesn't exist
image_folder = './Images';
if ~exist(image_folder, 'dir')
    mkdir(image_folder)
end

% Agent 1 (pretrained or fine-tuned)
agent1 = 'SAC_static_complex';
data1 = load(strcat('Agents_new\trainStats_', agent1));
trainStats1 = data1.trainStats;

% Agent 2 (scratch trained)
agent2 = 'SAC_static_complex_scratch';
data2 = load(strcat('Agents_new\trainStats_', agent2));
trainStats2 = data2.trainStats;

% Select number of episodes to visualize
num_episodes = 900;

% Create figure
figure()
hold on

% -------- Plot Agent 1 --------
% Dim raw episode reward (not in legend)
plot(trainStats1.EpisodeIndex(1:num_episodes), trainStats1.EpisodeReward(1:num_episodes), ...
    '-', 'Color', [0.6 0.85 0.95], 'LineWidth', 0.5)

% Bold average reward (in legend)
h1 = plot(trainStats1.EpisodeIndex(1:num_episodes), trainStats1.AverageReward(1:num_episodes), ...
    '-', 'Color', [0 0.2 0.6], 'LineWidth', 2.5);

% -------- Plot Agent 2 --------
% Dim raw episode reward (not in legend)
plot(trainStats2.EpisodeIndex(1:num_episodes), trainStats2.EpisodeReward(1:num_episodes), ...
    '-', 'Color', [0.95 0.75 0.75], 'LineWidth', 0.5)

% Bold average reward (in legend)
h2 = plot(trainStats2.EpisodeIndex(1:num_episodes), trainStats2.AverageReward(1:num_episodes), ...
    '-', 'Color', [0.5 0 0], 'LineWidth', 2.5);

% Formatting
xlabel('Episode Number', 'FontSize', 12)
ylabel('Reward', 'FontSize', 12)
legend([h1 h2], {'Avg Reward (Curriculum)', 'Avg Reward (Scratch)'}, ...
    'Location', 'northwest', 'FontSize', 10)

grid on
set(gca, 'FontSize', 11)

% Save figure
set(gcf, 'Position', [50, 50, 1200, 400])
image_file = 'comparison_SAC_agents.png';
image_save_path = fullfile(image_folder, image_file);
saveas(gcf, image_save_path)