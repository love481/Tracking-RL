% Set image folder and create if it doesn't exist
image_folder = './Images';
if ~exist(image_folder, 'dir')
    mkdir(image_folder)
end

% Load agent training data
agent = 'SAC_static';
data = load(strcat('Agents_new\trainStats_', agent));
trainStats = data.trainStats;

% Select number of episodes to display
num_episodes = 900;

% Create figure
figure()

% Plot raw episode rewards (light, thin)
plot(trainStats.EpisodeIndex(1:num_episodes), trainStats.EpisodeReward(1:num_episodes),...
    '-', 'Color', [0.7 0.9 0.95], 'LineWidth', 0.5)

hold on

% Plot average reward (dark, bold)
hAvg = plot(trainStats.EpisodeIndex(1:num_episodes), trainStats.AverageReward(1:num_episodes),...
    '-', 'Color', [0 0.2 0.6], 'LineWidth', 2.5, 'DisplayName', 'Average Reward');

hold off

% Add labels and legend
xlabel('Episode Number', 'FontSize', 12)
ylabel('Reward', 'FontSize', 12)
legend(hAvg, 'Average Reward', 'Location', 'northwest', 'FontSize', 10)
grid on
set(gca, 'FontSize', 11)

% Save figure
set(gcf, 'Position', [50, 50, 1200, 400])
image_file = strcat(agent, '.png');
image_save_path = fullfile(image_folder, image_file);
saveas(gcf, image_save_path)