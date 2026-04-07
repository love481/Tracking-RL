% Setting trajectory generation parameters
%% Create agent folder for saved agents
agent_folder = 'Agents_trained';
if ~exist(agent_folder, 'dir')
    mkdir(agent_folder)
end
sampleTime = 0.1;
time_allocated= 30; %in s
Total_error_eucl_dist_x_y = 0;
Total_error_theta = 0;
Total_success = [];
Num_sim_steps = 100;
% Randomly generated trajectory saved and loaded
ref=load('random_path.mat').list_paths_to_track;
for num_sim=1:Num_sim_steps
    total_ref_paths = size(ref,2)/3;
    path_index = randi(total_ref_paths,1);
    ref_path = ref(:,(path_index*3)-2:path_index*3);
    
    connection = simulation_openConnection(simulation_setup(), 0);
    omni_init(connection);
    bodyDiameter=1.000;
    wheelDiameter=0.100;
    coupling_matrix=(2/wheelDiameter)*[-0.7071  0.7071 -bodyDiameter/2; ...
                    0.7071  0.7071 -bodyDiameter/2; ...
                    0.7071  -0.7071 -bodyDiameter/2; ...
                    -0.7071  -0.7071 -bodyDiameter/2];
    % Creating Environment
    env = omniEnv_SAC(connection,sampleTime,coupling_matrix,ref_path);
    steps_per_episode = ceil((time_allocated)/sampleTime);
    agent_folder = 'Agents_trained';
    saved_agent_name = strcat('agent_', 'SAC_static_complex_scratch.mat');
    SaveAgentDirectory = fullfile(agent_folder,saved_agent_name);
    
    % % Training Agent in Environment
    trainStats = rlTrainingOptions(...
        'MaxEpisodes',2000,...
        'MaxStepsPerEpisode',steps_per_episode,...
        'StopTrainingCriteria',"AverageReward",...
        "ScoreAveragingWindowLength",10,...
        'SaveAgentDirectory', SaveAgentDirectory,...
        'StopTrainingValue',2000);
    load(trainStats.SaveAgentDirectory,'agent')
    
    
    simOpts = rlSimulationOptions('MaxSteps',steps_per_episode);
    
    experience = sim(env,agent, simOpts);
    pose=experience.Observation.RobotStates.Data(1:4,:);
    pose=pose';
    pose_size = size(pose, 1);
    error_eucl_dist_x_y=mean(sqrt(sum(pose(1:end,1:2) .^ 2,2)));
    error_theta=mean(abs(atan(pose(1:end,3)./pose(1:end,4))*2));
    success_r = sqrt(sum(pose(end,1:2).^ 2));
    fprintf('%f %f  %f  %f\n',num_sim ,error_eucl_dist_x_y, error_theta, success_r);
    Total_error_eucl_dist_x_y = Total_error_eucl_dist_x_y + error_eucl_dist_x_y;
    Total_error_theta = error_theta + Total_error_theta;
    Total_success = [success_r;Total_success];
end
Total_success_rate = sum(Total_success<1.0);
fprintf('%f  %f  %f\n',Total_error_eucl_dist_x_y/Num_sim_steps, Total_error_theta/Num_sim_steps, Total_success_rate/Num_sim_steps);