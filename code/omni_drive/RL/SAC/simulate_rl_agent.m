% Setting trajectory generation parameters
%rng(0);
resolution=50/10;
% %% Define a small map
bodyR=1.5/2;
sampleTime = 0.1;
map = false(50,50);
map(floor((3-bodyR)*resolution):floor((3+bodyR)*resolution),1:floor((4+bodyR)*resolution))=true;
map(floor((3-bodyR)*resolution):floor((3+bodyR)*resolution),floor((6-bodyR)*resolution):50)=true;
start_coords = [resolution,resolution];
dest_coords  = [45,5];

%% find the path_way from A* algorithm
[robot_pose_xy,p,q]= AStarGrid (map, start_coords, dest_coords);
robot_pose=zeros(size(robot_pose_xy,1),3);

robot_pose(:,1:2)=robot_pose_xy/resolution;
% for i=2:size(robot_pose_xy,1)
%     robot_pose(i,3)=normalizeAngle(atan2(robot_pose_xy(i,2)-robot_pose_xy(i-1,2),robot_pose_xy(i,1)-robot_pose_xy(i-1,1)));
% end
for i=1:size(robot_pose_xy,1)
    robot_pose(i,3)=0.0;
end
time_allocated=30; %in s
times =((cumsum(ones(size(robot_pose_xy,1),1))-1)/(size(robot_pose_xy,1)-1))*time_allocated;
tVec = 0.0:sampleTime:times(end);      % Time array
% ref = interp1(times,robot_pose,tVec,'spline'); % robot_pose=Function(times)
[ref, d_ref] = bsplinepolytraj(robot_pose',[tVec(1) tVec(end)],tVec);
ref=ref';

connection = simulation_openConnection(simulation_setup(), 0);
omni_init(connection);
bodyDiameter=1.000;
wheelDiameter=0.100;
coupling_matrix=(2/wheelDiameter)*[-0.7071  0.7071 -bodyDiameter/2; ...
                0.7071  0.7071 -bodyDiameter/2; ...
                0.7071  -0.7071 -bodyDiameter/2; ...
                -0.7071  -0.7071 -bodyDiameter/2];
% Creating Environment
env = omniEnv_SAC(connection,sampleTime,coupling_matrix,ref);
steps_per_episode = ceil((time_allocated)/sampleTime);
agent_folder = 'Agents_trained';
saved_agent_name = strcat('agent_', 'SAC_static_complex.mat');
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
fprintf('%f  %f  %f\n',error_eucl_dist_x_y, error_theta, success_r);