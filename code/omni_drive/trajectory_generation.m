resolution=50/10;
sampleTime =0.1;
bodyR=1.5/2;
time_allocated=30;
map = false(50,50);
map(floor((3-bodyR)*resolution):floor((3+bodyR)*resolution),1:floor((4+bodyR)*resolution))=true;
map(floor((3-bodyR)*resolution):floor((3+bodyR)*resolution),floor((6-bodyR)*resolution):50)=true;
start_coords = [resolution,resolution];
% Generate 10000 random trajectory
randomx_coordinates = randi([20 45],1,10000);
randomy_coordinates = randi([5 45],1,10000);
randomxy_coordinates = [randomx_coordinates; randomy_coordinates];
list_paths_to_track = [];
for each_des_index = 1:size(randomxy_coordinates,2)
    dest_coords = randomxy_coordinates(:,each_des_index)';
    [robot_pose_xy,p,q]= AStarGrid (map, start_coords, dest_coords);
    robot_pose=zeros(size(robot_pose_xy,1),3);
    robot_pose(:,1:2)=robot_pose_xy/resolution;
    times =((cumsum(ones(size(robot_pose_xy,1),1))-1)/(size(robot_pose_xy,1)-1))*time_allocated;
    tVec = 0.0:sampleTime:times(end);      % Time array
    [ref, d_ref] = bsplinepolytraj(robot_pose',[tVec(1) tVec(end)],tVec);
    ref=ref';
    list_paths_to_track = [list_paths_to_track,ref];
    disp(each_des_index);
end
save('random_path.mat','list_paths_to_track');