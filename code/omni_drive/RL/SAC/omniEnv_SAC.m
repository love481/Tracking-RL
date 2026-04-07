classdef omniEnv_SAC < rl.env.MATLABEnvironment
    %OMNIDRIVEENV: Template for defining custom environment in MATLAB.    
    
    %% Properties (set properties' attributes accordingly)
    properties
        % Specify and initialize environment's necessary properties    
        % Max input the input can apply
        MaxInput= 30;
        connection;
        ref;
        ref_path;
        N;
               
        % Sample time
        Ts;
        r;
        n=1;
        
        % Angle at which to fail the episode (radians)
        AngleThreshold = 0.5; % pi corresponding to maximum angle
        
        % Distance at which to fail the episode
        DisplacementThreshold = 4.0;
        ObjectDiam = 0.30;
        
        
        % Penalty when the robot fails to follow trajectory
        RewardForNotFollowing = -0.6;
    end
    
    properties
        % Initialize system state [ex, ey, cos_etheta, sin_etheta, x_dot, y_dot, theta_dot, ranges]'
        State = zeros(17,1)
    end
    
    properties(Access = protected)
        % Initialize internal flag to indicate episode termination
        IsDone = false;
        coupling_matrix;
        
        % Handle to figure
        %Figure
    end

    %% Necessary Methods
    methods              
        % Contructor method creates an instance of the environment
        % Change class name and constructor name accordingly
        function this = omniEnv_SAC(connection,sampleTime,coupling_matrix,ref)
            % Initialize Observation settings
            numObs = 17;
            ObservationInfo = rlNumericSpec([numObs 1]);
            ObservationInfo.Name = 'Robot states';
            ObservationInfo.Description = 'ex, ey, cos_etheta, sin_etheta, x_dot, y_dot, theta_dot, ranges';
            
            % Initialize Action settings
            numAct = 4;
            ActionInfo = rlNumericSpec([numAct 1]);
            ActionInfo.Name = 'Robot Action';
            
            % The following line implements built-in functions of RL env
            this = this@rl.env.MATLABEnvironment(ObservationInfo,ActionInfo);
            this.connection=connection;
            this.ref=ref;
            this.N=numel(this.ref(:,1));
            this.coupling_matrix=coupling_matrix;
            this.Ts=sampleTime;
            this.r=rateControl(1/this.Ts);
            % Initialize property values and pre-compute necessary values
            updateActionInfo(this);
        end
        
        % Apply system dynamics and simulates the environment with the 
        % given action for one step.
        function [Observation,Reward,IsDone,LoggedSignals] = step(this,Action)
            this.n=this.n+1;
            LoggedSignals = [];
            w=Action;
            omni_setWheelSpeeds(this.connection,w(1),w(2),w(3),w(4));
            waitfor(this.r);
            [x, y, theta] = omni_getPose(this.connection);
            [m1,m2,m3,m4] = omni_getWheelSpeeds(this.connection);
            f_v=this.coupling_matrix\[m1;m2;m3;m4];
            f_v = f_v/1.5;
            %fprintf('%f %f %f\n',f_v(1), f_v(2),f_v(3));
            theta=normalizeAngle(double(theta));
            x = double(x);
            y = double(y);
            theta = double(theta);
            [x_lid,y_lid]=omni_getLaserData(this.connection);
            x_lid=double(x_lid);
            y_lid=double(y_lid);
            dr=sqrt(x_lid.^2+y_lid.^2);
            topk_nearest_object_range = mink(dr,5);
            nearest_object_range=min(topk_nearest_object_range);
            ranges=[];
            for i=1:5
            [r c]=find(dr== topk_nearest_object_range(i));
            nearest_object_angle=normalizeAngle(atan2(y_lid(c),x_lid(c)))/pi;
           % ranges=[x_lid(c);y_lid(c);nearest_object_angle;ranges];
            ranges=[topk_nearest_object_range(i)/6.0;nearest_object_angle;ranges];
            end
            
            % Update system states
            ex= this.ref_path(this.n,1)-x;
            ey= this.ref_path(this.n,2)-y;
            etheta=normalizeAngle(this.ref_path(this.n,3)-theta);
            etheta_x=cos(this.ref_path(this.n,3))-cos(theta);
            etheta_y=sin(this.ref_path(this.n,3))-sin(theta);
            

            Observation = [ex;ey;etheta_x;etheta_y;double(f_v(1));double(f_v(2));double(f_v(3));ranges];
            this.State = Observation;
            % Check terminal condition
            IsDone = nearest_object_range < this.ObjectDiam || abs(ex) > this.DisplacementThreshold || abs(ey) > this.DisplacementThreshold || abs(etheta)> this.AngleThreshold || this.n >=this.N;
            this.IsDone = IsDone;
            
            % Get reward
            %fprintf('%f %f %f %f %f %f\n',x,y,theta,this.ref_path(this.n,1),this.ref_path(this.n,2),this.n);
            Reward = getReward(this,abs(ex),abs(ey),abs(etheta_x),abs(etheta_y),nearest_object_range);
            %fprintf('R: %f\n',Reward);
        end
        
        % Reset environment to initial state and output initial observation
        function InitialObservation = reset(this)
            simulation_stop(this.connection);
            simulation_start(this.connection);
            [x_lid,y_lid]=omni_getLaserData(this.connection);
            x_lid=double(x_lid);
            y_lid=double(y_lid);
            dr=sqrt(x_lid.^2+y_lid.^2);
            topk_nearest_object_range = mink(dr,5);
            ranges=[];
            for i=1:5
            [r c]=find(dr== topk_nearest_object_range(i));
            nearest_object_angle= normalizeAngle(atan2(y_lid(c),x_lid(c)))/pi;
            ranges=[topk_nearest_object_range(i)/6.0;nearest_object_angle;ranges];
            end
            [x, y ,theta] = omni_getPose(this.connection);
            x = double(x);
            y = double(y);
            theta = double(theta);

            total_ref_paths = size(this.ref,2)/3;
            path_index = randi(total_ref_paths,1);
            this.ref_path = this.ref(:,(path_index*3)-2:path_index*3);
            ex= this.ref_path(this.n,1)-x;
            ey= this.ref_path(this.n,2)-y;
            etheta_x=cos(this.ref_path(this.n,3))-cos(theta);
            etheta_y=sin(this.ref_path(this.n,3))-sin(theta);
            InitialObservation = [ex;ey;etheta_x;etheta_y;double(0.0);double(0.0);double(0.0);ranges];
            this.State = InitialObservation;
            
        end
    end
    %% Optional Methods (set methods' attributes accordingly)

    methods               
        % Helper methods to create the environment
        % update the action info based on max velocity
        function updateActionInfo(this)
            this.ActionInfo.LowerLimit = -this.MaxInput;
            this.ActionInfo.UpperLimit = this.MaxInput;
        end
        
        % Reward function
        function Reward = getReward(this,ex,ey,etheta_x,etheta_y,dr) 
	        if ~this.IsDone
		        %obstacle_penalty = exp(-(dr-0.9) * 10);  % Encourage keeping distance
		        tracking_reward = (exp(-ex*5) + exp(-ey*5) + exp(10*(-etheta_x-etheta_y))) / 3;
		        %progress_reward = -sqrt(ex^2 + ey^2);  % Encourage trajectory progress
                %Timely_reward = this.n/this.N;
                Reward = tracking_reward;
		        %Reward = tracking_reward - 0.5 * obstacle_penalty;
	        else
		        simulation_stop(this.connection);
                % progress_reward = -sqrt(ex^2 + ey^2);
		        % Reward = progress_reward/1.41;
                Reward =  this.RewardForNotFollowing/2;
		        if dr < this.ObjectDiam
		            Reward = this.RewardForNotFollowing;
		        end
		        if this.n >= this.N
		            Reward = (exp(-ex*5) + exp(-ey*5) + exp(10*(-etheta_x-etheta_y))) / 3;
		        end
		        this.n = 1;
	        end       
        end
     end
end
