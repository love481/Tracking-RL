function actor = build_actor_SAC(obsInfo, actInfo)
%build_actor Setups RL Actor
    commonPath = [
    featureInputLayer(prod(obsInfo.Dimension),Name="netObsIn")
    fullyConnectedLayer(512)
    reluLayer(Name="CommonRelu")
    ];

    % Define path for mean value
    meanPath = [
        fullyConnectedLayer(128,Name="meanIn")
        % reluLayer
        % fullyConnectedLayer(64)
        reluLayer
        fullyConnectedLayer(prod(actInfo.Dimension),Name="MeanOut")
        % scalingLayer(Name="MeanOut", ...
        % Scale=actInfo.UpperLimit) 
        ];

    % Define path for standard deviation
    stdPath = [
        fullyConnectedLayer(128,Name="stdIn")
        % reluLayer
        % fullyConnectedLayer(64)
        reluLayer
        fullyConnectedLayer(prod(actInfo.Dimension))
        softplusLayer(Name="StandardDeviationOut")
        ];



    %% LSTM approach
    % commonPath = [
    % sequenceInputLayer(prod(obsInfo.Dimension),Name="netObsIn")
    % fullyConnectedLayer(256)
    % lstmLayer(4)
    % reluLayer(Name="CommonRelu")
    % ];
    % 
    % % Define path for mean value
    % meanPath = [
    %     fullyConnectedLayer(128,Name="meanIn")
    %     reluLayer
    %     fullyConnectedLayer(prod(actInfo.Dimension),Name="MeanOut")
    %     % scalingLayer(Name="MeanOut", ...
    %     % Scale=actInfo.UpperLimit) 
    %     ];
    % 
    % % Define path for standard deviation
    % stdPath = [
    %     fullyConnectedLayer(128,Name="stdIn")
    %     reluLayer
    %     fullyConnectedLayer(prod(actInfo.Dimension))
    %     softplusLayer(Name="StandardDeviationOut")
    %     ];


    
    % Add layers to layerGraph object 
    actorNet = layerGraph(commonPath);
    actorNet = addLayers(actorNet,meanPath);
    actorNet = addLayers(actorNet,stdPath);
    
    % Connect layers
    actorNet = connectLayers(actorNet,"CommonRelu","meanIn/in");
    actorNet = connectLayers(actorNet,"CommonRelu","stdIn/in");


    
    % Convert to dlnetwork and display the number of weights.
    actorNet = dlnetwork(actorNet);
    actor = rlContinuousGaussianActor(actorNet, obsInfo, actInfo, ...
    ActionMeanOutputNames="MeanOut",...
    ActionStandardDeviationOutputNames="StandardDeviationOut",...
    ObservationInputNames="netObsIn",UseDevice="gpu");
end

