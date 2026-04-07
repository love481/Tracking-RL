function [critic1,critic2] = build_critic_SAC(obsInfo, actInfo)
%build_critic Setups RL critic
    % Observation path
    obsPath = [
        featureInputLayer(prod(obsInfo.Dimension),Name="obsPathIn")
        fullyConnectedLayer(512)
        reluLayer
        fullyConnectedLayer(128,Name="obsPathOut")
        ];
    
    % Action path
    actPath = [
        featureInputLayer(prod(actInfo.Dimension),Name="actPathIn")
        fullyConnectedLayer(128)
        reluLayer
        fullyConnectedLayer(64,Name="actPathOut")
        ];

    % Common path
    commonPath = [
        concatenationLayer(1,2,Name="concat")
        % reluLayer
        fullyConnectedLayer(128)
        reluLayer
        fullyConnectedLayer(1)
        ];

    %% LSTM part
    % obsPath = [
    %     sequenceInputLayer(prod(obsInfo.Dimension),Name="obsPathIn")
    %     fullyConnectedLayer(256)
    %     reluLayer
    %     fullyConnectedLayer(128,Name="obsPathOut")
    %     ];
    % 
    % % Action path
    % actPath = [
    %     sequenceInputLayer(prod(actInfo.Dimension),Name="actPathIn")
    %     fullyConnectedLayer(256)
    %     reluLayer
    %     fullyConnectedLayer(128,Name="actPathOut")
    %     ];
    % 
    % % Common path
    % commonPath = [
    %     concatenationLayer(1,2,Name="concat")
    %     lstmLayer(4)
    %     reluLayer
    %     fullyConnectedLayer(1)
    %     ];

    
    % Add layers to layergraph object
    criticNet = layerGraph;
    criticNet = addLayers(criticNet,obsPath);
    criticNet = addLayers(criticNet,actPath);
    criticNet = addLayers(criticNet,commonPath);
    
    % Connect layers
    criticNet = connectLayers(criticNet,"obsPathOut","concat/in1");
    criticNet = connectLayers(criticNet,"actPathOut","concat/in2");
    criticNet1 = dlnetwork(criticNet);
    criticNet2 = dlnetwork(criticNet);
    critic1 = rlQValueFunction(criticNet1,obsInfo,actInfo,ObservationInputNames="obsPathIn",ActionInputNames="actPathIn",UseDevice="gpu");
    critic2 = rlQValueFunction(criticNet2,obsInfo,actInfo,ObservationInputNames="obsPathIn",ActionInputNames="actPathIn",UseDevice="gpu");

end

