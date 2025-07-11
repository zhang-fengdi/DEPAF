clear all;
close all;

% Load DEPAF source code:
addpath(genpath('..\..\DEPAF_src_replication'));

for SNR = {'Low', 'Medium', 'High'}
    for rate = [1 2 3]
        predResDir = dir(['..\models\predRes*' num2str(rate) '_' SNR{1} '*.mat']);
        for i = 1:length(predResDir)

            % Load prediction results:
            predResName = predResDir(i).name;
            predResPath = fullfile(predResDir(i).folder, predResDir(i).name);
            predResData = load(predResPath, 'loc', 'modelID');
            predLoc = predResData.loc;
            modelID = predResData.modelID;

            % Load ground truth:
            GTDataName = ['data_rate_' num2str(rate) '_' SNR{1} '_SNR_with_GT.mat'];
            GTPath = ['..\datasets\' GTDataName];
            GTData = load(GTPath,'spikeLocGT');
            GTLoc = GTData.spikeLocGT;

            % Evaluate prediction results:
            traceNum = length(GTLoc);
            progressDisp(traceNum); % Initialize progress bar
            dists = zeros(traceNum,1);
            ERs = zeros(traceNum,1);
            for j = 1:traceNum
                singleGTLoc = GTLoc{j};
                singlePredLoc = predLoc{j}(:,2) - 150; % shift from spike center to spike onset
                singlePredLoc = singlePredLoc - 1; % switch to 0-based indexing
                dt = 0.025; % 40Hz acquisition
                singlePredLoc = singlePredLoc * dt;
                [dists(j), ERs(j)] = perfEvalSpikeInfer(singleGTLoc, singlePredLoc);

                % Refresh progress bar:
                progressDisp(0);
            end

            % Terminate progress bar:
            progressDisp(-1);

            % Compute overall evaluation metrics:
            dist = mean(dists);
            ER = mean(ERs);

            % Save results:
            result_filename = ['..\models\EvalRes_data_rate_' num2str(rate) ...
                '_' SNR{1} '_SNR_by_Mdl_' modelID '.mat'];
            save(result_filename, ...
                'dist', 'ER', ...
                'modelID', ...
                'predResName', 'GTDataName');
        end
    end
end