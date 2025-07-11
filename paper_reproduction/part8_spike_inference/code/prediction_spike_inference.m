clear all;
close all;

% Load DEPAF source code:
addpath(genpath('..\..\DEPAF_src_replication'));

for SNR = {'Low', 'Medium', 'High'}
    for rate = [1 2 3]
        mdlDir = dir(['..\models\Mdl_*' num2str(rate) '_' SNR{1} '*.mat']);
        for i = 1:length(mdlDir)
            modelPath = [mdlDir(i).folder '\' mdlDir(i).name]; % Path of the prediction model
            dataPath = ['..\datasets\data_rate_' num2str(rate) '_' SNR{1} '_SNR_only_I.mat']; % Path of the prediction data

            batchSize = 1000; % Batch size for single prediction
            patchSize = [1024 1]; % Patch size for block prediction (must be a power of 2)
            patchStride = [512 1]; % Stride for block prediction
            resSavePath = '..\models\'; % Path to save prediction results
            outputISyn = false; % Whether to predict and output noiseless synthetic images
            outputBG = false; % Whether to predict and output noiseless background images
            useGPU = true; % Whether to use GPU

            DEPAFPred(modelPath, dataPath, ...
                'batchSize', batchSize, ...
                'patchSize', patchSize, ...
                'patchStride', patchStride, ...
                'resSavePath', resSavePath, ...
                'outputISyn', outputISyn, ...
                'outputBG', outputBG, ...
                'useGPU', useGPU);
        end
    end
end