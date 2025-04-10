clear all;
close all;

% Load DEPAF source code:
addpath(genpath('..\..\DEPAF_src_replication'));

for density = [3775 7550 18874 33974 56623 113246]
    for int = [1000 7000 40000]
        mdlDir = dir(['..\models\Mdl_*' num2str(int) '_' num2str(density) '*.mat']);
        for i = 1:length(mdlDir)
            modelPath = [mdlDir(i).folder '\' mdlDir(i).name]; % Path of the prediction model
            dataPath = ['..\datasets\data_' num2str(int) '_' num2str(density) '_PerlinBG.tif']; % Path of the prediction data

            batchSize = 10; % Batch size for single prediction
            patchSize = [256 256]; % Patch size for block prediction (must be a power of 2)
            patchStride = [128 128]; % Stride for block prediction
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