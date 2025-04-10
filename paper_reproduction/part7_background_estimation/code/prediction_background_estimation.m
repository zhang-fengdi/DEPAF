clear all;
close all;

% Load DEPAF source code:
addpath(genpath('..\..\DEPAF_src_replication'));

density = 3775;
for int = [1000 7000 40000]
    for scale = [5 6.4 9 13.9 19.3]
        mdlDir = dir(['..\models\Mdl*' num2str(int) '_' num2str(density) '*' num2str(scale) '*.mat']);
        for i = 1:length(mdlDir)
            modelPath = [mdlDir(i).folder '\' mdlDir(i).name]; % Prediction model path
            dataPath = ['..\datasets\data_' num2str(int) '_' num2str(density) '_PerlinBG_' num2str(scale) '_onlyI.mat']; % Prediction data path

            batchSize = 80; % Batch size for single prediction
            patchSize = [256 256]; % Patch size for block prediction (must be a power of 2)
            patchStride = [128 128]; % Stride for block prediction
            resSavePath = '..\models\'; % Path to save prediction results
            outputISyn = false; % Output noise-free synthetic images
            outputBG = true; % Output background images
            useGPU = true; % Use GPU for computation

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