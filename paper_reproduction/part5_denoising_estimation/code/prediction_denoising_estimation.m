clear all;
close all;

% Load DEPAF source code:
addpath(genpath('..\..\DEPAF_src_replication'));

% Lookup table (LUT) to store training set parameters, including PSF density and intensity in the training set:
LUT = [3775, 1000
    3775, 7000
    56623, 1000];

for i = 1:size(LUT,1)
    density = LUT(i,1); % Retrieve PSF density for the training set
    int = LUT(i,2); % Retrieve PSF intensity for the training set
    dataDir = dir(['..\datasets\' num2str(density) '_SNR_' num2str(int) '*_X1_onlyI.mat']);
    dataPath = ['..\datasets\' dataDir(1).name]; % Image data path
    mdlDir = dir(['..\models\Mdl_*' num2str(density) '_SNR_' num2str(int) '*.mat']);
    for j = 1:length(mdlDir)
        modelPath = [mdlDir(j).folder '\' mdlDir(j).name]; % Prediction model path

        batchSize = 80; % Batch size per prediction
        patchSize = [256 256]; % Patch size for block prediction (must be a power of 2)
        patchStride = [128 128]; % Stride for block prediction
        resSavePath = '..\models\'; % Path to save prediction results
        outputISyn = true; % Whether to predict and output noiseless synthetic images
        outputBG = true; % Whether to predict and output noiseless background images
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