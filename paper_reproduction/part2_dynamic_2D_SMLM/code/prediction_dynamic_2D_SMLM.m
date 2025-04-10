clear all;
close all;

% Load DEPAF source code:
addpath(genpath('..\..\DEPAF_src_replication'));

modelPath = '..\models\Mdl_07-27-22-31-06_by_ROI#5_1_149_107_255_1_5000_bleaching_corrected.mat'; % Path of the prediction model
dataPath = '..\datasets\ROI#5_1_149_107_255_1_5000_bleaching_corrected.tif'; % Path of the prediction data

batchSize = 128; % Batch size for single prediction
patchSize = [224 224]; % Patch size for block prediction (must be a power of 2)
patchStride = [112 112]; % Stride for block prediction
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