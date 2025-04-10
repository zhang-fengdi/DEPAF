clear all;
close all;

% Load DEPAF source code:
addpath(genpath('..\..\DEPAF_src_replication'));

modelPath = '..\models\Mdl_08-15-06-22-20_by_sequence-as-stack-MT0.N2.HD-DH-Exp.mat'; % Path of the prediction model
dataPath = '..\datasets\sequence-as-stack-MT0.N2.HD-DH-Exp.tif'; % Path of the prediction data

batchSize = 64; % Batch size for single prediction
patchSize = [96 96]; % Patch size for block prediction (must be a power of 2)
patchStride = [48 48]; % Stride for block prediction
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