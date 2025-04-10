clear all;

% Load DEPAF source code:
addpath(genpath('..\..\DEPAF_src_replication'));

% Lookup table (LUT) to store training parameters, including PSF density, intensity, and regularization coefficient in the training set:
LUT = [3775, 1000, 0.07
    3775, 7000, 0.018
    56623, 1000, 0.005];

for i = 1:size(LUT,1)
    close all;

    density = LUT(i,1); % Retrieve PSF density for the training set
    int = LUT(i,2); % Retrieve PSF intensity for the training set
    dataDir = dir(['..\datasets\' num2str(density) '_SNR_' num2str(int) '*_X1_onlyI.mat']);
    dataPath = ['..\datasets\' dataDir(1).name]; % Image data path
    POIPath = '..\datasets\calibrated_PSF_denoising_estimation.mat'; % POI image data path
    lambda = LUT(i,3); % Regularization intensity coefficient

    % Data loading parameters:
    trainIdxRange = 1:78; % Training set patch sampling range index
    valIdxRange = 79:80; % Validation set patch sampling range index

    % Model saving parameters:
    modelSavePath = '..\models\'; % Path to save the model

    % Preprocessing parameters:
    upsamplRatio = [1.5 1.5 1]; % Upsampling ratio
    interpMethod = 'spline'; % Interpolation method, options: 'spline','linear','nearest','cubic'

    % Model structure parameters:
    encoderDepth = 2; % Depth of U-net encoder part (total depth is approximately twice this)

    % U-net model training parameters:
    patchSize = [256 256]; % Patch size for sampling
    trainPatchNum = 512; % Number of patches for training set
    valPatchNum = 64; % Number of patches for validation set
    learningRate = 1e-3; % Learning rate
    minLR = 1e-4; % Lower limit for learning rate decay
    miniBatchSize = 8; % Batch size per iteration
    maxEpochs = 1000; % Maximum training epochs
    valFreq = 20; % Validation frequency (validate every valFreq batches)
    maxPatience = 50; % Early stopping patience (stops training if validation loss does not decrease for this many validations)
    learnBG = true; % Whether to learn the background
    verbose = true; % Whether to display processing images
    useGPU = true; % Whether to use GPU

    % Bayesian estimation best threshold search parameters:
    useParallel = true; % Whether to use parallel computation
    parNum = 6; % Number of parallel pool workers
    patchNumForThreshSearch = 512; % Number of samples for best threshold search

    DEPAFTrain(dataPath, POIPath, lambda, ...
        'trainIdxRange', trainIdxRange, ...
        'valIdxRange', valIdxRange, ...
        'upsamplRatio', upsamplRatio, ...
        'interpMethod', interpMethod, ...
        'patchSize', patchSize, ...
        'trainPatchNum', trainPatchNum, ...
        'valPatchNum', valPatchNum, ...
        'encoderDepth', encoderDepth, ...
        'learningRate', learningRate, ...
        'minLR', minLR, ...
        'miniBatchSize', miniBatchSize, ...
        'maxEpochs', maxEpochs, ...
        'valFreq', valFreq, ...
        'maxPatience', maxPatience, ...
        'learnBG', learnBG, ...
        'verbose', verbose, ...
        'useGPU', useGPU, ...
        'useParallel', useParallel, ...
        'parNum', parNum, ...
        'patchNumForThreshSearch', patchNumForThreshSearch, ...
        'modelSavePath', modelSavePath);
end