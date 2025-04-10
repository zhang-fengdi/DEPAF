clear all;

% Load DEPAF source code:
addpath(genpath('..\..\DEPAF_src_replication'));

% Lookup table (LUT) used to store training parameters, including PSF density, intensity, and regularization coefficient in the training set:
LUT = [3775, 1000, 0.07
    3775, 7000, 0.018
    3775, 40000, 0.02];

for scale = [5 6.4 9 13.9 19.3] % Train on datasets with different background variation scales
    for i = 1:size(LUT,1)
        close all;

        density = LUT(i,1); % Retrieve PSF density in training set
        int = LUT(i,2); % Retrieve PSF intensity in training set
        dataPath = ['..\datasets\data_' num2str(int) '_' num2str(density) '_PerlinBG_' num2str(scale) '_onlyI.mat']; % Image data path
        POIPath = '..\datasets\calibrated_PSF_background_estimation.mat'; % POI image data path
        lambda = LUT(i,3); % Regularization strength coefficient

        % Data loading parameters:
        trainIdxRange = 1:78; % Range of Patch indices for training set
        valIdxRange = 79:80; % Range of Patch indices for validation set

        % Model saving parameters:
        modelSavePath = '..\models\'; % Path to save model

        % Preprocessing parameters:
        upsamplRatio = [1.5 1.5 1]; % Upsampling ratio
        interpMethod = 'spline'; % Interpolation algorithm, options: 'spline', 'linear', 'nearest', 'cubic'

        % Model architecture parameters:
        encoderDepth = 2; % Depth of U-net encoder (total depth is approximately double)

        % U-net model training parameters:
        patchSize = [256 256]; % Size of patches to be sampled
        trainPatchNum = 512; % Number of patches sampled for training set
        valPatchNum = 64; % Number of patches sampled for validation set
        learningRate = 1e-3; % Learning rate
        minLR = 1e-4; % Lower limit for learning rate decay
        miniBatchSize = 8; % Batch size for each iteration
        maxEpochs = 1000; % Maximum number of training epochs
        valFreq = 20; % Validation frequency (validation occurs every valFreq batches)
        maxPatience = 50; % Early stopping patience (training stops if validation loss does not decrease after this many epochs)
        learnBG = true; % Whether to learn background
        verbose = true; % Display processing as images
        useGPU = true; % Use GPU for training

        % Bayesian estimation best threshold search parameters:
        useParallel = true; % Use parallel computation
        parNum = 6; % Number of parallel pools
        patchNumForThreshSearch = 512; % Number of samples for best segmentation threshold search

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
end