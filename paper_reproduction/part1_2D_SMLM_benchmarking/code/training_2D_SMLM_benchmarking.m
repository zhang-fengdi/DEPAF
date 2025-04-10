clear all;

% Load DEPAF source code:
addpath(genpath('..\..\DEPAF_src_replication'));

for density = [3775 7550 18874 33974 56623 113246]
    for int = [1000 7000 40000]
        close all;

        dataPath = ['..\datasets\data_' num2str(int) '_' num2str(density) '_PerlinBG.tif']; % Image data path
        POIPath = '..\datasets\calibrated_PSF_2D_SMLM_benchmarking.mat'; % POI image data path
        lambda = lookupLambda(int, density); % Regularization intensity coefficient

        % Data loading parameters:
        trainIdxRange = 1:78; % Training set patch sampling range index
        valIdxRange = 79:80; % Validation set patch sampling range index

        % Model saving parameters:
        modelSavePath = '..\models\'; % Path to save model

        % Preprocessing parameters:
        upsamplRatio = [1.5 1.5 1]; % Upsampling ratio
        interpMethod = 'spline'; % Interpolation method, options: 'spline', 'linear', 'nearest', 'cubic'

        % Model structure parameters:
        encoderDepth = 2; % Depth of U-net encoder part (total depth is approximately twice this)

        % U-net model training parameters:
        patchSize = [256 256]; % Size of patches to sample
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
end


% Helper function: parameter lookup table.
function lambda = lookupLambda(int, density)
% Define lookup table:
LUT = [
    1000, 3775, 0.07;
    1000, 7550, 0.05;
    1000, 18874, 0.03;
    1000, 33974, 0.01;
    1000, 56623, 0.005;
    1000, 113246, 0.001;
    7000, 3775, 0.018;
    7000, 7550, 0.012;
    7000, 18874, 0.013;
    7000, 33974, 0.005;
    7000, 56623, 0.001;
    7000, 113246, 0.0005;
    40000, 3775, 0.02;
    40000, 7550, 0.009;
    40000, 18874, 0.008;
    40000, 33974, 0.009;
    40000, 56623, 0.0025;
    40000, 113246, 0.0001;
    ];

% Find matching row:
index = find(LUT(:, 1) == int & LUT(:, 2) == density);

% If a match is found, return the value in the third column:
if ~isempty(index)
    lambda = LUT(index, 3);
else
    error('No matching int and density values found.');
end
end