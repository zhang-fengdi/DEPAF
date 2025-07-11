clear all;

% Load DEPAF source code:
addpath(genpath('..\..\DEPAF_src_replication'));

for SNR = {'Low', 'Medium', 'High'}
    for rate = [1 2 3]
        close all;

        dataPath = ['..\datasets\data_rate_' num2str(rate) '_' SNR{1} '_SNR_only_I.mat']; % Image data path
        POIPath = '..\datasets\generated_spike_for_spike_inference.mat'; % POI image data path
        lambda = lookupLambda(SNR, rate); % Regularization intensity coefficient

        % Data loading parameters:
        trainIdxRange = 1 : 1600; % Training set patch sampling range index
        valIdxRange = 1601 : 2000; % Validation set patch sampling range index

        % Model saving parameters:
        modelSavePath = '..\models\'; % Path to save model

        % Preprocessing parameters:
        upsamplRatio = [1 1 1] ; % Upsampling ratio
        interpMethod = 'linear'; % Interpolation method, options: 'spline', 'linear', 'nearest', 'cubic'

        % Model structure parameters:
        encoderDepth = 5; % Depth of U-net encoder part (total depth is approximately twice this)

        % U-net model training parameters:
        patchSize = [1024 1]; % Size of patches to sample
        trainPatchNum = 1600; % Number of patches for training set
        valPatchNum = 400; % Number of patches for validation set
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
        patchNumForThreshSearch = 1600; % Number of samples for best threshold search

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
function lambda = lookupLambda(SNR, rate)
% Define lookup table:
LUT = {
    'Low', 1, 0.011;
    'Low', 2, 0.0045;
    'Low', 3, 0.0009;
    'Medium', 1, 0.012;
    'Medium', 2, 0.005;
    'Medium', 3, 0.0007;
    'High', 1, 0.015;
    'High', 2, 0.004;
    'High', 3, 0.001;
    };

% Find matching row:
index = find(strcmp(LUT(:,1), SNR) & (cell2mat(LUT(:,2)) == rate));

% If a match is found, return the value in the third column:
if ~isempty(index)
    lambda = LUT{index, 3};
else
    error('No matching SNR and spike rate values found.');
end
end