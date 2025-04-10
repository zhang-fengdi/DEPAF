clear all;
close all;

% Load DEPAF source code:
addpath(genpath('..\..\DEPAF_src_replication'));

% ---------------------------- Data Parameters ----------------------------
dataPath = '..\datasets\ROI#5_1_149_107_255_1_5000.tif'; % Image data path
POIPath = '..\datasets\calibrated_PSF_dynamic_2D_SMLM.mat'; % POI image data path
lambda = 0.013; % Regularization intensity coefficient

% Data loading parameters:
trainIdxRange = 1:5000; % Training set patch sampling range index
valIdxRange = 1:5000; % Validation set patch sampling range index

% Model saving parameters:
modelSavePath = '..\models\'; % Path to save the model

% Preprocessing parameters:
upsamplRatio = [1.5 1.5 1]; % Upsampling ratio
interpMethod = 'spline'; % Interpolation method, options: 'spline', 'linear', 'nearest', 'cubic'

% Model structure parameters:
encoderDepth = 2; % Depth of U-net encoder part (total depth is approximately twice this)

% U-net model training parameters:
patchSize = [224 224]; % Patch size for sampling
trainPatchNum = 5000; % Number of patches for training set
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

% ------------------------- Bleaching Correction --------------------------
% Decompose file path, file name, and extension:
[fileDir, fileName, fileExt] = fileparts(dataPath);

% Construct file path for bleaching-corrected data:
correctedDataPath = [fileDir '\' fileName, '_bleaching_corrected', fileExt];

% Check if bleaching-corrected file exists:
if ~isfile(correctedDataPath)
    disp('Performing bleaching correction...');

    % Load original image data and convert to uint16 format:
    I = uint16(loadData(dataPath));

    % Perform bleaching correction:
    progressDisp(size(I,3)); % Initialize progress bar
    for i = 1:size(I,3)
        % Apply bleaching correction to the i-th frame (not needed for the 1st frame):
        if i > 1
            I(:,:,i) = bleachCorrection(I(:,:,1), I(:,:,i));
        end

        % Save current frame; create file for the first frame, append subsequent frames:
        if i == 1
            imwrite(I(:,:,i), correctedDataPath, 'tif', 'Compression', 'none');
        else
            imwrite(I(:,:,i), correctedDataPath, 'tif', 'WriteMode', 'append', 'Compression', 'none');
        end
        progressDisp(0); % Refresh progress bar
    end
    progressDisp(-1); % End progress bar
end

% ----------------------------- Model Training ----------------------------
disp('Training model...');
dataPath = correctedDataPath; % Training data path
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