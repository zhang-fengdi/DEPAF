clear all;
close all;

% Load DEPAF source code:
addpath(genpath('..\..\DEPAF_src_replication'));

% ---------------------------- Data Parameters ----------------------------
datasetName = 'J123'; % Training data name
partName = 'part22'; % Part number of training data

trainDataPath = ['..\datasets\' datasetName '_' partName '.mat']; % Training data path
GTMaskPath = ['..\datasets\FinalMasks_' datasetName '_' partName '.mat']; % GTMask path for post-processing parameter optimization
saveFolderName = ['..\models\' datasetName '_' partName '_trainRes']; % Result save path
lambda1D = 0.026; % Lambda for 1D sample training
lambda2D = 0.004; % Lambda for 2D sample training
totalParNum = 6; % Total number of parallel pools

% ------------------------------ File Check -------------------------------
% Create an array containing all paths:
filePaths = {trainDataPath, GTMaskPath};

% Check each file path:
for i = 1:length(filePaths)
    if ~exist(filePaths{i}, 'file')
        error('Missing file: %s', filePaths{i});
    end
end

% -------------------------- Data Preprocessing ---------------------------
% Set preprocessing parameters:
select1DNum = 20000; % Number of 1D samples to collect
win = 1024; % Length of 1D sample to collect
select2DNum = 1000; % Number of 2D samples to collect

% Load data dimension information:
[widI, lenI, numI] = loadDataSize(trainDataPath);

% Extract file name:
[~, trainDataName] = fileparts(trainDataPath);

% Check if preprocessed files already exist; if not, perform data loading and preprocessing:
if ~exist(['..\datasets\' trainDataName '_I1D.mat'], 'file') || ...
        ~exist(['..\datasets\' trainDataName '_I1DSampled.mat'], 'file') || ...
        ~exist('..\datasets\fitted_spike_signal_2PCI_processing.mat', 'file')
    disp('Data preprocessing...');

    % Load data from training data path:
    I = loadData(trainDataPath);

    % Use median filtering to remove spike noise:
    I = movmedian(I, 20, 3);

    % Fit and generate 1D POI:
    fitPtsNum = 1000;
    tol = 0.9;
    spike = fit1DExpPOI(I, fitPtsNum, tol);
    save('..\datasets\fitted_spike_signal_2PCI_processing.mat', 'spike');

    % Normalize using noise level:
    I = I ./ (prctile(I, 50, 3) - prctile(I, 25, 3));

    % Initialize counter and 1D processed data array:
    count = 0;
    I1D = zeros(numI, 1, widI*lenI, 'single');

    % Convert 2D image data to 1D array:
    for i = 1:widI
        for j = 1:lenI
            count = count + 1;
            I1D(:, :, count) = I(i, j, :);
        end
    end

    % Save 1D data:
    save(['..\datasets\' trainDataName '_I1D.mat'], 'I1D', '-v7.3');
    clear I;

    % ------------------------ 1D Sample Selection ------------------------
    disp('Sampling 1D samples...');
    % Perform temporal filtering:
    [~,sortedIdx] = sort(sum(I1D, 3), 'descend');
    selectedTimeIdx = sortedIdx(1:select1DNum);

    % Perform spatial filtering:
    [~, sortedIdx] = sort(sum(I1D(selectedTimeIdx,1,:)-median(I1D,1), 1), 'descend');
    selectedPixelIdx = sortedIdx(1:select1DNum);

    % Extract samples:
    I1DSampled = zeros(win, 1, select1DNum, 'single');
    for i = 1:select1DNum
        % Select pixel:
        pixel = selectedPixelIdx(i);
        pixVals = I1D(selectedTimeIdx,1,pixel);

        % Select time point with maximum pixel value:
        [~,maxIdx] = max(pixVals);
        t = selectedTimeIdx(maxIdx);

        % Add random offset to time point:
        bias = randi([-round(win/4) round(win/4)]);
        t = t + bias;

        % Generate selected time range:
        if t - ceil((win-1)/2) < 1
            tRange = 1:win;
        elseif t + floor((win-1)/2) > numI
            tRange = (numI-win+1):numI;
        else
            tRange = t - ceil((win-1)/2) : t + floor((win-1)/2);
        end

        % Sample data:
        I1DSampled(:,1,i) = I1D(tRange,1,pixel);
    end

    % Save samples:
    save(['..\datasets\' trainDataName '_I1DSampled.mat'], 'I1DSampled', '-v7.3');
    clear I1DSampled I1D pixVals selectedPixelIdx selectedTimeIdx sortedIdx;
end

% Generate 2D POI:
if ~exist('..\datasets\standard_Gaussian_PSF_2PCI_processing.mat', 'file')
    sigma = 1;
    hsize = 2 * ceil(3 * sigma) + 1;
    PSF = fspecial('gaussian', hsize, sigma);
    save('..\datasets\standard_Gaussian_PSF_2PCI_processing.mat', 'PSF');
end

% --------------------------- 1D Model Training ---------------------------
disp('Training model with sampled 1D samples...');
% Data path and POI path:
dataPath = ['..\datasets\' trainDataName '_I1DSampled.mat']; % Image data path
POIPath = '..\datasets\fitted_spike_signal_2PCI_processing.mat'; % POI image data path

% Data loading related parameters:
trainIdxRange = 1 : select1DNum*0.8; % Training patch range
valIdxRange = select1DNum*0.8+1 : select1DNum; % Validation patch range

% Model saving related parameters:
modelSavePath = [saveFolderName '_1D']; % Path to save model

% Preprocessing related parameters:
upsamplRatio = [1 1 1] ; % Upsampling factor
interpMethod = 'linear'; % Interpolation method: options are 'spline','linear','nearest','cubic'

% Model structure parameters:
encoderDepth = 2; % Depth of U-net encoder (total depth is approximately twice)

% U-net model training parameters:
patchSize = [win 1]; % Patch size for sampling
trainPatchNum = select1DNum*0.8; % Number of patches for training
valPatchNum = select1DNum - select1DNum*0.8; % Number of patches for validation
learningRate = 1e-3; % Learning rate
minLR = 1e-4; % Lower bound for learning rate decay
miniBatchSize = 8; % Batch size per iteration
maxEpochs = 1000; % Maximum training epochs
valFreq = 20; % Validation frequency (validate once every valFreq batches)
maxPatience = 50; % Early stopping patience (stop training if validation loss doesn't decrease for this many steps)
learnBG = true; % Whether to learn background
verbose = true; % Whether to display process images
useGPU = true; % Whether to use GPU

% Bayesian estimation best threshold search parameters:
useParallel = true; % Whether to use parallel computing
parNum = totalParNum; % Number of parallel pools
patchNumForThreshSearch = trainPatchNum; % Number of samples for best threshold search

DEPAFTrain(dataPath, POIPath, lambda1D, ...
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

% -------------------------- 1D Model Prediction --------------------------
disp('Processing all 1D samples with model...');
% Model path and data path:
mdlDir = dir([saveFolderName '_1D\Mdl_*.mat']);
modelPath = [mdlDir(end).folder '\' mdlDir(end).name]; % Prediction model path
dataPath = ['..\datasets\' trainDataName '_I1D.mat']; % Prediction data path

% Other parameter settings:
batchSize = 1024; % Batch size for single prediction
patchSize = [win 1]; % Patch size for prediction (must be a power of 2)
patchStride = [round(win/2) 1]; % Stride for patch-based prediction
resSavePath = [saveFolderName '_1D']; % Path to save prediction results
outputISyn = true; % Predict and output noise-free synthetic images
outputBG = false; % Predict and output background
useGPU = true; % Use GPU for computation

DEPAFPred(modelPath, dataPath, ...
    'batchSize', batchSize, ...
    'patchSize', patchSize, ...
    'patchStride', patchStride, ...
    'resSavePath', resSavePath, ...
    'outputISyn', outputISyn, ...
    'outputBG', outputBG, ...
    'useGPU', useGPU);
close all;

% -------------------------- 1D to 2D Conversion --------------------------
disp('Converting 1D sample prediction results to 2D format...');
% Retrieve all files matching the pattern in the save path:
ISynDir = dir([saveFolderName '_1D\ISyn_*.mat']);

% Load data from the last matching file:
ISyn = loadData([ISynDir(end).folder '\' ISynDir(end).name]);

% Initialize a 3D array to store the converted 2D ISyn:
ISyn2D = zeros(widI, lenI, numI, 'single');

% Initialize counter:
count = 0;

% Iterate over each element, reorganizing 1D array data back to 2D format:
for i = 1:widI
    for j = 1:lenI
        count = count + 1;
        ISyn2D(i, j, :) = ISyn(:, :, count);  % Convert from 1D to 2D indexing
    end
end

% Check and create save path:
checkPath([saveFolderName '_2D']);

% Save 2D data:
save([saveFolderName '_2D\ISyn2D_' ISynDir(end).name(6:end)], 'ISyn2D', '-v7.3');
clear ISyn

% -------------------------- 2D Sample Selection --------------------------
disp('Sampling 2D samples...');
% Perform temporal filtering:
[~,sortedIdx] = sort(sum(ISyn2D, [1 2]), 'descend');
selectedTimeIdx = sortedIdx(1:select2DNum);
ISyn2DSampled = ISyn2D(:,:,selectedTimeIdx);

% Save samples:
save([saveFolderName '_2D\ISyn2DSampled_' ISynDir(end).name(6:end)], 'ISyn2DSampled', '-v7.3');
clear ISyn2D ISyn2DSampled

% --------------------------- 2D Model Training ---------------------------
disp('Training model with sampled 2D samples...');
% Data path and POI path:
dataDir = dir([saveFolderName '_2D\ISyn2DSampled_*.mat']);
dataPath = [dataDir(end).folder '\' dataDir(end).name];% Image data path
POIPath = '..\datasets\standard_Gaussian_PSF_2PCI_processing.mat'; % POI image data path

% Data loading related parameters:
trainIdxRange = 1 : select2DNum*0.8; % Training patch range
valIdxRange = select2DNum*0.8+1 : select2DNum; % Validation patch range

% Model saving related parameters:
modelSavePath = [saveFolderName '_2D']; % Path to save model

% Preprocessing related parameters:
upsamplRatio = [1 1 1] ; % Upsampling factor
interpMethod = 'spline'; % Interpolation method: options are 'spline','linear','nearest','cubic'

% Model structure parameters:
encoderDepth = 2; % Depth of U-net encoder (total depth is approximately twice)

% U-net model training parameters:
patchSize = 2^encoderDepth * floor(min(widI,lenI) / 2^encoderDepth) * [1 1]; % Patch size for sampling
trainPatchNum = 1024; % Number of patches for training
valPatchNum = 64; % Number of patches for validation
learningRate = 1e-3; % Learning rate
minLR = 1e-4; % Lower bound for learning rate decay
miniBatchSize = 8; % Batch size per iteration
maxEpochs = 1000; % Maximum training epochs
valFreq = 20; % Validation frequency (validate once every valFreq batches)
maxPatience = 50; % Early stopping patience (stop training if validation loss doesn't decrease for this many steps)
learnBG = true; % Whether to learn background
verbose = true; % Whether to display process images
useGPU = true; % Whether to use GPU

% Bayesian estimation best threshold search parameters:
useParallel = true; % Whether to use parallel computing
parNum = totalParNum; % Number of parallel pools
patchNumForThreshSearch = trainPatchNum; % Number of samples for best threshold search

DEPAFTrain(dataPath, POIPath, lambda2D, ...
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

% -------------------------- 2D Model Prediction --------------------------
disp('Processing all 2D samples with model...');
% Model path and data path:
mdlDir = dir([saveFolderName '_2D\Mdl_*.mat']);
modelPath = [mdlDir(end).folder '\' mdlDir(end).name]; % Prediction model path
dataDir = dir([saveFolderName '_2D\ISyn2D_*.mat']); % Prediction data path
dataPath = [dataDir(end).folder '\' dataDir(end).name];

% Other parameters:
batchSize = 256; % Batch size for single prediction
patchSize = 2^encoderDepth * floor(min(widI,lenI) / 2^encoderDepth) * [1 1]; % Patch size for sampling
patchStride = patchSize / 2; % Stride for patch-based prediction
resSavePath = [saveFolderName '_2D']; % Path to save prediction results
outputISyn = false; % Predict and output noise-free synthetic images
outputBG = false; % Predict and output background
useGPU = true; % Use GPU for computation

DEPAFPred(modelPath, dataPath, ...
    'batchSize', batchSize, ...
    'patchSize', patchSize, ...
    'patchStride', patchStride, ...
    'resSavePath', resSavePath, ...
    'outputISyn', outputISyn, ...
    'outputBG', outputBG, ...
    'useGPU', useGPU);
close all;

% -------------------- Optimization and Mask Generation -------------------
disp('Optimizing and generating segmentation mask...');
% Create parallel pool:
if useParallel
    createParPool(totalParNum, 20);
end

% Load data:
GTMasks = loadData(GTMaskPath);
predResDir = dir([saveFolderName '_2D\PredRes*.mat']);
loc = loadData([predResDir(end).folder '\' predResDir(end).name]);

% Define parameter range for Bayesian optimization:
optimVars = [
    optimizableVariable('epsDBSCAN',[0.1 40],'Type','real');
    optimizableVariable('minPtsDBSCAN',[1 40],'Type','integer');
    optimizableVariable('convexBound',[0 1],'Type','real');
    optimizableVariable('expDistBound',[0 10],'Type','real');
    optimizableVariable('minAreaMask',[0 300],'Type','integer');
    optimizableVariable('avgAreaMask',[200 400],'Type','integer');
    optimizableVariable('threshBinarizeMask',[0 1],'Type','real');
    optimizableVariable('threshCOM0',[0 20],'Type','real');
    optimizableVariable('threshCOM',[0 20],'Type','real');
    optimizableVariable('threshIoU',[0 1],'Type','real');
    optimizableVariable('threshConsume',[0 1],'Type','real');
    optimizableVariable('threshConsec',[1 50],'Type','integer');
    ];

% Objective function setup:
objFcn = @(x) iOptimizeNSResEval(x, loc, widI, lenI, GTMasks);

try
    % Run Bayesian optimization:
    results = bayesopt(objFcn, optimVars, ...
        'AcquisitionFunctionName', 'expected-improvement', ...
        'IsObjectiveDeterministic', true, ...
        'UseParallel', true, ...
        'MaxObjectiveEvaluations', 500);

    % Retrieve best parameters:
    bestParams = results.XAtMinObjective;

    % Calculate best mask:
    bestMasks = neuronSegMaskGeneration(loc, widI, lenI, ...
        bestParams.epsDBSCAN, bestParams.minPtsDBSCAN, ...
        bestParams.convexBound, bestParams.expDistBound, ...
        bestParams.minAreaMask, bestParams.avgAreaMask, ...
        bestParams.threshBinarizeMask, ...
        bestParams.threshCOM0, bestParams.threshCOM, bestParams.threshIoU, ...
        bestParams.threshConsume, bestParams.threshConsec);

    % Calculate best evaluation metrics:
    matchThresh = 0.5;
    [bestF1, bestmIoU] = neuronSegResEval(bestMasks, GTMasks, matchThresh);

    % Save best result:
    varOrder = {'bestParams', 'bestF1', 'bestmIoU', 'bestMasks'};
    save([saveFolderName '_2D\PostprocParams_' predResDir(end).name], ...
        'bestParams', 'bestF1', 'bestmIoU', 'bestMasks', 'varOrder');

catch ME
    % Output and save error message:
    errorMsg = ME.message;
    fprintf('Error during Bayesian optimization: %s\n', errorMsg);
    save([saveFolderName '_2D\PostprocParams_' predResDir(end).name], 'ME', 'errorMsg');
end

% Close all windows:
close all;

% Close all parallel pools:
delete(gcp('nocreate'));


% Helper function: Calculate Bayesian optimization loss.
function loss = iOptimizeNSResEval(params, loc, widMask, lenMask, GTMasks)
% Generate neuron segmentation mask based on given parameters:
masks = neuronSegMaskGeneration(loc, widMask, lenMask, ...
    params.epsDBSCAN, params.minPtsDBSCAN, ...
    params.convexBound, params.expDistBound, ...
    params.minAreaMask, params.avgAreaMask, ...
    params.threshBinarizeMask, ...
    params.threshCOM0, params.threshCOM, params.threshIoU, ...
    params.threshConsume, params.threshConsec);

% Evaluate generated neuron segmentation masks against groundtruth masks:
matchThresh = 0.5;
[F1, mIoU] = neuronSegResEval(masks, GTMasks, matchThresh);

% Bayesian optimization defaults to minimizing the objective function, so we adjust it:
F1Loss = 1 - F1;
mIoULoss = 1 - mIoU;

% Prioritize optimizing F1, then optimize mIoU after F1 is optimal:
loss = F1Loss + mIoULoss / (1 + 1000 * F1Loss);
end