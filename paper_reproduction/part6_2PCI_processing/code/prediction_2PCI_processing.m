clear all;
close all;

% Load DEPAF source code:
addpath(genpath('..\..\DEPAF_src_replication'));

% --------------------------- Parameter Settings --------------------------
datasetName = 'J123'; % Prediction data name
trainedPartName = 'part22';
model1DID = '07-12-04-09-39';
model2DID = '07-12-23-36-41';

for predPartName = {'part11', 'part12', 'part21', 'part22'} % Prediction data part number
    predPartName = predPartName{1};

    predDataPath = ['..\datasets\' datasetName '_' predPartName '.mat']; % Prediction data path
    saveFolderName = ['..\models\' datasetName '_' predPartName '_predRes']; % Result save path

    model1DDir = dir(['..\models\' datasetName '_' trainedPartName ...
        '_trainRes_1D\Mdl_' model1DID '_by_*.mat']);
    assert(numel(model1DDir) == 1, 'Expected one model with the selected 1D model ID.');
    model1DPath = fullfile(model1DDir(1).folder, model1DDir(1).name);

    model2DDir = dir(['..\models\' datasetName '_' trainedPartName ...
        '_trainRes_2D\Mdl_' model2DID '_by_*.mat']);
    assert(numel(model2DDir) == 1, 'Expected one model with the selected 2D model ID.');
    model2DPath = fullfile(model2DDir(1).folder, model2DDir(1).name);
    modelTrainingName = extractAfter(model2DDir(1).name, ['Mdl_' model2DID '_by_']);
    [~, postprocDataName] = fileparts(strrep(modelTrainingName, 'Sampled', ''));
    postprocParamsPath = fullfile(model2DDir(1).folder, ...
        ['PostprocParams_PredRes_' postprocDataName '_by_Mdl_' model2DID '.mat']);

    GTMaskPath = ['..\datasets\FinalMasks_' datasetName '_' predPartName '.mat']; % Groundtruth mask path for segmentation result evaluation

    % ---------------------------- File Check -----------------------------
    % Create an array containing all paths:
    filePaths = {predDataPath, model1DPath, model2DPath, postprocParamsPath, GTMaskPath};

    % Check each file path:
    for i = 1:length(filePaths)
        if ~exist(filePaths{i}, 'file')
            error('File not found: %s', filePaths{i});
        end
    end

    % ----------------------------- 2D to 1D ------------------------------
    % Load data dimension information:
    [widI, lenI, numI] = loadDataSize(predDataPath);

    % Extract file name:
    [~, predDataName] = fileparts(predDataPath);

    % Check if preprocessed file already exists, if not, perform data loading and preprocessing:
    if ~exist(['..\datasets\' predDataName '_I1D.mat'], 'file')
        disp('Converting 2D samples to 1D format...');

        % Load data from prediction data path:
        I = loadData(predDataPath);

        % Use median filtering to remove spike noise:
        I = movmedian(I, 20, 3);

        % Normalize using noise level:
        I = I ./ (prctile(I, 50, 3) - prctile(I, 25, 3));

        % Initialize counter and 1D processed data array:
        count = 0;
        I1D = zeros(numI, 1, widI*lenI, 'single');

        % Convert 2D image data to a 1D array:
        for i = 1:widI
            for j = 1:lenI
                count = count + 1;
                I1D(:, :, count) = I(i, j, :);
            end
        end

        % Save 1D data to a .mat file:
        save(['..\datasets\' predDataName '_I1D.mat'], 'I1D', '-v7.3');
        clear I;
    end

    % ------------------------ 1D Model Prediction ------------------------
    disp('Processing 1D samples with model...');
    % Model path and data path:
    modelPath = model1DPath; % Prediction model path
    dataPath = ['..\datasets\' predDataName '_I1D.mat']; % Prediction data path

    % Other parameter settings:
    batchSize = 1024; % Batch size for single prediction
    patchSize = [1024 1]; % Patch size for prediction (each non-singleton dimension must be a multiple of 2^encoderDepth)
    patchStride = [512 1]; % Stride during patch-based prediction
    resSavePath = [saveFolderName '_1D']; % Path to save prediction results
    outputISyn = true; % Predict and output model-synthesized images
    outputBG = true; % Predict and output background
    useGPU = true; % Use GPU for computation

    DEPAFPred(modelPath, dataPath, ...
        'batchSize', batchSize, ...
        'patchSize', patchSize, ...
        'patchStride', patchStride, ...
        'resSavePath', resSavePath, ...
        'outputISyn', outputISyn, ...
        'outputBG', outputBG, ...
        'useGPU', useGPU);

    % ----------------------------- 1D to 2D ------------------------------
    disp('Converting processed 1D samples back to 2D format...');
    ISynName = ['ISyn_' predDataName '_I1D_by_Mdl_' model1DID '.mat'];
    ISyn = loadData([saveFolderName '_1D\' ISynName]);

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
    save([saveFolderName '_2D\ISyn2D_' ISynName(6:end)], 'ISyn2D', '-v7.3');

    % ------------------------ 2D Model Prediction ------------------------
    disp('Processing 2D samples with model...');
    % Model path and data path:
    modelPath = model2DPath; % Prediction model path
    dataPath = [saveFolderName '_2D\ISyn2D_' ISynName(6:end)]; % Prediction data path

    % Other parameters:
    batchSize = 256; % Batch size for single prediction
    load(model2DPath, 'encoderDepth');
    patchSize = 2^encoderDepth * floor(min(widI,lenI) / 2^encoderDepth) * [1 1]; % Patch size for prediction
    patchStride = patchSize / 2; % Stride during patch-based prediction
    resSavePath = [saveFolderName '_2D']; % Path to save prediction results
    outputISyn = false; % Predict and output model-synthesized images
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

    % -------------------- Generate Segmentation Mask ---------------------
    disp('Generating segmentation mask...');
    % Load loc:
    [~, predictionDataName] = fileparts(dataPath);
    predResName = ['PredRes_' predictionDataName '_by_Mdl_' model2DID '.mat'];
    loc = loadData([saveFolderName '_2D\' predResName]);

    % Load post-processing parameters:
    params = loadData(postprocParamsPath);

    % Load Groundtruth Mask:
    GTMasks = loadData(GTMaskPath);

    % Generate segmentation result mask:
    masks = neuronSegMaskGeneration(loc, widI, lenI, ...
        params.epsDBSCAN, params.minPtsDBSCAN, ...
        params.convexBound, params.expDistBound, ...
        params.minAreaMask, params.avgAreaMask, ...
        params.threshBinarizeMask, ...
        params.threshCOM0, params.threshCOM, params.threshIoU, ...
        params.threshConsume, params.threshConsec);

    % ------------------------- Result Evaluation -------------------------
    disp('Evaluating results...');
    % Calculate optimal evaluation metrics:
    matchThresh = 0.5;
    [F1, mIoU] = neuronSegResEval(masks, GTMasks, matchThresh);

    % Save optimal results:
    save([saveFolderName '_2D\NeuronSegPredRes_' predResName], ...
        'F1', 'mIoU', 'masks');
    close all;
end