clear all;
close all;

% -------------------------- Parameter Settings ---------------------------
sliceNum = 16; % Number of slices
bitNum = 16; % Number of encoding bits
frameInterval = 5; % Frame interval for slices
widI = 1536; % Image width
lenI = 1536; % Image length
savePath = '..\models\'; % Path to save models
trainDataSelectNum = 64; % Number of training data images randomly selected from the dataset
codebookPath = '..\datasets\CodeBookSubPool1_fetalLiver.mat';
filePath = '..\datasets\sequential\'; % Path of data files
plotRes = true; % Whether to plot and save result images
beadFileNamesForBit = { % Calibration bead image filenames
    'BEAD_00_126';
    'BEAD_00_126';
    'BEAD_01_126';
    'BEAD_01_126';
    'BEAD_02_126';
    'BEAD_02_126';
    'BEAD_03_126';
    'BEAD_03_126';
    'BEAD_04_126';
    'BEAD_04_126';
    'BEAD_05_126';
    'BEAD_05_126';
    'BEAD_06_126';
    'BEAD_06_126';
    'BEAD_07_126';
    'BEAD_07_126'};
signalFileNamesForBit = { % Signal image filenames
    'Cy5_00_126';
    'Cy7_00_126';
    'Cy7_01_126';
    'Cy5_01_126';
    'Cy5_02_126';
    'Cy7_02_126';
    'Cy5_03_126';
    'Cy7_03_126';
    'Cy7_04_126';
    'Cy5_04_126';
    'Cy5_05_126';
    'Cy7_05_126';
    'Cy5_06_126';
    'Cy7_06_126';
    'Cy7_07_126';
    'Cy5_07_126'};

% --------------------------- Data Organization ---------------------------
% Load DEPAF source code:
addpath(genpath('..\..\DEPAF_src_replication'));

% Check if data files exist to determine whether data organization is needed:
dataFilesMissing = false;
if ~exist('..\datasets\data_all.mat', 'file')
    disp('Training data file missing, preparing to generate...');
    dataFilesMissing = true;
end
for bitIdx = 1:length(signalFileNamesForBit)
    if ~exist(['..\datasets\data_bit', num2str(bitIdx, '%02d'), '.mat'], 'file')
        disp('Some bit data files are missing, preparing to generate...');
        dataFilesMissing = true;
        break;
    end
end

POIFilesMissing = false;
if isempty(dir(['..\datasets\fitted_elliptical_Gaussian_PSF_' ...
        'MERFISH_analysis_xSigma*_ySigma*.mat']))
    disp('POI data files are missing, preparing to generate...');
    POIFilesMissing = true;
end

if dataFilesMissing
    disp('Organizing and generating data...');
    % Initialize image matrix and maximum feature points:
    beadI = zeros(widI, lenI, sliceNum);
    maxFeaturePts = 0;
    maxFeaturePtsIdx = 0;

    % Load and process calibration bead images, also calculate feature points:
    disp('Loading and processing calibration bead images...');
    for sliceIdx = 1:sliceNum
        % Load image data of the current slice:
        singleBeadI = ReadDax([filePath beadFileNamesForBit{sliceIdx} '.dax'], ...
            'startFrame', 1, 'endFrame', 40, 'verbose', false);

        % Obtain the total image by averaging:
        singleBeadI = mean(singleBeadI, 3);

        % Normalize image data to the range [0, 1]:
        singleBeadI = (singleBeadI - min(singleBeadI, [], 'all')) / ...
            (max(singleBeadI, [], 'all') - min(singleBeadI, [], 'all'));

        % Save processed image data to beadI matrix:
        beadI(:, :, sliceIdx) = singleBeadI;

        % Detect KAZE feature points in the current image:
        featurePoints = detectKAZEFeatures(beadI(:, :, sliceIdx));

        % Update maximum feature points and the corresponding slice index:
        if featurePoints.Count > maxFeaturePts
            maxFeaturePts = featurePoints.Count;
            maxFeaturePtsIdx = sliceIdx;
        end
    end

    % Calculate transformation matrices for registration:
    disp('Calculating transformation matrices for registration...');
    transforms = cell(sliceNum, 1);
    referenceImage = beadI(:, :, maxFeaturePtsIdx); % Select the image with the most feature points as the reference
    for sliceIdx = 1:sliceNum
        % Calculate the transformation matrix between the current slice image and the reference image:
        transforms{sliceIdx} = getTform(referenceImage, beadI(:, :, sliceIdx));
    end

    % Load and process signal images, and save data for training and each bit for prediction:
    disp('Loading and processing signal images and saving...');
    numIAll = sliceNum * bitNum;
    IAll = zeros(widI, lenI, numIAll, 'uint16');
    count = 0;
    for bitIdx = 1:bitNum
        I = zeros(widI, lenI, sliceNum, 'uint16');
        for sliceIdx = 1:sliceNum
            % Calculate start and end frames for the current slice:
            startFrame = (sliceIdx - 1) * frameInterval + 3;
            endFrame = (sliceIdx - 1) * frameInterval + 6;

            % Load image data for the current slice:
            singleI = ReadDax([filePath signalFileNamesForBit{bitIdx} '.dax'], ...
                'startFrame', startFrame, 'endFrame', endFrame, 'verbose', false);

            % Obtain total image by averaging:
            singleI = mean(singleI, 3);

            % Apply image registration transformation:
            singleI = imwarp(singleI, transforms{bitIdx}, 'OutputView', imref2d([widI, lenI]));

            % Scale image data to the range [0, 65535] to utilize uint16 range fully:
            singleI = (singleI - min(singleI, [], 'all')) / ...
                (max(singleI, [], 'all') - min(singleI, [], 'all')) * 65535;
            singleI = uint16(singleI);

            % Perform bleach correction using histogram matching:
            count = count + 1;
            if count == 1
                refImg = singleI;
            else
                singleI = bleachCorrection(refImg, singleI);
            end

            % Record data:
            I(:, :, sliceIdx) = singleI;
            IAll(:, :, count) = singleI;
        end

        % Save data for each bit:
        save(['..\datasets\data_bit', num2str(bitIdx, '%02d'), '.mat'], 'I');
    end

    % Randomly select training data:
    IAll = IAll(:, :, randperm(trainDataSelectNum));

    % Save all data:
    save('..\datasets\data_all.mat', 'IAll');
    clear I IAll;
end

% --------------------------- Generate POI Data ---------------------------
if POIFilesMissing
    disp('Fitting and generating POI data...');
    % Load training data:
    IAll = loadData('..\datasets\data_all.mat');

    % Fit 2D Gaussian POI:
    fitPointsNumPerI = 30; % Set the number of fit points per image
    tol = 0.99; % Set tolerance
    [PSF, xSigma, ySigma] = fit2DGaussPOI(IAll, fitPointsNumPerI, tol);

    % Save POI data:
    POISaveFileName = sprintf(['fitted_elliptical_Gaussian_PSF_' ...
        'MERFISH_analysis_xSigma%.2f_ySigma%.2f.mat'], xSigma, ySigma);
    save(['..\datasets\' POISaveFileName], 'PSF');
    clear IAll;
end

% ---------------------------- Model Training -----------------------------
disp('Training model...');
% Data path and POI path:
dataPath = '..\datasets\data_all.mat'; % Image data path
POIDir = dir(['..\datasets\fitted_elliptical_Gaussian_PSF_' ...
    'MERFISH_analysis_xSigma*_ySigma*.mat']);
POIPath = [POIDir(1).folder '\' POIDir(1).name]; % POI image data path
lambda = 0.005; % Regularization intensity coefficient

% Data loading parameters:
trainIdxRange = 1 : round(trainDataSelectNum * 0.8); % Training set patch sampling range index
valIdxRange = round(trainDataSelectNum * 0.8) + 1 : trainDataSelectNum; % Validation set patch sampling range index

% Model saving parameters:
modelSavePath = savePath; % Path to save the model

% Preprocessing parameters:
upsamplRatio = [1.5 1.5 1]; % Upsampling ratio
interpMethod = 'spline'; % Interpolation method, options: 'spline', 'linear', 'nearest', 'cubic'

% Model structure parameters:
encoderDepth = 2; % Depth of U-net encoder part (total depth is approximately twice this)

% U-net model training parameters:
patchSize = [256 256]; % Patch size for sampling
trainPatchNum = 512; % Number of patches for training set
valPatchNum = 64; % Number of patches for validation set
learningRate = 1e-3; % Learning rate
minLR = 1e-4; % Lower limit for learning rate decay
miniBatchSize = 8; % Batch size per iteration
maxEpochs = 10; % Maximum training epochs
valFreq = 20; % Validation frequency (validate every valFreq batches)
maxPatience = 50; % Early stopping patience (stops training if validation loss does not decrease for this many validations)
learnBG = true; % Whether to learn the background
verbose = true; % Whether to display processing images
useGPU = true; % Whether to use GPU

% Bayesian estimation best threshold search parameters:
useParallel = true; % Whether to use parallel computation
parNum = 6; % Number of parallel pool workers
patchNumForThreshSearch = trainPatchNum; % Number of samples for best threshold search

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
close all;

% --------------------------- Model Prediction ----------------------------
disp('Processing image data with the model...');
% Model path and data path:
mdlDir = dir([savePath '\Mdl_*.mat']);
modelPath = [mdlDir(end).folder '\' mdlDir(end).name]; % Path of prediction model
dataDir = dir('..\datasets\data_bit*.mat'); % Path of prediction data
for i = 1:length(dataDir)
    dataPath = [dataDir(i).folder '\' dataDir(i).name];

    % Other parameters:
    batchSize = 16; % Batch size for prediction
    patchSize = [256 256]; % Patch size for sampling
    patchStride = [128 128]; % Stride for block prediction
    resSavePath = savePath; % Path to save prediction results
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
end

% ----------------------------- Gene Decoding -----------------------------
try
    disp('Decoding localization results into binary codes...');
    allPredResDir = dir([savePath '\PredRes*.mat']);
    modelNum = allPredResDir(end).name(end-17:end-4);
    predResDir = dir([savePath '\PredRes*' modelNum '.mat']);

    allBitLoc = cell(bitNum, 1);
    words = cell(bitNum, 1);
    for bitIdx = 1:bitNum
        % Load data:
        [allBitLoc{bitIdx}, widI, lenI] = loadData([predResDir(bitIdx).folder '\' predResDir(bitIdx).name]);

        % Get number of slices:
        sliceNum = length(allBitLoc{bitIdx});

        % Process points in each slice and initialize words:
        words{bitIdx} = cell(sliceNum, 1);
        for sliceIdx = 1:sliceNum
            % Filter points near the edges:
            allBitLoc{bitIdx}{sliceIdx}(allBitLoc{bitIdx}{sliceIdx}(:,1) <= 10, :) = [];
            allBitLoc{bitIdx}{sliceIdx}(allBitLoc{bitIdx}{sliceIdx}(:,1) >= lenI - 10, :) = [];
            allBitLoc{bitIdx}{sliceIdx}(allBitLoc{bitIdx}{sliceIdx}(:,2) <= 10, :) = [];
            allBitLoc{bitIdx}{sliceIdx}(allBitLoc{bitIdx}{sliceIdx}(:,2) >= widI - 10, :) = [];

            % Initialize words:
            words{bitIdx}{sliceIdx} = false(size(allBitLoc{bitIdx}{sliceIdx}, 1), bitNum);
            words{bitIdx}{sliceIdx}(:, bitIdx) = true;
        end
    end

    % Start matching:
    for sliceIdx = 1:sliceNum
        for i = 1:bitNum
            % Match with the subsequent bits:
            for j = i+1:bitNum
                % Perform matching:
                matchThresh = 3;
                assignments = fastMatchPairs(allBitLoc{i}{sliceIdx}, allBitLoc{j}{sliceIdx}, matchThresh);

                % If there are successful matches, record them:
                if ~isempty(assignments)
                    % Update words:
                    words{i}{sliceIdx}(assignments(:,1), :) = ...
                        words{i}{sliceIdx}(assignments(:,1), :) + words{j}{sliceIdx}(assignments(:,2), :);

                    % Update centroids:
                    allBitLoc{i}{sliceIdx}(assignments(:,1), :) = ...
                        (allBitLoc{i}{sliceIdx}(assignments(:,1), :) + allBitLoc{j}{sliceIdx}(assignments(:,2), :)) / 2;

                    % Delete matched points:
                    words{j}{sliceIdx}(assignments(:,2), :) = [];
                    allBitLoc{j}{sliceIdx}(assignments(:,2), :) = [];
                end
            end
        end
    end

    % Convert format to final decoding result:
    words = cell2mat(cellfun(@(x) vertcat(x{:}), words, 'UniformOutput', false));
    allBitLoc = cell2mat(cellfun(@(x) vertcat(x{:}), allBitLoc, 'UniformOutput', false));

catch ME
    % Output and save error information:
    errorMsg = ME.message;
    fprintf('Error occurred: %s\n', errorMsg);
    save([savePath, '.\EvalRes_Mdl_' modelNum '.mat'], 'ME', 'errorMsg');
end

% --------------------------- Result Evaluation ---------------------------
disp('Evaluating results...');
% Load codebook data:
Codebook = loadData(codebookPath);
codeDic = logical(cell2mat({Codebook.Code}') - 48); % Retrieve codes and convert to logical array
geneNameDic = {Codebook.GeneShortName}'; % Retrieve gene names

% Initialize variables:
wordNum = size(words, 1); % Number of words and bits
correctionFlag = false(wordNum, 1); % Initialize correction flag
decodedGeneNames = cell(wordNum, 1); % Initialize decoded gene names

% Calculate Hamming distance:
hammingDist = pdist2(double(words), double(codeDic), 'hamming') * bitNum;

% Find the minimum Hamming distance and corresponding index:
[minVal, minIdx] = min(hammingDist, [], 2);

% Decode only words with Hamming distance <= 1:
decodedGeneNames(minVal <= 1) = geneNameDic(minIdx(minVal <= 1));
correctionFlag(minVal == 1) = true; % Mark words needing correction

% Generate ground truth words:
GTWords = false(wordNum, bitNum);
GTWords(minVal <= 1, :) = codeDic(minIdx(minVal <= 1), :);

% Generate decoding result:
decodeRes = struct('Gene', decodedGeneNames, ...
    'Position', num2cell(allBitLoc, 2), ...
    'GroundtruthWord', cellstr(char(GTWords + 48)), ...
    'decodedWord', cellstr(char(words + 48)), ...
    'Correction', num2cell(correctionFlag));

% Delete decoding results with Hamming distance > 1:
decodeRes(minVal > 1) = [];

% Calculate correction rate:
correctionFlag = correctionFlag(minVal <= 1);
correctionRate = sum(correctionFlag) / length(correctionFlag);

% Calculate per-bit correction rate:
words = words(minVal <= 1, :);
GTWords = GTWords(minVal <= 1, :);
perBitCorrectionRate = sum(words ~= GTWords) ./ size(words, 1);

% Calculate 0-to-1 and 1-to-0 correction rates per bit:
zeroToOneErrs = sum((GTWords == 0) & (words == 1));
oneToZeroErrs = sum((GTWords == 1) & (words == 0));
zeroTotal = sum(GTWords == 0);
oneTotal = sum(GTWords == 1);
zeroToOneErrRate = zeroToOneErrs ./ zeroTotal;
oneToZeroErrRate = oneToZeroErrs ./ oneTotal;

% Generate perfect decoding result:
perfectDecodeRes = decodeRes(~[decodeRes.Correction]);

% Calculate decoded gene counts for corrected and perfect decoding:
geneCountDic = dictionary(geneNameDic, zeros(length(geneNameDic), 1));
perfectGeneCountDic = dictionary(geneNameDic, zeros(length(geneNameDic), 1));
for i = 1:length(decodeRes)
    geneName = {decodeRes(i).Gene};
    geneCountDic(geneName) = geneCountDic(geneName) + 1;
end
for i = 1:length(perfectDecodeRes)
    geneName = {perfectDecodeRes(i).Gene};
    perfectGeneCountDic(geneName) = perfectGeneCountDic(geneName) + 1;
end
decodedGeneNum = values(geneCountDic);
perfectDecodedGeneNum = values(perfectGeneCountDic);

% Retrieve FPKM values:
FPKM = [Codebook.FPKM]';

% Calculate log correlation between decoded gene count and FPKM:
[FPKMCorr, p] = corr(log10(decodedGeneNum), log10(FPKM));
[perfectFPKMCorr, perfectp] = corr(log10(perfectDecodedGeneNum), log10(FPKM));

% Save results:
save([savePath, '\EvalRes_Mdl_' modelNum '.mat'], ...
    'decodeRes', 'perfectDecodeRes', ...
    'correctionRate', 'perBitCorrectionRate', ...
    'zeroToOneErrRate', 'oneToZeroErrRate', ...
    'FPKMCorr', 'p', 'perfectFPKMCorr', 'perfectp', ...
    'decodedGeneNum', 'perfectDecodedGeneNum', 'FPKM', 'matchThresh');