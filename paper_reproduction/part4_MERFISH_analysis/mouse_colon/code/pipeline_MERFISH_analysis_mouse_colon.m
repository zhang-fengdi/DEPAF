clear all;
close all;

% -------------------------- Parameter Settings ---------------------------
fovId = 173; % Default mouse colon FOV bundled with this reproduction folder
savePath = '..\models\'; % Path to save models and generated prediction/evaluation results
trainDataSelectNum = 210; % Number of training images randomly selected from the dataset
dataOrgPath = '..\datasets\settings\data_organization.csv';
codebookCsvPath = '..\datasets\settings\Fibl2v1E1_codebook.csv';
fpkmMatbPath = '..\datasets\settings\FPKM.matb';
colorTransformPath = '..\datasets\settings\Transforms.matb';
filePath = '..\datasets\raw\'; % Path of raw mouse colon TIFF files
datasetRoot = '..\datasets\processed\'; % Generated DEPAF input files
cameraOrientation = [0 0 1]; % Match the conventional MERFISH decoder orientation

% --------------------------- Data Organization ---------------------------
% Load DEPAF source code:
addpath(genpath('..\..\..\DEPAF_src_replication'));
addpath(pwd, '-begin');

% Check if required raw data and setting files exist:
ensureDirectory(datasetRoot);
ensureDirectory(savePath);
requireDirectory(filePath);
requireFile(dataOrgPath);
requireFile(codebookCsvPath);
requireFile(fpkmMatbPath);
requireFile(colorTransformPath);

colorTransforms = LoadByteStreamLocal(colorTransformPath);
bitInfo = loadMouseColonBitInfo(dataOrgPath);
bitNum = numel(bitInfo);
sliceNum = numel(bitInfo(1).signalFrames);
[widI, lenI] = getMouseColonImageSize(filePath, fovId, bitInfo(1));

% Check if data files exist to determine whether data organization is needed:
dataAllPath = fullfile(datasetRoot, 'data_all.mat');
bitDataPaths = cell(bitNum, 1);
for bitIdx = 1:bitNum
    bitDataPaths{bitIdx} = fullfile(datasetRoot, sprintf('data_bit%02d.mat', bitIdx));
end

datasetInfoPath = fullfile(datasetRoot, 'dataset_info.mat');
bitManifestPath = fullfile(datasetRoot, 'bit_manifest.csv');

dataFilesMissing = exist(dataAllPath, 'file') ~= 2;
for bitIdx = 1:bitNum
    if exist(bitDataPaths{bitIdx}, 'file') ~= 2
        dataFilesMissing = true;
        break;
    end
end

if dataFilesMissing
    disp('Organizing and generating mouse colon image data...');
    [selectedTrainingIndices, referenceBitIndex] = prepareMouseColonDatasets( ...
        filePath, fovId, bitInfo, widI, lenI, ...
        trainDataSelectNum, colorTransforms, cameraOrientation, ...
        dataAllPath, bitDataPaths);

    bitManifest = buildBitManifestTable(bitInfo, fovId);
    writetable(bitManifest, bitManifestPath);
    save(datasetInfoPath, 'fovId', 'bitInfo', 'bitNum', 'sliceNum', ...
        'widI', 'lenI', 'selectedTrainingIndices', 'referenceBitIndex');
else
    disp('Mouse colon data files already exist; reusing prepared .mat inputs.');
end

% --------------------------- Generate POI Data ---------------------------
poiGlob = dir(fullfile(datasetRoot, ...
    'fitted_elliptical_Gaussian_PSF_MERFISH_analysis_xSigma*_ySigma*.mat'));
POIFilesMissing = isempty(poiGlob);

if POIFilesMissing
    disp('Fitting and generating POI data...');
    IAll = loadData(dataAllPath);
    fitPointsNumPerI = 30;
    tol = 0.99;
    [PSF, xSigma, ySigma] = fit2DGaussPOI(IAll, fitPointsNumPerI, tol);
    POISaveFileName = sprintf('fitted_elliptical_Gaussian_PSF_MERFISH_analysis_xSigma%.2f_ySigma%.2f.mat', ...
        xSigma, ySigma);
    save(fullfile(datasetRoot, POISaveFileName), 'PSF');
    clear IAll;
end

% ---------------------------- Codebook Setup -----------------------------
[Codebook, bitNames, isBlank] = loadMouseColonCodebook(codebookCsvPath, fpkmMatbPath);
if numel(bitNames) ~= bitNum
    error('Codebook bit count (%d) does not match mouse colon data organization (%d).', ...
        numel(bitNames), bitNum);
end

dataBitNames = string({bitInfo.bitName});
if ~isequal(dataBitNames(:), bitNames(:))
    error(['Mouse colon data_organization bit order does not match the codebook bit_names order. ', ...
        'This would invalidate decoding; please inspect the bit ordering.']);
end

% ----------------------------- Model Setup -------------------------------
disp('Using bundled mouse colon model.');
% Load the bundled mouse colon model:
modelPath = fullfile(savePath, 'Mdl_04-27-15-58-00_by_data_all.mat');
requireFile(modelPath);
modelInfo = load(modelPath, 'modelID');
modelID = string(modelInfo.modelID);

% --------------------------- Model Prediction ----------------------------
disp('Processing image data with the model...');
% Set prediction parameters:
patchSize = [256 256];
batchSize = 16;
patchStride = [128 128];
outputISyn = false;
outputBG = false;
useGPU = true;

predPaths = cell(bitNum, 1);
for bitIdx = 1:bitNum
    predPaths{bitIdx} = fullfile(savePath, ...
        sprintf('PredRes_data_bit%02d_by_Mdl_%s.mat', bitIdx, char(modelID)));
    if exist(predPaths{bitIdx}, 'file') ~= 2
        DEPAFPred(modelPath, bitDataPaths{bitIdx}, ...
            'batchSize', batchSize, ...
            'patchSize', patchSize, ...
            'patchStride', patchStride, ...
            'resSavePath', savePath, ...
            'outputISyn', outputISyn, ...
            'outputBG', outputBG, ...
            'useGPU', useGPU);
    end
end

% ----------------------------- Gene Decoding -----------------------------
disp('Building post-pairwise candidates and decoding localization results...');
% Build pairwise localization candidates:
matchThresh = 1.0;
cropBorder = 10;
candidatePath = fullfile(savePath, 'pairwise_amp_candidates.mat');
evalPath = fullfile(savePath, sprintf('EvalRes_Mdl_%s.mat', char(modelID)));

if exist(candidatePath, 'file') ~= 2
    buildPairwiseAmplitudeCandidates(predPaths, matchThresh, cropBorder, candidatePath);
else
    disp('Pairwise candidate file already exists; reusing saved candidates.');
end

% ------------------ Post-pairwise Coordinate Descent ---------------------
disp('Optimizing bit-level amplitude pruning thresholds...');
% Optimize bit-level amplitude thresholds and decode genes:
percentileCandidates = [0 1 2 3 4 5 6 8 10 12 15 18 20 25 30 35 40 45 50];
maxRounds = 6;
thresholdPath = fullfile(savePath, 'bit_level_cd_best_thresholds.csv');
summaryPath = fullfile(savePath, 'bit_level_cd_best_summary.csv');
acceptedPath = fullfile(savePath, 'bit_level_cd_accepted_steps.csv');
trialPath = fullfile(savePath, 'bit_level_cd_trials.csv');
decodedPath = fullfile(savePath, 'decoded_barcodes_bit_level_cd_best.csv');
metricsPath = fullfile(savePath, 'depaf_baseline_aligned_metrics.csv');

if exist(thresholdPath, 'file') ~= 2 || exist(evalPath, 'file') ~= 2
    decodeResult = runPostPairwiseBitLevelCoordinateDescent( ...
        candidatePath, Codebook, isBlank, percentileCandidates, maxRounds, ...
        evalPath, summaryPath, thresholdPath, acceptedPath, trialPath, decodedPath);
else
    disp('Coordinate-descent outputs already exist; reusing saved EvalRes file.');
    decodeResult = load(evalPath);
end

computeBaselineAlignedMetrics( ...
    decodeResult.perfectDecodedGeneNum, ...
    decodeResult.decodedGeneNum - decodeResult.perfectDecodedGeneNum, ...
    Codebook, isBlank, metricsPath);
disp('Mouse colon MERFISH reproduction finished.');


function requireFile(filePath)
if exist(filePath, 'file') ~= 2
    error('Required file is missing: %s', filePath);
end
end


function requireDirectory(dirPath)
if exist(dirPath, 'dir') ~= 7
    error('Required directory is missing: %s', dirPath);
end
end


function ensureDirectory(dirPath)
if exist(dirPath, 'dir') ~= 7
    mkdir(dirPath);
end
end


function bitInfo = loadMouseColonBitInfo(dataOrgPath)
raw = readcell(dataOrgPath, 'Delimiter', ',');
headers = string(raw(1, :));
rows = raw(2:end, :);

bitNumberCol = find(headers == "bitNumber", 1);
bitNameCol = find(headers == "bitName", 1);
imageTypeCol = find(headers == "imageType", 1);
imagingRoundCol = find(headers == "imagingRound", 1);
imagingColorCol = find(headers == "color", 1);
imagingCameraIDCol = find(headers == "imagingCameraID", 1);
frameCol = find(headers == "frame", 1);
zPosCol = find(headers == "zPos", 1);
fidImageTypeCol = find(headers == "fiducialImageType", 1);
fidImagingRoundCol = find(headers == "fiducialImagingRound", 1);
fidFrameCol = find(headers == "fiducialFrame", 1);
fidCameraIDCol = find(headers == "fiducialCameraID", 1);

validIdx = false(size(rows, 1), 1);
bitNumbers = nan(size(rows, 1), 1);
for i = 1:size(rows, 1)
    bitNumbers(i) = str2double(string(rows{i, bitNumberCol}));
    bitName = string(rows{i, bitNameCol});
    isMerfishBit = startsWith(bitName, "RS");
    validIdx(i) = isfinite(bitNumbers(i)) && isMerfishBit;
end

rows = rows(validIdx, :);
bitNumbers = bitNumbers(validIdx);
[bitNumbers, sortIdx] = sort(bitNumbers);
rows = rows(sortIdx, :);

bitInfo = repmat(struct( ...
    'bitNumber', 0, ...
    'bitName', "", ...
    'imageType', "", ...
    'imagingRound', 0, ...
    'imagingColor', "", ...
    'imagingCameraID', "", ...
    'signalFrames', [], ...
    'zPos', [], ...
    'fiducialImageType', "", ...
    'fiducialImagingRound', 0, ...
    'fiducialFrame', 0, ...
    'fiducialCameraID', ""), numel(bitNumbers), 1);

for i = 1:numel(bitNumbers)
    bitInfo(i).bitNumber = bitNumbers(i);
    bitInfo(i).bitName = string(rows{i, bitNameCol});
    bitInfo(i).imageType = string(rows{i, imageTypeCol});
    bitInfo(i).imagingRound = str2double(string(rows{i, imagingRoundCol}));
    bitInfo(i).imagingColor = strtrim(string(rows{i, imagingColorCol}));
    bitInfo(i).imagingCameraID = string(rows{i, imagingCameraIDCol});
    bitInfo(i).signalFrames = parseNumberList(rows{i, frameCol});
    bitInfo(i).zPos = parseNumberList(rows{i, zPosCol});
    bitInfo(i).fiducialImageType = string(rows{i, fidImageTypeCol});
    bitInfo(i).fiducialImagingRound = str2double(string(rows{i, fidImagingRoundCol}));
    bitInfo(i).fiducialFrame = str2double(string(rows{i, fidFrameCol}));
    bitInfo(i).fiducialCameraID = string(rows{i, fidCameraIDCol});
end

sliceCounts = arrayfun(@(x) numel(x.signalFrames), bitInfo);
if numel(unique(sliceCounts)) ~= 1
    error('Mouse colon bit slices are not consistent across bits.');
end
end


function numbers = parseNumberList(value)
if isnumeric(value)
    numbers = value(:)';
    return;
end

value = strtrim(char(string(value)));
if isempty(value)
    numbers = [];
    return;
end

tokens = regexp(value, '[-+]?\d*\.?\d+', 'match');
numbers = str2double(tokens);
end


function [widI, lenI] = getMouseColonImageSize(rawDataDir, fovId, bitSpec)
samplePath = buildMouseColonImagePath(rawDataDir, fovId, bitSpec.imageType, ...
    bitSpec.imagingRound, bitSpec.imagingCameraID);
[widI, lenI] = loadDataSize(samplePath);
end


function [selectedTrainingIndices, referenceBitIndex] = prepareMouseColonDatasets( ...
    rawDataDir, fovId, bitInfo, widI, lenI, ...
    trainDataSelectNum, colorTransforms, cameraOrientation, ...
    dataAllPath, bitDataPaths)
bitNum = numel(bitInfo);
sliceNum = numel(bitInfo(1).signalFrames);

% Initialize image matrix and maximum feature points:
beadI = zeros(widI, lenI, bitNum, 'single');
featureCounts = zeros(bitNum, 1);

% Load and process fiducial images, also calculate feature points:
disp('Loading and processing calibration bead / fiducial images...');
for bitIdx = 1:bitNum
    spec = bitInfo(bitIdx);
    fiducialPath = buildMouseColonImagePath(rawDataDir, fovId, ...
        spec.fiducialImageType, spec.fiducialImagingRound, spec.fiducialCameraID);
    singleBeadI = loadData(fiducialPath, 'all', 'all', spec.fiducialFrame);
    singleBeadI = squeeze(singleBeadI(:, :, 1));
    singleBeadI = normalizeUnitRange(singleBeadI);
    beadI(:, :, bitIdx) = singleBeadI;
    featureCounts(bitIdx) = detectKAZEFeatures(singleBeadI).Count;
end

if ~any(featureCounts > 0)
    error('No KAZE features were detected in mouse colon fiducial images.');
end

[~, referenceBitIndex] = max(featureCounts);
referenceImage = beadI(:, :, referenceBitIndex);

% Calculate transformation matrices for registration:
disp('Calculating transformation matrices for registration...');
transforms = cell(bitNum, 1);
for bitIdx = 1:bitNum
    if bitIdx == referenceBitIndex
        transforms{bitIdx} = affine2d(eye(3));
    else
        transforms{bitIdx} = getTform(referenceImage, beadI(:, :, bitIdx));
    end
end

% Load and process signal images, and save data for prediction:
disp('Loading and processing signal images and saving...');
numIAll = bitNum * sliceNum;
IAll = zeros(widI, lenI, numIAll, 'uint16');
count = 0;

for bitIdx = 1:bitNum
    spec = bitInfo(bitIdx);
    signalPath = buildMouseColonImagePath(rawDataDir, fovId, ...
        spec.imageType, spec.imagingRound, spec.imagingCameraID);
    signalStack = loadData(signalPath, 'all', 'all', spec.signalFrames);
    I = zeros(widI, lenI, sliceNum, 'uint16');

    for sliceIdx = 1:sliceNum
        singleI = signalStack(:, :, sliceIdx);
        localTransforms = selectColorTransformsForBit(colorTransforms, spec.imagingColor);
        singleI = applyColorTransformsLikeBaseline(singleI, localTransforms);
        singleI = imwarp(singleI, transforms{bitIdx}, ...
            'OutputView', imref2d([widI, lenI]));
        singleI = applyCameraOrientationLikeBaseline(singleI, cameraOrientation);
        singleI = scaleToUint16(singleI);

        count = count + 1;
        if count == 1
            refImg = singleI;
            refCdf = cumsum(imhist(refImg, 65536));
            refCdf = refCdf ./ refCdf(end);
        else
            singleI = bleachCorrection(refImg, singleI, refCdf);
        end

        I(:, :, sliceIdx) = singleI;
        IAll(:, :, count) = singleI;
    end

    save(bitDataPaths{bitIdx}, 'I');
end

% Randomly select training data:
selectedCount = min(trainDataSelectNum, numIAll);
selectedTrainingIndices = randperm(numIAll, selectedCount);
IAll = IAll(:, :, selectedTrainingIndices);
save(dataAllPath, 'IAll');
end


function tableOut = buildBitManifestTable(bitInfo, fovId)
numBits = numel(bitInfo);
fovCol = repmat(fovId, numBits, 1);
bitNumberCol = reshape([bitInfo.bitNumber], [], 1);
bitNameCol = reshape(string({bitInfo.bitName}), [], 1);
imageTypeCol = reshape(string({bitInfo.imageType}), [], 1);
imagingRoundCol = reshape([bitInfo.imagingRound], [], 1);
imagingColorCol = reshape(string({bitInfo.imagingColor}), [], 1);
cameraCol = reshape(string({bitInfo.imagingCameraID}), [], 1);
frameListCol = strings(numBits, 1);
fidImageTypeCol = reshape(string({bitInfo.fiducialImageType}), [], 1);
fidRoundCol = reshape([bitInfo.fiducialImagingRound], [], 1);
fidFrameCol = reshape([bitInfo.fiducialFrame], [], 1);
fidCameraCol = reshape(string({bitInfo.fiducialCameraID}), [], 1);

for i = 1:numBits
    frameListCol(i) = sprintf('%d ', bitInfo(i).signalFrames);
    frameListCol(i) = strtrim(frameListCol(i));
end

tableOut = table(fovCol, bitNumberCol, bitNameCol, imageTypeCol, ...
    imagingRoundCol, imagingColorCol, cameraCol, frameListCol, fidImageTypeCol, ...
    fidRoundCol, fidFrameCol, fidCameraCol, ...
    'VariableNames', {'fov_id', 'bit_number', 'bit_name', 'image_type', ...
    'imaging_round', 'imaging_color', 'imaging_camera_id', 'signal_frames', ...
    'fiducial_image_type', 'fiducial_imaging_round', ...
    'fiducial_frame', 'fiducial_camera_id'});
end


function filePath = buildMouseColonImagePath(rawDataDir, fovId, imageType, imagingRound, cameraID)
if imagingRound < 0
    fileName = sprintf('%s_%03d_%s.tif', char(imageType), fovId, char(cameraID));
else
    fileName = sprintf('%s_%03d_%02d_%s.tif', ...
        char(imageType), fovId, imagingRound, char(cameraID));
end
filePath = fullfile(rawDataDir, fileName);
requireFile(filePath);
end


function normalizedImage = normalizeUnitRange(imageData)
imageData = single(imageData);
imgMin = min(imageData(:));
imgMax = max(imageData(:));
if imgMax > imgMin
    normalizedImage = (imageData - imgMin) ./ (imgMax - imgMin);
else
    normalizedImage = zeros(size(imageData), 'single');
end
end


function imageOut = scaleToUint16(imageIn)
imageIn = single(imageIn);
imgMin = min(imageIn(:));
imgMax = max(imageIn(:));
if imgMax > imgMin
    imageOut = uint16((imageIn - imgMin) ./ (imgMax - imgMin) * 65535);
else
    imageOut = zeros(size(imageIn), 'uint16');
end
end


function localTransforms = selectColorTransformsForBit(colorTransforms, imagingColor)
if isempty(colorTransforms)
    localTransforms = struct('color', {}, 'type', {}, 'transform', {});
    return;
end

imagingColor = char(strtrim(string(imagingColor)));
if isempty(imagingColor)
    localTransforms = struct('color', {}, 'type', {}, 'transform', {});
    return;
end

matchIdx = strcmp({colorTransforms.color}, imagingColor);
localTransforms = colorTransforms(matchIdx);
end


function imageOut = applyColorTransformsLikeBaseline(imageIn, localTransforms)
imageOut = imageIn;
if isempty(localTransforms)
    return;
end

ra = imref2d(size(imageOut));
for t = 1:numel(localTransforms)
    transformType = lower(char(localTransforms(t).type));
    switch transformType
        case {'similarity', 'euclidean'}
            trans = affine2d(localTransforms(t).transform);
            imageOut = imwarp(imageOut, trans, 'OutputView', ra);
        case 'invert'
            if localTransforms(t).transform(1)
                imageOut = imageOut(:, end:-1:1);
            end
            if numel(localTransforms(t).transform) >= 2 && localTransforms(t).transform(2)
                imageOut = imageOut(end:-1:1, :);
            end
        otherwise
            error('Unsupported color transform type: %s', localTransforms(t).type);
    end
end
end


function imageOut = applyCameraOrientationLikeBaseline(imageIn, cameraOrientation)
imageOut = imageIn;
if cameraOrientation(3)
    imageOut = transpose(imageOut);
end
if cameraOrientation(1)
    imageOut = flip(imageOut, 2);
end
if cameraOrientation(2)
    imageOut = flip(imageOut, 1);
end
end


function [Codebook, bitNames, isBlank] = loadMouseColonCodebook(codebookCsvPath, fpkmMatbPath)
raw = readcell(codebookCsvPath, 'Delimiter', ',');
headerRows = string(raw(:, 1));

bitNameRow = find(headerRows == "bit_names", 1);
dataHeaderRow = find(headerRows == "name", 1);
if isempty(bitNameRow) || isempty(dataHeaderRow)
    error('Unexpected mouse colon codebook format: %s', codebookCsvPath);
end

bitNameCells = raw(bitNameRow, 2:end);
bitNameMask = ~cellfun(@(x) isempty(x) || strlength(string(x)) == 0, bitNameCells);
bitNames = string(bitNameCells(bitNameMask));

dataRows = raw(dataHeaderRow+1:end, 1:3);
validRows = ~cellfun(@(x) isempty(x) || strlength(string(x)) == 0, dataRows(:, 1));
dataRows = dataRows(validRows, :);

abundData = LoadByteStreamLocal(fpkmMatbPath);
abundNames = string({abundData.geneName}');
abundValues = [abundData.FPKM]';

Codebook = repmat(struct( ...
    'GeneShortName', '', ...
    'GeneID', '', ...
    'Code', '', ...
    'FPKM', nan), size(dataRows, 1), 1);
isBlank = false(size(dataRows, 1), 1);

for i = 1:size(dataRows, 1)
    geneName = normalizeCellString(dataRows{i, 1});
    geneId = normalizeCellString(dataRows{i, 2});
    barcodeText = char(normalizeCellString(dataRows{i, 3}));
    codeBits = regexp(barcodeText, '[01]', 'match');
    codeChar = char([codeBits{:}]);

    if numel(codeChar) ~= numel(bitNames)
        error('Code length mismatch for %s: expected %d bits, found %d.', ...
            geneName, numel(bitNames), numel(codeChar));
    end

    Codebook(i).GeneShortName = char(geneName);
    Codebook(i).GeneID = char(geneId);
    Codebook(i).Code = codeChar;
    isBlank(i) = startsWith(char(geneName), 'Blank-');

    fpkmIdx = find(abundNames == geneName, 1);
    if isempty(fpkmIdx)
        Codebook(i).FPKM = nan;
    else
        Codebook(i).FPKM = abundValues(fpkmIdx);
    end
end
end


function out = normalizeCellString(value)
if isempty(value)
    out = "";
elseif isstring(value)
    if any(ismissing(value))
        out = "";
    else
        out = value(1);
    end
elseif ischar(value)
    out = string(value);
elseif isnumeric(value)
    if any(isnan(value(:)))
        out = "";
    else
        out = string(value(1));
    end
else
    out = string(value);
    if any(ismissing(out))
        out = "";
    else
        out = out(1);
    end
end

out = strtrim(out);
end


function data = LoadByteStreamLocal(filePath)
fid = fopen(filePath, 'r');
if fid < 0
    error('Could not open byte-stream file: %s', filePath);
end
cleanupObj = onCleanup(@() fclose(fid));
byteStream = fread(fid, inf, '*uint8');
data = getArrayFromByteStream(byteStream);
clear cleanupObj;
end


function buildPairwiseAmplitudeCandidates(predPaths, matchThresh, cropBorder, candidatePath)
bitNum = numel(predPaths);

% Initialize candidate words and amplitude traces:
allBitLoc = cell(bitNum, 1);
words = cell(bitNum, 1);
ampTraces = cell(bitNum, 1);
seedSlice = cell(bitNum, 1);
sliceNum = [];
widI = [];
lenI = [];

for bitIdx = 1:bitNum
    [locCells, localWidI, localLenI, amplitudeCells] = loadData(predPaths{bitIdx});
    if isempty(sliceNum)
        sliceNum = numel(locCells);
        widI = localWidI;
        lenI = localLenI;
    elseif sliceNum ~= numel(locCells)
        error('Slice count mismatch at bit %d.', bitIdx);
    end

    allBitLoc{bitIdx} = cell(sliceNum, 1);
    words{bitIdx} = cell(sliceNum, 1);
    ampTraces{bitIdx} = cell(sliceNum, 1);
    seedSlice{bitIdx} = cell(sliceNum, 1);

    for sliceIdx = 1:sliceNum
        loc = locCells{sliceIdx};
        amp = amplitudeCells{sliceIdx};
        amp = amp(:);

        if isempty(loc)
            loc = zeros(0, 3, 'single');
            amp = zeros(0, 1, 'single');
        end
        if size(loc, 1) ~= numel(amp)
            error('loc/amplitude row mismatch at bit %d slice %d.', bitIdx, sliceIdx);
        end

        keepMask = loc(:, 1) > cropBorder & loc(:, 1) < localLenI - cropBorder & ...
            loc(:, 2) > cropBorder & loc(:, 2) < localWidI - cropBorder;
        loc = single(loc(keepMask, :));
        amp = single(amp(keepMask));
        keptCount = size(loc, 1);

        allBitLoc{bitIdx}{sliceIdx} = loc;
        words{bitIdx}{sliceIdx} = false(keptCount, bitNum);
        words{bitIdx}{sliceIdx}(:, bitIdx) = true;
        ampTraces{bitIdx}{sliceIdx} = zeros(keptCount, bitNum, 'single');
        ampTraces{bitIdx}{sliceIdx}(:, bitIdx) = amp;
        seedSlice{bitIdx}{sliceIdx} = repmat(uint16(sliceIdx), keptCount, 1);
    end
end

% Merge matched detections across bits:
for sliceIdx = 1:sliceNum
    for i = 1:bitNum
        for j = i+1:bitNum
            assignments = fastMatchPairs(allBitLoc{i}{sliceIdx}, allBitLoc{j}{sliceIdx}, matchThresh);
            if isempty(assignments)
                continue;
            end

            rowI = assignments(:, 1);
            rowJ = assignments(:, 2);
            words{i}{sliceIdx}(rowI, :) = words{i}{sliceIdx}(rowI, :) | words{j}{sliceIdx}(rowJ, :);
            ampTraces{i}{sliceIdx}(rowI, :) = ...
                ampTraces{i}{sliceIdx}(rowI, :) + ampTraces{j}{sliceIdx}(rowJ, :);
            allBitLoc{i}{sliceIdx}(rowI, :) = ...
                0.5 * (allBitLoc{i}{sliceIdx}(rowI, :) + allBitLoc{j}{sliceIdx}(rowJ, :));

            words{j}{sliceIdx}(rowJ, :) = [];
            ampTraces{j}{sliceIdx}(rowJ, :) = [];
            allBitLoc{j}{sliceIdx}(rowJ, :) = [];
            seedSlice{j}{sliceIdx}(rowJ, :) = [];
        end
    end
end

binaryWords = vertcatNested(words);
rawAmplitudeTraces = vertcatNested(ampTraces);
seedPositions = vertcatNested(allBitLoc);
seedSlice = vertcatNested(seedSlice);
seedX = seedPositions(:, 1);
seedY = seedPositions(:, 2);
if size(seedPositions, 2) >= 3
    seedZ_or_internal = seedPositions(:, 3);
else
    seedZ_or_internal = ones(size(seedPositions, 1), 1, 'single');
end

metadata = struct();
metadata.matchThresh = matchThresh;
metadata.cropBorder = cropBorder;
metadata.bitNum = bitNum;
metadata.sliceNum = sliceNum;
metadata.widI = widI;
metadata.lenI = lenI;
metadata.candidateCount = size(binaryWords, 1);
metadata.createdAt = char(datetime('now'));

% Save candidate matrices for decoding:
save(candidatePath, 'binaryWords', 'rawAmplitudeTraces', ...
    'seedX', 'seedY', 'seedZ_or_internal', 'seedSlice', 'metadata', '-v7.3');
disp('Saved pairwise candidates.');
end


function decodeResult = runPostPairwiseBitLevelCoordinateDescent( ...
    candidatePath, Codebook, isBlank, percentileCandidates, maxRounds, ...
    evalPath, summaryPath, thresholdPath, acceptedPath, trialPath, decodedPath)
loaded = load(candidatePath, 'binaryWords', 'rawAmplitudeTraces', ...
    'seedX', 'seedY', 'seedZ_or_internal', 'seedSlice', 'metadata');
binaryWords = logical(loaded.binaryWords);
amp = single(loaded.rawAmplitudeTraces);
bitNum = size(binaryWords, 2);

geneNames = string({Codebook.GeneShortName}');
codeDic = logical(cell2mat(cellfun(@(x) x - '0', {Codebook.Code}', 'UniformOutput', false)));
FPKM = [Codebook.FPKM]';

[lookupKeys, lookupGeneIdx, lookupHamming] = buildHammingLookup(codeDic);
thresholdTable = computeBitThresholdTable(amp, binaryWords, percentileCandidates);
percentileIdx = ones(bitNum, 1);
minObjectiveGain = 1e-6;

% Search bit-level percentile thresholds by coordinate descent:
[currentSummary, ~] = decodeForPercentileState( ...
    amp, binaryWords, percentileIdx, thresholdTable, percentileCandidates, ...
    lookupKeys, lookupGeneIdx, lookupHamming, geneNames, isBlank, FPKM, ...
    loaded, false);
currentScore = currentSummary.fpkm_corr_exact;

acceptedTable = struct2table(addCdFields(currentSummary, 0, 0, 0, 0, 0));
trialTable = table();

for roundIdx = 1:maxRounds
    roundStartScore = currentScore;
    roundChanges = 0;

    for bitIdx = 1:bitNum
        incumbentIdx = percentileIdx(bitIdx);
        bestIdx = incumbentIdx;
        bestSummary = currentSummary;
        bestScore = currentScore;

        for pctIdx = 1:numel(percentileCandidates)
            if pctIdx == incumbentIdx
                continue;
            end

            trialIdx = percentileIdx;
            trialIdx(bitIdx) = pctIdx;
            [trialSummary, ~] = decodeForPercentileState( ...
                amp, binaryWords, trialIdx, thresholdTable, percentileCandidates, ...
                lookupKeys, lookupGeneIdx, lookupHamming, geneNames, isBlank, FPKM, ...
                loaded, false);
            trialScore = trialSummary.fpkm_corr_exact;
            wouldAccept = trialScore > bestScore + minObjectiveGain;

            trialRow = addTrialFields(trialSummary, roundIdx, bitIdx, ...
                percentileCandidates(incumbentIdx), percentileCandidates(pctIdx), wouldAccept, trialScore);
            trialTable = appendTable(trialTable, struct2table(trialRow));

            if wouldAccept
                bestIdx = pctIdx;
                bestSummary = trialSummary;
                bestScore = trialScore;
            end
        end

        if bestIdx ~= incumbentIdx
            previousScore = currentScore;
            percentileIdx(bitIdx) = bestIdx;
            currentSummary = bestSummary;
            currentScore = bestScore;
            roundChanges = roundChanges + 1;
            acceptedRow = addCdFields(currentSummary, roundIdx, bitIdx, ...
                percentileCandidates(incumbentIdx), percentileCandidates(bestIdx), ...
                currentScore - previousScore);
            acceptedTable = appendTable(acceptedTable, struct2table(acceptedRow));
        end
    end

    roundGain = currentScore - roundStartScore;
    if roundChanges == 0 || roundGain < minObjectiveGain
        break;
    end
end

[finalSummary, finalDecoded] = decodeForPercentileState( ...
    amp, binaryWords, percentileIdx, thresholdTable, percentileCandidates, ...
    lookupKeys, lookupGeneIdx, lookupHamming, geneNames, isBlank, FPKM, ...
    loaded, true);
thresholdValues = thresholdTable(sub2ind(size(thresholdTable), (1:bitNum)', percentileIdx));

thresholdRows = table((1:bitNum)', percentileCandidates(percentileIdx(:))', thresholdValues, ...
    'VariableNames', {'bit', 'percentile', 'threshold'});
writetable(thresholdRows, thresholdPath);
writetable(struct2table(finalSummary), summaryPath);
writetable(acceptedTable, acceptedPath);
if ~isempty(trialTable)
    writetable(trialTable, trialPath);
end
writetable(finalDecoded.decodedTable, decodedPath);

decodeRes = finalDecoded.decodeRes;
perfectDecodeRes = finalDecoded.perfectDecodeRes;
decodedGeneNum = finalDecoded.decodedGeneNum;
perfectDecodedGeneNum = finalDecoded.perfectDecodedGeneNum;
correctionRate = finalSummary.correction_rate;
FPKMCorr = finalSummary.fpkm_corr_all;
perfectFPKMCorr = finalSummary.fpkm_corr_exact;
matchThresh = loaded.metadata.matchThresh;
save(evalPath, 'decodeRes', 'perfectDecodeRes', 'decodedGeneNum', ...
    'perfectDecodedGeneNum', 'FPKM', 'correctionRate', 'FPKMCorr', ...
    'perfectFPKMCorr', 'matchThresh', 'thresholdRows');

decodeResult = load(evalPath);
end


function [summary, decoded] = decodeForPercentileState( ...
    amp, binaryWords, percentileIdx, thresholdTable, percentileCandidates, ...
    lookupKeys, lookupGeneIdx, lookupHamming, geneNames, isBlank, FPKM, ...
    candidateData, keepDecodedTable)
bitNum = size(binaryWords, 2);
numGenes = numel(geneNames);
thresholdValues = thresholdTable(sub2ind(size(thresholdTable), (1:bitNum)', percentileIdx(:)));
prunedWords = binaryWords & bsxfun(@ge, amp, thresholdValues(:)');
keys = packWords(prunedWords);
[acceptedMask, lookupPos] = ismember(keys, lookupKeys);
acceptedIdx = find(acceptedMask);
lookupPos = lookupPos(acceptedMask);
geneIdx = double(lookupGeneIdx(lookupPos));
hammingDist = double(lookupHamming(lookupPos));
exactMask = hammingDist == 0;

decodedGeneNum = accumarray(geneIdx, 1, [numGenes, 1], @sum, 0);
perfectDecodedGeneNum = accumarray(geneIdx(exactMask), 1, [numGenes, 1], @sum, 0);
totalDecoded = numel(geneIdx);
exactCount = sum(exactMask);
blankCount = sum(isBlank(geneIdx));
exactBlankCount = sum(isBlank(geneIdx(exactMask)));
candidateOnBits = sum(prunedWords, 2);
candidateHist = histcounts(double(candidateOnBits), -0.5:1:(bitNum + 0.5));

summary = struct();
summary.threshold_min = min(thresholdValues);
summary.threshold_median = median(thresholdValues);
summary.threshold_max = max(thresholdValues);
summary.candidate_count = size(prunedWords, 1);
summary.candidate_onbit_0 = candidateHist(1);
summary.candidate_onbit_1 = candidateHist(2);
summary.candidate_onbit_2 = candidateHist(3);
summary.candidate_onbit_3 = candidateHist(4);
summary.candidate_onbit_4 = candidateHist(5);
summary.candidate_onbit_gt4 = sum(candidateHist(6:end));
summary.total_decoded = totalDecoded;
summary.exact_count = exactCount;
summary.corrected_count = totalDecoded - exactCount;
summary.blank_count = blankCount;
summary.blank_fraction = safeDivide(blankCount, totalDecoded);
summary.exact_blank_count = exactBlankCount;
summary.exact_blank_fraction = safeDivide(exactBlankCount, exactCount);
summary.nonblank_count = totalDecoded - blankCount;
summary.correction_rate = safeDivide(totalDecoded - exactCount, totalDecoded);
summary.fpkm_corr_exact = computeFpkmCorrelationNoOffset(perfectDecodedGeneNum, FPKM, isBlank);
summary.fpkm_corr_all = computeFpkmCorrelationNoOffset(decodedGeneNum, FPKM, isBlank);
summary.bit_percentiles = strjoin(string(percentileCandidates(percentileIdx(:))), ';');

decoded = struct();
decoded.decodedGeneNum = decodedGeneNum;
decoded.perfectDecodedGeneNum = perfectDecodedGeneNum;
if ~keepDecodedTable
    decoded.decodeRes = struct('Gene', {}, 'Position', {}, 'GroundtruthWord', {}, 'decodedWord', {}, 'Correction', {});
    decoded.perfectDecodeRes = decoded.decodeRes;
    decoded.decodedTable = table();
    return;
end

acceptedWords = prunedWords(acceptedIdx, :);
positionMat = [candidateData.seedX(acceptedIdx), candidateData.seedY(acceptedIdx), ...
    candidateData.seedZ_or_internal(acceptedIdx)];
decodedWordText = cellstr(char(double(acceptedWords) + '0'));
groundtruthWordText = cellstr(char(double(codeWordsFromGeneIndex(geneIdx, numGenes, lookupKeys, lookupGeneIdx, lookupHamming, bitNum)) + '0'));
geneText = cellstr(geneNames(geneIdx));
correctionFlag = hammingDist == 1;

decoded.decodeRes = struct('Gene', geneText, ...
    'Position', num2cell(positionMat, 2), ...
    'GroundtruthWord', groundtruthWordText, ...
    'decodedWord', decodedWordText, ...
    'Correction', num2cell(correctionFlag));
decoded.perfectDecodeRes = decoded.decodeRes(exactMask);

decoded.decodedTable = table(acceptedIdx, geneIdx, geneNames(geneIdx), isBlank(geneIdx), ...
    exactMask, correctionFlag, hammingDist, candidateData.seedX(acceptedIdx), ...
    candidateData.seedY(acceptedIdx), candidateData.seedZ_or_internal(acceptedIdx), ...
    candidateData.seedSlice(acceptedIdx), sum(acceptedWords, 2), ...
    'VariableNames', {'candidate_index', 'gene_index', 'gene_name', 'is_blank', ...
    'is_exact', 'is_corrected', 'hamming_distance', 'seed_x', 'seed_y', ...
    'seed_z_or_internal', 'seed_slice', 'pruned_on_bit_count'});
end


function thresholdTable = computeBitThresholdTable(amp, binaryWords, percentileCandidates)
bitNum = size(binaryWords, 2);
thresholdTable = zeros(bitNum, numel(percentileCandidates), 'single');
for bitIdx = 1:bitNum
    values = amp(binaryWords(:, bitIdx) & isfinite(amp(:, bitIdx)) & amp(:, bitIdx) > 0, bitIdx);
    if isempty(values)
        continue;
    end
    thresholdTable(bitIdx, :) = single(prctile(double(values), percentileCandidates));
    thresholdTable(bitIdx, percentileCandidates == 0) = 0;
end
end


function [lookupKeys, lookupGeneIdx, lookupHamming] = buildHammingLookup(codeDic)
[numGenes, bitNum] = size(codeDic);
weights = uint32(2 .^ (0:bitNum-1));
exactKeys = packWords(codeDic);
allKeys = zeros(numGenes * (bitNum + 1), 1, 'uint32');
allGeneIdx = zeros(numGenes * (bitNum + 1), 1, 'uint16');
allHamming = zeros(numGenes * (bitNum + 1), 1, 'uint8');
row = 0;
for geneIdx = 1:numGenes
    row = row + 1;
    allKeys(row) = exactKeys(geneIdx);
    allGeneIdx(row) = geneIdx;
    allHamming(row) = 0;
end
for geneIdx = 1:numGenes
    for bitIdx = 1:bitNum
        row = row + 1;
        allKeys(row) = bitxor(exactKeys(geneIdx), weights(bitIdx));
        allGeneIdx(row) = geneIdx;
        allHamming(row) = 1;
    end
end
[uniqueKeys, firstIdx] = unique(allKeys, 'stable');
uniqueGeneIdx = allGeneIdx(firstIdx);
uniqueHamming = allHamming(firstIdx);
[lookupKeys, order] = sort(uniqueKeys);
lookupGeneIdx = uniqueGeneIdx(order);
lookupHamming = uniqueHamming(order);
end


function keys = packWords(words)
bitNum = size(words, 2);
weights = uint32(2 .^ (0:bitNum-1));
keys = zeros(size(words, 1), 1, 'uint32');
for bitIdx = 1:bitNum
    mask = words(:, bitIdx);
    keys(mask) = bitor(keys(mask), weights(bitIdx));
end
end


function codeWords = codeWordsFromGeneIndex(geneIdx, numGenes, lookupKeys, lookupGeneIdx, lookupHamming, bitNum)
% Reconstruct exact codewords for saved EvalRes compatibility.
codeWords = false(numel(geneIdx), bitNum);
exactMask = lookupHamming == 0;
exactKeys = zeros(numGenes, 1, 'uint32');
exactKeys(double(lookupGeneIdx(exactMask))) = lookupKeys(exactMask);
weights = uint32(2 .^ (0:bitNum-1));
for bitIdx = 1:bitNum
    codeWords(:, bitIdx) = bitand(exactKeys(geneIdx), weights(bitIdx)) ~= 0;
end
end


function rho = computeFpkmCorrelationNoOffset(counts, fpkm, isBlank)
valid = ~isBlank(:) & counts(:) > 0 & fpkm(:) > 0 & isfinite(fpkm(:));
if sum(valid) < 3
    rho = nan;
    return;
end
rho = corr(log10(double(counts(valid))), log10(double(fpkm(valid))));
end


function metrics = computeBaselineAlignedMetrics(exactCounts, correctedCounts, Codebook, isBlank, metricsPath)
FPKM = [Codebook.FPKM]';
valid = ~isBlank(:) & exactCounts(:) > 0 & FPKM(:) > 0 & isfinite(FPKM(:));
if sum(valid) >= 3
    fpkmCorrExact = corr(log10(double(exactCounts(valid)) + 1), log10(double(FPKM(valid)) + 1));
else
    fpkmCorrExact = nan;
end
validCorrected = ~isBlank(:) & correctedCounts(:) > 0 & FPKM(:) > 0 & isfinite(FPKM(:));
if sum(validCorrected) >= 3
    fpkmCorrCorrected = corr(log10(double(correctedCounts(validCorrected)) + 1), ...
        log10(double(FPKM(validCorrected)) + 1));
else
    fpkmCorrCorrected = nan;
end

blankExact = sum(exactCounts(isBlank));
totalExact = sum(exactCounts);
geneExact = sum(exactCounts(~isBlank));
blankCounts = exactCounts(isBlank);
if isempty(blankCounts)
    maxBlankCount = nan;
else
    maxBlankCount = max(blankCounts);
end

metrics = table("DEPAF mouse colon post-pairwise CD", "completed", ...
    totalExact, sum(correctedCounts), geneExact, blankExact, ...
    safeDivide(blankExact, totalExact), sum(exactCounts(~isBlank) > 0), ...
    maxBlankCount, sum(exactCounts(~isBlank) > maxBlankCount), ...
    fpkmCorrExact, fpkmCorrCorrected, ...
    'VariableNames', {'method', 'status', 'total_exact', 'total_corrected', ...
    'gene_exact', 'blank_exact', 'blank_fraction', 'genes_detected', ...
    'max_blank_count', 'genes_above_max_blank', 'fpkm_corr_exact', ...
    'fpkm_corr_corrected'});
writetable(metrics, metricsPath);
end


function value = safeDivide(num, den)
if den == 0
    value = nan;
else
    value = double(num) ./ double(den);
end
end


function row = addCdFields(summary, roundIdx, bitIdx, previousPercentile, acceptedPercentile, gain)
row = summary;
row.round = roundIdx;
row.bit = bitIdx;
row.previous_percentile = previousPercentile;
row.accepted_percentile = acceptedPercentile;
row.objective_gain = gain;
end


function row = addTrialFields(summary, roundIdx, bitIdx, previousPercentile, testedPercentile, wouldAccept, objectiveValue)
row = summary;
row.round = roundIdx;
row.bit = bitIdx;
row.previous_percentile = previousPercentile;
row.tested_percentile = testedPercentile;
row.would_accept = wouldAccept;
row.objective_fpkm_corr_exact = objectiveValue;
end


function out = appendTable(existingRows, newRows)
if isempty(existingRows) || width(existingRows) == 0
    out = newRows;
else
    out = [existingRows; newRows];
end
end


function out = vertcatNested(nestedCells)
out = cell2mat(cellfun(@(x) vertcat(x{:}), nestedCells, 'UniformOutput', false));
end
