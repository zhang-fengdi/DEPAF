function DEPAFTrain(varargin)
% DEPAFTrain runs the complete DEPAF training pipeline.
%
%  This function performs the complete DEPAF training process, including data loading and preprocessing,
%  model construction, training, validation, and saving the training results. The function provides
%  several customizable parameters, allowing users to adjust the training process according to specific needs.
%
%  Input Parameters:
%    dataPath - Path containing training data.
%    lambda - Regularization coefficient.
%    POIPath - Path containing POI sample.
%    Other custom parameters, including training and validation index ranges, upsampling ratios,
%    patch size, number of training and validation patches, encoder depth, etc.
%
%  Output:
%    No direct output. The trained model and related parameters will be saved to the specified path.

% Parse parameters:
userSetParams = iParseInputs(varargin{:});
dataPath = userSetParams.Results.dataPath;
lambda = userSetParams.Results.lambda;
POIPath = userSetParams.Results.POIPath;
trainIdxRange = userSetParams.Results.trainIdxRange;
valIdxRange = userSetParams.Results.valIdxRange;
upsamplRatio = userSetParams.Results.upsamplRatio;
interpMethod = userSetParams.Results.interpMethod;
patchSize = userSetParams.Results.patchSize;
trainPatchNum = userSetParams.Results.trainPatchNum;
valPatchNum = userSetParams.Results.valPatchNum;
encoderDepth = userSetParams.Results.encoderDepth;
learningRate = userSetParams.Results.learningRate;
minLR = userSetParams.Results.minLR;
miniBatchSize = userSetParams.Results.miniBatchSize;
maxEpochs = userSetParams.Results.maxEpochs;
valFreq = userSetParams.Results.valFreq;
maxPatience = userSetParams.Results.maxPatience;
learnBG = userSetParams.Results.learnBG;
verbose = userSetParams.Results.verbose;
useGPU = userSetParams.Results.useGPU;
useParallel = userSetParams.Results.useParallel;
parNum = userSetParams.Results.parNum;
patchNumForThreshSearch =  userSetParams.Results.patchNumForThreshSearch;
modelSavePath = userSetParams.Results.modelSavePath;

% Load POI:
disp('[Training] Loading POI...');
POI = loadData(POIPath);

% Upsample and normalize POI:
disp('[Training] Upsampling and normalizing POI...');
POI = POI - min(POI, [], [1 2]);
disp(['[Training] Performing ' num2str(upsamplRatio(2)) 'x upsampling in x, ' ...
    num2str(upsamplRatio(1)) 'x in y, and ' num2str(upsamplRatio(3)) 'x in z for POI...']);
for dim = 1:3
    if mod(ceil(size(POI,dim)*upsamplRatio(dim)),2) == 0
        upsamplRatio(dim) = ...
            (ceil(size(POI,dim)*upsamplRatio(dim))+0.5)/size(POI,dim);
        disp(['[Training] Detected even dimension after upsampling POI in ' char('x'+(dim-1)) ...
            ' direction; adjusting upsampling ratio to ' num2str(upsamplRatio(dim)) 'x.']);
    end
end
POI = max(0, imgResizeCenterKeep(POI, upsamplRatio, interpMethod));
POI = POI ./ max(POI, [], [1 2]);
POI = POI ./ mean(sum(POI,[1 2]));
POI(isnan(POI)) = 0;

% Generate background POI:
if learnBG
    BGPOI = BGPOIGeneration(POI);
else
    BGPOI = [];
end

% Randomly sample and load training data based on specified training and validation patch sample counts:
disp('[Training] Loading training data...');
whichI = iSampleSequence(trainIdxRange, trainPatchNum);
uniqueWhichI = unique(whichI);
I = loadData(dataPath, 'all', 'all', uniqueWhichI);
whichValI = iSampleSequence(valIdxRange, valPatchNum);
uniqueWhichValI = unique(whichValI);
valI = loadData(dataPath, 'all', 'all', uniqueWhichValI);

% Upsample and normalize data:
disp('[Training] Upsampling and normalizing data...');
normMin = min(I, [], 'all');
I = I - normMin;
valI = valI - normMin;
disp(['[Training] Performing ' num2str(upsamplRatio(2)) 'x upsampling in x and ' ...
    num2str(upsamplRatio(1)) 'x in y for data...']);
I = max(0, imgResizeCenterKeep(I, [upsamplRatio(1:2) 1], interpMethod));
valI = max(0, imgResizeCenterKeep(valI, [upsamplRatio(1:2) 1], interpMethod));
normMax = prctile(I, 99.9, 'all');
I = I ./ normMax;
valI = valI ./ normMax;

% Get dimensions after upsampling:
[widI, lenI] = size(I,[1 2]);

% Collect training set patches:
disp('[Training] Collecting training patches...');
IPatch = zeros(patchSize(1), patchSize(2), 1, trainPatchNum, 'single');
for i = 1:trainPatchNum
    iPatch = randi(widI-patchSize(1)+1,1,1);
    jPatch = randi(lenI-patchSize(2)+1,1,1);
    IPatch(:,:,1,i) = I(iPatch:iPatch+patchSize(1)-1, ...
        jPatch:jPatch+patchSize(2)-1, whichI(i)==uniqueWhichI);
end
clear I;

% Collect validation set patches:
disp('[Training] Collecting validation patches...');
valIPatch = zeros(patchSize(1), patchSize(2), 1, valPatchNum, 'single');
for i = 1:valPatchNum
    iPatch = randi(widI-patchSize(1)+1,1,1);
    jPatch = randi(lenI-patchSize(2)+1,1,1);
    valIPatch(:,:,1,i) = valI(iPatch:iPatch+patchSize(1)-1,...
        jPatch:jPatch+patchSize(2)-1, whichValI(i)==uniqueWhichValI);
end
clear valI;

% Reshape POI into 5-D format for grouped convolution (width × height × channels per group × number of POIs per group × group count):
[widPOI, lenPOI, channelNum] = size(POI);
POI = reshape(POI, widPOI, lenPOI, 1, 1, channelNum);

% Train POI fitting and denoising model:
disp('[Training] Training POI fitting and denoising model...');
[dlnet, finalEpoch, minValLoss, bestEpoch] = trainModel(IPatch, ...
    valIPatch, POI, BGPOI, learningRate, miniBatchSize, maxEpochs, ...
    encoderDepth, lambda, valFreq, maxPatience, ...
    minLR, verbose, useGPU);
clear valIPatch;

% Save model:
disp('[Training] Saving model...');
checkPath(modelSavePath); % Check path
[~, dataName, dataExt] = fileparts(dataPath); % Get data file name
modelID = char(datetime('now','Format','MM-dd-HH-mm-ss')); % Model ID
savePath = [modelSavePath '\Mdl_' modelID '_by_' dataName '.mat'];
save(savePath, 'dlnet', 'finalEpoch', 'minValLoss', 'bestEpoch');

% Randomly select samples for optimal segmentation threshold search (using all samples may cause memory overflow):
patchIdxForThreshSearching = randperm(trainPatchNum, min(trainPatchNum, patchNumForThreshSearch));
IPatch = IPatch(:,:,:,patchIdxForThreshSearching);

% Predict training set images using the model in batches:
disp('[Training] Predicting training images using the model...');
progressDisp(patchNumForThreshSearch); % Initialize progress bar
IFitPatch = zeros(patchSize(1), patchSize(2), channelNum, patchNumForThreshSearch, 'single');
IDenoisedPatch = zeros(patchSize(1), patchSize(2), channelNum, patchNumForThreshSearch, 'single');
for i = 1:miniBatchSize:patchNumForThreshSearch
    % Batch range:
    batchRange = i:min(i+miniBatchSize-1, patchNumForThreshSearch);
    % Get batch:
    IPatchBatch = IPatch(:,:,:,batchRange);
    IPatchBatch = dlarray(IPatchBatch,'SSCB');
    if useGPU
        IPatchBatch = gpuArray(IPatchBatch);
    end

    % Predict training batch using model:
    [IFitPatchBatch, IDenoisedPatchBatch, BGPatchBatch] = ...
        modelPred(dlnet, IPatchBatch, POI, BGPOI, [], 'predict');

    % Remove background:
    IPatchBatch = IPatchBatch - BGPatchBatch;

    % Record predictions:
    IPatch(:,:,:,batchRange) = gather(extractdata(IPatchBatch));
    IFitPatch(:,:,:,batchRange) = gather(extractdata(IFitPatchBatch));
    IDenoisedPatch(:,:,:,batchRange) = gather(extractdata(IDenoisedPatchBatch));

    % Refresh progress bar:
    progressDisp(0, min(miniBatchSize, patchNumForThreshSearch-i+1));
end
progressDisp(-1); % End progress bar
clear dlnet IPatchBatch IFitPatchBatch IDenoisedPatchBatch BGPatchBatch;

% Sum channels for IDenoisedPatch:
IDenoisedPatch = sum(IDenoisedPatch, 3);

% Reshape POI back to 3-D format:
POI = reshape(POI, widPOI, lenPOI, channelNum);

% Create parallel pool:
if useParallel
    disp('[Training] Creating parallel pool...');
    createParPool(parNum, 20);
end

% Perform Bayesian optimization for coarse optimal segmentation threshold search:
disp('[Training] Performing Bayesian optimization for coarse threshold search...');
if verbose
    plotFcn = {@plotObjectiveModel};
else
    plotFcn = [];
end
threshRangeUpLimit = prctile(sum(IFitPatch, 3), 99.99, 'all');
threshRangeDownLimit = prctile(sum(IFitPatch, 3), 0.01, 'all');
threshRange = [threshRangeDownLimit threshRangeUpLimit];
thresh = optimizableVariable('thresh',threshRange);
fun = @(x) threshSearchingLoss( ...
    IPatch, ...
    IFitPatch, ...
    IDenoisedPatch, ...
    POI, ...
    x.thresh, ...
    interpMethod);
results = bayesopt(fun, thresh, ...
    'AcquisitionFunctionName', 'expected-improvement', ...
    'IsObjectiveDeterministic', true, ...
    'UseParallel', useParallel, ...
    'MaxObjectiveEvaluations', 80, ...
    'MaxTime', 3600, ...
    'Verbose', 0, ...
    'plotFcn', plotFcn);
suboptimalThresh = results.XAtMinObjective{1,1};

% Use grid search for fine optimal segmentation threshold search:
disp('[Training] Performing grid search for fine threshold search...');
stepSize = (threshRangeUpLimit - threshRangeDownLimit) / 3000;
stepNum = 100;
halfThreshRange = stepNum / 2 * stepSize;
for attemp = 1:5 % Multiple attempts to prevent minimum loss point from being outside range
    threshRange = [max(suboptimalThresh-halfThreshRange,threshRangeDownLimit) ...
        min(suboptimalThresh+halfThreshRange-stepSize,threshRangeUpLimit)];
    [optimalThresh, BOLoss, BOThreshList] = threshSearching( ...
        IPatch, ...
        IFitPatch, ...
        IDenoisedPatch, ...
        POI, threshRange, stepSize, interpMethod, useParallel);
    if min(BOLoss) == BOLoss(1) && threshRange(1) > threshRangeDownLimit
        suboptimalThresh = BOThreshList(1) - halfThreshRange + 10 * stepSize;
    elseif min(BOLoss) == BOLoss(end) && threshRange(2) < threshRangeUpLimit
        suboptimalThresh = BOThreshList(end) + halfThreshRange - 10 * stepSize;
    else
        break;
    end
end

% Display threshold search loss:
if verbose
    figure,plot(BOThreshList,BOLoss,'LineWidth',3,'Color','r');
    title('Threshold Search Loss');
    ylabel('Loss');
    xlabel('Threshold');
    legend('Train Loss');
end

% Fit amplitude distribution parameters:
disp('[Training] Fitting amplitude distribution...');
[~, amplitude] = getLocAndAmp( ...
    IPatch, IFitPatch, IDenoisedPatch, POI, optimalThresh, interpMethod);
amplitude = cell2mat(amplitude);
try
    amplitudeDistMdl = fitdist(amplitude, 'Normal');
catch ME
    fprintf('Error in fitting: %s\n', ME.message);
    amplitudeDistMdl = ME.message;
end

% Save key parameters:
disp('[Training] Saving key parameters...');
dataFileName = [dataName dataExt];
[~, POIName, POIExt] = fileparts(POIPath); % Get POI file name
POIFileName = [POIName POIExt];
save(savePath, ...
    'POI', 'upsamplRatio', 'interpMethod', 'normMax', 'normMin', ...
    'encoderDepth', 'optimalThresh', ...
    'modelID', 'dataFileName', 'POIFileName', ...
    'learnBG', 'lambda', 'userSetParams', 'amplitudeDistMdl', ...
    'BOLoss', 'BOThreshList', ...
    '-append');
end


% Helper function 1: Random sampling to maximize data utilization.
function sampledIdx = iSampleSequence(dataIdx, sampleSize)
% Calculate basic sampling count:
basicSamplesPerElement = floor(sampleSize / length(dataIdx));

% Ensure each number is sampled the basic number of times:
sampledIdx = repmat(dataIdx, 1, basicSamplesPerElement);

% Calculate additional sampling count:
additionalSamples = sampleSize - length(sampledIdx);

% Randomly select the remaining unique sampled elements:
if additionalSamples > 0
    additionalSampledData = dataIdx(randperm(length(dataIdx), additionalSamples));
    sampledIdx = [sampledIdx additionalSampledData];
end

% Final shuffle:
sampledIdx = sampledIdx(randperm(length(sampledIdx)));
end


% Helper function 2: Input parameter validation and parsing.
function params = iParseInputs(varargin)
% Create inputParser instance:
params = inputParser;

% Define basic data type and size validation functions:
validNonEmptyChar = @(x) validateattributes(x, {'char'}, {'nonempty'});
validNonNegReal = @(x) validateattributes(x, {'numeric'}, {'real', 'nonnegative'});
validPosInt = @(x) validateattributes(x, {'numeric'}, {'integer', 'positive'});
validPosReal = @(x) validateattributes(x, {'numeric'}, {'real', 'positive'});
validBool = @(x) validateattributes(x, {'logical'}, {'binary'});
validRealVecAboveOne = @(x) validateattributes(x, {'numeric'}, {'real', 'size', [1 3], '>=', 1});
validIntVecAboveOne = @(x) validateattributes(x, {'numeric'}, {'integer', 'size', [1 2], '>=', 1});

% Add required parameters:
addRequired(params, 'dataPath', validNonEmptyChar);
addRequired(params, 'POIPath', validNonEmptyChar);
addRequired(params, 'lambda', validNonNegReal);

% Parse and validate required parameters:
parse(params, varargin{1:3});

% Complex validation for required parameter dataPath:
if ~endsWith(params.Results.dataPath, {'.mat', '.tif', '.tiff'})
    error('''dataPath'' must point to a file of type ''.mat'', ''.tif'', or ''.tiff''.');
end
if ~isfile(params.Results.dataPath)
    error(['File not found: ''' params.Results.dataPath ''' for ''dataPath''.']);
end

% Complex validation for required parameter POIPath:
if ~endsWith(params.Results.POIPath, {'.mat', '.tif', '.tiff'})
    error('''POIPath'' must be a file of type ''.mat'', ''.tif'', or ''.tiff''.');
end
if ~isfile(params.Results.POIPath)
    error(['File not found: ''' params.Results.POIPath ''' for ''POIPath''.']);
end

% Get data and POI dimensions:
[widI, lenI, numI] = loadDataSize(params.Results.dataPath);
[widPOI, lenPOI] = loadDataSize(params.Results.POIPath);

% Check if dimensions of POI and data match:
if widI == 1 && widPOI ~= 1 || widI ~= 1 && widPOI == 1 || ...
        lenI == 1 && lenPOI ~= 1 || lenI ~= 1 && lenPOI == 1
    error('Shape mismatch between data and POI sample.');
end

% Set default values for optional parameters:
trainIdxRangeDefault = randperm(numI, round(0.8*numI));
valIdxRangeDefault = setdiff(1:numI, trainIdxRangeDefault);
upsamplRatioDefault = [min(1.5,widI) min(1.5,lenI)];
patchSizeDefault = [min(512,widI) min(512,lenI)];

% Add optional parameters:
addParameter(params, 'trainIdxRange', trainIdxRangeDefault, validPosInt);
addParameter(params, 'valIdxRange', valIdxRangeDefault, validPosInt);
addParameter(params, 'modelSavePath', '.\', validNonEmptyChar);
addParameter(params, 'upsamplRatio', upsamplRatioDefault, validRealVecAboveOne);
addParameter(params, 'interpMethod', 'spline', validNonEmptyChar);
addParameter(params, 'encoderDepth', 2, validPosInt);
addParameter(params, 'patchSize', patchSizeDefault, validIntVecAboveOne);
addParameter(params, 'trainPatchNum', 512, validPosInt);
addParameter(params, 'valPatchNum', 64, validPosInt);
addParameter(params, 'learningRate', 1e-3, validPosReal);
addParameter(params, 'minLR', 1e-4, validPosReal);
addParameter(params, 'miniBatchSize', 8, validPosInt);
addParameter(params, 'maxEpochs', 1000, validPosInt);
addParameter(params, 'valFreq', 20, validPosInt);
addParameter(params, 'maxPatience', 20, validPosInt);
addParameter(params, 'verbose', true, validBool);
addParameter(params, 'learnBG', true, validBool);
addParameter(params, 'useGPU', true, validBool);
addParameter(params, 'useParallel', true, validBool);
addParameter(params, 'parNum', 6, validPosInt);
addParameter(params, 'patchNumForThreshSearch', 512, validPosInt);

% Parse and validate optional parameters:
parse(params, varargin{:});

% Complex validation for optional parameter trainIdxRange:
if any(params.Results.trainIdxRange > numI)
    error('''trainIdxRange'' contains indices larger than the total number of images %d.', numI);
end

% Complex validation for optional parameter valIdxRange:
if any(params.Results.valIdxRange > numI)
    error('''valIdxRange'' contains indices larger than the total number of images %d.', numI);
end

% Warning for overlapping training and validation indices:
if ~isempty(params.Results.trainIdxRange) && ~isempty(params.Results.valIdxRange)
    if any(intersect(params.Results.trainIdxRange, params.Results.valIdxRange))
        warning('trainIdxRange and valIdxRange have overlapping indices.');
    end
end

% Complex validation for optional parameter modelSavePath:
if any(ismember(params.Results.modelSavePath, '<>:"|?*'))
    error('Invalid characters in ''modelSavePath''.');
end

% Complex validation for optional parameter upsamplRatio:
if widI == 1 && params.Results.upsamplRatio(1) ~= 1
    error('''upsamplRatio'' attempts to upsample 1D data along y-axis.');
end
if lenI == 1 && params.Results.upsamplRatio(2) ~= 1
    error('''upsamplRatio'' attempts to upsample 1D data along x-axis.');
end

% Complex validation for optional parameter interpMethod:
validInterpMethods = {'spline', 'linear', 'nearest', 'cubic'};
if ~ismember(params.Results.interpMethod, validInterpMethods)
    error('Invalid interpolation method: ''%s''. Interpolation method must be one of: ''%s''.', ...
        params.Results.interpMethod, strjoin(validInterpMethods, ''', '''));
end

% Complex validation for optional parameter patchSize:
if params.Results.patchSize(1) > ceil(widI * params.Results.upsamplRatio(1))
    error('''patchSize'' exceeds upsampled image dimensions along y-axis, should be <= %d.', ceil(widI * params.Results.upsamplRatio(1)));
end
if params.Results.patchSize(2) > ceil(lenI * params.Results.upsamplRatio(2))
    error('''patchSize'' exceeds upsampled image dimensions along x-axis, should be <= %d.', ceil(lenI * params.Results.upsamplRatio(2)));
end
if params.Results.patchSize(1) ~= 1 && mod(params.Results.patchSize(1), 2^params.Results.encoderDepth) ~= 0
    error('''patchSize'' along y-axis must be a multiple of 2^encoderDepth = %d.', 2^params.Results.encoderDepth);
end
if params.Results.patchSize(2) ~= 1 && mod(params.Results.patchSize(2), 2^params.Results.encoderDepth) ~= 0
    error('''patchSize'' along x-axis must be a multiple of 2^encoderDepth = %d.', 2^params.Results.encoderDepth);
end

% Complex validation for optional parameter patchNumForThreshSearch:
if params.Results.patchNumForThreshSearch > params.Results.trainPatchNum
    error('''patchNumForThreshSearch'' exceeds ''trainPatchNum'' value %d.', params.Results.trainPatchNum);
end
end