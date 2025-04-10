function DEPAFPred(varargin)
% DEPAFPred uses a pre-trained model for predictions.
%
%  This function loads a pre-trained neural network model and related preprocessing parameters
%  and performs predictions on the provided dataset. It supports batch processing of image data
%  and offers multiple output options, including generated background images and synthetic images.
%
%  Input Parameters:
%    modelPath - Path to the .mat file containing the pre-trained model and preprocessing parameters.
%    dataPath - Path containing the data to be predicted, supports .mat, .tif, and .tiff formats.
%    batchSize - Number of images per batch for batch prediction.
%    patchSize - Patch size used for prediction.
%    patchStride - Stride size used for patch-based prediction.
%    resSavePath - Path to save the results.
%    outputISyn - Whether to output noiseless synthetic images.
%    outputBG - Whether to output background images.
%    useGPU - Whether to use GPU for prediction.
%
%  Output:
%    No direct output. All prediction results are saved in a .mat file in the specified path.

% Parse parameters:
userSetParams = iParseInputs(varargin{:});
modelPath = userSetParams.Results.modelPath;
dataPath = userSetParams.Results.dataPath;
batchSize = userSetParams.Results.batchSize;
patchSize = userSetParams.Results.patchSize;
patchStride = userSetParams.Results.patchStride;
resSavePath = userSetParams.Results.resSavePath;
outputISyn = userSetParams.Results.outputISyn;
outputBG = userSetParams.Results.outputBG;
useGPU = userSetParams.Results.useGPU;
clear userSetParams;

% Load model:
disp('[Prediction] Loading model...');
load(modelPath,'dlnet', 'POI', 'upsamplRatio', 'normMax', 'normMin', ...
    'learnBG', 'optimalThresh', 'interpMethod', 'modelID');

% Generate background POI:
if learnBG
    BGPOI = BGPOIGeneration(POI);
else
    BGPOI = [];
end

% Load the number of images:
[widI, lenI, numI] = loadDataSize(dataPath);

% Start batch prediction:
disp('[Prediction] Predicting...');
progressDisp(numI); % Initialize progress bar
loc = cell(numI, 1);
amplitude = cell(numI, 1);
if outputISyn
    ISyn = zeros(widI, lenI, numI, 'single');
end
if outputBG
    BG = zeros(widI, lenI, numI, 'single');
end
for idxI = 1:batchSize:numI

    % Read data for prediction:
    IBatch = loadData(dataPath, 'all', 'all', ...
        idxI:min(idxI+batchSize-1,numI));

    % Normalize and upsample:
    IBatch = IBatch - normMin;
    IBatch = max(0, imgResizeCenterKeep(IBatch, [upsamplRatio(1:2) 1], interpMethod));
    IBatch = IBatch ./ normMax;

    % Reshape IBatch to 4-D format:
    [widIUpsampled, lenIUpsampled, numIBatch] = size(IBatch);
    IBatch = reshape(IBatch, widIUpsampled, lenIUpsampled, 1, numIBatch);

    % Predict using the model:
    [IFitBatch, IDenoisedBatch, BGBatch] = GaussWeightsPatchJointPred( ...
        IBatch, dlnet, POI, BGPOI, patchSize, patchStride, useGPU);

    % Remove background:
    IBatch = IBatch - BGBatch;

    % Record background:
    if outputBG
        BGBatch = reshape(BGBatch, widIUpsampled, lenIUpsampled, numIBatch);
        BGBatch = BGBatch .* normMax;
        BGBatch = imgResizeCenterKeep(BGBatch, [1./upsamplRatio(1:2) 1], interpMethod);
        BGBatch = BGBatch + normMin;
        BG(:,:,idxI:min(idxI+batchSize-1,numI)) = BGBatch;
    end
    clear BGBatch;

    % Locate and extract amplitude:
    [locBatch, amplitudeBatch] = getLocAndAmp(IBatch, ...
        IFitBatch, IDenoisedBatch, POI, optimalThresh, interpMethod);
    clear IBatch IFitBatch IDenoisedBatch;

    % Generate synthetic image based on location and amplitude:
    if outputISyn
        ISynBatch = zeros(widIUpsampled, lenIUpsampled, numIBatch, 'single');
        for i = 1:numIBatch
            if isempty(locBatch{i})
                continue;
            end
            ISynBatch(:,:,i) = imgSynthesis([widIUpsampled lenIUpsampled], ...
                locBatch{i}, amplitudeBatch{i}, POI, interpMethod);
        end
        ISynBatch = ISynBatch .* normMax;
        ISynBatch = imgResizeCenterKeep(ISynBatch, [1./upsamplRatio(1:2) 1], interpMethod);
        ISyn(:,:,idxI:min(idxI+batchSize-1,numI)) = ISynBatch;
        clear ISynBatch;
    end

    % Restore and record location results based on the upsampling ratio:
    loc(idxI:min(idxI+batchSize-1, numI)) = ...
        locResizeCenterKeep(locBatch, ...
        [widIUpsampled lenIUpsampled size(POI,3)], 1./upsamplRatio);

    % Record amplitude:
    amplitude(idxI:min(idxI+batchSize-1, numI)) = amplitudeBatch;

    % Refresh progress bar:
    progressDisp(0, min(batchSize, numI-idxI+1));
end

% End progress bar:
progressDisp(-1);

% Save prediction results:
disp('[Prediction] Saving prediction results...');
checkPath(resSavePath); % Check path
[~, dataName, dataExt] = fileparts(dataPath); % Get prediction data name
dataFileName = [dataName dataExt]; % Get prediction data name
predResSavePath = [resSavePath '\PredRes_' dataName ...
    '_by_Mdl_' modelID '.mat']; % Get save path name
varOrder = {'loc', 'widI', 'lenI', 'amplitude', 'modelID', 'dataFileName'};
save(predResSavePath, 'loc', 'widI', 'lenI', 'amplitude', ...
    'modelID', 'dataFileName', 'varOrder');

% Save noiseless synthetic image:
if outputISyn
    disp('[Prediction] Saving noiseless synthetic image...');
    savePathISyn = [resSavePath '\ISyn_' dataName '_by_Mdl_' modelID '.mat'];
    save(savePathISyn, 'ISyn', '-v7.3');
end

% Save background image:
if outputBG
    disp('[Prediction] Saving background image...');
    savePathBG = [resSavePath '\BG_' dataName '_by_Mdl_' modelID '.mat'];
    save(savePathBG, 'BG', '-v7.3');
end
end


% Helper Function: Input parameter validation and parsing.
function params = iParseInputs(varargin)
% Create inputParser instance:
params = inputParser;

% Define basic data type and size validation functions:
validNonEmptyChar = @(x) validateattributes(x, {'char'}, {'nonempty'});
validPosInt = @(x) validateattributes(x, {'numeric'}, {'integer', 'positive'});
validIntVecAboveOne = @(x) validateattributes(x, {'numeric'}, {'integer', 'size', [1 2], '>=', 1});
validBool = @(x) validateattributes(x, {'logical'}, {'binary'});

% Add required parameters:
addRequired(params, 'modelPath', validNonEmptyChar);
addRequired(params, 'dataPath', validNonEmptyChar);

% Parse and validate required parameters:
parse(params, varargin{1:2});

% Complex validation for required parameter modelPath:
if ~endsWith(params.Results.modelPath, {'.mat'})
    error('''modelPath'' must point to a file of type ''.mat''.');
end
if ~isfile(params.Results.modelPath)
    error(['File not found: ''' params.Results.modelPath ''' for ''modelPath''.']);
end

% Complex validation for required parameter dataPath:
if ~endsWith(params.Results.dataPath, {'.mat', '.tif', '.tiff'})
    error('''dataPath'' must point to a file of type ''.mat'', ''.tif'', or ''.tiff''.');
end
if ~isfile(params.Results.dataPath)
    error(['File not found: ''' params.Results.dataPath ''' for ''dataPath''.']);
end

% Get data dimensions:
[widI, lenI] = loadDataSize(params.Results.dataPath);

% Get neural network depth and upsampling ratio:
encoderDepth = load(params.Results.modelPath,'encoderDepth');
encoderDepth = encoderDepth.encoderDepth;
upsamplRatio = load(params.Results.modelPath,'upsamplRatio');
upsamplRatio = upsamplRatio.upsamplRatio;

% Set default values for optional parameters:
patchSizeDefault = [min(256,widI) min(256,lenI)]; % Patch size for prediction (must be a power of 2)
patchStrideDefault = ceil(patchSizeDefault/2); % Stride for patch-based prediction

% Add optional parameters:
addParameter(params, 'resSavePath', '.\', validNonEmptyChar);
addParameter(params, 'batchSize', 8, validPosInt);
addParameter(params, 'patchSize', patchSizeDefault, validIntVecAboveOne);
addParameter(params, 'patchStride', patchStrideDefault, validIntVecAboveOne);
addParameter(params, 'outputISyn', false, validBool);
addParameter(params, 'outputBG', false, validBool);
addParameter(params, 'useGPU', true, validBool);

% Parse and validate optional parameters:
parse(params, varargin{:});

% Complex validation for optional parameter patchSize:
if params.Results.patchSize(1) > ceil(widI * upsamplRatio(1))
    error('''patchSize'' exceeds image dimensions along y-axis after upsampling, should be <= %d.', ceil(widI * upsamplRatio(1)));
end
if params.Results.patchSize(2) > ceil(lenI * upsamplRatio(2))
    error('''patchSize'' exceeds image dimensions along x-axis after upsampling, should be <= %d.', ceil(lenI * upsamplRatio(2)));
end
if params.Results.patchSize(1) ~= 1 && mod(params.Results.patchSize(1), 2^encoderDepth) ~= 0
    error('''patchSize'' along y-axis must be a multiple of 2^encoderDepth = %d.', 2^encoderDepth);
end
if params.Results.patchSize(2) ~= 1 && mod(params.Results.patchSize(2), 2^encoderDepth) ~= 0
    error('''patchSize'' along x-axis must be a multiple of 2^encoderDepth = %d.', 2^encoderDepth);
end

% Complex validation for optional parameter patchStride:
if params.Results.patchStride(1) > params.Results.patchSize(1)
    error('''patchStride'' along y-axis exceeds ''patchSize'', should be <= %d.', params.Results.patchSize(1));
end
if params.Results.patchStride(2) > params.Results.patchSize(2)
    error('''patchStride'' along x-axis exceeds ''patchSize'', should be <= %d.', params.Results.patchSize(2));
end

% Complex validation for optional parameter resSavePath:
if any(ismember(params.Results.resSavePath, '<>"|?*'))
    error('Invalid characters in ''resSavePath''.');
end
end