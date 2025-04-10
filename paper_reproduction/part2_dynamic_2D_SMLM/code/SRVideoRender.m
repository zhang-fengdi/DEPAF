function SRVideoRender(rndrPredLocPath, ...
    rndrSMLMVideoPath, SMLMVideoSamplNum, ...
    SRRatio, renderMethod, rndrColor, SMLMFrameRate, recFrameNum, ...
    normMode, normRange, cropPct, SRVideoSavePath, fileFormat, quality)
% SRVideoRender renders super-resolution video based on single-molecule localization results.
%
%  This function renders a super-resolution video from provided localization results using 
%  different rendering methods including nearest-neighbor, bilinear, and Gaussian. The rendered 
%  super-resolution video can also be merged with the original SMLM video for comparison.
%
%  Input parameters:
%    rndrPredLocPath - Path to the localization results file.
%    rndrSMLMVideoPath - Path to the original SMLM video file.
%    SMLMVideoSamplNum - Sampling number for the original SMLM video.
%    SRRatio - Magnification ratio for super-resolution.
%    renderMethod - Rendering method, 'nearest', 'bilinear', or 'Gauss'.
%    rndrColor - Rendering color, 'hot' or 'gray'.
%    SMLMFrameRate - Frame rate of the original SMLM video.
%    recFrameNum - Number of frames used for reconstruction.
%    normMode - Normalization mode, 'percent' or 'absolute'.
%    normRange - Normalization range; if normMode is 'percent', represents percentage range.
%    cropPct - Cropping percentage.
%    SRVideoSavePath - Path to save the super-resolution video.
%    fileFormat - File format, '.avi' or '.mp4'.
%    quality - Video quality.

% Load localization results for rendering:
[predLoc, oriWid, oriLen, ~, modelID, predDataName] = ...
    loadData(rndrPredLocPath);

% Calculate super-resolution video dimensions:
SRWid = ceil(oriWid*SRRatio);
SRLen = ceil(oriLen*SRRatio);

% Only use the first multiple of recFrameNum frames from the SMLM video for reconstruction:
SMLMFrameNum = length(predLoc) - mod(length(predLoc), recFrameNum);

% Calculate super-resolution video:
disp('[Rendering] Calculating super-resolution video...');
progressDisp(floor(SMLMFrameNum/recFrameNum)); % Initialize progress bar
SRVideo = zeros(SRWid, SRLen, SMLMFrameNum/recFrameNum, 'single');
for frameNum = 1 : recFrameNum : SMLMFrameNum
    oriLoc = predLoc(frameNum:frameNum+recFrameNum-1, 1);
    oriLoc = cell2mat(oriLoc);
    SRLoc = oriLoc;
    SRLoc(:,1) = SRLoc(:,1) * (SRLen/oriLen);
    SRLoc(:,2) = SRLoc(:,2) * (SRWid/oriWid);
    SRLoc(:,1) = min(SRLoc(:,1), SRLen-1);
    SRLoc(:,2) = min(SRLoc(:,2), SRWid-1);
    SRFrameNum = ceil(frameNum/recFrameNum);
    switch renderMethod
        case 'nearest'
            SRVideo(:,:,SRFrameNum) = nearestRender(SRLoc, SRWid, SRLen);
        case 'bilinear'
            SRVideo(:,:,SRFrameNum) = bilinearRender(SRLoc, SRWid, SRLen);
        case 'Gauss'
            SRVideo(:,:,SRFrameNum) = bilinearRender(SRLoc, SRWid, SRLen);
            GaussKernel = fspecial('gaussian', 7, 1);
            SRVideo(:,:,SRFrameNum) = conv2(SRVideo(:,:,SRFrameNum), GaussKernel, 'same');
        otherwise
            error('Only ''nearest'', ''bilinear'', and ''Gauss'' rendering methods are supported.');
    end

    % Update progress bar:
    progressDisp(0);
end

% End progress bar:
progressDisp(-1);

% Crop edges:
SRVideo = edgeCropByPercent(SRVideo, cropPct);

% Normalize super-resolution video:
switch normMode
    case 'percent'
        normMin = prctile(SRVideo, normRange(1), 'all');
        normMax = prctile(SRVideo, normRange(2), 'all');
    case 'absolute'
        normMin = normRange(1);
        normMax = normRange(2);
    otherwise
        error('Only ''percent'' and ''absolute'' normalization modes are supported.');
end
SRVideo = mat2gray(SRVideo, double([normMin normMax]));

% Create video object:
checkPath(SRVideoSavePath); % Check path
savePath = [SRVideoSavePath, '\SRVideo-' renderMethod '_' ...
    predDataName '_by_Mdl_' modelID];
switch fileFormat
    case '.avi'
        v = VideoWriter(savePath, 'Motion JPEG AVI');
    case '.mp4'
        v = VideoWriter(savePath, 'MPEG-4');
    otherwise
        error('Only ''.avi'' and ''.mp4'' output formats are supported.');
end

% Set rendering quality:
v.Quality = quality;

% Output video:
if isempty(rndrSMLMVideoPath) % Render only super-resolution video:
    disp('[Rendering] Outputting video...');

    % Initialize progress bar:
    progressDisp(SRFrameNum);

    % Set frame rate:
    v.FrameRate = SMLMFrameRate / recFrameNum;

    % Start rendering:
    open(v)
    for i = 1:SRFrameNum
        % Convert to uint8 format:
        renderRes = im2uint8(SRVideo(:,:,i));

        % Convert to RGB format:
        switch rndrColor
            case 'hot'
                renderRes = ind2rgb(renderRes, hot(256));
            case 'gray'
                renderRes = ind2rgb(renderRes, gray(256));
            otherwise
                error('Only ''hot'' and ''gray'' rendering colors are supported.');
        end

        % Add timestamp and frame display:
        note = sprintf( ...
            ' Time: %4.2f ms, SR Frame: %2d', 1000*i/v.FrameRate, i);
        renderRes = insertText(renderRes, [0 0], note, ...
            'TextColor', 'white', 'BoxColor', 'black', 'FontSize', 24);

        % Render and write frame:
        renderRes = im2frame(renderRes);
        writeVideo(v, renderRes);

        % Display rendering result:
        imshow(renderRes.cdata, 'border', 'tight');

        % Update progress bar:
        progressDisp(0);
    end

    % End progress bar:
    progressDisp(-1);

else % Add SMLM video for comparison:
    % Load SMLM video for comparison:
    disp('[Rendering] Loading SMLM video for comparison...');
    IRaw = loadData(rndrSMLMVideoPath);

    % Crop edges:
    IRaw = edgeCropByPercent(IRaw, cropPct);

    % Normalize SMLM video:
    IRaw = mat2gray(IRaw, double([prctile(IRaw,0.01,'all') ...
        prctile(IRaw,99.99,'all')]));

    disp('[Rendering] Outputting video...');

    % Initialize progress bar:
    progressDisp(ceil(SMLMFrameNum / SMLMVideoSamplNum));

    % Set frame rate:
    v.FrameRate = SMLMFrameRate / SMLMVideoSamplNum;

    % Start rendering:
    open(v)
    for i = 1:SMLMVideoSamplNum:SMLMFrameNum
        % Load single SMLM video frame:
        singleSMLMFrame = IRaw(:,:,i);

        % Load single super-resolution video frame:
        singleSRFrame = SRVideo(:,:,ceil(i/recFrameNum));

        % Align SMLM video frame to super-resolution frame dimensions:
        singleSMLMFrame = imresize(singleSMLMFrame, size(singleSRFrame), 'nearest');

        % Generate render frame:
        rndrFrame = [singleSMLMFrame singleSRFrame];

        % Convert to uint8 format:
        renderRes = im2uint8(rndrFrame);

        % Convert to RGB format:
        switch rndrColor
            case 'hot'
                renderRes = ind2rgb(renderRes, hot(256));
            case 'gray'
                renderRes = ind2rgb(renderRes, gray(256));
            otherwise
                error('Only ''hot'' and ''gray'' rendering colors are supported.');
        end

        % Add timestamp and frame display:
        note = sprintf( ...
            ' Time: %4.2f ms, SMLM Frame: %4d, SR Frame: %2d', ...
            1000*i/SMLMFrameRate, i, ceil(i/recFrameNum));
        renderRes = insertText(renderRes, [0 0], note, ...
            'TextColor', 'white', 'BoxColor', 'black', 'FontSize', 24);

        % Render and write frame:
        renderRes = im2frame(renderRes);
        writeVideo(v, renderRes);

        % Display rendering result:
        imshow(renderRes.cdata, 'border', 'tight');

        % Update progress bar:
        progressDisp(0);
    end

    % End progress bar:
    progressDisp(-1);
end
close(v);

% Backup rendering parameters:
paramsSavePath = [SRVideoSavePath, '\SRVideo-' ...
    renderMethod '_' predDataName '_by_Mdl_' modelID '_RndrParams.txt'];
writeParamsToTxt(paramsSavePath, rndrPredLocPath, ...
    rndrSMLMVideoPath, SMLMVideoSamplNum, ...
    SRRatio, renderMethod, rndrColor, SMLMFrameRate, recFrameNum, ...
    normMode, normRange, cropPct, SRVideoSavePath, fileFormat, quality);
end


% Helper function 1: Render super-resolution image frame using bilinear interpolation.
function SRFrame = bilinearRender(predLoc, SRWid, SRLen)
% Initialize parameters and preallocate memory:
x0 = floor(predLoc(:,1));
y0 = floor(predLoc(:,2));
locNum = size(predLoc,1);
SRFrame = zeros(SRWid, SRLen, 'single');

% Calculate weights:
wx = predLoc(:,1) - x0;
wy = predLoc(:,2) - y0;
M = [(1-wy).*(1-wx), (1-wy).*wx, wy.*(1-wx), wy.*wx];

% Render super-resolution frame:
for i = 1:locNum
    SRFrame(y0(i), x0(i)) = SRFrame(y0(i), x0(i)) + M(i,1);
    SRFrame(y0(i), x0(i)+1) = SRFrame(y0(i), x0(i)+1) + M(i,2);
    SRFrame(y0(i)+1, x0(i)) = SRFrame(y0(i)+1, x0(i)) + M(i,3);
    SRFrame(y0(i)+1, x0(i)+1) = SRFrame(y0(i)+1, x0(i)+1) + M(i,4);
end
end


% Helper function 2: Render super-resolution image frame using nearest neighbor interpolation.
function SRFrame = nearestRender(predLoc, SRWid, SRLen)
SRFrame = zeros(SRWid, SRLen, 'single');
for k = 1:size(predLoc,1)
    x = round(predLoc(k,1));
    y = round(predLoc(k,2));
    SRFrame(y, x) = SRFrame(y, x) + 1;
end
end


% Helper function 3: Crop edges by percentage.
function I = edgeCropByPercent(I, cropPct)
[widI, lenI, numI] = size(I);
win = centerCropWindow3d([widI, lenI, numI], ...
    [ceil(widI*cropPct) ceil(lenI*cropPct) numI]);
I = imcrop3(I, win);
end


% Helper function 4: Backup parameters to .txt file.
function writeParamsToTxt(varargin)
% Set path name:
filePath = varargin{1};

% Check path name:
isValidTxtPath = isTxtFilePath(filePath);
if ~isValidTxtPath
    error('Specified .txt file path is invalid.');
end

% Open file:
fileID = fopen(filePath, 'w');

% Check if file opened successfully:
if fileID == -1
    error('File cannot be opened.');
end

% Write all input parameters to file:
for i = 2:length(varargin)
    % Get variable name:
    varName = inputname(i);
    if isempty(varName)
        varName = sprintf('Var%d', i-1); % If no variable name, create a default one
    end

    % Detect data type and convert to string:
    if iscell(varargin{i})
        varValue = cellfun(@mat2str, varargin{i}, 'UniformOutput', false);
        varValue = strjoin(varValue, ', ');
    else
        varValue = mat2str(varargin{i});
    end

    % Write variable name and value:
    fprintf(fileID, '%s: %s\n', varName, varValue);
end

% Close file:
fclose(fileID);
end


% Helper function 5: Check if .txt file path is valid.
function isValidTxtPath = isTxtFilePath(filePath)
% Check if it is a character array or string:
if ~ischar(filePath) && ~isstring(filePath)
    isValidTxtPath = false;
    return;
end

% Check for invalid characters in path:
invalidChars = '<>|"?*';
if any(ismember(filePath, invalidChars))
    isValidTxtPath = false;
    return;
end

% Check if file extension is .txt:
[~, ~, ext] = fileparts(filePath);
isValidTxtPath = strcmpi(ext, '.txt');
end