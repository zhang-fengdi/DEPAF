function [loc, amplitude] = getLocAndAmp(I, IFit, IDenoised, ...
    POI, thresh, interpMethod)
% getLocAndAmp segments the input image based on a threshold and returns POI locations and amplitudes.
%
%  This function extracts the position and amplitude of POIs from a given image.
%
%  Input Parameters:
%    I - Original image data, which can contain multiple frames.
%    IFit - Fitting map.
%    IDenoised - Denoised image data.
%    POI - POI sample used to normalize the noise map.
%    thresh - Threshold for POI segmentation.
%    interpMethod - Interpolation method, such as 'spline', 'nearest', 'linear', 'cubic'.
%
%  Output Parameters:
%    loc - Detected POI locations as an array of coordinates.
%    amplitude - Corresponding amplitude of each POI.

% Extract and normalize the noise map:
INoise = I - IDenoised;
INoise = INoise ./ max(POI, [], 'all');
clear I IDenoised;

% Merge channels in IFit to create a 2D map:
IFit2D = sum(IFit, 3);

% Synthesize a noisy IFit map:
IFitNoisy = max(IFit2D, INoise);
clear INoise;

% POI localization and amplitude extraction:
numI = size(IFit, 4);
mask = IFit2D > thresh;
loc = cell(numI, 1);
amplitude = cell(numI, 1);
for k = 1:numI
    singleIFit = IFit(:, :, :, k);
    singleIFit2D = IFit2D(:, :, 1, k);
    singleIFitNoisy = IFitNoisy(:, :, 1, k);
    singleMask = mask(:, :, 1, k);

    % Compute 8-connected regions:
    CC = bwconncomp(singleMask, 8);
    ROIIdxList = CC.PixelIdxList;
    imSize = CC.ImageSize;

    % Further split rough ROI:
    for i = 1:length(ROIIdxList)
        ROIIdxList{i} = ROIPartitioning(imSize, ROIIdxList{i}, ...
            singleIFit2D(ROIIdxList{i}));
    end

    % Flatten emitterIdxList:
    ROINum = 0;
    for i = 1:length(ROIIdxList)
        ROINum = ROINum + numel(ROIIdxList{i});
    end
    count = 0;
    ROIIdxListFlat = cell(1,ROINum);
    for i = 1:length(ROIIdxList)
        singleIFitEachChannel = ROIIdxList{i};
        for j = 1:length(ROIIdxList{i})
            count = count + 1;
            ROIIdxListFlat{count} = singleIFitEachChannel{j};
        end
    end

    % Calculate subpixel positions and amplitudes of POIs in the map:
    singleLoc = zeros(length(ROIIdxListFlat), 3, 'single');
    singleAmplitude = zeros(length(ROIIdxListFlat), 1, 'single');
    for i = 1:length(ROIIdxListFlat)
        [singleLoc(i,:), singleAmplitude(i)] = ...
            getLocAndAmpForEachROI(imSize, ROIIdxListFlat{i}, ...
            singleIFit, ...
            singleIFit2D(ROIIdxListFlat{i}), ...
            singleIFitNoisy(ROIIdxListFlat{i}), ...
            POI, ...
            interpMethod);
    end

    % Record results:
    loc{k,1} = [loc{k,1}; singleLoc];
    amplitude{k,1} = [amplitude{k,1}; singleAmplitude];
end
end