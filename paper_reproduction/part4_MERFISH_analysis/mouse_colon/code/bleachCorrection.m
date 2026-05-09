function corrImg = bleachCorrection(refImg, tgtImg, refCdf)
% Bleach correction using histogram matching
%
% Inputs:
%   refImg - Reference image
%   tgtImg - Target image to be corrected
%   refCdf - Optional precomputed reference cumulative histogram
%
% Output:
%   corrImg - Corrected image

% Determine the type of the reference image:
refImgClass = class(refImg);

% Check if the reference and target images have the same type:
if ~strcmp(refImgClass, class(tgtImg))
    error('The reference and target images must have the same type.');
end

% Determine the number of histogram bins:
switch class(refImg)
    case 'uint8'
        histBinCount = 256;
    case 'uint16'
        histBinCount = 65536;
    otherwise
        error('Unsupported bit depth.');
end

% Compute the histogram and cumulative distribution function (CDF) of the reference image:
if nargin < 3 || isempty(refCdf)
    refHist = imhist(refImg, histBinCount);
    refCdf = cumsum(refHist) / sum(refHist);
end

% Process the target image:
tgtHist = imhist(tgtImg, histBinCount);
tgtCdf = cumsum(tgtHist) / sum(tgtHist);
mapFunc = zeros(1, histBinCount, 'uint32');

% Compute the monotonic pixel mapping in linear time over histogram bins.
histIdx = 1;
for pxVal = 1:histBinCount
    while histIdx <= histBinCount && refCdf(histIdx) < tgtCdf(pxVal)
        histIdx = histIdx + 1;
    end
    mapFunc(pxVal) = histIdx - 1;
end

% Apply pixel mapping function for bleach correction:
mapFunc = cast(mapFunc, refImgClass); % Convert the type of the mapping function based on the input image type
corrImg = intlut(tgtImg, mapFunc);
end
