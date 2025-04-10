function corrImg = bleachCorrection(refImg, tgtImg)
% Bleach correction using histogram matching
%
% Inputs:
%   refImg - Reference image
%   tgtImg - Target image to be corrected
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
refHist = imhist(refImg, histBinCount);
refCdf = cumsum(refHist) / sum(refHist);

% Process the target image:
tgtHist = imhist(tgtImg, histBinCount);
tgtCdf = cumsum(tgtHist) / sum(tgtHist);
mapFunc = zeros(1, histBinCount);

% Compute pixel mapping function:
for pxVal = 1:histBinCount
    histIdx = histBinCount;
    while histIdx > 0 && tgtCdf(pxVal) <= refCdf(histIdx)
        mapFunc(pxVal) = histIdx - 1;
        histIdx = histIdx - 1;
    end
end

% Apply pixel mapping function for bleach correction:
mapFunc = cast(mapFunc, refImgClass); % Convert the type of the mapping function based on the input image type
corrImg = intlut(tgtImg, mapFunc);
end