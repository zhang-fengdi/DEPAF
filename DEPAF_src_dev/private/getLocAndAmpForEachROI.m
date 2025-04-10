function [loc, amplitude] = getLocAndAmpForEachROI(imSize, ROIInd, ...
    IFit, grayValIFit2D, grayValIFitNoisy, POI, interpMethod)
% getLocAndAmpForEachROI calculates the subpixel coordinates and amplitude for each ROI.
%
%  This function uses a pixel-weighted centroid algorithm to calculate the subpixel coordinates
%  of the POI within each ROI and estimates the POI amplitude using the maximum value after
%  convolving the ROI with the POI.
%
%  Input Parameters:
%    imSize - Size of the original image.
%    ROIInd - Indices of the ROI in the image.
%    IFit - Fitting map.
%    grayValIFit2D - 2D grayscale values within the ROI.
%    grayValIFitNoisy - 2D grayscale values within the noisy ROI.
%    POI - POI sample.
%    interpMethod - Interpolation method, such as 'spline', 'nearest', 'linear', 'cubic'.
%
%  Output Parameters:
%    loc - Calculated subpixel coordinates of the POI.
%    amplitude - Calculated amplitude of the POI.

% Restore ROI subscript coordinates:
[subi, subj] = ind2sub(imSize, ROIInd);

% Calculate subpixel coordinates along the z-axis:
channelNum = size(IFit, 3);
if channelNum > 1
    ROISumValInEachChannel = zeros(channelNum, 1, 'single');
    for pix = 1:length(subi)
        ROISumValInEachChannel = ROISumValInEachChannel + squeeze(IFit(subi(pix),subj(pix),:));
    end
    zLoc = sum(ROISumValInEachChannel .* (1:channelNum)') / sum(ROISumValInEachChannel);
else
    zLoc = 1;
end

% Reconstruct 2D ROI:
iBias = min(subi) - 1;
jBias = min(subj) - 1;
subi = subi - iBias;
subj = subj - jBias;
ROIFit = zeros(2, 2, 'single');
ind = sub2ind([2 2], subi, subj);
ROIFit(ind) = grayValIFit2D;

% Calculate subpixel coordinates along the xy-axes:
X = ROIFit(1,1) + ROIFit(1,2)*2 + ROIFit(2,1) + ROIFit(2,2)*2;
Y = ROIFit(1,1) + ROIFit(1,2) + ROIFit(2,1)*2 + ROIFit(2,2)*2;
sumVal = ROIFit(1,1) + ROIFit(1,2) + ROIFit(2,1) + ROIFit(2,2);
xLoc = X / sumVal;
yLoc = Y / sumVal;
loc = [xLoc + jBias, yLoc + iBias, zLoc];

% Generate subpixel POI along the z-axis:
if channelNum > 1
    [widPOI, lenPOI] = size(POI, [1 2]);
    [Xq, Yq, Zq] = meshgrid(1:lenPOI, 1:widPOI, zLoc);
    Xq = single(Xq);
    Yq = single(Yq);
    Zq = single(Zq);
    POI = interp3(POI, Xq, Yq, Zq, interpMethod);
end

% Amplitude estimation:
ROIFitNoisy = zeros(2, 2, 'single');
ROIFitNoisy(ind) = grayValIFitNoisy;
% conv2 function is different from dlconv; conv2 does not flip POI
ROIFitNoisyConv = conv2(ROIFitNoisy, POI, 'full');
amplitude = max(ROIFitNoisyConv, [], 'all');
end