function [IFit, IDenoised, BG] = ...
    GaussWeightsPatchJointPred(I, dlnet, POI, BGPOI, ...
    patchSize, stride, useGPU)
% GaussWeightsPatchJointPred performs patch-based prediction on an image with Gaussian weighting and stitching.
%
%  This function processes large images by dividing them into smaller patches, predicting each patch
%  separately, and then using Gaussian weights to stitch the patches back together into the original image.
%  This method reduces memory usage and edge effects, resulting in smoother and more accurate predictions.
%
%  Input Parameters:
%    I - Input image data.
%    dlnet - Deep learning model.
%    POI - POI sample.
%    BGPOI - Background POI sample.
%    patchSize - Size of each patch.
%    stride - Stride for patch extraction.
%    useGPU - Indicates whether to use GPU for prediction.
%
%  Output Parameters:
%    IFit - Fitting map.
%    IDenoised - Denoised image.
%    BG - Predicted background image.

% Reshape POI to 5-D format for grouped convolution (width × length × channels per group × number of POIs × groups):
[widPOI, lenPOI, channelNum] = size(POI);
POI = reshape(POI, widPOI, lenPOI, 1, 1, channelNum);

% Calculate shape parameters:
[widI, lenI, numI] = size(I);

% Generate Gaussian weight mask:
sigma = max(patchSize) / 4;
weights = fspecial('gaussian', patchSize, sigma);
weights = weights ./ max(weights, [], 'all');

% Patch-based prediction:
IFit = zeros(widI, lenI, channelNum, numI, 'single');
IDenoised = zeros(widI, lenI, channelNum, numI, 'single');
BG = zeros(widI, lenI, 1, numI, 'single');
norm = zeros(widI, lenI, 'single');
for i = 1 : stride(1) : widI
    i = min(i, widI-patchSize(1)+1); %#ok<*FXSET> % Prevent edge overflow
    for j = 1 : stride(2) : lenI
        j = min(j, lenI-patchSize(2)+1); % Prevent edge overflow

        % Extract patch:
        IPatch = I(i:i+patchSize(1)-1, j:j+patchSize(2)-1, 1, :);
        IPatch = dlarray(IPatch,'SSCB');
        if useGPU
            IPatch = gpuArray(IPatch);
        end

        % Perform model prediction:
        [IFitPatch, IDenoisedPatch, BGPatch] = ...
            modelPred(dlnet, IPatch, POI, BGPOI, [], 'predict');

        % Extract data:
        IFitPatch = gather(extractdata(IFitPatch));
        IDenoisedPatch = gather(extractdata(IDenoisedPatch));
        BGPatch = gather(extractdata(BGPatch));

        % Apply Gaussian weighting:
        IFitPatch = IFitPatch .* weights;
        IDenoisedPatch = IDenoisedPatch .* weights;
        BGPatch = BGPatch .* weights;

        % Add predicted patches and Gaussian weights to the main image:
        IFit(i:i+patchSize(1)-1,j:j+patchSize(2)-1,:,:) = ...
            IFit(i:i+patchSize(1)-1,j:j+patchSize(2)-1,:,:) + ...
            IFitPatch;
        IDenoised(i:i+patchSize(1)-1,j:j+patchSize(2)-1,:,:) =...
            IDenoised(i:i+patchSize(1)-1,j:j+patchSize(2)-1,:,:) + ...
            IDenoisedPatch;
        BG(i:i+patchSize(1)-1,j:j+patchSize(2)-1,:,:) =...
            BG(i:i+patchSize(1)-1,j:j+patchSize(2)-1,:,:) + ...
            BGPatch;
        norm(i:i+patchSize(1)-1,j:j+patchSize(2)-1) = ...
            norm(i:i+patchSize(1)-1,j:j+patchSize(2)-1) + ...
            weights;

        % Break loop if edge is reached:
        if j == lenI - patchSize(2) + 1
            break;
        end
    end
    if i == widI - patchSize(1) + 1
        break;
    end
end

IFit = IFit ./ norm;
IDenoised = IDenoised ./ norm;
BG = BG ./ norm;
end