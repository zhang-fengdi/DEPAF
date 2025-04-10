function loss = threshSearchingLoss( ...
    I, IFit, IDenoised, POI, thresh, interpMethod)
% threshSearchingLoss Loss for threshold search in Bayesian estimation.
%
%  This function calculates the loss between the predicted results and actual images
%  at a given threshold. It first locates the POIs in the image based on the threshold
%  and calculates their amplitude, then simulates an image and calculates the difference
%  between this synthetic image and the denoised image to determine the loss value.
%  This method is used in Bayesian optimization to search for the optimal threshold.
%
%  Input Parameters:
%    I - Original image data.
%    IFit - Fitting map.
%    IDenoised - Denoised image data.
%    POI - POI sample.
%    thresh - Threshold used in testing.
%    interpMethod - Interpolation method, such as 'spline', 'nearest', 'linear', 'cubic'.
%
%  Output Parameters:
%    loss - Loss value calculated at the given threshold.

% Get number of images:
numI = size(I,4);

% Locate and calculate the central amplitude of each POI:
[loc, amplitude] = getLocAndAmp(I, IFit, IDenoised, POI, thresh, interpMethod);
clear I IFit thresh;

% Synthesize an image based on location and intensity:
ISyn = zeros(size(IDenoised), 'single');
for i = 1:numI
    if isempty(loc{i})
        continue;
    end
    ISyn(:,:,1,i) = imgSynthesis(size(ISyn, [1 2]), ...
        loc{i}, amplitude{i}, POI, interpMethod);
end

% Calculate batch-normalized loss (to match the calculation in the neural network training phase):
loss = sum(abs(ISyn - IDenoised),'all') / numI;
end