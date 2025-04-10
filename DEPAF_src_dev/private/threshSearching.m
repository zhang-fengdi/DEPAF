function [optimalThresh, loss, thresholdList] = threshSearching( ...
    I, IFit, IDenoised, POI, threshRange, stepSize, interpMethod, useParallel)
% threshSearching Searches for optimal threshold using grid search.
%
%  This function is used to search for the optimal threshold in image processing.
%  It tests each threshold within a specified range, at a certain step size, and
%  evaluates the loss for each threshold to find the best one.
%  Parallel processing is supported to speed up the threshold search process.
%
%  Input Parameters:
%    I - Original image data.
%    IFit - Fitting map.
%    IDenoised - Denoised image data.
%    POI - POI sample.
%    threshRange - Range of threshold search, formatted as [minThreshold maxThreshold].
%    stepSize - Step size for threshold during the search process.
%    interpMethod - Interpolation method, such as 'spline', 'nearest', 'linear', 'cubic'.
%    useParallel - Whether to use parallel processing.
%
%  Output Parameters:
%    optimalThresh - Found optimal threshold.
%    loss - Loss values corresponding to different thresholds.
%    thresholdList - List of tested thresholds.

% Calculate necessary parameters:
if threshRange(1) > threshRange(2)
    error('Threshold range is set incorrectly.');
end
thresholdList = threshRange(1) : stepSize : threshRange(2);
thresholdNum = length(thresholdList);

% Begin searching for optimal threshold:
progressDisp(thresholdNum); % Initialize progress bar
loss = zeros(thresholdNum, 1, 'single');
if useParallel
    parfor i = 1:thresholdNum
        loss(i) = threshSearchingLoss(I, IFit, ...
            IDenoised, POI, thresholdList(i), interpMethod);

        % Refresh progress bar:
        progressDisp(0);
    end
else
    for i = 1:thresholdNum
        loss(i) = threshSearchingLoss(I, IFit, ...
            IDenoised, POI, thresholdList(i), interpMethod);

        % Refresh progress bar:
        progressDisp(0);
    end
end

% Terminate progress bar:
progressDisp(-1);

optimalThresh = thresholdList(find(loss == min(loss),1,'last'));
end