function [RMSE, corr, PSNR] = perfEvalBGEst(gtBG, gtI, predBG, predI)
% perfEvalBGEst  Evaluates background estimation performance metrics.
%
%   [RMSE, corr, PSNR] = perfEvalBGEst(gtBG, gtI, predBG, predI)
%
%   Input parameters:
%     gtBG     - Ground-truth background image.
%     gtI      - Ground-truth image with background removed.
%     predBG   - Predicted background image.
%     predI    - Predicted image with background removed.
%
%   Output:
%     RMSE  - Mean RMSE across all images.
%     corr  - Mean correlation across all images.
%     PSNR  - Mean PSNR across all images.

% Normalize all inputs to the [0,1] range using min-max normalization:
gtBG    = minMaxNorm(gtBG);
gtI     = minMaxNorm(gtI);
predBG  = minMaxNorm(predBG);
predI   = minMaxNorm(predI);

% Determine the number of images from the third dimension of gtI:
numI = size(gtI, 3);

% Preallocate vectors to store per-image metrics:
rmseAll = zeros(numI, 1);
corrAll = zeros(numI, 1);
psnrAll = zeros(numI, 1);

progressDisp(numI); % Initialize progress bar

% Compute RMSE, correlation, and PSNR:
for i = 1:numI
    % RMSE between predicted background and ground-truth background:
    rmseAll(i) = sqrt(mean((predBG(:, :, i) - gtBG).^2, 'all'));
    % Pearson correlation coefficient between predicted and ground-truth background:
    corrAll(i) = corr2(predBG(:, :, i), gtBG);
    % PSNR between predicted and ground-truth image with background removed:
    psnrAll(i) = psnr(predI(:, :, i), gtI(:, :, i));

    % Refresh progress bar:
    progressDisp(0);
end

% Terminate progress bar:
progressDisp(-1);

% Compute mean values across all images:
RMSE = mean(rmseAll);
corr = mean(corrAll);
PSNR = mean(psnrAll);
end


% Helper function: min-max normalization.
function normalizedImage = minMaxNorm(image)
% If the image has three dimensions, normalize each slice independently:
if ndims(image) == 3
    normalizedImage = zeros(size(image));
    for i = 1:size(image, 3)
        % Compute min and max for the current slice:
        minVal = min(image(:, :, i), [], 'all');
        maxVal = max(image(:, :, i), [], 'all');
        % Apply min-max normalization to the slice:
        normalizedImage(:, :, i) = (image(:, :, i) - minVal) / (maxVal - minVal);
    end
else
    % For a 2D image, compute min and max over all pixels:
    minVal = min(image(:));
    maxVal = max(image(:));
    % Normalize the 2D image:
    normalizedImage = (image - minVal) / (maxVal - minVal);
end
end