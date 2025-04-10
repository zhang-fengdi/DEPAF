function [eff, JacIdx, rmseLateral, rmseAxial] = perfEval3DSMLM(predLoc, tarLoc, matchThresh, pixNMSize)
% perfEval3D evaluates the performance of 3D localization predictions
%
% Inputs:
%   predLoc - Nx3 matrix representing predicted positions (N: number of predicted points)
%   tarLoc - Mx3 matrix representing target positions (M: number of target points)
%   matchThresh - Maximum allowed distance to match predictions with targets
%   pixNMSize - Pixel size in nanometers (for RMSE calculation)
%
% Outputs:
%   eff - Performance efficiency score
%   JacIdx - Jaccard index (measure of similarity between predictions and targets)
%   rmseLateral - Root mean square error of lateral (XY) positions
%   rmseAxial - Root mean square error of axial (Z) positions

% Get the number of predicted and target points:
predNum = size(predLoc, 1);  % Number of predicted points
tarNum = size(tarLoc, 1);    % Number of target points

% Calculate the Euclidean distance matrix between predicted and target locations:
distMatrix = pdist2(single(predLoc), single(tarLoc));

% Set elements with distances greater than the match threshold to infinity:
distMatrix(distMatrix > matchThresh) = Inf;

% Perform optimal matching using the distance matrix:
assignment = matchpairs(distMatrix, 1000);

% Calculate True Positives (TP), False Positives (FP), and False Negatives (FN):
TP = size(assignment, 1);    % Number of true positive matches
FP = predNum - TP;           % False Positives: unmatched predicted points
FN = tarNum - TP;            % False Negatives: unmatched target points

% Calculate the Jaccard index:
JacIdx = TP / (TP + FN + FP);

% Extract matched predicted and target locations based on optimal assignment:
predLocMatched = predLoc(assignment(:,1), :);  % Matched predicted locations
tarLocMatched = tarLoc(assignment(:,2), :);    % Matched target locations

% Calculate root mean square error of lateral (XY) positions:
rmseLateral = sqrt(mean((predLocMatched(:,1:2) - tarLocMatched(:,1:2)).^2, 'all'));

% Calculate root mean square error of axial (Z) positions:
rmseAxial = sqrt(mean((predLocMatched(:,3) - tarLocMatched(:,3)).^2, 'all'));

% Calculate performance efficiency (eff):
eff = 100 - sqrt((100 - JacIdx * 100)^2 + (rmseLateral * pixNMSize)^2 + (0.5 * rmseAxial * pixNMSize)^2);
end