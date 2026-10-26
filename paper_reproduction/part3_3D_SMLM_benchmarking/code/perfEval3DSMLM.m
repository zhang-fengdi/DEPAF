function [eff, JacIdx, lateralError, axialError] = perfEval3DSMLM(predLoc, tarLoc, matchThresh, pixNMSize)
% perfEval3DSMLM evaluates the performance of 3D localization predictions
%
% Inputs:
%   predLoc - Nx3 matrix representing predicted positions (N: number of predicted points)
%   tarLoc - Mx3 matrix representing target positions (M: number of target points)
%   matchThresh - Maximum allowed distance to match predictions with targets
%   pixNMSize - Nanometres per coordinate unit; used once in efficiency
%
% Outputs:
%   eff - Performance efficiency score
%   JacIdx - Jaccard index (measure of similarity between predictions and targets)
%   lateralError - Mean Euclidean XY distance in input coordinate units
%   axialError - Mean absolute Z distance in input coordinate units

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

% Calculate mean lateral distance:
lateralError = mean(sqrt(sum((predLocMatched(:,1:2) - tarLocMatched(:,1:2)).^2, 2)));

% Calculate mean absolute axial distance:
axialError = mean(abs(predLocMatched(:,3) - tarLocMatched(:,3)));

% Calculate performance efficiency (eff):
eff = 100 - 0.5 * ( ...
    sqrt((100 - JacIdx * 100)^2 + (lateralError * pixNMSize)^2) + ...
    sqrt((100 - JacIdx * 100)^2 + (0.5 * axialError * pixNMSize)^2));
end