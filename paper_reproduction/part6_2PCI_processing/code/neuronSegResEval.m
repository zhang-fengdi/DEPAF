function [F1, mIoU] = neuronSegResEval(predMasks, GTMasks, matchThresh)
% neuronSegResEval Evaluates generated neuron segmentation masks against ground truth masks.
%
% Inputs:
%   predMasks - 3D logical array where each slice is a predicted mask for a neuron.
%   GTMasks - 3D logical array where each slice is a ground truth mask for a neuron.
%   matchThresh - IoU threshold for considering a match between predicted and GT masks.
%
% Outputs:
%   F1 - F1 score, representing the harmonic mean of precision and recall.
%   mIoU - Mean IoU for matched masks.

% Initialize variables:
numGT = size(GTMasks, 3);
numPred = size(predMasks, 3);
costmatrix = zeros(numGT, numPred);
IoUs = zeros(numGT, numPred);

% Compute IoU values for all mask pairs:
for i = 1:numGT
    for j = 1:numPred
        mask1 = GTMasks(:,:,i);
        mask2 = predMasks(:,:,j);
        intersection = sum(sum(mask1 & mask2));
        union = sum(sum(mask1 | mask2));
        IoUs(i,j) = intersection / union;
        costmatrix(i,j) = 1 - IoUs(i,j);
    end
end
costmatrix(costmatrix > matchThresh) = 2;

% Use Hungarian algorithm for mask matching:
% Note: Official documentation specifies that costOfNonAssignment is set to half the 
%       maximum possible cost of a successful assignment.
costOfNonAssignment = 0.9;
assignments = assignmunkres(costmatrix, costOfNonAssignment);

% Return directly if no matches are found:
TP = size(assignments, 1);
if TP == 0
    F1 = 0;
    mIoU = 0;
    return;
end

% Calculate F1 score:
recall = TP / numGT;
precision = TP / numPred;
F1 = 2 * recall * precision / (recall + precision);

% Calculate mean IoU for matched mask pairs:
matchedIoUs = zeros(TP, 1);
for i = 1:TP
    matchedIoUs(i) = IoUs(assignments(i,1), assignments(i,2));
end
mIoU = mean(matchedIoUs);
end