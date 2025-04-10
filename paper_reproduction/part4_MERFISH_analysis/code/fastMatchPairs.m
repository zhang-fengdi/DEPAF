function assignments = fastMatchPairs(loc1, loc2, matchThresh)
% fastMatchPairs performs efficient linear matching between two sets of points within a specified threshold.
%
% Inputs:
%   loc1 - NxM matrix representing the first set of locations (N: number of points, M: dimensions)
%   loc2 - PxM matrix representing the second set of locations (P: number of points, M: dimensions)
%   matchThresh - Maximum allowed distance to match points between loc1 and loc2
%
% Output:
%   assignments - Matrix where each row represents a matched pair of indices. 
%                 The first column corresponds to indices in loc1, and the 
%                 second column corresponds to indices in loc2.

% Compute distance matrix:
distanceMatrix = pdist2(loc1, loc2, 'euclidean');

% Compute validity matrix:
validMat = distanceMatrix <= matchThresh;

% Retain rows and columns with at least one value below threshold:
validRowIdx = find(any(validMat, 2));
validColIdx = find(any(validMat, 1));
validMat = validMat(validRowIdx, validColIdx);

% Perform matching if points below threshold exist:
assignments = [];
if ~isempty(validMat)

    % Match directly if no conflicts:
    uniqueMatches = validMat;
    uniqueMatches(sum(validMat, 2) ~= 1, :) = false;
    uniqueMatches(:, sum(validMat, 1) ~= 1) = false;
    [uniqueMatchRow, uniqueMatchCol] = find(uniqueMatches);
    assignments1 = [validRowIdx(uniqueMatchRow) validColIdx(uniqueMatchCol)'];
    validRowIdx(uniqueMatchRow) = [];
    validColIdx(uniqueMatchCol) = [];

    % If conflicts arise, use hierarchical clustering to separate independent conflict clusters,
    % and apply linear assignment to resolve conflicts within each cluster:
    assignments2 = [];
    conflictLoc = [loc1(validRowIdx,:); loc2(validColIdx,:)];
    if ~isempty(conflictLoc)
        clusterIdx = clusterdata(conflictLoc, 'Linkage', 'single', 'criterion', 'distance', 'cutoff', matchThresh);
        for k = 1:max(clusterIdx)
            % Extract positions within the cluster that belong to the two sets:
            singleClusterIdx = find(clusterIdx == k);
            singleClusterIdx1 = singleClusterIdx(singleClusterIdx <= size(validRowIdx,1));
            singleClusterIdx2 = singleClusterIdx(singleClusterIdx > size(validRowIdx,1)) - size(validRowIdx,1);
            singleClusterLoc1 = loc1(validRowIdx(singleClusterIdx1),:);
            singleClusterLoc2 = loc2(validColIdx(singleClusterIdx2),:);

            % Resolve assignment conflicts using linear assignment:
            singleDistanceMatrix = pdist2(singleClusterLoc1, singleClusterLoc2, 'euclidean');
            singleDistanceMatrix(singleDistanceMatrix > matchThresh) = inf;
            costOfNonAssignment = 1000;
            singleAssignments = matchpairs(singleDistanceMatrix, costOfNonAssignment);
            assign1 = validRowIdx(singleClusterIdx1(singleAssignments(:,1)));
            assign2 = validColIdx(singleClusterIdx2(singleAssignments(:,2)));
            assignments2 = [assignments2; assign1 assign2']; %#ok<*AGROW>
        end
    end

    % Merge matching results:
    assignments = [assignments1; assignments2];

    % Sort by the second column for conventional ordering:
    [~, sortedIdx] = sort(assignments(:,2), 'ascend');
    assignments = assignments(sortedIdx, :);
end
end