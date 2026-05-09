function assignments = fastMatchPairs(loc1, loc2, matchThresh)
% fastMatchPairs matches two point sets within a distance threshold.
%
% This implementation keeps the original conflict-resolution semantics but
% replaces the initial dense pdist2 radius search with KD-tree rangesearch.
%
% Inputs:
%   loc1 - NxM matrix representing the first set of locations
%   loc2 - PxM matrix representing the second set of locations
%   matchThresh - Maximum allowed distance to match points between loc1 and loc2
%
% Output:
%   assignments - Matrix where each row is [indexInLoc1, indexInLoc2]

assignments = [];
if isempty(loc1) || isempty(loc2)
    return;
end

% Use KD-tree range search to avoid materializing the full N-by-P distance matrix.
neighborIdx = rangesearch(loc2, loc1, matchThresh);
validRowIdx = find(~cellfun(@isempty, neighborIdx));
if isempty(validRowIdx)
    return;
end

validColIdx = unique([neighborIdx{validRowIdx}])';

% Build the same logical validity matrix as the original implementation, but
% only for rows/columns that have at least one candidate edge.
edgeCounts = cellfun(@numel, neighborIdx(validRowIdx));
edgeStart = [0; cumsum(edgeCounts(:))];
rowSub = zeros(edgeStart(end), 1);
colSub = zeros(edgeStart(end), 1);
for r = 1:numel(validRowIdx)
    cols = neighborIdx{validRowIdx(r)};
    edgeIdx = edgeStart(r) + 1:edgeStart(r + 1);
    [~, mappedCols] = ismember(cols(:), validColIdx);
    rowSub(edgeIdx) = r;
    colSub(edgeIdx) = mappedCols;
end
validMat = sparse(rowSub, colSub, true, numel(validRowIdx), numel(validColIdx));

% Match directly if no conflicts:
rowCounts = full(sum(validMat, 2));
colCounts = full(sum(validMat, 1));
uniqueMatches = validMat;
uniqueMatches(rowCounts ~= 1, :) = false;
uniqueMatches(:, colCounts ~= 1) = false;
[uniqueMatchRow, uniqueMatchCol] = find(uniqueMatches);
assignments1 = [validRowIdx(uniqueMatchRow), validColIdx(uniqueMatchCol)];
validRowIdx(uniqueMatchRow) = [];
validColIdx(uniqueMatchCol) = [];

% If conflicts arise, use hierarchical clustering to separate independent
% conflict clusters, and apply linear assignment to resolve each cluster.
assignments2 = [];
conflictLoc = [loc1(validRowIdx, :); loc2(validColIdx, :)];
if ~isempty(conflictLoc)
    clusterIdx = clusterdata(conflictLoc, 'Linkage', 'single', ...
        'criterion', 'distance', 'cutoff', matchThresh);
    for k = 1:max(clusterIdx)
        singleClusterIdx = find(clusterIdx == k);
        singleClusterIdx1 = singleClusterIdx(singleClusterIdx <= numel(validRowIdx));
        singleClusterIdx2 = singleClusterIdx(singleClusterIdx > numel(validRowIdx)) - numel(validRowIdx);
        if isempty(singleClusterIdx1) || isempty(singleClusterIdx2)
            continue;
        end

        singleClusterLoc1 = loc1(validRowIdx(singleClusterIdx1), :);
        singleClusterLoc2 = loc2(validColIdx(singleClusterIdx2), :);

        singleDistanceMatrix = pdist2(singleClusterLoc1, singleClusterLoc2, 'euclidean');
        singleDistanceMatrix(singleDistanceMatrix > matchThresh) = inf;
        singleAssignments = matchpairs(singleDistanceMatrix, 1000);
        if isempty(singleAssignments)
            continue;
        end

        assign1 = validRowIdx(singleClusterIdx1(singleAssignments(:, 1)));
        assign2 = validColIdx(singleClusterIdx2(singleAssignments(:, 2)));
        assignments2 = [assignments2; assign1, assign2]; %#ok<AGROW>
    end
end

% Merge matching results and keep the original output ordering.
assignments = [assignments1; assignments2];
if ~isempty(assignments)
    [~, sortedIdx] = sort(assignments(:, 2), 'ascend');
    assignments = assignments(sortedIdx, :);
end
end
