function assignments = fastMatchPairs(loc1, loc2, matchThresh)
% Preserve the historical double-precision matching and assignment order.
assignments = [];
if isempty(loc1) || isempty(loc2)
    return;
end

loc1 = double(loc1);
loc2 = double(loc2);

neighborIdx = rangesearch(loc2, loc1, matchThresh);
validRowMask = ~cellfun(@isempty, neighborIdx);
validRowIdx = find(validRowMask);
if isempty(validRowIdx)
    return;
end

validColIdx = unique([neighborIdx{validRowMask}]);
validColIdx = validColIdx(:)';
if isempty(validColIdx)
    return;
end

colPosition = zeros(1, size(loc2, 1));
colPosition(validColIdx) = 1:numel(validColIdx);
validMat = sparse(numel(validRowIdx), numel(validColIdx));
for rowPos = 1:numel(validRowIdx)
    cols = neighborIdx{validRowIdx(rowPos)};
    validMat(rowPos, colPosition(cols)) = true;
end

uniqueMatches = validMat;
uniqueMatches(sum(validMat, 2) ~= 1, :) = false;
uniqueMatches(:, sum(validMat, 1) ~= 1) = false;
[uniqueMatchRow, uniqueMatchCol] = find(uniqueMatches);
assignments1 = [validRowIdx(uniqueMatchRow) validColIdx(uniqueMatchCol)'];

validRowIdx(uniqueMatchRow) = [];
validColIdx(uniqueMatchCol) = [];

assignments2 = [];
conflictLoc = [loc1(validRowIdx, :); loc2(validColIdx, :)];
if ~isempty(conflictLoc)
    clusterIdx = radiusConnectedComponents(conflictLoc, matchThresh);
    for k = 1:max(clusterIdx)
        singleClusterIdx = find(clusterIdx == k);
        singleClusterIdx1 = singleClusterIdx(singleClusterIdx <= size(validRowIdx, 1));
        singleClusterIdx2 = singleClusterIdx(singleClusterIdx > size(validRowIdx, 1)) - size(validRowIdx, 1);
        singleClusterLoc1 = loc1(validRowIdx(singleClusterIdx1), :);
        singleClusterLoc2 = loc2(validColIdx(singleClusterIdx2), :);

        if isempty(singleClusterLoc1) || isempty(singleClusterLoc2)
            continue;
        end
        singleDistanceMatrix = pdist2(singleClusterLoc1, singleClusterLoc2, 'euclidean');
        singleDistanceMatrix(singleDistanceMatrix > matchThresh) = inf;
        costOfNonAssignment = 1000;
        singleAssignments = matchpairs(singleDistanceMatrix, costOfNonAssignment);
        assign1 = validRowIdx(singleClusterIdx1(singleAssignments(:, 1)));
        assign2 = validColIdx(singleClusterIdx2(singleAssignments(:, 2)));
        assignments2 = [assignments2; assign1 assign2']; %#ok<AGROW>
    end
end

assignments = [assignments1; assignments2];
[~, sortedIdx] = sort(assignments(:, 2), 'ascend');
assignments = assignments(sortedIdx, :);
end


function clusterIdx = radiusConnectedComponents(points, radius)
numPoints = size(points, 1);
clusterIdx = zeros(numPoints, 1);
if numPoints == 0
    return;
end

neighborIdx = rangesearch(points, points, radius);
visited = false(numPoints, 1);
currentCluster = 0;

for startIdx = 1:numPoints
    if visited(startIdx)
        continue;
    end

    currentCluster = currentCluster + 1;
    stack = startIdx;
    visited(startIdx) = true;

    while ~isempty(stack)
        pointIdx = stack(end);
        stack(end) = [];
        clusterIdx(pointIdx) = currentCluster;

        neighbors = neighborIdx{pointIdx};
        neighbors = neighbors(~visited(neighbors));
        if isempty(neighbors)
            continue;
        end

        visited(neighbors) = true;
        stack = [stack; neighbors(:)]; %#ok<AGROW>
    end
end
end
