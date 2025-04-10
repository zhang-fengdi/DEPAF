function masks = neuronSegMaskGeneration(loc, widMask, lenMask, ...
    epsDBSCAN, minPtsDBSCAN, ...
    convexBound, expDistBound, ...
    minAreaMask, avgAreaMask, ...
    threshBinarizeMask, ...
    threshCOM0, threshCOM, threshIoU, threshConsume, threshConsec)
% neuronSegMaskGeneration generates neuron segmentation masks based on DEPAF localization results.
%
% Inputs:
%   loc - Cell array where each cell contains Nx2 matrices of neuron localization points for a frame.
%   widMask - Width of the output mask.
%   lenMask - Length of the output mask.
%   epsDBSCAN - Radius parameter for DBSCAN clustering.
%   minPtsDBSCAN - Minimum points required for DBSCAN clustering.
%   convexBound - Convexity parameter used in boundary detection.
%   expDistBound - Expansion distance for boundary modification.
%   minAreaMask - Minimum area threshold for retaining a mask.
%   avgAreaMask - Average area threshold for retaining a mask after merging.
%   threshBinarizeMask - Binarization threshold for masks during the merging steps.
%   threshCOM0 - Initial threshold for merging based on center of mass (COM).
%   threshCOM - Secondary threshold for merging based on COM distance.
%   threshIoU - IoU threshold for mask merging based on overlap.
%   threshConsume - Threshold for merging masks based on consume rate (overlap as a fraction of area).
%   threshConsec - Threshold for consecutive frames to retain in mask merging.
%
% Outputs:
%   masks - 3D logical array where each slice represents a binary mask for a neuron.

% Calculate dimensions:
numLoc = length(loc);

% Initialize mask container:
masks = false(widMask, lenMask, numLoc);
% Initialize container to track merged indices:
mergedIdxs = cell(numLoc, 1);
maskCount = 0;
for idx = 1:numLoc
    % Extract location for a single image:
    singleLoc = double(loc{idx}(:,1:2));

    % Skip if localization result is empty:
    if isempty(singleLoc)
        continue;
    end

    % Perform DBSCAN clustering:
    [labels, ~] = dbscan(singleLoc, epsDBSCAN, minPtsDBSCAN);

    % Calculate boundaries for each cluster:
    uniqueLabels = unique(labels);
    for i = 1:length(uniqueLabels)
        label = uniqueLabels(i);
        if label ~= -1 % Exclude noise points
            % Get points in a single cluster:
            clusterPts = singleLoc(labels == label, :);

            % Calculate boundary points:
            k = boundary(clusterPts(:,1), clusterPts(:,2), convexBound);
            if isempty(k)
                continue;
            end
            boundPts = clusterPts(k,:)';

            % Compute boundary curve using cubic spline:
            splineCurve = cscvn(boundPts);

            % Calculate tangent for each point on the curve:
            tangents = fnder(splineCurve);
            splineBreaks = splineCurve.breaks;
            tangentVals = ppval(tangents, splineBreaks);
            normals = [-tangentVals(2,:); tangentVals(1,:)];
            normals = normals ./ vecnorm(normals);

            % Offset curve by moving along normals:
            newBoundPts = boundPts - expDistBound * normals;
            splineCurve = cscvn(newBoundPts);

            % Generate curve coordinates:
            curvePts = fnplt(splineCurve, 'y', 2);

            % Convert curve to mask:
            maskCount = maskCount + 1;
            masks(:,:,maskCount) = poly2mask(curvePts(1,:), curvePts(2,:), widMask, lenMask);
            mergedIdxs{maskCount} = idx;
        end
    end
end
masks = masks(:,:,1:maskCount);
mergedIdxs = mergedIdxs(1:maskCount);

% Convert mask type to support operations that yield real values:
masks = uint16(masks);

% Step 1: Filter out masks that are too small based on minAreaMask.
numMask = size(masks, 3);
validMaskIdxs = true(numMask,1);
for i = 1:numMask
    if sum(masks(:,:,i), 'all') <= minAreaMask
        validMaskIdxs(i) = false;
    end
end
% Update result:
masks = masks(:,:,validMaskIdxs);
mergedIdxs = mergedIdxs(validMaskIdxs);
validMaskIdxs = validMaskIdxs(validMaskIdxs);
numMask = size(masks, 3);

% Step 2: Perform non-unique sequential merging based on a small COM threshold threshCOM0.
% Logic: Merge multiple masks as long as the conditions are met.
% Higher-order masks merge first; after merging, masks become invalid, ending related chain merging.
COMs = iCenterOfMass(masks);
for i = 1:numMask
    if ~validMaskIdxs(i)
        continue;
    end
    idxsToBeMerged = false(numMask,1);
    for j = i+1:numMask
        if ~validMaskIdxs(j)
            continue;
        end
        if norm(COMs(i,:)-COMs(j,:),2) <= threshCOM0
            idxsToBeMerged(j) = true;
        end
    end
    masks(:,:,i) = masks(:,:,i) + sum(masks(:,:,idxsToBeMerged), 3, 'native');
    mergedIdxs{i} = [mergedIdxs{i}; vertcat(mergedIdxs{idxsToBeMerged})];
    validMaskIdxs(idxsToBeMerged) = false;
end
% Update result:
masks = masks(:,:,validMaskIdxs);
mergedIdxs = mergedIdxs(validMaskIdxs);
validMaskIdxs = validMaskIdxs(validMaskIdxs);
numMask = size(masks, 3);

% Step 3: Perform unique, order-independent merging based on a larger COM threshold threshCOM.
% Logic: If multiple masks meet the condition, only one "may" merge.
% Merge condition: Target mask must be the closest COM for all candidate masks.
tempMasks = masks >= max(masks,[],[1 2]) * threshBinarizeMask;
COMs = iCenterOfMass(tempMasks);
mergeMatrix = false(numMask);
COMDistMatrix = inf(numMask);
for i = 1:numMask
    for j = i+1:numMask
        COMDist = norm(COMs(i,:)-COMs(j,:),2);
        if COMDist <= threshCOM && i ~= j
            mergeMatrix(i,j) = true;
            COMDistMatrix(i,j) = COMDist;
        end
    end
end
% Resolve conflicts:
for j = 1:numMask
    candidateMasks = find(mergeMatrix(:,j));
    if numel(candidateMasks) >= 2
        [~, i] = min(COMDistMatrix(candidateMasks, j));
        candidateMasks(i) = [];
        mergeMatrix(candidateMasks,:) = false;
        COMDistMatrix(candidateMasks,:) = inf;
    end
end
% Execute merging:
for i = 1:numMask
    idxsToBeMerged = mergeMatrix(i,:);
    if any(idxsToBeMerged)
        masks(:,:,i) = masks(:,:,i) + sum(masks(:,:,idxsToBeMerged),3, 'native');
        mergedIdxs{i} = [mergedIdxs{i}; vertcat(mergedIdxs{idxsToBeMerged})];
        validMaskIdxs(idxsToBeMerged) = false;
    end
end
% Update result:
masks = masks(:,:,validMaskIdxs);
mergedIdxs = mergedIdxs(validMaskIdxs);
validMaskIdxs = validMaskIdxs(validMaskIdxs);
numMask = size(masks, 3);

% Step 4: Perform non-unique sequential merging based on IoU threshold threshIoU.
% Logic: Merge multiple masks as long as the conditions are met.
tempMasks = masks >= max(masks,[],[1 2]) * threshBinarizeMask;
for i = 1:numMask
    if ~validMaskIdxs(i)
        continue;
    end
    idxsToBeMerged = false(numMask,1);
    for j = i+1:numMask
        if ~validMaskIdxs(j)
            continue;
        end
        intersection = sum(tempMasks(:,:,i) & tempMasks(:,:,j), 'all');
        union = sum(tempMasks(:,:,i) | tempMasks(:,:,j), 'all');
        IoU = intersection / union;
        if IoU >= threshIoU
            idxsToBeMerged(j) = true;
        end
    end
    masks(:,:,i) = masks(:,:,i) + sum(masks(:,:,idxsToBeMerged), 3, 'native');
    mergedIdxs{i} = [mergedIdxs{i}; vertcat(mergedIdxs{idxsToBeMerged})];
    validMaskIdxs(idxsToBeMerged) = false;
end
% Update result:
masks = masks(:,:,validMaskIdxs);
mergedIdxs = mergedIdxs(validMaskIdxs);
validMaskIdxs = validMaskIdxs(validMaskIdxs);
numMask = size(masks, 3);

% Step 5: Perform merging based on consume threshold threshConsume.
% Logic: Compute consume matrix, then pick all pairs with consume above threshold.
% If any mask has area larger than avgAreaMask, delete the larger mask; otherwise, merge.
tempMasks = masks >= max(masks,[],[1 2]) * threshBinarizeMask;
% Determine merging:
mergeMatrix = false(numMask);
for i = 1:numMask
    for j = 1:numMask
        if i ~= j
            intersection = sum(tempMasks(:,:,i) & tempMasks(:,:,j), 'all');
            area = sum(tempMasks(:,:,i), 'all');
            consume = intersection / area;
            mergeMatrix(i,j) = consume >= threshConsume;
        end
    end
end
% Identify invalid masks:
for i = 1:numMask
    for j = 1:numMask
        if mergeMatrix(i,j)
            area1 = sum(tempMasks(:,:,i), 'all');
            area2 = sum(tempMasks(:,:,j), 'all');
            if max(area1, area2) > avgAreaMask
                if area1 > area2
                    validMaskIdxs(i) = false;
                else
                    validMaskIdxs(j) = false;
                end
            end
        end
    end
end
% Remove invalid masks:
mergeMatrix = mergeMatrix(validMaskIdxs,validMaskIdxs);
masks = masks(:,:,validMaskIdxs);
mergedIdxs = mergedIdxs(validMaskIdxs);
validMaskIdxs = validMaskIdxs(validMaskIdxs);
numMask = size(masks, 3);
% Use dynamic programming to establish merging relationships (supports chain merging):
[i,j] = find(mergeMatrix);
mergeAim = 1:numMask; % Target mask index for merging, initially set to self.
for k = 1:length(i)
    iMergeAim = mergeAim(i(k));
    jMergeAim = mergeAim(j(k));
    if iMergeAim < jMergeAim
        mergeAim(j(k)) = iMergeAim;
    else
        mergeAim(i(k)) = jMergeAim;
    end
end
% Execute merging:
for i = 1:numMask
    idxsToBeMerged = find(mergeAim == i);
    idxsToBeMerged = setdiff(idxsToBeMerged, i);
    if ~isempty(idxsToBeMerged)
        masks(:,:,i) = masks(:,:,i) + sum(masks(:,:,idxsToBeMerged),3, 'native');
        mergedIdxs{i} = [mergedIdxs{i}; vertcat(mergedIdxs{idxsToBeMerged})];
        validMaskIdxs(idxsToBeMerged) = false;
    end
end
% Update result:
masks = masks(:,:,validMaskIdxs);
mergedIdxs = mergedIdxs(validMaskIdxs);
validMaskIdxs = validMaskIdxs(validMaskIdxs);
numMask = size(masks, 3);

% Step 6: Filter based on continuous frames (threshConsec) using mergedIdxs.
for i = 1:numMask
    activeIdxs = mergedIdxs{i};
    activeIdxs = sort(activeIdxs, 'ascend');
    idxGaps = activeIdxs(threshConsec+1:end) - activeIdxs(1:end-threshConsec);
    if ~any(idxGaps == threshConsec)
        validMaskIdxs(i) = false;
    end
end
% Update result:
masks = masks(:,:,validMaskIdxs);

% Generate final masks:
masks = masks >= max(masks,[],[1 2]) * threshBinarizeMask;
end


% Helper function: Calculate Mask center of mass (COM).
function COMs = iCenterOfMass(masks)
numMask = size(masks,3);
COMs = zeros(numMask, 2);
for i = 1:numMask
    [rows, cols] = find(masks(:,:,i));
    COMs(i,:) = mean([cols rows]);
end
end