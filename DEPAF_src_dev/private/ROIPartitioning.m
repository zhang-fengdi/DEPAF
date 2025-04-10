function ROIIdxList = ROIPartitioning(imSize,oriInd,grayVal)
% ROIPartitioning Divides the image into 2x2 ROIs based on predefined rules.
%
%  This function divides a specific region of the image into 2x2 subregions (ROIs).
%  It uses a method based on pixel values and the number of neighboring pixels to
%  partition the ROI.
%
%  Input Parameters:
%    imSize - Size of the image.
%    oriInd - Pixel indices of the ROI in the original image.
%    grayVal - Gray values corresponding to oriInd.
%
%  Output Parameters:
%    ROIIdxList - List of indices for the partitioned ROIs, with each element representing
%                 the pixel indices within a subregion.

% Recover subscript coordinates:
[subi,subj] = ind2sub(imSize,oriInd);

% If distributed within a 2x2 image block, do not split:
if max(subi)-min(subi)<=1 && max(subj)-min(subj)<=1
    ROIIdxList = {oriInd};
    return;
end

% Reconstruct a small patch for analysis, adding a ring of zeros around it
% to prevent overflow during quadrant assignment:
iBias = min(subi) - 2;
jBias = min(subj) - 2;
subi = subi - iBias;
subj = subj - jBias;

wid = max(subi) + 1;
len = max(subj) + 1;
mask = false(wid, len);
I = zeros(wid, len, 'single');

ind = sub2ind([wid len], subi, subj);
mask(ind) = true;
I(ind) = grayVal;

% Create an index map of each pixel for easy spatial indexing:
indMap = zeros(size(I), 'single');
indMap(1:wid*len) =  1 : wid*len;

ROIIdxList = cell(numel(ind), 1);
count = 0;
while 1
    % Use the 8-connected pixel count as traversal priority:
    priority = pix8ConnNumMap(mask);
    priority = priority(ind);
    [subi, subj] = ind2sub([wid len], ind);

    % Allocate coordinates with the highest priority, then recalculate:
    subLoc = [subi subj];
    subLoc = subLoc(priority == min(priority), :);

    % Traverse the 2x2 blocks centered on this pixel, grouping the block with the highest pixel sum:
    for i = 1:size(subLoc, 1)
        ci = subLoc(i, 1);
        cj = subLoc(i, 2);
        if I(ci,cj) == 0
            continue;
        end
        lu = sum(I(ci-1:ci, cj-1:cj), 'all');
        ru = sum(I(ci-1:ci, cj:cj+1), 'all');
        ld = sum(I(ci:ci+1, cj-1:cj), 'all');
        rd = sum(I(ci:ci+1, cj:cj+1), 'all');
        [~,maxIdx] = max([lu ru ld rd]);
        switch maxIdx
            case 1
                selectedPatch = indMap(ci-1:ci, cj-1:cj);
            case 2
                selectedPatch = indMap(ci-1:ci, cj:cj+1);
            case 3
                selectedPatch = indMap(ci:ci+1, cj-1:cj);
            case 4
                selectedPatch = indMap(ci:ci+1, cj:cj+1);
        end

        % Remove zero pixels in the selected Patch:
        selectedPatch = selectedPatch(mask(selectedPatch));

        % Remove assigned coordinates:
        mask(selectedPatch) = false;
        I(selectedPatch) = 0;
        ind = ind(mask(ind));

        % Record the assigned coordinate groups:
        [resi, resj] = ind2sub([wid len], selectedPatch);
        count = count + 1;
        ROIIdxList{count} = [resi resj];
    end

    % If all coordinates are assigned, exit the loop:
    if isempty(ind)
        % Remove empty cells:
        ROIIdxList(count+1:end) = [];

        % Convert assigned coordinate groups to original indices by adding bias:
        for i = 1:length(ROIIdxList)
            ROIIdxList{i} = sub2ind(imSize, ROIIdxList{i}(:,1)+iBias, ...
                ROIIdxList{i}(:,2)+jBias);
        end

        % Exit the while loop:
        break;
    end
end
end