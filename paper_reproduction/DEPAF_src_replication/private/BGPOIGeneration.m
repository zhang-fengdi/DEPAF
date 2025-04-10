function BGPOI = BGPOIGeneration(POI)
% BGPOIGeneration generates a POI for learning inhomogeneous background.
%
%  This function takes a signal POI as input and calculates a central region covering
%  95% of the total sum of the signal POI. Then, based on the area of this central region,
%  it generates a background POI matrix (a homogeneous patch) that is 3 times the area of the center.
%
%  Input Parameters:
%    POI - A 2D or multidimensional signal POI matrix. If it is a multidimensional matrix,
%          the function first reduces it to a 2D matrix for processing.
%
%  Output Parameters:
%    BGPOI - The generated background POI. This is a homogeneous patch with an area that is
%            3 times the central region, which covers 95% of the total sum of the signal POI.

% Find the 95% central region of the POI:
POI = sum(POI, 3);
[widPOI, lenPOI] = size(POI);
centerWid = (widPOI + 1) / 2;
centerLen = (lenPOI + 1) / 2;
thresh = 0.95 * sum(POI, 'all');
for offset = 0:max(widPOI,lenPOI)
    % Calculate the boundary of the rectangular region:
    top = max(centerWid - offset, 1);
    bottom = min(centerWid + offset, widPOI);
    left = max(centerLen - offset, 1);
    right = min(centerLen + offset, lenPOI);

    % Calculate the sum of pixels within the region:
    regionSum = sum(POI(top:bottom, left:right), 'all');

    % Check if the threshold is reached:
    if regionSum >= thresh
        break;
    end
end

% Generate background POI:
scaleFactor = 3;
BGPOISize = [bottom-top+1 right-left+1];
BGPOISize(BGPOISize ~= 1) = BGPOISize(BGPOISize ~= 1) * scaleFactor;
BGPOI = fspecial('average', BGPOISize);
end