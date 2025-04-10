function loc = locResizeCenterKeep(loc, oriSize, scale)
% locResizeCenterKeep Obtain center-aligned resized coordinates.
%
%  This function adjusts a set of coordinates according to a specified scaling factor,
%  ensuring that the scaled coordinates remain center-aligned.
%
%  Input Parameters:
%    loc - Set of original coordinates, stored as a cell array where each cell contains
%          an N×3 matrix of coordinates.
%    oriSize - Dimensions of the original data, given as [Height Width Depth].
%    scale - Scaling factor, given as [Height Scale Width Scale Depth Scale].
%
%  Output Parameters:
%    loc - Set of scaled coordinates, format remains consistent with input.

% Get original image dimensions:
oriWid = oriSize(1);
oriLen = oriSize(2);
oriChannel = oriSize(3);

% Calculate the original image center:
oriXCenter = mean(1:oriLen);
oriYCenter = mean(1:oriWid);
oriZCenter = mean(1:oriChannel);

% Calculate the center-aligned coordinates for the scaled image:
if scale(1) >= 1
    tarWid = ceil(oriWid * scale(1));
else
    tarWid = floor(oriWid * scale(1));
end
if scale(2) >= 1
    tarLen = ceil(oriLen * scale(2));
else
    tarLen = floor(oriLen * scale(2));
end
if scale(3) >= 1
    tarChannel = ceil(oriChannel * scale(3));
else
    tarChannel = floor(oriChannel * scale(3));
end

% Calculate the center of the scaled image:
tarXCenter = mean(1:tarLen);
tarYCenter = mean(1:tarWid);
tarZCenter = mean(1:tarChannel);

% Compute the scaled coordinates:
for i = 1:length(loc)
    X = loc{i}(:,1);
    Y = loc{i}(:,2);
    Z = loc{i}(:,3);
    X = (X - oriXCenter) * scale(2) + tarXCenter;
    Y = (Y - oriYCenter) * scale(1) + tarYCenter;
    Z = (Z - oriZCenter) * scale(3) + tarZCenter;
    loc{i} = [X Y Z];
end
end