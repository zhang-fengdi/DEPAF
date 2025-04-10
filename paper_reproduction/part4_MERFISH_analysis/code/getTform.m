function tform = getTform(fixed, moving)
% getTform estimates an affine transformation matrix to align two images using feature matching.
%
% Inputs:
%   fixed - The reference (fixed) image.
%   moving - The image to be aligned (moving image).
%
% Output:
%   tform - Affine transformation object representing the transformation matrix that aligns 
%           the moving image to the fixed image.

% Detect feature points:
ptsFixed = detectKAZEFeatures(fixed);
ptsMoving = detectKAZEFeatures(moving);

% Extract features at feature points:
[featuresFixed, validPtsFixed] = extractFeatures(fixed, ptsFixed);
[featuresMoving, validPtsMoving] = extractFeatures(moving, ptsMoving);

% Match feature points:
indexPairs = matchFeatures(featuresFixed, featuresMoving);
matchedFixed  = validPtsFixed(indexPairs(:,1));
matchedMoving = validPtsMoving(indexPairs(:,2));

% Estimate transformation matrix:
tform = estgeotform2d(matchedMoving, matchedFixed, 'affine');
end