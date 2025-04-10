function res = pix8ConnNumMap(mask)
% pix8ConnNumMap calculates the number of non-zero pixels in the 8-neighborhood of each pixel in the mask.
%
%  This function calculates the number of non-zero pixels in the 8-neighborhood (i.e., the surrounding 8 pixels)
%  of each pixel in a binary image.
%
%  Input Parameters:
%    mask - Input binary image, where pixels with a value of 1 represent specific regions of interest.
%
%  Output Parameters:
%    res - Output image where each pixel's value represents the number of non-zero pixels in the 8-neighborhood
%          of that pixel in the input image.

mask = logical(mask);
res = conv2(mask, ones(3,3,'single'), 'same');
end