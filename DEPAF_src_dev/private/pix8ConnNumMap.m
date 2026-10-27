function res = pix8ConnNumMap(mask)
% pix8ConnNumMap calculates the number of non-zero pixels in the 3-by-3 neighborhood, including the center pixel.
%
%  This function calculates the number of non-zero pixels in the 3-by-3 neighborhood
%  of each pixel in a binary image, including the center pixel.
%
%  Input Parameters:
%    mask - Input binary image, where pixels with a value of 1 represent specific regions of interest.
%
%  Output Parameters:
%    res - Output image where each pixel's value represents the number of non-zero pixels in the 3-by-3 neighborhood
%          of that pixel, including the center pixel.

mask = logical(mask);
res = conv2(mask, ones(3,3,'single'), 'same');
end