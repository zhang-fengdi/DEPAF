function ISyn = imgSynthesis(ISize, loc, amplitude, POI, interpMethod)
% imgSynthesis Generates a noise synthesis image based on location, amplitude, and POI.
%
%  This function generates a synthesized image based on specified location, amplitude,
%  and POI sample.
%
%  Input Parameters:
%    ISize - Size of the synthesized image, represented as a two-element vector [height width].
%    loc - List of POI location coordinates, represented as an N×3 matrix.
%    amplitude - List of POI amplitudes.
%    POI - Shape description data for the POI.
%    interpMethod - Interpolation method, such as 'spline', 'nearest', 'linear', 'cubic'.
%
%  Output Parameters:
%    ISyn - Generated synthesized image.

% Calculate the size of the POI:
[widPOI, lenPOI] = size(POI, [1 2]);

% Calculate the radius of the POI:
xRadius = (lenPOI - 1) / 2;
yRadius = (widPOI - 1) / 2;

% Calculate margin length to prevent overflow:
lenMargin = ceil(xRadius);
widMargin = ceil(yRadius);

% Generate the image and add margins to prevent overflow:
ISyn = zeros(ISize + 2*[widMargin lenMargin], 'single');

% Add the POI to the canvas:
POIPadded = padarray(POI, [1 1 1], 'replicate', 'both');
for idx = 1:length(amplitude)
    % Get coordinates from the list:
    x = loc(idx,1);
    y = loc(idx,2);
    z = loc(idx,3);

    % Snap to the nearest integer or 0.5 coordinate based on radius being even or odd:
    xNearest = round(x + mod(xRadius,1)) - mod(xRadius,1);
    yNearest = round(y + mod(yRadius,1)) - mod(yRadius,1);

    % Calculate bias between the original and snapped coordinates:
    xBias = x - xNearest;
    yBias = y - yNearest;

    % Obtain biased POI based on interpolation (add a surrounding pixel layer to prevent boundary overflow):
    [Xq, Yq, Zq] = meshgrid((1:lenPOI)-xBias+1, (1:widPOI)-yBias+1, z+1);
    POIBiased = interp3(POIPadded, Xq, Yq, Zq, interpMethod);

    % Normalize the biased POI:
    POIBiased = POIBiased / max(POIBiased, [], 'all');

    % Add the biased POI to the canvas:
    yRange = (yNearest-yRadius : yNearest+yRadius) + widMargin;
    xRange = (xNearest-xRadius : xNearest+xRadius) + lenMargin;
    ISyn(yRange,xRange) = ISyn(yRange,xRange) + ...
        amplitude(idx) * POIBiased;
end

% Crop out the margin:
win = centerCropWindow2d(size(ISyn),ISize);
ISyn = imcrop(ISyn,win);
end