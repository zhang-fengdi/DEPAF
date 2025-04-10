function [POI, xSigma, ySigma] = fit2DGaussPOI(I, fitPointsNumPerI, tol)
% fit2DGaussPOI fits an elliptical Gaussian POI from the input image sequence.
%
%  Input parameters:
%    I - Input image sequence.
%    fitPointsNumPerI - Number of points to fit in each image.
%    tol - Goodness-of-fit threshold to assess fit quality.
%
%  Outputs:
%    POI - Estimated elliptical Gaussian POI sample.
%    xSigma - Estimated x-dimension sigma of Gaussian fit.
%    ySigma - Estimated y-dimension sigma of Gaussian fit.

% Remove background to prevent interference:
BGEstLevel = 5;
I = removeBGByWaveletTrans(double(I), BGEstLevel);

% Fit 2D elliptical Gaussian POI:
numI = size(I, 3);
xSigmaAll = [];
ySigmaAll = [];
for k = 1:numI
    singleI = I(:, :, k);

    % Get 8-connected regional maxima:
    BW = imregionalmax(singleI);

    % Calculate centroids of 8-connected regions:
    CC = bwconncomp(BW, 8);
    S = regionprops(CC, 'Centroid', 'Area');

    % Adjust singleFitPointsNumPerI to prevent overflow:
    singleFitPointsNumPerI = min(fitPointsNumPerI, length(S));

    % Select regions with the highest amplitudes for fitPointsNumPerI:
    centerAmplitude = zeros(1, length(S), 'single');
    for i = 1:length(S)
        y0 = round(S(i).Centroid(2));
        x0 = round(S(i).Centroid(1));
        centerAmplitude(i) = singleI(y0, x0);
    end
    [~, idx] = sort(centerAmplitude, 'descend');
    S = S(idx(1:singleFitPointsNumPerI));

    % Obtain sigma parameters of the Gaussian PSF using fitting:
    fitDataSize = 11;
    count = 0;
    xSigma = zeros(length(S), 1, 'single');
    ySigma = zeros(length(S), 1, 'single');
    for i = 1:length(S)
        % Extract central coordinates:
        y0 = round(S(i).Centroid(2));
        x0 = round(S(i).Centroid(1));

        % Define start position for fitting, adjusting to prevent boundary overflow:
        y0Start = max(1, y0 - fitDataSize);
        y0End = min(size(singleI, 1), y0 + fitDataSize);
        x0Start = max(1, x0 - fitDataSize);
        x0End = min(size(singleI, 2), x0 + fitDataSize);

        % Extract cross-sectional row and column data at PSF center for fitting:
        data1 = singleI(y0Start:y0End, x0);
        data2 = singleI(y0, x0Start:x0End);

        % Normalize to prevent low gradient values stopping iteration below tolerance:
        data1 = (data1 - min(data1)) ./ (max(data1) - min(data1));
        data2 = (data2 - min(data2)) ./ (max(data2) - min(data2));

        % Define Gaussian template:
        GaussEqu = 'a * exp(-(x-b)^2/(2*c^2)) + d';

        % Set initial parameters: intensity (1 after normalization), position, variance, offset:
        startValue1 = [1 y0 1 0];
        startValue2 = [1 x0 1 0];

        % Perform fitting:
        [f1, gof1] = fit((y0Start:y0End)', data1, GaussEqu, ...
            'Start', startValue1, 'TolFun', 1e-12, 'TolX', 1e-12);
        [f2, gof2] = fit((x0Start:x0End)', data2', GaussEqu, ...
            'Start', startValue2, 'TolFun', 1e-12, 'TolX', 1e-12);

        % Record sigma fit values:
        if gof1.adjrsquare > tol && gof2.adjrsquare > tol
            count = count + 1;
            ySigma(count) = f1.c;
            xSigma(count) = f2.c;
        end
    end

    % Display progress message:
    fprintf('Image %d/%d, window size: %d×%d, adjusted R^2 threshold: %.2f, successful fits: %d/%d\n', ...
        k, numI, fitDataSize, fitDataSize, tol, count, length(S));

    % Record sigma values for each image:
    ySigma(ySigma == 0) = [];
    xSigma(xSigma == 0) = [];
    ySigmaAll = [ySigmaAll; ySigma]; %#ok<*AGROW>
    xSigmaAll = [xSigmaAll; xSigma];
end

% Throw an error if no successful fits:
if isempty(xSigmaAll) || isempty(ySigmaAll)
    error('No POI meeting requirements found in the image; please lower tolerance.');
end

% Remove outliers:
ySigmaAll(isoutlier(ySigmaAll, 'median')) = [];
xSigmaAll(isoutlier(xSigmaAll, 'median')) = [];
ySigma = median(ySigmaAll);
xSigma = median(xSigmaAll);

% Generate POI:
x = 1 : 2 * ceil(3 * max(xSigma, ySigma)) + 1;
y = x';
POI = exp(-(x - mean(x)).^2 / (2 * xSigma^2) - (y - mean(y)).^2 / (2 * ySigma^2));

% Post-process POI:
POI = POI - min(POI, [], 'all');
POI = POI ./ max(POI, [], 'all');
end
