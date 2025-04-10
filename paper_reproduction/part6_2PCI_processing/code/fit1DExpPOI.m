function POI = fit1DExpPOI(I, fitPtsNum, tol)
% fit1DExpPOI fits an exponential spike signal POI from the input image sequence.
%
% Input parameters:
%   I - Input image sequence.
%   fitPtsNum - Number of points with the highest intensity to fit.
%   tol - Tolerance for the goodness-of-fit, used to filter poor fits.
%
% Output:
%   POI - Estimated 1D exponential spike signal POI.

% Retrieve the top fitPtsNum maximum values and their indices:
[~, maxIdx] = maxk(I(:), fitPtsNum);
[xMaxIdx, yMaxIdx, zMaxIdx] = ind2sub(size(I), maxIdx);

% Disable warnings:
warning('off', 'signal:findpeaks:largeMinPeakHeight');

% Initialize fitting parameters:
count = 0;
fitLen = 200;
spikes = zeros(fitLen/2*3+1, fitPtsNum);

for k = 1:fitPtsNum
    % Boundary check to prevent overflow:
    if zMaxIdx(k) - 100 < 1 || max(zMaxIdx(k) + fitLen, zMaxIdx(k) + 100) > size(I, 3)
        continue;
    end

    % Prevent duplicate fitting:
    maxVal = I(xMaxIdx(k), yMaxIdx(k), zMaxIdx(k));
    maxValNbrhd = I(xMaxIdx(k), yMaxIdx(k), zMaxIdx(k)-100:zMaxIdx(k)+100);
    if any(maxVal < maxValNbrhd)
        continue;
    end

    % Extract data for fitting:
    y = I(xMaxIdx(k), yMaxIdx(k), zMaxIdx(k):zMaxIdx(k)+fitLen);
    y = double(reshape(y, [], 1));

    % Define fitting type (single exponential decay):
    fitType = fittype('A * exp(-lambda * x) + B', 'independent', 'x', ...
        'coefficients', {'A', 'B', 'lambda'});

    % Set fitting options:
    fitOptions = fitoptions('Method', 'NonlinearLeastSquares', ...
        'StartPoint', [1 1 1], ...     % Initial guesses
        'Lower', [-inf -inf 1e-3]);    % Lower bounds

    % Perform fitting using fit function:
    [fitResult, gof] = fit((1:length(y))', y, fitType, fitOptions);

    % Record fitted spike if goodness-of-fit meets tolerance:
    if gof.adjrsquare > tol
        spike = I(xMaxIdx(k), yMaxIdx(k), zMaxIdx(k)-fitLen/2:zMaxIdx(k)+fitLen);
        spike = spike - fitResult.B; % Background subtraction
        spike = spike / max(spike); % Normalization

        % Check for nearby spikes and ignore if any are found:
        peaks = findpeaks(reshape(spike(1:fitLen/2), [], 1), 'MinPeakHeight', 0.2);
        if isempty(peaks)
            count = count + 1;
            spikes(:,count) = spike;
        end
    end
end

% Output fitting result information:
if count == 0
    error('No POI meeting the fit tolerance found in the image; consider reducing the tolerance.');
else
    fprintf('Fitting window length: %d, adjusted R-squared threshold: %.2f, successful fits: %d/%d\n', ...
        fitLen+1, tol, count, fitPtsNum);
end

% Obtain the final POI:
spikes = spikes(:,1:count);
POI = mean(spikes, 2);
POI = POI - min(POI);
POI = POI / max(POI);
end