function [dist, errRate] = perfEvalSpikeInfer(A, B, q, tol)
% perfEvalSpikeInfer Evaluates spike train similarity metrics.
%
%   This function computes dissimilarity and error metrics between two spike trains, including
%   the Victor–Purpura distance and error rate based on F1-score. It is typically used to assess
%   the performance of spike inference algorithms.
%
%   [dist, errRate] = perfEvalSpikeInfer(A, B, q, tol)
%
%   Input parameters:
%     A        - Ground truth spike train, as a row vector of spike times (in seconds or other units).
%     B        - Predicted spike train, as a row vector of spike times.
%     q        - (Optional) Victor–Purpura cost parameter (unit: 1/second), controlling the penalty for shifting spikes.
%     tol      - (Optional) Matching tolerance in seconds. If specified, spikes within this range are considered matched.
%                Default is 0, which requires exact matches.
%
%   Output:
%     dist     - Victor–Purpura distance between A and B, indicating temporal dissimilarity.
%     errRate  - Error Rate, computed as 1 − F1 score, where F1 = 2 × |A ∩ B| / (|A| + |B|).

if nargin < 3
    q = 1;       % Default movement cost parameter
end
if nargin < 4
    tol = 0.5;   % Default matching tolerance
end

% Input preprocessing:
A = A(:)';  
B = B(:)';
A = A(isfinite(A));
B = B(isfinite(B));
A = sort(A);
B = sort(B);

% Victor–Purpura distance:
m = numel(A);
n = numel(B);
% Initialize dynamic programming matrix D of size (m+1)×(n+1):
D = zeros(m+1, n+1);
D(:,1) = (0:m)';    % Cost of deleting all spikes
D(1,:) = 0:n;       % Cost of inserting all spikes

for i = 1:m
    for j = 1:n
        shiftCost = q * abs(A(i) - B(j));
        D(i+1,j+1) = min([
            D(i,  j)   + shiftCost, ...  % Move A(i) to B(j)
            D(i+1,j)   + 1,         ...  % Insert B(j)
            D(i,  j+1) + 1            % Delete A(i)
            ]);
    end
end
dist = D(m+1, n+1);

% Compute |A ∩ B|:
if tol > 0
    % Tolerance-based matching: each A(i) can match only one unmatched B(j)
    matchedB = false(1,n);
    interCount = 0;
    for i = 1:m
        idx = find(~matchedB & abs(B - A(i)) <= tol, 1);
        if ~isempty(idx)
            matchedB(idx) = true;
            interCount = interCount + 1;
        end
    end
else
    % Exact matching:
    interCount = numel(intersect(A, B));
end

% Error Rate:
if m + n > 0
    F1 = 2 * interCount / (m + n);
else
    F1 = 0;
end
errRate = 1 - F1;
end