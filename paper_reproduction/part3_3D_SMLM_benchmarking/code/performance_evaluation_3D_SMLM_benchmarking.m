clear all;
close all;

% Load DEPAF source code:
addpath(genpath('..\..\DEPAF_src_replication'));

% Set path to save evaluation results:
evalResSavePath = '..\models\';

% Set file paths:
tarLocPath = '..\datasets\positions.csv';
predResPath = '..\models\PredRes_sequence-as-stack-MT0.N2.HD-DH-Exp_by_Mdl_08-15-06-22-20.mat';

% Load ground truth location data:
disp('[Evaluation] Loading prediction results and ground truth labels...');
tarLoc = readmatrix(tarLocPath);

% Preprocess ground truth data:
tarLoc(:, 1) = 64 - tarLoc(:, 1) / 100; % Convert X-axis
tarLoc(:, 2:3) = tarLoc(:, 2:3) / 100;  % Convert Y and Z axes

% Filter coordinates within specified range:
tarLoc = filterWithinRange(tarLoc, 5, 60, 5, 60);

% Load prediction result data:
load(predResPath, 'loc', 'amplitude', 'modelID');

% Convert predicted locations and amplitudes to matrix format:
loc = cell2mat(loc);
amplitude = cell2mat(amplitude);

% Filter data based on amplitude percentile:
highAmplitudeIdx = amplitude > prctile(amplitude, 42);
loc = loc(highAmplitudeIdx, :);

% Manually align coordinate systems:
loc(:, 1) = 64 - loc(:, 1);   % Flip X-axis
loc(:, 3) = (loc(:, 3) - 76) * 10 / 100; % Convert Z-axis
loc(:, 1) = loc(:, 1) + 0.2;  % Shift X-axis
loc(:, 2) = loc(:, 2) - 0.2;  % Shift Y-axis

% Filter predicted points within specified range:
loc = filterWithinRange(loc, 5, 60, 5, 60);

% Evaluate performance:
disp('[Evaluation] Starting evaluation...')
matchThresh = 1;
pixNMSize = 100;
[eff, JacIdx, rmseLateral, rmseAxial] = perfEval3DSMLM(loc, tarLoc, 1, 100);

% Save evaluation results:
disp('[Evaluation] Saving evaluation results...');
checkPath(evalResSavePath);
[~, tarLocDataName] = fileparts(tarLocPath); % Get data file name
[~, predResName] = fileparts(predResPath); % Get data file name
savePath = [evalResSavePath '\EvalRes_' predResName(9:end-4) ...
    '_by_Mdl_' modelID '.mat'];
save(savePath, 'eff', 'JacIdx', 'rmseLateral', 'rmseAxial', ...
    'modelID', 'predResName', 'tarLocDataName');


% Helper function: Filter points within boundary range.
function filteredData = filterWithinRange(data, xMin, xMax, yMin, yMax)
isInRange = data(:, 1) >= xMin & data(:, 1) <= xMax & ...
    data(:, 2) >= yMin & data(:, 2) <= yMax;
filteredData = data(isInRange, :);
end