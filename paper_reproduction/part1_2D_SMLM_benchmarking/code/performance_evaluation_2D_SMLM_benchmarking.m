clear all;
close all;

% Load DEPAF source code:
addpath(genpath('..\..\DEPAF_src_replication'));

for density = [3775 7550 18874 33974 56623 113246]
    for int = [1000 7000 40000]
        predLocDir = dir(['..\models\predRes*' num2str(int) '_' num2str(density) '*.mat']);
        for i = 1:length(predLocDir)
            evalDataPath = ['..\datasets\data_' num2str(int) '_' num2str(density) '_PerlinBG.tif']; % Path of predicted data (for extracting brightness information for matching)
            predLocPath = [predLocDir(i).folder '\' predLocDir(i).name]; % Path of prediction results
            tarLocPath = ['..\datasets\data_' num2str(int) '_' num2str(density) '_tarLoc.mat']; % Path of ground truth labels
            matchThresh = 1; % Matching threshold between predicted points and ground truth points when evaluating results (unit: pixel)
            pixNMSize = 120; % Pixel size (unit: nm)
            evalResSavePath = '..\models\'; % Path to save evaluation parameters
            evalUseParallel = true; % Whether to use parallel computation
            evalParNum = 6; % Number of parallel pool workers

            perfEval2DSMLM(evalDataPath, predLocPath, tarLocPath, matchThresh, pixNMSize, ...
                evalResSavePath, evalUseParallel, evalParNum);
        end
    end
end