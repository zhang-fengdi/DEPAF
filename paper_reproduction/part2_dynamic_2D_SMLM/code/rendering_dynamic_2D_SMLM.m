clear all;
close all;

% Load DEPAF source code:
addpath(genpath('..\..\DEPAF_src_replication'));

rndrPredLocPath = '..\models\PredRes_ROI#5_1_149_107_255_1_5000_bleaching_corrected_by_Mdl_07-27-22-31-06.mat'; % Path to localization results
rndrSMLMVideoPath = '..\datasets\ROI#5_1_149_107_255_1_5000_bleaching_corrected.tif'; % Original SMLM video data path for comparison, leave empty if not adding
SMLMVideoSamplNum = 57; % Sampling interval of the original SMLM video (to reduce render volume), 1 means no sampling
SRRatio = 4; % Resolution multiplier for super-resolution rendering
renderMethod = 'bilinear'; % Rendering method, options: 'nearest', 'bilinear', 'Gauss'
rndrColor = 'hot'; % Render color, options: 'gray' or 'hot'
SMLMFrameRate = 781; % Frames per second of the SMLM video
recFrameNum = 60; % Number of SMLM frames used to reconstruct one super-resolution video frame
normMode = 'percent'; % Normalization mode, options: 'percent' or 'absolute'
normRange = [0 99.8]; % Normalization range percentage or absolute values based on normMode
cropPct = 0.9; % Edge cropping percentage
SRVideoSavePath = '..\models\'; % Path to save the super-resolution video
fileFormat = '.avi'; % Output file format, only supports '.avi' and '.mp4'
quality = 100; % Output video quality, between 0-100

SRVideoRender(rndrPredLocPath, rndrSMLMVideoPath, SMLMVideoSamplNum, ...
    SRRatio, renderMethod, rndrColor, SMLMFrameRate, recFrameNum, ...
    normMode, normRange, cropPct, SRVideoSavePath, fileFormat, quality);