function [compPOI, tarActChanLogicIdx, oriActChanLogicIdx, compRelship] = ...
    compressPOI(POI, tarCompLevel, oriCompLevel)
% compressPOI performs channel dimension compression on the POI.
%
%  This function compresses the POI sample along the channel dimension, reducing the number of channels.
%  The operation is achieved by recursively bisecting each channel and selectively activating specific channels
%  at a given compression level.
%
%  Input Parameters:
%    POI - Five-dimensional matrix representing the POI sample to be compressed.
%    tarCompLevel - Target compression level, specifying the target number of channels after POI compression.
%    oriCompLevel - (Optional) Original compression level. If not specified, it defaults to the target compression level.
%
%  Output Parameters:
%    compPOI - Compressed POI sample.
%    tarActChanLogicIdx - Logical index of activated channels at the target compression level.
%    oriActChanLogicIdx - Logical index of activated channels at the original compression level.
%    compRelship - Compression relationship matrix, describing the relationship between original and target compression levels.

% Calculate the number of channels:
channelNum = size(POI, 5);

% Set default value for oriCompLevel:
if ~exist('oriCompLevel', 'var')
    oriCompLevel = tarCompLevel;
end

bisecIdx = ones(channelNum, 1, 'single');
for level = 1:max(oriCompLevel, tarCompLevel)
    % Generate bisection channel indices:
    if 2^level > channelNum
        bisecIdx = (1:channelNum)';
    else
        newBisecIdx = zeros(size(bisecIdx), 'single');
        for idx = unique(bisecIdx)'
            secRange = find(bisecIdx == idx);
            mid = ceil(length(secRange) / 2);
            newBisecIdx(secRange(1) : secRange(1)+mid-1) = 2 * idx - 1;
            newBisecIdx(secRange(1)+mid : secRange(end)) = 2 * idx;
        end
        bisecIdx = newBisecIdx;
    end

    % Generate activation channel index:
    actChanLogicIdx = false(size(bisecIdx));
    for idx = unique(bisecIdx)'
        secRange = find(bisecIdx == idx);
        actChanLogicIdx(ceil((secRange(1)+secRange(end))/2)) = true;
    end

    if level == oriCompLevel
        % Calculate compression relationship matrix between starting compression level and all channels:
        actChanNumIdxList = find(actChanLogicIdx);
        oriCompRelship = zeros(channelNum, length(actChanNumIdxList), 'single');
        for channel = 1:channelNum
            oriCompRelship(channel, bisecIdx(channel)) = 1;
        end

        % Record current compression level's activated channels as original activated channels:
        oriActChanLogicIdx = actChanLogicIdx;
    end

    if level == tarCompLevel
        % Calculate compression relationship matrix between target compression level and all channels:
        actChanNumIdxList = find(actChanLogicIdx);
        tarCompRelship = zeros(channelNum, length(actChanNumIdxList), 'single');
        for channel = 1:channelNum
            tarCompRelship(channel, bisecIdx(channel)) = 1;
        end

        % Record current compression level's activated channels as target activated channels:
        tarActChanLogicIdx = actChanLogicIdx;

        % Record current compression level's bisected activated channels as target bisected activated channels:
        tarBisecIdx = bisecIdx;
    end
end

% Calculate decompression relationship matrix between starting and target compression levels:
compRelship = oriCompRelship' * tarCompRelship;
compRelship = compRelship ./ sum(compRelship, 2);

% Compress POI based on target activation channel index:
tarActChanNumIdxList = find(tarActChanLogicIdx);
compPOI = zeros(size(POI), 'single');
compNum = zeros(1, 1, 1, 1, channelNum, 'single');
for channel = 1:channelNum
    % Accumulate POI of the current channel to the nearest target activated channel:
    tarActChanNumIdx = tarActChanNumIdxList(tarBisecIdx(channel));
    compPOI(:,:,1,1,tarActChanNumIdx) = ...
        compPOI(:,:,1,1,tarActChanNumIdx) + POI(:,:,1,1,channel);
    compNum(:,:,1,1,tarActChanNumIdx) = compNum(:,:,1,1,tarActChanNumIdx) + 1;
end

% Normalize compressed POI:
compPOI = compPOI ./ compNum;
compPOI(isnan(compPOI)) = 0;
end