function perfEval2DSMLM(dataPath, ...
    locPath, tarLocPath, matchThresh, pixNMSize, ...
    resSavePath, useParallel, parNum)
% perfEval Evaluates localization performance metrics.
%
%  This function evaluates the error between the predicted localization results and the actual labels, calculating various metrics,
%  such as average localization error, Jaccard index, and efficiency.
%
%  Input parameters:
%    dataPath - Path to the original image data.
%    locPath - Path to the predicted localization results.
%    tarLocPath - Path to the actual labels (target locations).
%    matchThresh - Matching threshold for determining if predicted positions are correctly matched.
%    pixNMSize - Nanometer size corresponding to each pixel.
%    resSavePath - Path to save evaluation results.
%    useParallel - Whether to use parallel computation.
%    parNum - Number of threads for parallel computation.
%
%  Output:
%    The evaluation results will be saved to the specified path.

% Load predicted results and actual labels:
disp('[Evaluation] Loading predicted results and actual labels...');
[predLoc, ~, ~, ~, modelID, predDataName] = loadData(locPath);
tarLoc = loadData(tarLocPath);

% Initialize parameters:
numI = length(predLoc);
locErr = zeros(numI, 1, 'single');
JacIdx = zeros(numI, 1, 'single');

% Create parallel pool:
if useParallel
    disp('[Evaluation] Creating parallel pool...');
    createParPool(parNum, 20);
end

% Begin evaluation:
disp('[Evaluation] Starting evaluation...')
progressDisp(numI); % Initialize progress bar
if useParallel
    parfor k = 1:numI
        % Sort by intensity to prioritize matching higher intensity points:
        IRaw = loadData(dataPath,'all','all',k);
        amplitude = interp2(IRaw,tarLoc{k}(:,1),tarLoc{k}(:,2),'spline');
        [~,idx] = sort(amplitude,'descend');
        tarLoc{k} = tarLoc{k}(idx,:);
        amplitude = interp2(IRaw,predLoc{k}(:,1),predLoc{k}(:,2),'spline');
        [~,idx] = sort(amplitude,'descend');
        predLoc{k} = predLoc{k}(idx,1:2);

        % Calculate evaluation results for each image:
        tarNum = size(tarLoc{k},1);
        predNum = size(predLoc{k},1);
        RMSECount = 0;
        matchNum = 0;
        for i = 1:predNum
            RMSE = sqrt(sum((predLoc{k}(i,1:2) - tarLoc{k}).^2,2));
            [minRMSE,idx] = min(RMSE);
            if minRMSE <= matchThresh
                RMSECount = RMSECount + minRMSE;
                matchNum = matchNum + 1;
                tarLoc{k}(idx,:) = inf;
            end
        end
        locErr(k) = RMSECount / matchNum;
        TP = matchNum;
        FP = predNum - matchNum;
        FN = tarNum - matchNum;
        JacIdx(k) = TP / (TP+FN+FP);

        % Refresh progress bar:
        progressDisp(0);
    end
else
    for k = 1:numI
        % Sort by intensity to prioritize matching higher intensity points:
        IRaw = loadData(dataPath,'all','all',k);
        amplitude = interp2(IRaw,tarLoc{k}(:,1),tarLoc{k}(:,2),'spline');
        [~,idx] = sort(amplitude,'descend');
        tarLoc{k} = tarLoc{k}(idx,:);
        amplitude = interp2(IRaw,predLoc{k}(:,1),predLoc{k}(:,2),'spline');
        [~,idx] = sort(amplitude,'descend');
        predLoc{k} = predLoc{k}(idx,1:2);

        % Calculate evaluation results for each image:
        tarNum = size(tarLoc{k},1);
        predNum = size(predLoc{k},1);
        RMSECount = 0;
        matchNum = 0;
        for i = 1:predNum
            RMSE = sqrt(sum((predLoc{k}(i,:) - tarLoc{k}).^2,2));
            [minRMSE,idx] = min(RMSE);
            if minRMSE <= matchThresh
                RMSECount = RMSECount + minRMSE;
                matchNum = matchNum + 1;
                tarLoc{k}(idx,:) = inf;
            end
        end
        locErr(k) = RMSECount / matchNum;
        TP = matchNum;
        FP = predNum - matchNum;
        FN = tarNum - matchNum;
        JacIdx(k) = TP / (TP+FN+FP);

        % Refresh progress bar:
        progressDisp(0);
    end
end

% Terminate progress bar:
progressDisp(-1);

% Calculate average evaluation results:
disp('[Evaluation] Calculating average evaluation results...');
locErr = mean(locErr);
JacIdx = mean(JacIdx);
eff = 100 - sqrt((100-JacIdx*100)^2 + (locErr*pixNMSize)^2);

% Save evaluation results:
disp('[Evaluation] Saving evaluation results...');
checkPath(resSavePath);
[~, tarLocDataName] = fileparts(tarLocPath); % Retrieve data file name
savePath = [resSavePath '\EvalRes_' predDataName(1:end-4) ...
    '_by_Mdl_' modelID '.mat'];
save(savePath, 'locErr', 'JacIdx', 'eff', ...
    'modelID', 'predDataName', 'tarLocDataName');
end