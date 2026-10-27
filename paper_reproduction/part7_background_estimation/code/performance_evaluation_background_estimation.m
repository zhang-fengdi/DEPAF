clear all;
close all;

% Load DEPAF source code:
addpath(genpath('..\..\DEPAF_src_replication'));

density = 3775;
for int = [1000 7000 40000]
    for scale = [5 6.4 9 13.9 19.3]
        
        % Load ground truth:
        gtDataName = ['data_' num2str(int) '_' num2str(density) '_PerlinBG_' num2str(scale) '.mat'];
        gtPath = ['..\datasets\' gtDataName];
        gtData = load(gtPath);

        modelDataName = ['data_' num2str(int) '_' num2str(density) ...
            '_PerlinBG_' num2str(scale) '_onlyI'];
        mdlDir = dir(['..\models\Mdl_*_by_' modelDataName '.mat']);
        assert(numel(mdlDir) == 1, 'Expected one model for the selected intensity and background scale.');
        modelInfo = load(fullfile(mdlDir(1).folder, mdlDir(1).name), 'modelID');
        predResDir = dir(['..\models\BG_' modelDataName '_by_Mdl_' modelInfo.modelID '.mat']);
        assert(numel(predResDir) == 1, 'The background result for the selected model is missing.');
        for i = 1:length(predResDir)

            % Load prediction results:
            predResName = predResDir(i).name;
            predResPath = fullfile(predResDir(i).folder, predResName);
            predRes = load(predResPath);
            
            % Compute ground-truth and predicted background and foreground images
            gtBG = gtData.PerlinBG;
            gtI = gtData.I - gtBG;
            predBG = predRes.BG;
            predI = gtData.I - predBG;

            % Evaluate prediction results:
            [RMSE, corr, PSNR] = perfEvalBGEst(gtBG, gtI, predBG, predI);

            % Save results:
            saveName = ['..\models\EvalRes_' predResName];
            save(saveName, 'RMSE', 'corr', 'PSNR', ...
                'predResName', 'gtDataName');
        end
    end
end
