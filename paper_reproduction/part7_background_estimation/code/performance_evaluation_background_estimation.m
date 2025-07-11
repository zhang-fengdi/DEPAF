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

        predResDir = dir(['..\models\BG_data_' num2str(int) '_' num2str(density) '_PerlinBG_' num2str(scale) '*.mat' ]);
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
