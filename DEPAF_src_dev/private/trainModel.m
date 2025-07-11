function [bestdlnet, epoch, minValLoss, bestEpoch] = trainModel( ...
    I, valI, POI, BGPOI, ...
    learningRate, miniBatchSize, maxEpochs, encoderDepth, ...
    lambda, valFreq, maxPatience, minLR, verbose, useGPU)
% trainModel trains a model for POI fitting and denoising.
%
%  This function trains a deep learning network for POI fitting and denoising of images.
%  The model is trained iteratively, evaluating performance on a validation set after each epoch
%  and monitoring the loss function.
%
%  Input Parameters:
%    I - Training set image data.
%    valI - Validation set image data.
%    POI - POI sample.
%    BGPOI - Background POI sample.
%    learningRate - Learning rate.
%    miniBatchSize - Size of the mini-batch.
%    maxEpochs - Maximum number of epochs.
%    encoderDepth - Depth of the encoder.
%    lambda - Regularization coefficient.
%    valFreq - Validation frequency.
%    maxPatience - Patience level.
%    minLR - Minimum learning rate.
%    verbose - Whether to display the training process.
%    useGPU - Whether to use GPU for computation.
%
%  Output Parameters:
%    bestdlnet - Best deep learning network after training.
%    epoch - Epoch at which the best network was achieved.
%    minValLoss - Minimum validation loss during training.

% Get number of channels:
channelNum = size(POI, 5);

% If the number of channels is greater than 2, compress POI and gradually decompress during training for convergence stability:
tarCompLevel = 1;
[compPOI, tarActChanLogicIdx] = compressPOI(POI, tarCompLevel);

% Reshape for channel filtering:
tarActChanLogicIdx = reshape(tarActChanLogicIdx,1,1,[]);

% Preprocess training and validation samples:
[widI, lenI, numI] = size(I);
valI = dlarray(valI,'SSCB');
if useGPU
    valI = gpuArray(valI);
end

% Create network:
imageSize = [widI lenI];
if ~isempty(BGPOI)
    firstLayerFilterNum = max(64, channelNum + 1);
    outputNumChannels = channelNum + 1;
else
    firstLayerFilterNum = max(64, channelNum);
    outputNumChannels = channelNum;
end
lgraph = createUNet(imageSize, encoderDepth, firstLayerFilterNum, outputNumChannels);
dlnet = dlnetwork(lgraph);

% Create display window for training process:
if verbose
    screenSize = get(0, 'screensize');
    figHeight = screenSize(4) * 0.85;
    if ~isempty(BGPOI)
        if channelNum == 1
            gridDim = [3 2];
        else
            gridDim = [4 2];
        end
    else
        if channelNum == 1
            gridDim = [2 2];
        else
            gridDim = [3 2];
        end
    end
    figure(1);
    set(gcf,'Position',[0 0 figHeight/gridDim(1)*gridDim(2) figHeight]);
    movegui('north');
    subplot(gridDim(1),gridDim(2),1);
    xlabel("Iteration");
    ylabel("Loss");
    grid on;
    lineTrainLoss = animatedline('Color','r');
    lineValLoss = animatedline('Color','b');
    legend('Train Loss', 'Val Loss', 'Location', 'northeast');
end

% Initialize and set secondary hyperparameters:
trailingAvg = [];
trailingAvgSq = [];
gradDecay = 0.9;
gradDecaySq = 0.99;

% Begin training:
iteration = 0;
for epoch = 1:maxEpochs

    % Shuffle data every epoch:
    I = I(:,:,:,randperm(numI));

    % Batch loop:
    for batch = 1:floor(numI / miniBatchSize)
        iteration = iteration + 1;

        % Get batch:
        IBatch = I(:,:,:,(batch-1)*miniBatchSize+1 : batch*miniBatchSize);
        IBatch = dlarray(IBatch,'SSCB');
        if useGPU
            IBatch = gpuArray(IBatch);
        end

        % Soft-start lambda schedule:
        if iteration < 2500
            lambdaCurr = 0.01 * lambda;
        elseif iteration < 5000
            lambdaCurr = 0.1 * lambda;
        else
            lambdaCurr = lambda;
        end
        if iteration == 1
            disp("[Iter " + iteration + "] Lambda warm-up started: " + lambdaCurr + " (1/100 of target)");
            minValLoss = inf;
        elseif iteration == 2500
            disp("[Iter " + iteration + "] Lambda warm-up: " + lambdaCurr + " (1/10 of target)");
            minValLoss = inf;
        elseif iteration == 5000
            disp("[Iter " + iteration + "] Lambda warm-up complete: " + lambdaCurr);
            minValLoss = inf;
        end

        % Calculate gradients using automatic differentiation and return loss function:
        [gradients,loss] = dlfeval(@modelGradients, dlnet, IBatch, ...
            compPOI, BGPOI, lambdaCurr, tarActChanLogicIdx, verbose);
        loss = double(gather(extractdata(loss)));

        % Update neural network weights:
        [dlnet.Learnables,trailingAvg,trailingAvgSq] = ...
            adamupdate(dlnet.Learnables,gradients, trailingAvg, ...
            trailingAvgSq,iteration,learningRate,gradDecay,gradDecaySq);

        % Calculate validation loss at specified frequency and plot:
        if mod(iteration, valFreq) == 1

            % Use model to predict on validation set:
            [valIFit, valIDenoised, valBG] = ...
                modelPred(dlnet, valI, compPOI, BGPOI, tarActChanLogicIdx, 'predict');

            % Calculate validation loss:
            valLoss = modelLoss(valI - valBG, valIFit, valIDenoised, lambdaCurr);
            valLoss = double(gather(extractdata(valLoss)));
            clear valIFit valIDenoised valBG;

            % Save the best model based on validation loss:
            if valLoss < minValLoss
                bestdlnet = dlnet;
                minValLoss = valLoss;
                bestEpoch = epoch;
                patience = 0;
            end

            % Patience for non-decreasing validation loss:
            patience = patience + 1;
            if patience > maxPatience
                return;
            end

            % Take action when patience reaches half of maxPatience:
            if patience == maxPatience && learningRate > minLR

                % Reset patience:
                patience = 1;

                % Prefer POI decompression:
                if 2^tarCompLevel < channelNum

                    % POI decompression:
                    oriCompLevel = tarCompLevel;
                    tarCompLevel = tarCompLevel + 1;
                    [compPOI, tarActChanLogicIdx, oriActChanLogicIdx, compRelship] ...
                        = compressPOI(POI, tarCompLevel, oriCompLevel);

                    % Initialize weights for newly activated channels in neural network based on decompression relation matrix:
                    finalLayerWeightsIdx = ...
                        (dlnet.Learnables.Layer == "Final-ConvolutionLayer") &...
                        (dlnet.Learnables.Parameter == "Weights");
                    finalLayerWeights = dlnet.Learnables{finalLayerWeightsIdx, 'Value'}{1};
                    finalLayerWeights = reshape(finalLayerWeights,size(finalLayerWeights,3),size(finalLayerWeights,4));
                    dlnet.Learnables{finalLayerWeightsIdx, 'Value'}{1}(:,:,:,tarActChanLogicIdx) = ...
                        finalLayerWeights(:,oriActChanLogicIdx) * compRelship;
                    clear finalLayerWeights;

                    % Initialize biases for newly activated channels in neural network based on decompression relation matrix:
                    finalLayerBiasIdx = ...
                        (dlnet.Learnables.Layer == "Final-ConvolutionLayer") &...
                        (dlnet.Learnables.Parameter == "Bias");
                    finalLayerBias = dlnet.Learnables{finalLayerBiasIdx, 'Value'}{1};
                    finalLayerBias = reshape(finalLayerBias,size(finalLayerBias,2),size(finalLayerBias,3));
                    dlnet.Learnables{finalLayerBiasIdx, 'Value'}{1}(:,:,tarActChanLogicIdx) = ...
                        finalLayerBias(:,oriActChanLogicIdx) * compRelship;
                    clear finalLayerBias;

                    % Reshape for channel filtering:
                    tarActChanLogicIdx = reshape(tarActChanLogicIdx,1,1,[]);
                    fprintf('POI compression level reduced to: %d, Current POI channel activations: %d\n', ...
                        tarCompLevel, min(2^tarCompLevel, channelNum));

                    % Reset validation loss:
                    minValLoss = inf;

                else % If POI is fully decompressed, only apply learning rate decay:
                    learningRate = max(learningRate * 0.5, minLR);
                    disp("[Iter " + iteration + "] Learning rate decayed to: " + learningRate);
                end
            end

            % Plot:
            if verbose
                if iteration > 20
                    addpoints(lineValLoss,iteration,valLoss);
                end
            else
                disp("[Iter " + iteration + "] " + ...
                    "Epoch: " + epoch + ...
                    ", TrainLoss: " + loss + ...
                    ", ValLoss: " + valLoss + ...
                    ", MinValLoss: " + minValLoss + ...
                    ", Patience: " + patience);
            end
        end

        % Calculate training loss and plot:
        if verbose
            if iteration > 20
                addpoints(lineTrainLoss,iteration,loss);
            end
            figure(1),subplot(gridDim(1),gridDim(2),1);
            title("Epoch: " + epoch + ...
                ", TrainLoss: " + loss + ...
                ", ValLoss: " + valLoss, ...
                "minValLoss: " + minValLoss + ...
                ", Patience: " + patience);
            drawnow;
        end
    end
end
end