function [gradients, loss] = ...
    modelGradients(dlnet, I, POI, BGPOI, lambda, tarActChanLogicIdx, verbose)
% modelGradients Calculates gradients for updating neural network parameters.
%
%  This function computes the parameter update gradients for a neural network model,
%  suitable for use in deep learning training. It first predicts the input image using the model,
%  then calculates the loss based on the predicted results and the target output,
%  and finally computes the gradient of the loss with respect to the network parameters.
%
%  Input Parameters:
%    dlnet - Deep learning network object.
%    I - Input image data, formatted as a 4D array.
%    POI - POI sample.
%    BGPOI - Background POI sample.
%    lambda - Regularization coefficient.
%    tarActChanLogicIdx - Logical index for the target active channel.
%    verbose - Flag to indicate whether to display intermediate results.
%
%  Output Parameters:
%    gradients - Gradients of the network parameters.
%    loss - Calculated loss value.

% Use the model to predict on the training set:
[IFit, IDenoised, BG] = ...
    modelPred(dlnet, I, POI, BGPOI, tarActChanLogicIdx, 'forward');

% Calculate loss:
loss = modelLoss(I - BG, IFit, IDenoised, lambda);

% Compute gradients:
gradients = dlgradient(loss,dlnet.Learnables);

% Display results:
if verbose
    figure(1);

    % Prepare data:
    I = gather(extractdata(I(:,:,1,1)));
    IFit = gather(extractdata(IFit(:,:,tarActChanLogicIdx,1)));
    IDenoised = gather(extractdata(IDenoised(:,:,tarActChanLogicIdx,1)));
    if ~isempty(BGPOI) % In case of learning the background
        BG = gather(extractdata(BG(:,:,1,1)));
    end

    % Display results:
    displayModelTraining(I, IFit, IDenoised, BG, POI, BGPOI);
end