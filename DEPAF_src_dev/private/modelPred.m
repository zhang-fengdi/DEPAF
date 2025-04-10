function [IFit, IDenoised, BG] = ...
    modelPred(dlnet, I, POI, BGPOI, actChanLogicIdx, inferenceMode)
% modelPred performs predictions using the model.
%
%  This function uses a pre-trained deep learning model to perform predictions on input images.
%  It supports two inference modes: 'forward' mode is used for calculating gradients during training,
%  while 'predict' mode is used for inference without calculating gradients.
%  The function can also output a background image depending on whether background POI (BGPOI) data is provided.
%
%  Input Parameters:
%    dlnet - Pre-trained deep learning network object.
%    I - Input image data.
%    POI - POI sample.
%    BGPOI - Background POI sample.
%    actChanLogicIdx - Logical index for activating channels.
%    inferenceMode - Inference mode, either 'forward' or 'predict'.
%
%  Output Parameters:
%    IFit - Fitting map.
%    IDenoised - Denoised image portion from the prediction result.
%    BG - Background image portion from the prediction result.

% Calculate model output:
switch inferenceMode
    case 'forward' % Calculate gradients
        dlnetOutputs = forward(dlnet, I);
    case 'predict' % No gradient calculation
        dlnetOutputs = predict(dlnet, I);
    otherwise
        error('Only ''forward'' and ''predict'' inference modes are supported.');
end
dlnetOutputs = abs(dlnetOutputs);

% Obtain respective outputs depending on whether background learning is enabled:
if ~isempty(BGPOI)
    IFit = dlnetOutputs(:,:,1:end-1,:);
    BG = dlconv(dlnetOutputs(:,:,end,:), flip(flip(BGPOI,1),2), 0, ...
        'Padding', 'same', 'PaddingValue', 'symmetric-exclude-edge');
else
    IFit = dlnetOutputs;
    BG = zeros(size(I), 'single');
end
clear dlnetOutputs;

% Activate channels if activation index is specified:
if ~isempty(actChanLogicIdx)
    IFit = IFit .* actChanLogicIdx;
end

% Compute denoised image dlXDenoised after convolution with POI:
IDenoised = dlconv(IFit, flip(flip(POI,1),2), 0, 'Padding', 'same', ...
    'PaddingValue', 'symmetric-exclude-edge');
end