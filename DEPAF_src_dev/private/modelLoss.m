function loss = modelLoss(I, IFit, IDenoised, lambda)
% modelLoss calculates the model training loss function.
%
%  This function calculates the loss function during the training process. The loss function consists of two parts:
%  one part is the L2 loss between the IDenoised image and the original image I;
%  the other part is the sum of the absolute values of the IFit image (L1 loss) multiplied by the regularization parameter lambda.
%
%  Input Parameters:
%    I - Original image data.
%    IFit - Fitting map.
%    IDenoised - Denoised image data.
%    lambda - Regularization coefficient.
%
%  Output Parameters:
%    loss - The calculated total loss value.

loss1 = l2loss(sum(IDenoised,3), I, 'NormalizationFactor', 'batch-size');
loss2 = sum(IFit,'all') / size(IFit,4);
loss = loss1 + lambda * loss2;
end