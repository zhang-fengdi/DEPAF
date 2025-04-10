function displayModelTraining(I, IFit, IDenoised, BG, POI, BGPOI)
% displayModelTraining displays image processing results during model training.
%
%  This function is used to visualize intermediate results in image processing or model training,
%  providing insights into the original image, background processing, POI processing, etc.
%  It supports displaying a single or multiple POIs and adjusts the display strategy
%  based on the presence of background information.
%
%  Input Parameters:
%    I - Original image.
%    IFit - Fitting map.
%    IDenoised - Denoised image data.
%    BG - Background image data.
%    POI - POI sample.
%    BGPOI - Background POI sample.
%
%  Output Parameters:
%    No return value; results are displayed directly in a graphical interface.

if ~isempty(BGPOI) % When background learning is enabled
    if size(POI,5) == 1 % Single POI case
        gridDim = [3 2];
        iDisplayImageOrPlot(BG, gridDim, 3, ...
            '$I_{\rm BG}$', isvector(BG));
        iDisplayImageOrPlot(I - BG, gridDim, 4, ...
            '$I_{\rm ori} - I_{\rm BG}$', isvector(I - BG));
        iDisplayImageOrPlot(IFit, gridDim, 5, ...
            '$I_{\rm fit}$', isvector(IFit));
        iDisplayImageOrPlot(IDenoised, gridDim, 6, ...
            '$I_{\rm fit} \ast \bar{\mathcal{P}}$', isvector(IDenoised));
    else % Multiple POI case
        gridDim = [4 2];
        iDisplayImageOrPlot(BG, gridDim, 3, ...
            '$I_{\rm BG}$', isvector(BG));
        iDisplayImageOrPlot(I - BG, gridDim, 4, ...
            '$I_{\rm ori} - I_{\rm BG}$', isvector(I - BG));
        if isvector(I)
            isVector = true;
            iDisplayImageOrPlot(sum(IFit,3), gridDim, 5, ...
                '$I_{\rm fit}$', isVector);
            iDisplayImageOrPlot(sum(IDenoised,3), gridDim, 6, ...
                '$I_{\rm fit} \ast \bar{\mathcal{P}}$', isVector);
            iDisplayImageOrPlot(squeeze(IFit), gridDim, 7, ...
                '$I_{\rm fit}$', isVector);
            iDisplayImageOrPlot(squeeze(IDenoised), gridDim, 8, ...
                '$I_{\rm fit} \ast \bar{\mathcal{P}}$', isVector);
        else
            isVector = false;
            iDisplayImageOrPlot(sum(IFit,3), gridDim, 5, ...
                '$I_{\rm fit}$ (xy view)', isVector);
            iDisplayImageOrPlot(sum(IDenoised,3), gridDim, 6, ...
                '$I_{\rm fit} \ast \bar{\mathcal{P}}$ (xy view)', isVector);
            IFit = imresize(flip(transpose(squeeze(sum(IFit,1)))), ...
                size(IFit,[1 2]),'nearest');
            IDenoised = imresize(flip(transpose(squeeze(sum(IDenoised,1)))), ...
                size(IDenoised,[1 2]),'nearest');
            iDisplayImageOrPlot(IFit, gridDim, 7, ...
                '$I_{\rm fit}$ (xz view)', isVector);
            iDisplayImageOrPlot(IDenoised, gridDim, 8, ...
                '$I_{\rm fit} \ast \bar{\mathcal{P}}$ (xz view)', isVector);
        end
    end
    iDisplayImageOrPlot(I, gridDim, 2, '$I_{\rm ori}$', isvector(I));

else % When background learning is disabled
    if size(POI,5) == 1 % Single POI case
        gridDim = [2 2];
        iDisplayImageOrPlot(IFit, gridDim, 3, ...
            '$I_{\rm fit}$', isvector(IFit));
        iDisplayImageOrPlot(IDenoised, gridDim, 4, ...
            '$I_{\rm fit} \ast \bar{\mathcal{P}}$', isvector(IDenoised));
    else % Multiple POI case
        gridDim = [3 2];
        if isvector(I)
            isVector = true;
            iDisplayImageOrPlot(sum(IFit,3), gridDim, 3, ...
                '$I_{\rm fit}$', isVector);
            iDisplayImageOrPlot(sum(IDenoised,3), gridDim, 4, ...
                '$I_{\rm fit} \ast \bar{\mathcal{P}}$', isVector);
            iDisplayImageOrPlot(squeeze(IFit), gridDim, 5, ...
                '$I_{\rm fit}$', isVector);
            iDisplayImageOrPlot(squeeze(IDenoised), gridDim, 6, ...
                '$I_{\rm fit} \ast \bar{\mathcal{P}}$', isVector);
        else
            isVector = false;
            iDisplayImageOrPlot(sum(IFit,3), gridDim, 3, ...
                '$I_{\rm fit}$ (xy view)', isVector);
            iDisplayImageOrPlot(sum(IDenoised,3), gridDim, 4, ...
                '$I_{\rm fit} \ast \bar{\mathcal{P}}$ (xy view)', isVector);
            IFit = imresize(flip(transpose(squeeze(sum(IFit,1)))), ...
                size(IFit,[1 2]),'nearest');
            IDenoised = imresize(flip(transpose(squeeze(sum(IDenoised,1)))), ...
                size(IDenoised,[1 2]),'nearest');
            iDisplayImageOrPlot(IFit, gridDim, 5, ...
                '$I_{\rm fit}$ (xz view)', isVector);
            iDisplayImageOrPlot(IDenoised, gridDim, 6, ...
                '$I_{\rm fit} \ast \bar{\mathcal{P}}$ (xz view)', isVector);
        end
    end
    iDisplayImageOrPlot(I, gridDim, 2, '$I_{\rm ori}$', isvector(I));
end
end


% Helper function: Display results based on data dimensions.
function iDisplayImageOrPlot(data, gridDim, subplotPosition, titleText, isVector)
subplot(gridDim(1), gridDim(2), subplotPosition);
if isVector
    plot(data);
    xlim([1 length(data)]);
else
    imshow(data, []);
end
title(titleText, 'interpreter', 'latex', 'FontSize', 15);
end