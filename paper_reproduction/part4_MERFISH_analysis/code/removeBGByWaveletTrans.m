function I = removeBGByWaveletTrans(I, BGEstLevel)
% removeBGByWaveletTrans removes background from an image using dual-tree wavelet transform.
%
%  Input parameters:
%    I - Input image or sequence of images.
%    BGEstLevel - Background estimation level. 0 indicates no background removal.
%
%  Output:
%    I - Image or image sequence with background removed.

% BGEstLevel of 0 means no background removal:
if BGEstLevel == 0
    return;
end

% Calculate dimensions of input image:
[widI, lenI, numI] = size(I);

% Begin processing:
for i = 1:numI
    % Perform dual-tree wavelet transform:
    if isvector(I(:,:,i))
        [A, D] = dualtree(I(:,:,i), 'Level', BGEstLevel);
    else
        [A, D] = dualtree2(I(:,:,i), 'Level', BGEstLevel);
    end

    % Set coefficient matrices to zero:
    for j = 1:BGEstLevel
        D{j} = zeros(size(D{j}), 'single');
    end

    % Obtain background through inverse wavelet transform:
    if isvector(I(:,:,i))
        BG = idualtree(A, D);
    else
        BG = idualtree2(A, D);
    end

    % Remove background:
    I(:,:,i) = I(:,:,i) - BG(1:widI, 1:lenI);
end
end