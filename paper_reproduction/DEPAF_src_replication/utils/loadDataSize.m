function [widI,lenI,numI] = loadDataSize(filePath)
% loadDataSize retrieves the dimensions of a data file.
%
%  This function reads the size information of a specified file, supporting .mat and .tif/.tiff file formats.
%  It returns the width, length, and number of images contained in the file.
%
%  Input Parameters:
%    filePath - Path to the data file.
%
%  Output Parameters:
%    widI - Width of the image.
%    lenI - Length of the image.
%    numI - Number of images contained in the file.
%
%  Notes:
%    - For .mat files, it is assumed that the file contains only one variable, which is a 2D or 3D array.
%    - For .tif/.tiff files, the function retrieves the dimensions and number of all image frames.

% Check if the file exists:
if ~exist(filePath, 'file')
    error(['File not found: ''' filePath '''.']);
end

% Retrieve image count:
[~,~,ext] = fileparts(filePath);
switch ext
    case '.mat'
        matObj = matfile(filePath);
        info = whos(matObj);
        if length(info) > 1
            error('For .mat format, the file must contain only one variable.');
        else
            widI = info(1).size(1);
            lenI = info(1).size(2);
            if length(info(1).size) == 2
                numI = 1;
            else
                numI = info(1).size(3);
            end
        end
    case {'.tif','.tiff'}
        t = Tiff(filePath, 'r');
        widI = getTag(t, 'ImageWidth');     % Retrieve width
        lenI = getTag(t, 'ImageLength');    % Retrieve height
        numI = 1;
        while ~lastDirectory(t)
            nextDirectory(t);
            numI = numI + 1;
        end
        close(t);
end
end