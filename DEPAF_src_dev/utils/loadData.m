function varargout = loadData(filePath, varargin)
% loadData loads data files, supporting full or partial loading of .mat and .tif files.
%
%  This function loads data files from a specified path, supporting both .mat and .tif files.
%  It allows for loading the entire content or a specified portion of the data, suitable for
%  flexible loading of large data files.
%
%  Parameters:
%    - filePath (string): Full path to the data file.
%    - varargin (optional): Parameters for partial loading, specified as [row range, column range, z-axis range].
%                           Use 'all' to load the entire range.
%
%  Returns:
%    - varargout (variable count): Loaded data content. The number and type of variables returned depend on
%                                  file type and loading mode.
%
%  Supported file formats and loading modes:
%    - .mat files: Can load all variables in the file or specified portions of a single matrix. For .mat
%                  files with multiple variables, a 'varOrder' variable is required to indicate loading order.
%    - .tif files: Supports full loading or loading of specified row, column, and z-axis ranges of the image.
%
%  Example Usage:
%    % Load an entire .mat file
%    data = loadData('example.mat');
%
%    % Load a specific range from a .tif file
%    imgPart = loadData('example.tif', 100:200, 100:200, 'all');
%
%  Notes:
%    - For .tif files, row, column, and z-axis ranges are optional parameters.
%    - For .mat files with multiple variables, a 'varOrder' variable is required to specify the loading order.
%    - Row and column ranges must be continuous.
%
%  Error Handling:
%    - If the file path does not exist or the specified range is not continuous, the function will return
%      an error message.
%    - Unsupported file extensions will throw an error.

% Check if the file exists:
if ~exist(filePath, 'file')
    error(['File not found: ''' filePath '''.']);
end

% Check if the number of input arguments is 1 or 4:
if nargin ~= 1 && nargin ~= 4
    error('Incorrect number of input parameters.');
end

% Check if the row and column ranges are continuous:
for i = 1:nargin-2
    range = varargin{i};
    diffArr = diff(range);
    if ~strcmp(range, 'all') && ~all(diffArr == 1)
        error('Row and column ranges must be continuous.');
    end
end

% Get file extension:
[~,~,ext] = fileparts(filePath);

% Determine how to load data based on file extension:
switch ext
    case '.mat' % For .mat files
        matObj = matfile(filePath);
        info = whos(matObj);
        if nargin > 1 % Partial loading mode
            % Format validation:
            if length(info) > 1
                error('For partial loading, .mat files can only contain one variable.');
            end
            if length(info(1).size) ~= 2 && length(info(1).size) ~= 3
                error('Matrix size in .mat files must be 2D or 3D.');
            end

            % Get data loading range:
            rowRange = varargin{1};
            colRange = varargin{2};
            zRange = varargin{3};

            % Use 'all' to load the entire range:
            if strcmp(rowRange, 'all')
                rowRange = 1:info(1).size(1);
            end
            if strcmp(colRange, 'all')
                colRange = 1:info(1).size(2);
            end
            if length(info(1).size) == 2
                zRange = 1;
            elseif strcmp(zRange, 'all')
                zRange = 1:info(1).size(3);
            end

            % Return loaded data:
            varargout{1} = matObj.(info(1).name);
            varargout{1} = varargout{1}(rowRange, colRange, zRange);

        else % Load all variables in .mat file
            if length(info) == 1
                varargout{1} = matObj.(info(1).name);
            elseif ~ismember('varOrder', who(matObj))
                error('Multi-variable .mat files must contain a ''varOrder'' variable to specify order.');
            else
                varOrder = matObj.('varOrder');
                for i = 1:length(varOrder)
                    varargout{i} = matObj.(varOrder{i}); %#ok<AGROW>
                end
            end
        end
    case {'.tif','.tiff'} % For .tif files
        info = imfinfo(filePath);
        numImages = numel(info);
        if nargin > 1 % Partial loading mode
            % Get data loading range:
            rowRange = varargin{1};
            colRange = varargin{2};
            zRange = varargin{3};

            % Use 'all' to load the entire range:
            if strcmp(rowRange, 'all')
                rowRange = 1:info(1).Height;
            end
            if strcmp(colRange, 'all')
                colRange = 1:info(1).Width;
            end
            if strcmp(zRange, 'all')
                zRange = 1:numImages;
            end

            % Return loaded data:
            varargout{1} = zeros(length(rowRange), length(colRange), length(zRange), 'single');
            for i = 1:length(zRange)
                varargout{1}(:,:,i) = imread(filePath, zRange(i), ...
                    'PixelRegion', {[rowRange(1) rowRange(end)], ...
                    [colRange(1) colRange(end)]});
            end
        else % Load entire .tif file
            varargout{1} = zeros(info(1).Height, info(1).Width, numImages, 'single');
            for i = 1:numImages
                varargout{1}(:,:,i) = imread(filePath, i);
            end
        end
    otherwise % Throw error for unsupported file extensions
        error('Unsupported file extension.');
end

% Convert to single data type if only one variable is output:
if length(varargout)==1 && isnumeric(varargout{1}) && ...
        ~isa(varargout{1}, 'single')
    varargout{1} = single(varargout{1});
end
end