function [movie, infoFile] = ReadDax(fileName, varargin)
% ReadDax reads complete frames from a 16-bit DAX movie.
% fileName is the path to a .dax or its associated .inf file.
% Options: startFrame (1), endFrame (last frame), verbose (true).
% movie is a uint16 row-by-column-by-frame array; infoFile contains the
% metadata used to read it. Spatial subregions and ND2 orientation are not supported.
%
% Adapted from ReadDax by Jeffrey Moffitt, September 7, 2012.
% jeffmoffitt@gmail.com
% Original reader updates: ANB and JRM, 2013-2015.

startFrame = 1;
endFrame = [];
verbose = true;
if mod(numel(varargin), 2) ~= 0
    error('ReadDax:ParameterPairs', 'Optional parameters must be name-value pairs.');
end
for k = 1:2:numel(varargin)
    value = varargin{k+1};
    switch varargin{k}
        case 'startFrame'
            validateattributes(value, {'numeric'}, {'scalar','integer','positive','finite'}, mfilename, 'startFrame');
            startFrame = value;
        case 'endFrame'
            if ~isempty(value)
                validateattributes(value, {'numeric'}, {'scalar','integer','positive','finite'}, mfilename, 'endFrame');
            end
            endFrame = value;
        case 'verbose'
            validateattributes(value, {'logical','numeric'}, {'scalar','binary'}, mfilename, 'verbose');
            verbose = value;
        otherwise
            error('ReadDax:UnknownParameter', 'Unknown parameter: %s.', string(varargin{k}));
    end
end

infoFile = ReadInfoFile(fileName);
framesInDax = infoFile.numberOfFrames;
validateattributes(framesInDax, {'numeric'}, {'scalar','integer','positive','finite'}, mfilename, 'numberOfFrames');
if isempty(endFrame)
    endFrame = framesInDax;
elseif endFrame > framesInDax
    if verbose
        warning('ReadDax:EndFrameClipped', 'endFrame exceeds the available frames; using the last available frame.');
    end
    endFrame = framesInDax;
end
if startFrame > endFrame
    error('ReadDax:FrameRange', 'startFrame must not exceed the last selected frame.');
end
framesToLoad = endFrame - startFrame + 1;
validateattributes(infoFile.frameDimensions, {'numeric'}, {'vector','numel',2,'positive','finite'}, mfilename, 'frameDimensions');
validateattributes(infoFile.binning, {'numeric'}, {'vector','numel',2,'positive','finite'}, mfilename, 'binning');
frameDim = [infoFile.frameDimensions(1)/infoFile.binning(1), ...
            infoFile.frameDimensions(2)/infoFile.binning(2)];
validateattributes(frameDim, {'numeric'}, {'integer','positive','finite'}, mfilename, 'binned frame dimensions');
frameSize = prod(frameDim);

if contains(infoFile.dataType, 'little endian')
    binaryFormat = 'l';
else
    binaryFormat = 'b';
end
daxPath = fullfile(infoFile.localPath, [infoFile.localName(1:end-4) '.dax']);
if verbose
    disp(['Loading ' daxPath]);
end
fid = fopen(daxPath, 'r');
if fid < 0
    error('ReadDax:OpenFailed', 'Cannot open DAX file: %s.', daxPath);
end
fileCleanup = onCleanup(@() fclose(fid));
if fseek(fid, frameSize*(startFrame-1)*2, 'bof') ~= 0
    error('ReadDax:SeekFailed', 'Cannot seek to the selected start frame.');
end
dataSize = frameSize*framesToLoad;
[movie, count] = fread(fid, dataSize, '*uint16', binaryFormat);
if count ~= dataSize
    error('ReadDax:IncompleteData', 'The DAX file contains fewer pixels than required by the selected frames.');
end
if framesToLoad == 1
    movie = reshape(movie, frameDim)';
else
    movie = permute(reshape(movie, [frameDim framesToLoad]), [2 1 3]);
end
if verbose
    disp(['Loaded ' daxPath]);
    fprintf('%d %d x %d frames loaded\n', framesToLoad, frameDim(1), frameDim(2));
end
end
