function infoFile = ReadInfoFile(fileName)
% ReadInfoFile reads the INF metadata needed for full-frame DAX loading.
% fileName is an explicit .inf path or the path of its associated .dax file.
% Returned fields describe the local file, dimensions, binning, frame count
% and data type, plus image bounds when present. Other metadata are ignored.
%
% Adapted from ReadInfoFile by Jeffrey Moffitt, September 5, 2012.
% jeffmoffitt@gmail.com

if ~(ischar(fileName) && isrow(fileName)) && ~(isstring(fileName) && isscalar(fileName))
    error('ReadInfoFile:FileName', 'Specify one INF or DAX file path.');
end
fileName = char(fileName);
[folder, name, extension] = fileparts(fileName);
if strcmpi(extension, '.dax')
    fileName = fullfile(folder, [name '.inf']);
elseif ~strcmpi(extension, '.inf')
    error('ReadInfoFile:Extension', 'The file path must end in .inf or .dax.');
end
fid = fopen(fileName, 'r');
if fid < 0
    error('ReadInfoFile:OpenFailed', 'Cannot open INF file: %s.', fileName);
end
fileCleanup = onCleanup(@() fclose(fid));
[folder, name, extension] = fileparts(fileName);
if isempty(folder)
    folder = '.';
end
infoFile.localName = [name extension];
infoFile.localPath = [folder filesep];

while true
    line = fgetl(fid);
    if ~ischar(line)
        break;
    end
    equals = strfind(line, '=');
    if isempty(equals)
        continue;
    end
    equals = equals(1);
    key = regexprep(lower(strtrim(line(1:equals-1))), '[^a-z]', '');
    switch key
        case 'framedimensions'
            fieldName = 'frameDimensions';
        case 'numberofframes'
            fieldName = 'numberOfFrames';
        case 'datatype'
            fieldName = 'dataType';
        case {'binning','hstart','hend','vstart','vend'}
            fieldName = key;
        otherwise
            continue;
    end
    value = strtrim(line(equals+1:end));
    if strcmp(fieldName, 'dataType')
        infoFile.dataType = value;
    else
        fieldValue = str2num(strrep(value, ' x ', ' '), 'Evaluation', 'restricted'); %#ok<ST2NM>
        if isempty(fieldValue)
            error('ReadInfoFile:NumericValue', 'Invalid numeric value for INF field %s.', fieldName);
        end
        infoFile.(fieldName) = fieldValue;
    end
end
required = {'frameDimensions','binning','numberOfFrames','dataType'};
for k = 1:numel(required)
    if ~isfield(infoFile, required{k})
        error('ReadInfoFile:MissingField', 'Missing required INF field: %s.', required{k});
    end
end
if any(infoFile.frameDimensions == 0)
    bounds = {'hstart','hend','vstart','vend'};
    if ~all(isfield(infoFile, bounds))
        error('ReadInfoFile:MissingBounds', 'Zero frame dimensions require hstart, hend, vstart and vend.');
    end
    warning('matlabSTORM:corruptedInfoFile', 'Unexpected frame dimensions');
    infoFile.frameDimensions = [infoFile.hend-infoFile.hstart+1, ...
                                infoFile.vend-infoFile.vstart+1];
end
end
