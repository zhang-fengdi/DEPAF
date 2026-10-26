function saverFilePath = progressDisp(flag, times, saverFilePath)
% progressDisp displays progress in the form of a progress bar.
%
%  Input Parameters:
%    flag - Indicates the type of operation. 0 updates progress, -1 terminates display,
%           -2 silently cleans up, and any positive integer initializes the progress bar.
%    times - Optional parameter indicating the number of progress increments, default is 1.

%    saverFilePath - Progress file returned on initialization; pass it to workers.
%                    If omitted, serial calls use the most recently initialized bar.

narginchk(1,3);

% Default increment per progress update is 1:
if ~exist('times','var')
    times = 1;
end

% Set progress bar width and path to save progress information (automatically in temp file path):
barWidth = 50;
persistent defaultFilePath
if nargin < 3 || isempty(saverFilePath)
    if flag > 0
        defaultFilePath = [tempname '.tmp'];
    end
    saverFilePath = defaultFilePath;
end
if flag == -2
    if ~isempty(saverFilePath) && isfile(saverFilePath)
        delete(saverFilePath);
    end
    return;
end
if isempty(saverFilePath) || (flag <= 0 && ~isfile(saverFilePath))
    error('DEPAF:ProgressNotInitialized', 'Initialize the progress bar and pass its file path to workers.');
end
if flag > 0 && (isfile(saverFilePath) || isfolder(saverFilePath))
    error('DEPAF:ProgressFileExists', 'Progress file already exists: %s', saverFilePath);
end

switch flag
    case 0 % Increase progress
        % Write '1' for each 'times' to file, indicating progress increments:
        f = fopen(saverFilePath, 'a');
        if f < 0
            error('DEPAF:ProgressFileAccess', 'Cannot access progress file: %s', saverFilePath);
        end
        fprintf(f, repmat('1\n', 1, times));
        fclose(f);

        % Read progress data:
        f = fopen(saverFilePath, 'r');
        if f < 0
            error('DEPAF:ProgressFileAccess', 'Cannot access progress file: %s', saverFilePath);
        end
        progress = fscanf(f, '%d');
        fclose(f);

        % Display progress:
        processedNum = length(progress) - 1;
        totalNum = progress(1);
        showProgress(1, processedNum, totalNum, barWidth);

    case -1 % Terminate display
        % Read progress data:
        f = fopen(saverFilePath, 'r');
        if f < 0
            error('DEPAF:ProgressFileAccess', 'Cannot access progress file: %s', saverFilePath);
        end
        progress = fscanf(f, '%d');
        fclose(f);

        % Delete progress save file:
        delete(saverFilePath);

        % Display progress:
        totalNum = progress(1);
        showProgress(1, totalNum, totalNum, barWidth);

    otherwise % Initialize
        % Write total count in the first line of the file:
        f = fopen(saverFilePath, 'w');
        if f < 0
            error('DEPAF:ProgressFileAccess', 'Cannot access progress file: %s', saverFilePath);
        end
        fprintf(f, '%d\n', flag);
        fclose(f);

        % Display progress:
        showProgress(0, 0, flag, barWidth);
end
end


% Helper function: Display progress.
function showProgress(eraseFlag, processedNum, totalNum, barWidth)
% Format display for processed count and total count:
totalNumLen = length(num2str(totalNum));
numDisp = sprintf( ...
    ['%' num2str(totalNumLen) 'd/%' num2str(totalNumLen) 'd'], ...
    processedNum, totalNum);

% Format display for progress percentage:
percent = processedNum/totalNum*100;
percDisp = sprintf('%3.0f%%', percent);

% Construct and display progress bar:
if processedNum == totalNum
    lastSymbol = '=';
else
    lastSymbol = '>';
end
fprintf([repmat(char(8),1,eraseFlag*(barWidth+2*totalNumLen+10)) ...
    '%s %s[%s%s%s]\n'], ...
    numDisp, ...
    percDisp, ...
    repmat('=',1,round(percent/100*barWidth)), ...
    lastSymbol, ...
    repmat(' ',1,barWidth-round(percent/100*barWidth)));
end