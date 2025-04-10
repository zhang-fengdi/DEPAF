function progressDisp(flag, times)
% progressDisp displays progress in the form of a progress bar.
%
%  Input Parameters:
%    flag - Indicates the type of operation. 0 updates progress, -1 terminates display,
%           and any positive integer initializes the progress bar.
%    times - Optional parameter indicating the number of progress increments, default is 1.

% Check if the number of input arguments is 1:
narginchk(1,2);

% Default increment per progress update is 1:
if ~exist('times','var')
    times = 1;
end

% Set progress bar width and path to save progress information (automatically in temp file path):
barWidth = 50;
saverFilePath = fullfile(tempdir, 'progressDispSaver.tmp');

switch flag
    case 0 % Increase progress
        % Write '1' for each 'times' to file, indicating progress increments:
        f = fopen(saverFilePath, 'a');
        fprintf(f, repmat('1\n', 1, times));
        fclose(f);

        % Read progress data:
        f = fopen(saverFilePath, 'r');
        progress = fscanf(f, '%d');
        fclose(f);

        % Display progress:
        processedNum = length(progress) - 1;
        totalNum = progress(1);
        showProgress(1, processedNum, totalNum, barWidth);

    case -1 % Terminate display
        % Read progress data:
        f = fopen(saverFilePath, 'r');
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