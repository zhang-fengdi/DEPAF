function checkPath(path)
% checkPath checks if the specified path exists, and if not, creates it.
%
%  This function ensures that the given folder path exists. If the specified path does not exist,
%  the function will create it and display a message confirming that the path has been created.
%
%  Input Parameters:
%    path - A string representing the folder path to be checked or created.

if ~exist(path, 'dir')
    mkdir(path);
    disp(['Path "' path '" has been created.']);
end
end