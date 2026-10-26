function assertNewOutputFiles(paths)
% assertNewOutputFiles refuses to overwrite any requested output file or folder.
for i = 1:numel(paths)
    if isfile(paths{i}) || isfolder(paths{i})
        error('DEPAF:OutputExists', 'Output already exists; choose a new output directory: %s', paths{i});
    end
end
end
