function createParPool(parNum, maxRetries)
% createParPool creates a new parallel pool.
%
%  This function creates a parallel pool, allowing tasks to be executed in a parallel computing environment.
%  It creates a parallel pool with the specified number of workers, with multiple attempts allowed until the pool is successfully created.
%  If a parallel pool already exists and has the required number of workers, the function will not create a new pool.
%
%  Input Parameters:
%    parNum - Desired number of parallel workers.
%    maxRetries - Maximum number of attempts to create the parallel pool.

currentPool = gcp('nocreate');
desiredParNum = min(parNum, parcluster('Processes').NumWorkers);
if isempty(currentPool) || currentPool.NumWorkers ~= desiredParNum
    numRetries = 0;
    poolCreated = false;
    while ~poolCreated && numRetries < maxRetries
        try
            % Attempt to create parallel pool:
            delete(gcp('nocreate'));
            parpool('Processes', desiredParNum);
            poolCreated = true;  % If successful, set flag to exit loop
        catch ME
            disp(ME.message);
            % If creation fails, wait and retry:
            disp(['Attempt ', num2str(numRetries + 1), ...
                ' to create parallel pool failed. ' ...
                'Retrying in 10 seconds...']);
            pause(10);
            numRetries = numRetries + 1;  % Increment retry count
        end
    end
end
end