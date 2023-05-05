function [model] = saveModelPhaseIntervalTimepoints(model, filename)

    initialTable = getTablePhaseIntervalTimepoints("initial", model.intervals.initialTimepointsList, ...
        model.f_initial_list, model.intervals.initialPhaseMapping);

    topTable = getTablePhaseIntervalTimepoints("top", model.intervals.topTimepointsList, ...
        model.f_top_list, model.intervals.topPhaseMapping);

    bottomTable = getTablePhaseIntervalTimepoints("bottom", model.intervals.bottomTimepointsList, ...
        model.f_bottom_list, model.intervals.bottomPhaseMapping);

    writetable([initialTable; topTable; bottomTable], filename, 'Delimiter', ',');


function [saveTable] = getTablePhaseIntervalTimepoints(branch, timepointsList, intervalsList, phaseMapping)

    variableNames = cellstr({"branch", "phase", "index", "timepoint"});
    variableTypes = {'string', 'string', 'double', 'double'};
    saveTable = table('Size', [0, length(variableNames)], 'VariableNames', variableNames, 'VariableTypes', variableTypes);

    for index = 1:size(phaseMapping, 2)
        
        % Get the index of the intervals for the current phase
        % and convert it to a cell array
        intervals = intervalsList{index};
        intervals = num2cell(intervals);
       
        % Get the interval time values
        timepoints = num2cell(timepointsList{index}(1:end-1));

        % The number of rows we are going to add
        numCur = length(intervals);
       
        % Get the phase name and branch name and repeat it by the number of rows
        cellCyclePhaseName = phaseMapping{index}{1};
        repPhaseNames = cellstr(repmat(cellCyclePhaseName, numCur, 1))';
        repBranchNames = cellstr(repmat(branch, numCur, 1))';

        % Append to the cell array
        saveTable(end+1:end+numCur, :) = [repBranchNames; repPhaseNames; intervals; timepoints]';
    end
