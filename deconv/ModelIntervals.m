classdef ModelIntervals
	% A model class to deconvolve gene expression data from CLOCCS cell cycle
	% parameters

	properties (Access = public)

		% TODO: These data structures need to be refactored some how, but it's unclear
		% how they are used. Right now everything is indexed by list position, it may
		% be more clear if we start using key value mappings instead
		% cell cycle phase -> branches -> intervals
		% branches -> cell cycle phases
		%
		% getPhasesForBranch
		% getIntervalsForBranch
		% getPhases
		% getIntervalForBranchPhase
		%

		% A set of lists that describes the make up of each cell phase.
		%
		% e.g.     {CG1  i   1   t   0}
		%
		relations

		% The initial, top, and bottom branches have a set of timepoints that define
		% the higher resolution time series that we want to estimate.
		% These timepoints define the subintervals that exist on these branches
		%
		% 	e.g. initial: {-101.9, -101.05, ...} {-27, 26, ...} {13.53, ...}
		%
		initialTimepointsList, topTimepointsList, bottomTimepointsList

		% These intervals define which of the cell-cycle phases 
		%
		% 	e.g. initial: {R 1} {CG1 2} {postG1 4}
		%
 		initialPhaseMapping, topPhaseMapping, bottomPhaseMapping
	end

	properties (Access = private)
		parameters
	end

	methods
		function intervals = ModelIntervals(modelpath, model)
			[parameters, relations, initialTimepointsList, ...
			 topTimepointsList, bottomTimepointsList, initialPhaseMapping, topPhaseMapping, bottomPhaseMapping] = readModelFormat(modelpath, model);

			intervals.parameters = parameters;
			intervals.relations = relations;
			intervals.initialTimepointsList = initialTimepointsList;
			intervals.topTimepointsList = topTimepointsList;
			intervals.bottomTimepointsList = bottomTimepointsList;
			intervals.initialPhaseMapping = initialPhaseMapping;
			intervals.topPhaseMapping = topPhaseMapping;
			intervals.bottomPhaseMapping = bottomPhaseMapping;
		end

		function parameters = getCellCycleParameters(model)
			% The CLOCCS cell cycle parameters: mu0, lambda, delta, alpha, sigma0, sigmav
			parameters = model.parameters;
		end
	 end
end
