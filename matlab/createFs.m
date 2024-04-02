function [f_initial, f_top, f_bottom, f_initial_list, f_top_list, f_bottom_list] = createFs(model)

	H = model.H;
	Hsize = size(H, 2);

	% Construct the f index vector (f_initial) for the first
	[f_initial, f_initial_list] = createFBranch(model.Hpos, model.intervals.initialPhaseMapping, Hsize);

	% Construct the f index vector (f_top) for the first
	[f_top, f_top_list] = createFBranch(model.Hpos, model.intervals.topPhaseMapping, Hsize);

	% Construct the f index vector (f_b) for the second
	[f_bottom, f_bottom_list] = createFBranch(model.Hpos, model.intervals.bottomPhaseMapping, Hsize);
end

function [f_partial, f_partial_list] = createFBranch(HsubintervalStartEnds, phaseMapping, last)

	f_partial = [];
	f_partial_list = {};
	for phaseMappingIdx = 1:length(phaseMapping)

		% Get the phase name and column set in H's set of subintervals
		phaseName = phaseMapping{phaseMappingIdx}{1};
		HsubintervalIndex = phaseMapping{phaseMappingIdx}{2};

		% Generate the indices for the subinterval
		subintervalStartEnd = HsubintervalStartEnds{HsubintervalIndex};
		indices = subintervalStartEnd(1):subintervalStartEnd(2);

		% Add to the partial list of f
		f_partial = [f_partial indices];
		f_partial_list{phaseMappingIdx} = indices;
	end
end
