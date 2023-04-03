function [f_initial, f_top, f_bottom, f_initial_list, f_top_list, f_bottom_list, padding] = createFs(model)

	H = model.H;
	Hsize = size(H, 2);

	% Construct the f index vector (f_initial) for the first, pad if necessary
	[f_initial, f_initial_list, initialPadding] = createFBranch(model.Hpos, model.intervals.initialPhaseMapping, Hsize);

	% Construct the f index vector (f_top) for the first, pad if necessary
	[f_top, f_top_list, topPadding] = createFBranch(model.Hpos, model.intervals.topPhaseMapping, Hsize);

	% Construct the f index vector (f_b) for the second, pad if necessary
	[f_bottom, f_bottom_list, bottomPadding] = createFBranch(model.Hpos, model.intervals.bottomPhaseMapping, Hsize);

	% We only need to pad H by the larger of the two paddings
	padding = max(initialPadding, bottomPadding);
end

function [padding] = getPaddingF(f_partial)
	% If the f_partial vector is not a power of two, return how much padding is needed to pad the vector
	fsize = size(f_partial, 2);
	logfsize = log2(fsize);
	padding = 0;
	if (rem(logfsize, 1) ~= 0)
		closest2Power = ceil(logfsize);
		paddedSize = 2^closest2Power;
		padding = paddedSize-fsize;
	end
end

function [f_partial, f_partial_list, padding] = createFBranch(HsubintervalStartEnds, phaseMapping, last)
	% Construct a part of the f vector list that maps back to the subinterval indices in the H matrix
	% Each subinterval set we are smoothing over will need to reference which columns of the H matrix we want
	% to smooth over, in this case we will be using the set of subintervals within a branch to define
	% the columns we want to smooth

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

	% If padding is necessary, add more items to the vector to reach
	% a power of two size, padding is indexed by the last index in the unmodified H matrix
	padding = getPaddingF(f_partial);
	if (padding > 0)
		f_padding = last+[1:padding];
		f_partial = [f_partial f_padding];
	end
end