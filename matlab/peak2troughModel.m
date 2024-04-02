function [m20, Cr20, Dr20] = peak2troughModel(model)

	% Function to compute the deconvolved peak to trough ratio
	% for a given model which has deconvolved one gene.
	curf = model.f;

	% The indicies of the subsets of f corresponding to
	% the top branch (mother) and bottom branch (daughter)
	t_y_idx = model.f_top;
	b_y_idx = model.f_bottom;

	% The time intervals corresponding to mother and daughter
	t_x = model.intervals.topTimepoints;
	b_x = model.intervals.bottomTimepoints;

	% Get the top and bottom branch subsets of f
	% We'll call them current C (mother) and current D (daughter)
	curC = curf(t_y_idx);
	curD = curf(b_y_idx);

	% After looking over the rescale function, it seems
	% to be interpolating the values of x (timepoints?)
	% to be a vector of the same length as the curC/curD
	% I think. Maybe we can rename it to matchVecLengthInterpolate
	% or something like that.
	curC_r = rescale(t_x, curC);
	curD_r = rescale(b_x, curD);
	
	[m20, Cr20, Dr20] = peak2trough(curC_r, curD_r);

return
