function [m20, Cr20, Dr20] = peak2trough(model);

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
	
	% Compute various quantiles of the rescaled C and D
	% vectors, note that we will only use 20 and 80 for
	% the final result, other values are likely for unpublished 
	% analyses.
	ptrC = quantile(curC_r, [5 10 20 80 90 95]./100);
	ptrD = quantile(curD_r, [5 10 20 80 90 95]./100);

	% Combine the mother and daughter
	% PTR scales as defined in the paper as
	% [ ptr_C^2 * ptr_D^1] ^ (1/3)
	%
	% So, weight should be equal to 2/3
	% (referring to ptr_C's exponent)
	%
	Weight = 2./3.;
	Cr20 = ptrC(4)/ptrC(3);
	Dr20 = ptrD(4)/ptrD(3);
	m20 = ptrScore(Cr20, Dr20, Weight);

return

% This function appears to combine C and D using the Weight 
% (ratio of C to D) to get the final combined PTR score. 
%
% This appears a little different than how the paper describes the
% combining.. What is weight's value? 
%
%    [  (ptrC ^ 2) (ptrD) ^ 1  ] ^ (1/3) =
%    [  (ptrC ^ 2/3) (ptrD) ^ 1/3  ]
% 
%    So Weight = 2/3
%
function score = ptrScore(C, D, Weight)
	score = power(C, Weight)*power(D, 1-Weight);
return;

