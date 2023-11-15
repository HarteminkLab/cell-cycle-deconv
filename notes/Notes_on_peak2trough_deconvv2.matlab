
	% Documentation of the peaktotrough.m function in deconv.v2 
	% in analysis/union/temp
	%
	% I believe this to be the latest version of the code
	% but don't have any signs of the function being called or used
	% in the the deconv.v2 code base.
	%

	% Let's get the f vector for the current ORF
	% and call it curf (current f)
	curf = allF(orfid,:);

	% Then the raw data as g1, and g2 (the g for replicate 1
	% and for replicate 2)
	g1 = dataset1(orfid, :)';
	g2 = dataset2(orfid, :)';

	% g1_range and g2_range appear to be
	%
	% the first cell cycle's range in the
	% raw data's timepoints 
	%
	% mu0 --------- mu0+lambda      
	%
	g1_range = find(WT1_TP>=mu0_1 & 
		WT1_TP<=mu0_1+lambda_1);
	g2_range = find(WT2_TP>=mu0_2 & 
		WT2_TP<=mu0_2+lambda_2);

	% To compute peak to trough on the raw data
	% we only want to examine the timepoints
	% associated with the first cell cycle
	g1 = g1(g1_range);
	g2 = g2(g2_range);

	% The original peak to trough ratio is computed as
	% the max over the min for g1 and g2 (and then taken
	% as the mean, divided by 2. Interestingly it is not
	% 80/20 like in the deconvolved PTR.
	%
	% ******* 
	%         We can check compute this calculation ourselves 
	%         then compare with published to check if this
	%         version of peak to trough matches the paper's version
	% *******
	%
	orig_p2t = (max(g1)/min(g1) + max(g2)/min(g2))/2;

	% Append to running list of original peak to trough values
	% for all genes
	orig_array = [orig_array orig_p2t];

	% Get the top and bottom branch subsets of f
	% We'll call them current C (mother) and current D (daughter)
	curC = curf(t_y_idx);
	curD = curf(b_y_idx);

	% After looking over the rescale function, it seems
	% to be interpolating the values of x (timepoints?)
	% to be a vector of the same length as the curC/curD
	% I think. Maybe we can rename it to matchVecLengthInterpolate
	% or something like that.
	%
	curC_r = rescale(t_x, curC);
	curD_r = rescale(b_x, curD);
	
	% Compute various quantiles of the rescaled C and D
	% vectors, note that we will only use 20 and 80 for
	% the final result, other values are likely for unpublished 
	% analyses.
	ptrC = quantile(curC_r, [5 10 20 80 90 95]./100);
	ptrD = quantile(curD_r, [5 10 20 80 90 95]./100);

	% Compute the 95/5 peak to trough ratio
	Cr5 = ptrC(6)/ptrC(1);
	Dr5 = ptrD(6)/ptrD(1);

	% Combine the mother and daughter
	% PTR scales as defined in the paper as
	% [ ptr_C^2 * ptr_D^1] ^ (1/3)
	%
	% So, weight should be equal to 2/3
	% (referring to ptr_C's exponent)
	%
	m5 = ptrScore(Cr5, Dr5, Weight);

	% Same thing for 90/10
	Cr10 = ptrC(5)/ptrC(2);
	Cr10_array = [Cr10_array Cr10];
	Dr10 = ptrD(5)/ptrD(2);
	Dr10_array = [Dr10_array Dr10];
	m10 = ptrScore(Cr10, Dr10, Weight);

	% Same thing for 90/20
	Cr20 = ptrC(4)/ptrC(3);
	Dr20 = ptrD(4)/ptrD(3);
	m20 = ptrScore(Cr20, Dr20, Weight);


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

