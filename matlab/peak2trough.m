function [m20, Cr20, Dr20] = peak2trough(common_values, daughter_values)

	% Compute various quantiles of the rescaled C and D
	% vectors, note that we will only use 20 and 80 for
	% the final result, other values are likely for unpublished 
	% analyses.
	quantilesC = quantile(common_values, [5 10 20 80 90 95]./100);
	quantilesD = quantile(daughter_values, [5 10 20 80 90 95]./100);

	% Combine the mother and daughter
	% PTR scales as defined in the paper as
	% [ ptr_C^2 * ptr_D^1] ^ (1/3)
	%
	% So, weight should be equal to 2/3
	% (referring to ptr_C's exponent)
	%
    quantileC80 = quantilesC(4);
    quantileC20 = quantilesC(3);

    quantileD80 = quantilesD(4);
    quantileD20 = quantilesD(3);

    % To prevent division by zero, let's set a minimum threshold as
    % 1.
    quantileD20 = max(quantileD20, 1);
    quantileC20 = max(quantileC20, 1);

	Weight = 2./3.;
	Cr20 = quantileC80/quantileC20;
	Dr20 = quantileD80/quantileD20;
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

