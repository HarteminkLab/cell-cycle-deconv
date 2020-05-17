function [model] = findOptimal4Mutant(model, fig_flag)

% model structure
% model.orfname
% model.orfid
% model.datatype
% model.alpha
% model.modeltype
% model.g
% model.timepoints
% model.lengths
% model.relations
% model.iList
% model.tList
% model.i_intervals
% model.t_intervals
% model.H
% model.Hsegments
% model.elbow
global DECONV_MU0POS;
global DECONV_LAMBDAPOS;
global DECONV_ALPHAPOS;

local_slience = 0;

ELBOW_BINS = 20;

optarg = 2;
if nargin < optarg
	error('Needs (model, fig_flag) in findOptimal4Mutant function');
end

% gamma boundary
GAMMA_MIN = log10(4e-3);
%GAMMA_MIN = log10(8e-3);
GAMMA_MAX = log10(0.2);

% find best fit
model.gm = 0;
model = deconvModel4Mutant(model);
best_err = model.err;
model.err0 = best_err;

DEFAULT_CUTOFF = 0.3;
DEFAULT_GM = 0.005;
skip = 0;
if best_err > DEFAULT_CUTOFF;
	model.gm = DEFAULT_GM;
	skip = 1; % no need for binary search
end

if ~local_slience
	disp(sprintf('...Best err_fit = %0.5g', best_err));
end

if ~skip
	if model.rn > 1e-5
		gm_left = GAMMA_MIN;
		gm_right = GAMMA_MAX;

		if ~local_slience
			disp(sprintf('...Initial search region [%0.5f, %0.5f]', 10^gm_left, 10^gm_right));
		end

		% find left boundary
		err_rate = 1.06;
		min_err = 0.009;

		err_left = min(err_rate*best_err, best_err+min_err);
		leftr = (err_left/best_err-1)*100;
		if ~local_slience
			disp(sprintf('\n... Search err_left, goal = %0.5f, rate = %0.5f', err_left, leftr));
		end
		[model, runs] = binarysearch(model, gm_left, gm_right, err_left);
		gm_left = log10(model.gm);

		% find right boundary
	%	err_rate = 1.42;
	%	max_err = 0.036;

		err_rate = 1.20;
		max_err = 0.030;

	%	err_right = max(err_rate*best_err, best_err+max_err);
		err_right = min(err_rate*best_err, best_err+max_err);
	%	err_right = err_rate*best_err;
		rightr = (err_right/best_err-1)*100;
		if ~local_slience
			disp(sprintf('\n...Search err_right, err = %0.5f, rate = %0.1f', err_right, rightr));
		end
		[model, runs] = binarysearch(model, gm_left, gm_right, err_right);
		gm_right = log10(model.gm);

		if ~local_slience
			disp(sprintf('\n...error rate range: [%0.5f, %0.5f]', err_left, err_right));
			disp(sprintf('...Search gm in [%0.5f %0.5f]', 10^gm_left, 10^gm_right));
		end

		if (abs(gm_right-gm_left) < 4e-5) % gm_right == gm_left
			model.gm = 10^gm_right;
		else
			diff = (gm_right-gm_left)/ELBOW_BINS;
			gamma_array = gm_left:diff:gm_right;
			[model.elbow] = findElbow4Mutant(model, 10.^gamma_array, fig_flag);
			model.gm = model.elbow;
	%		model.gm = (gm_right+gm_left)/2;
		end
	else % if rn is too tiny, use gamma_min 
		model.elbow = 10^GAMMA_MIN;
		model.gm = model.elbow;
	end
end

%if ~local_slience
	disp(sprintf('\n...gamma = %0.5f\n', model.gm));
%end
[model] = deconvModel4Mutant(model);

if fig_flag
	[model] = plotOptimal_general_parts4Mutant(model, fig_flag);
end

return;

% ===============================
% ===============================
% ===============================

function [model, runs] = binarysearch(model, gamma_min, gamma_max, err)

%fit = (err)^2*length(model.g);
global local_slience;

FIT_SMALL = 5e-4;
LR_SMALL = 5e-4;

left = gamma_min;
right = gamma_max;
runs = 0;
% find the fit_left point
while right-left > LR_SMALL
	cur_gamma = (left+right)/2;
	runs = runs+1;
	
	model.gm = 10^cur_gamma;
	
	[model] = deconvModel4Mutant(model);
	if abs(model.err-err) <= FIT_SMALL
		break;
	elseif model.err > err
		right = cur_gamma;
	else
		left = cur_gamma;
	end

	err_rate = model.err/model.err0-1;
	if ~local_slience
		disp(sprintf('cur_err = %0.3f, err_rate = %0.1f', model.err, err_rate*100));
	end

end
return;
