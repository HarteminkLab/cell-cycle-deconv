function [model] = findOptimal4AF(model, fig_flag)

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
global SLIENCE;

global log_space;
global local_slience;

if ~SLIENCE
	local_slience = 0;
else
	local_slience = 1;
end

ELBOW_BINS = 10;

log_space = 0;

optarg = 2;
if nargin < optarg
	error('Needs (model, fig_flag) in findOptimal4AF function');
end

% gamma boundary
GM_MIN = 0.007;
%GM_MIN = 0.01;
GM_MAX = 0.07;

if log_space
	gm_min = log10(GM_MIN);
	gm_max = log10(GM_MAX);
else
	gm_min = GM_MIN;
	gm_max = GM_MAX;
end

% some settings
DEFAULT_CUTOFF = 0.3;
DEFAULT_GM = 0.04;
skip = 0;

% find best fit
model.gm = 0;
model = deconvModel4AF(model);
best_err = model.err;
model.err0 = best_err;

if best_err > DEFAULT_CUTOFF;
	model.gm = DEFAULT_GM;
	skip = 1; % no need for binary search
end

if ~local_slience
	disp(sprintf('...Best err_fit = %0.5g', best_err));
end

if ~skip
	if model.rn > 1e-5
		gm_left = gm_min;
		gm_right = gm_max;

		if ~local_slience
			if log_space
				disp(sprintf('...Initial search region [%0.5f, %0.5f]', 10^gm_left, 10^gm_right));
			else
				disp(sprintf('...Initial search region [%0.5f, %0.5f]', gm_left, gm_right));
			end
		end

		% find left boundary
		err_rate = 1.05;
		min_err = 0.016;

		err_left = min(err_rate*best_err, best_err+min_err);
		leftr = (err_left/best_err-1)*100;
		if ~local_slience
			disp(sprintf('\n... Search err_left, rn_goal = %0.5f, rate = %0.5f%%', err_left, leftr));
		end
		[model, runs] = binarysearch(model, gm_left, gm_right, err_left);
		if log_space
			gm_left = log10(model.gm);
		else
			gm_left = model.gm;
		end

		% find right boundary
		err_rate = 1.30;
		max_err = 0.096;

		err_right = max(err_rate*best_err, best_err+max_err);
		rightr = (err_right/best_err-1)*100;
		if ~local_slience
			disp(sprintf('\n...Search err_right, err = %0.5f, rate = %0.1f', err_right, rightr));
		end
		[model, runs] = binarysearch(model, gm_left, gm_right, err_right);
		if log_space
			gm_right = log10(model.gm);
		else
			gm_right = model.gm;
		end

		if ~local_slience
			disp(sprintf('\n...error rate range: [%0.5f, %0.5f]', err_left, err_right));
			if log_space
				disp(sprintf('...Search gm in [%0.5f %0.5f]', 10^gm_left, 10^gm_right));
			else
				disp(sprintf('...Search gm in [%0.5f %0.5f]', gm_left, gm_right));
			end
		end

		if (abs(gm_right-gm_left) < 4e-5) % gm_right == gm_left
			if log_space
				model.gm = 10^gm_right;
			else
				model.gm = gm_right;
			end
		else
			diff = (gm_right-gm_left)/ELBOW_BINS;
			gamma_array = gm_left:diff:gm_right;
			if log_space
				[model.elbow] = findElbow4AF(model, 10.^gamma_array, fig_flag);
			else
				[model.elbow] = findElbow4AF(model, gamma_array, fig_flag);
			end
			model.gm = model.elbow;
	%		model.gm = (gm_right+gm_left)/2;
		end
	else % if rn is too tiny, use gamma_min 
		if log_space
			model.gm = 10^gm_min;
		else
			model.gm = gm_min;
		end
	end
end

%if ~local_slience
	disp(sprintf('\n...gamma = %0.5f\n', model.gm));
%end
[model] = deconvModel4AF(model);

if fig_flag
	[model] = plotOptimal_general_branch4AF(model, fig_flag);
end

return;

% ===============================
% ===============================
% ===============================

function [model, runs] = binarysearch(model, gamma_min, gamma_max, err)

%fit = (err)^2*length(model.g);
global local_slience;
global log_space;

FIT_SMALL = 5e-4;
LR_SMALL = 5e-4;

left = gamma_min;
right = gamma_max;
runs = 0;
% find the fit_left point
while right-left > LR_SMALL
	cur_gamma = (left+right)/2;
	runs = runs+1;
	
	if log_space
		model.gm = 10^cur_gamma;
	else
		model.gm = cur_gamma;
	end
	
	[model] = deconvModel4AF(model);
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
