function [model] = findOptimalBud(model, fig_flag)

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
% model.bList
% model.i_intervals
% model.t_intervals
% model.b_intervals 
% model.H
% model.Hsegments
% model.elbow

global DECONV_JOINT;
global DECONV_WT1;
global DECONV_WT2;

global WAVETYPE;
global WAVEPAR;

ELBOW_BINS = 20;

optarg = 2;
if nargin < optarg
	disp('Needs (model, fig_flag) in findOptimal function');
	return;
end

% gamma boundary
GAMMA_MIN = log10(5e-3);
GAMMA_MAX = log10(0.2);

% find best fit
model.gm = 0;
model = deconvModelBud(model);
best_err = model.err;

disp(sprintf('\nbest err_fit = %0.5g\n', best_err));

if model.rn > 1e-5
	gm_left = GAMMA_MIN;
	gm_right = GAMMA_MAX;

	disp(sprintf('\nInitial search region [%0.5g, %0.5g]\n', 10^gm_left, 10^gm_right));

	% find left boundary
	err_rate = 1.1;
	min_err = 0.01;

	err_left = min(err_rate*best_err, best_err+min_err);
	leftr = err_left/best_err;
	disp(sprintf('\nSearch err_left, goal = %0.5g, rate = %0.5g\n', err_left, leftr));
	[model, runs] = binarysearch(model, gm_left, gm_right, err_left);
	gm_left = log10(model.gm);

	% find right boundary
	err_rate = 1.40;
	max_err = 0.04;

	err_right = max(err_rate*best_err, best_err+max_err);
%	err_right = err_rate*best_err;
	rightr = err_right/best_err;
	disp(sprintf('\nSearch err_right, goal = %0.5g, rate = %0.5g\n', err_right, rightr));
	[model, runs] = binarysearch(model, gm_left, gm_right, err_right);
	gm_right = log10(model.gm);

	disp(sprintf('\nError range: [%0.5g, %0.5g]\n', err_left, err_right));
	disp(sprintf('\nSearch region: gm = [%0.5g %0.5g]\n', 10^gm_left, 10^gm_right));

	if (abs(gm_right-gm_left) < 4e-5) % gm_right == gm_left
		model.gm = 10^gm_right;
	else
		diff = (gm_right-gm_left)/ELBOW_BINS;
		gamma_array = gm_left:diff:gm_right;
		[model.elbow] = findElbowBud(model, 10.^gamma_array, fig_flag);
		model.gm = model.elbow;
	end
else % if rn is too tiny, use gamma_min 
	model.elbow = 10^GAMMA_MIN;
	model.gm = model.elbow;
end

disp(sprintf('\ngamma = %0.5g\n', model.gm));
[model] = deconvModelBud(model);

if fig_flag
	[model] = plotOptimal(model, fig_flag);
end

return;

% ===============================
% ===============================
% ===============================

function [model, runs] = binarysearch(model, gamma_min, gamma_max, err)

fit = (err)^2*length(model.g);

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
	
	[model] = deconvModelBud(model);
	if abs(model.rn-fit) <= FIT_SMALL
		break;
	elseif model.rn > fit
		right = cur_gamma;
	else
		left = cur_gamma;
	end
end
return;
