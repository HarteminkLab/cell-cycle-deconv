function [model, flag, rn, sn, gammas, elbow_gamma] = findOptimal(model, fig_flag)

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
global DECONV_MU0POS;
global DECONV_LAMBDAPOS;
global DECONV_DELTAPOS;
global DECONV_ALPHAPOS;
global FROM_FINDOPTIMAL;

global DEFAULT_RN_CUTOFF;

global SLIENCE;

SLIENCE = 0;

if ~SLIENCE
	disp('findOptimal');
end

ELBOW_BINS = 10;

SMALL = 5e-5;

FROM_FINDOPTIMAL = 1;

optarg = 2;
if nargin < optarg
	error('Needs (model, fig_flag) in findOptimal function');
end

% ======================
% some settings

DEFAULT_RN_CUTOFF = 10;		% base_rn is too_large
DEFAULT_GM = 0.004;			% default gamma if base_rn is larger than cutoff

% gamma boundary
GAMMA_MIN = 0.001;
GAMMA_MAX = 0.01;

% left boundary
rn_rate_left = 1.10;
left_rn = 0.08;


% right boundary
rn_rate_right = 1.40;
right_rn = 0.32;

% End of the settings
% ======================

% find best fit
model.gm = 0;
model = deconvolve(model);
base_rn = model.rn;
model.base_rn = base_rn;

if ~SLIENCE
	disp(sprintf('  ... base_rn = %0.4f', base_rn));
end

flag = 1; % not using the default_gm
if base_rn >= DEFAULT_RN_CUTOFF
	model.gm = DEFAULT_GM;
	flag = 0;
	if ~SLIENCE % OK
		disp(sprintf('  ... step1: base_rn is too large, use default %0.4f', model.gm));
	end
end

% left boundary search
if flag
	gm_left = GAMMA_MIN;
	gm_right = GAMMA_MAX;

	if ~SLIENCE
		disp(sprintf('  ... gamma in [%0.4f, %0.4f]', gm_left, gm_right));
	end

	rn_left = min(rn_rate_left*base_rn, base_rn+left_rn);
	leftr = (rn_left/base_rn-1)*100;
	if ~SLIENCE
		disp(sprintf('  ...  search left, rn_goal = %0.4f, rate = %0.1f', rn_left, leftr));
	end
	model.gm = gm_left;
	[model] = deconvolve(model);

	if model.rn >= DEFAULT_RN_CUTOFF;
		model.gm = DEFAULT_GM;
		flag = 0;
		if ~SLIENCE
			disp(sprintf('  ... left: base_rn is too large, use default %0.4f', model.gm));
		end
	end
end

if flag
	bs_flag = 1;
	if model.rn < rn_left
		[model, runs, bs_flag] = binarysearch(model, gm_left, gm_right, rn_left);
	end

	if bs_flag == 0
		flag = 0;
		model.gm = DEFAULT_GM;
		if ~SLIENCE
			disp(sprintf('  ... left_boundary: base_rn is too large in search, use default %0.4f', model.gm));
		end
	else
		gm_left = model.gm;
		rn_left = model.rn;
	end
end

% right boundary search
if flag
	rn_right = max(rn_rate_right*base_rn, base_rn+right_rn);
	rightr = (rn_right/base_rn-1)*100;
	if ~SLIENCE
		disp(sprintf('  ...  search right, rn_goal = %0.4f, rate = %0.1f', rn_right, rightr));
	end
	model.gm = gm_right;
	[model] = deconvolve(model);
	if model.rn >= DEFAULT_RN_CUTOFF;
		model.gm = DEFAULT_GM;
		flag = 0;
		if ~SLIENCE
			disp(sprintf('  ... right: base_rn is too large, use default %0.4f', model.gm));
		end
	end
end

if flag
	bs_flag = 1;
	if model.rn > rn_right
		[model, runs, bs_flag] = binarysearch(model, gm_left, gm_right, rn_right);
	end

	if bs_flag == 0
		flag = 0;
		model.gm = DEFAULT_GM;
		if ~SLIENCE % OK
			disp(sprintf('  ... right_boundary: base_rn is too large in search, use default %0.4f', model.gm));
		end
	else
		gm_right = model.gm;
		rn_right = model.rn;
	end
end

if flag
	if ~SLIENCE
		disp(sprintf('  ... rn range: [%0.4f, %0.4f]', rn_left, rn_right));
		disp(sprintf('  ... search gamma in [%0.4f %0.4f] for elbow', gm_left, gm_right));
	end

	bs_flag = 1;
	if (abs(gm_right-gm_left) < SMALL) % gm_right == gm_left
		model.gm = (gm_right+gm_left)/2;
	else
		step = (gm_right-gm_left)/ELBOW_BINS;
		gamma_array = gm_left:step:gm_right;
		[elbow_gamma, flag, gammas, rn, sn] = findElbow(model, gamma_array, fig_flag);
		model.gm = elbow_gamma;
	end

	if bs_flag == 0
		flag = 0;
		model.gm = DEFAULT_GM;
		if ~SLIENCE
			disp(sprintf('  ... findElbow: base_rn is too large or something wrong in search, use default %0.4f', model.gm));
		end
	end
end

disp(sprintf('%s: ... final gamma = %0.5f\n', model.orig_orfname, model.gm));
[model] = deconvolve(model);

return;

% ===============================
% ===============================
% ===============================

function [model, runs, flag] = binarysearch(model, gamma_min, gamma_max, rn_goal)
global SLIENCE;
global DEFAULT_RN_CUTOFF;

RN_SMALL = 2e-4;
LR_SMALL = 5e-4;
flag = 1;

left = gamma_min;
right = gamma_max;
runs = 0;
% find the fit_left point
while right-left > LR_SMALL
	cur_gamma = (left+right)/2;
	runs = runs+1;
	
	model.gm = cur_gamma;
	
	[model] = deconvolve(model);

	if model.rn >= DEFAULT_RN_CUTOFF
		flag = 0;
		return;
	end

	if abs(model.rn-rn_goal) <= RN_SMALL
		break;
	elseif model.rn > rn_goal
		right = cur_gamma;
	else
		left = cur_gamma;
	end

	rn_rate = (model.rn/model.base_rn-1)*100;
	if ~SLIENCE
		disp(sprintf('  ...   gm = %0.4f, rn = %0.4f, rate = %0.1f', model.gm, model.rn, rn_rate));
	end

end
return;
