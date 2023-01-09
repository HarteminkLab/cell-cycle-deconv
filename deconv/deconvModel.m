function [model, pred_g] = deconvModel(model)

global DECONV_KERNEL;
global DECONV_WAVELET;
global DECONV_DIFF;

global DECONV_MU0POS;
global DECONV_LAMBDAPOS;
global DECONV_DELTAPOS;
global DECONV_ALPHAPOS;

global WAVETYPE;
global WAVEPAR;

global DECONV_JOINT;
global DECONV_WT1;
global DECONV_WT2;

global NORMALIZING;
global FROM_FINDOPTIMAL;

%%%%%%%%%%%%%%%%%%%%%%%
% settings
NORMALIZING = 0;
TURN_OFF = 0;
%%%%%%%%%%%%%%%%%%%%%%%

% Symmlet
WAVETYPE = 'Symmlet';
WAVEPAR = 5;

mu0 = model.lengths(DECONV_MU0POS);
lambda = model.lengths(DECONV_LAMBDAPOS);
delta = model.lengths(DECONV_DELTAPOS);
alpha = model.lengths(DECONV_ALPHAPOS);
alpha_avg = mean(alpha);

SMALL = 1e-6;

IBT_SMOOTH = 10;
IT_B_SMOOTH = 45;
IB_SMOOTH = 20;
IT_SMOOTH = 30;
I_SMOOTH = 15;
IBT_RCG1_SPARSE_CONDITION = 40;
IBT_RDG1_SPARSE_CONDITION = 50;
IB_CDG1_SPARSE_CONDITION = 70;

NOTHING_BASELINE = -1;
NOTHING = 0;
PADDING = 1;
MIRROR = 2;
SMALLPADDING = 3;
DOUBLE_MIRROR = 4;
% submodel in DOUBLE_MIRROR
DOUBLE_MIRROR_REFINE = 1;
PADDINGSIZE = 30;

% Settings
EDGE_EFFECTS = NOTHING;
TRIAL = IB_SMOOTH;
modeltype = upper(model.modeltype);

g = model.g;
mean_g = mean(g);
H = model.H;
gamma = model.gm;

% model structure so far
% model:
%		- lengths: array
%		- relations: cell array
%		- i/t/b List: cell arrays
%		- i/t/b_intervals: string arrays
%		- Hsegments: cell arrays
%		- timepoints: array
% require: i and b are smooth

% wavelet; actually using
if DECONV_KERNEL == DECONV_WAVELET && TRIAL == IB_SMOOTH && ~TURN_OFF && EDGE_EFFECTS == NOTHING

	f_i = [];
	wf_i = [];
	i_gap = [];
	for i = 1:length(model.i_intervals)
		idx = model.i_intervals{i}{2};
		se = model.Hpos{idx};
		f_i = [f_i se(1):1:se(2)];
		gap = model.iList{i}(2)-model.iList{i}(1);
		i_gap = [i_gap gap];
		wf_i = [wf_i ones(1, se(2)-se(1)+1)*gap];
	end

	f_b = [];
	wf_b = [];
	b_gap = [];
	for i = 1:length(model.b_intervals)
		idx = model.b_intervals{i}{2};
		se = model.Hpos{idx};
		f_b = [f_b se(1):1:se(2)];
		gap = model.bList{i}(2)-model.bList{i}(1);
		b_gap = [b_gap gap];
		wf_b = [wf_b ones(1, se(2)-se(1)+1)*gap];
	end

	Hsize = size(H, 2);

	W1 = getWaveletKernel(WAVETYPE, length(f_i), WAVEPAR);
	W2 = getWaveletKernel(WAVETYPE, length(f_b), WAVEPAR);
	W1 = addWeight(W1, wf_i);
	W2 = addWeight(W2, wf_b);

	cvx_begin
		cvx_quiet(true);

		variable f(Hsize);

		minimize(...
			square_pos(norm(H*f./g'-1, 2)) ... % fit error
			+ gamma*(norm(W1*f([f_i]),1) + norm(W2*f([f_b]),1))/mean_g ... % smooth error
		);

		subject to
			f>=0;
	cvx_end

	sn = (norm(W1*f([f_i]),1) + norm(W2*f([f_b]),1))/mean_g;
	rn = square_pos(norm(H*f./g-1, 2));
end

% ..........................
% sn: solution norm
% rn: residual norm
% ..........................
pred_g = H*f;

model.f = f;
model.sn = sn;
model.rn = rn;
model.err = sqrt(model.rn/length(model.g));
g_avg = mean(model.g);
if strcmp(model.datatype, DECONV_JOINT)
	glen = length(model.g);
	pred_g = model.H*model.f;
	g1 = model.g(1:glen/2);
	g2 = model.g(glen/2+1:glen);
	pred_g1 = pred_g(1:glen/2);
	pred_g2 = pred_g(glen/2+1:glen);

	tm1 = model.timepoints(1,:);
	tm2 = model.timepoints(2,:);
else
	pred_g = model.H*model.f;
end

model.pred_g = pred_g;

return;
