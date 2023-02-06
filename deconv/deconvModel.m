function [model, pred_g] = deconvModel(model)

global DECONV_MU0POS;
global DECONV_LAMBDAPOS;
global DECONV_DELTAPOS;
global DECONV_ALPHAPOS;
global DECONV_SIGMA0POS;
global DECONV_SIGMAVPOS;
global DECONV_BETAPOS;

global DECONV_KERNEL;
global DECONV_WAVELET;
global DECONV_DIFF;
global DECONV_JOINT;

DECONV_MU0POS = 1;
DECONV_LAMBDAPOS = 2;
DECONV_DELTAPOS = 3;
DECONV_SIGMA0POS = 4;
DECONV_SIGMAVPOS = 5;
DECONV_ALPHAPOS = 6;
DECONV_BETAPOS = 7;

DECONV_WAVELET = 1;
DECONV_DIFF = 2;
% 1: wavelet (uneven intervals)
% 2: 1st-order different operator (even intervals)
DECONV_KERNEL = DECONV_WAVELET;

% Symmlet
WAVETYPE = 'Symmlet';
WAVEPAR = 5;
DECONV_JOINT = 'JOINT';

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
EDGE_EFFECTS = DOUBLE_MIRROR;
TRIAL = IT_B_SMOOTH;
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
% require: it and b are smooth

if DECONV_KERNEL == DECONV_WAVELET && TRIAL == IT_B_SMOOTH && EDGE_EFFECTS == DOUBLE_MIRROR

	Hsize = size(H, 2);

	f_it = [];
	% f_i
	for i = 1:length(model.i_intervals)
		idx = model.i_intervals{i}{2};
		se = model.Hpos{idx};
		f_it = [f_it se(1):1:se(2)];
	end
	% f_t
	for i = 1:length(model.t_intervals)
		idx = model.t_intervals{i}{2};
		se = model.Hpos{idx};
		f_it = [f_it se(1):1:se(2)];
	end

	f_b = [];
	% f_b
	for i = 1:length(model.b_intervals)
		idx = model.b_intervals{i}{2};
		se = model.Hpos{idx};
		f_b = [f_b se(1):1:se(2)];
	end

	f_final = zeros(Hsize,1);

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% right mirroring
	f_b_mirror = [f_b f_b];
	f_it_mirror = [f_it flip(f_it)];
	factor_fb = 1.5;

	W1 = getWaveletKernel(WAVETYPE, length(f_it_mirror), WAVEPAR);
	W2 = getWaveletKernel(WAVETYPE, length(f_b), WAVEPAR);
	W2pad = zeros(length(f_b));
	W2 = [W2 W2pad; W2pad fliplr(W2)];

	cvx_begin
		cvx_quiet(true);

		variable f(Hsize);

		minimize(...
			square_pos(norm(H*f./g'-1, 2)) ... % fit error
			+ gamma*(norm(W1*f(f_it_mirror),1) + ...
            factor_fb*norm(W2*f(f_b_mirror),1))/mean_g ... % smooth error
		);
    
		subject to
			f>=0;
	cvx_end

	f_final(f_it(1:end/2)) = f(f_it(1:end/2));
	f_b_1 = f(f_b);
	f_it_1 = f(f_it);

	%% left mirroring
	f_it_mirror = [flip(f_it) f_it];

	cvx_begin
		cvx_quiet(true);

		variable f(Hsize);

		minimize(...
			square_pos(norm(H*f./g'-1, 2)) ... % fit error
			+ gamma*(norm(W1*f([f_it_mirror]),1) + factor_fb*norm(W2*f([f_b_mirror]),1))/mean_g ... % smooth error
		);

		subject to
			f>=0;
	cvx_end

	f_final(f_it(end/2+1:end)) = f(f_it(end/2+1:end));
	f_b_2 = f(f_b);
	f_it_2 = f(f_it);


	f_final(f_it) = (f_it_1+f_it_2)/2;
	f_final(f_b) = (f_b_1+f_b_2)/2;

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	f = f_final;

	W1 = getWaveletKernel(WAVETYPE, length(f_it), WAVEPAR);
	W2 = getWaveletKernel(WAVETYPE, length(f_b), WAVEPAR);
	sn = ( norm(W1*f([f_it]),1) + norm(W2*f([f_b]),1) )/mean_g;
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
model.f_it = f_it;
model.f_b = f_b;

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
