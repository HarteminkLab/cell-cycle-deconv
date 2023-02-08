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
PADDINGSIZE = 42;

% Settings
EDGE_EFFECTS = DOUBLE_MIRROR;
TRIAL = IT_B_SMOOTH;
modeltype = upper(model.modeltype);

g = model.g;
mean_g = mean(g);
H = model.H;
gamma = model.gm;

g = model.g;
mean_g = mean(g);
H = model.H;
gamma = model.gm;
Hsize = size(H, 2);

% Unused in optimzation. For recording the top
% indices for plotting
f_t = [];
f_t_list = {};
for i = 1:length(model.t_intervals)
	idx = model.t_intervals{i}{2};
	se = model.Hpos{idx};
	indices = se(1):1:se(2);
	f_t = [f_t indices];
	f_t_list{i} = indices;
end

% Construct the f index vector (f_i) for the first
% Wavelet smoothing criteria W1
% Initial with padding to make it a power of 2
f_i = [];
last = Hsize;
f_i_front = last+[1:1:PADDINGSIZE];
last = last+PADDINGSIZE;
f_i_after = last+[1:1:PADDINGSIZE];
last = last+PADDINGSIZE;
f_i_list = {};
for i = 1:length(model.i_intervals)
	idx = model.i_intervals{i}{2};
	se = model.Hpos{idx};
	indices = se(1):1:se(2);
	f_i = [f_i indices];
	f_i_list{i} = indices;

end

f_i_pad = [f_i_front f_i f_i_after];

% Construct the f index vector (f_b) for the second
% Wavelet smoothing criteria W2
f_b = [];
for i = 1:length(model.b_intervals)
	idx = model.b_intervals{i}{2};
	se = model.Hpos{idx};
	indices = se(1):1:se(2);
	f_b = [f_b indices];
	f_b_list{i} = indices;

end

% Construct the Wavelets
Hsize = size(H, 2);
W1 = getWaveletKernel(WAVETYPE, length(f_i_pad), WAVEPAR);
W2 = getWaveletKernel(WAVETYPE, length(f_b), WAVEPAR);

PADDINGSIZE = 42;
% Error
% square_pos(norm(H*f(PADDINGSIZE:Hsize+PADDINGSIZE-1)./g'-1, 2)) ... % fit errors
% 		+ gamma*(norm(W1*f([f_i_pad]),1) + norm(W2*f([f_b]),1))/mean_g ...

% Working
		% square_pos(norm(H*f(PADDINGSIZE:Hsize+PADDINGSIZE-1)./g'-1, 2)) ... % fit errors
		% + gamma*(norm(W1*f([1:PADDINGSIZE+130  PADDINGSIZE+217:Hsize+PADDINGSIZE*2]),1) + ...
		% 		 norm(W2*f([PADDINGSIZE+131:PADDINGSIZE+258                       ]),1) ...
		% 		 )/mean_g ...

fixed_f_i_pad = [1:PADDINGSIZE+130  PADDINGSIZE+217:Hsize+PADDINGSIZE*2];
model.fixed_f_i_pad = fixed_f_i_pad;
model.f_i_pad = f_i_pad;

% Enforce smoothness of the entire padded array
W = getWaveletKernel(WAVETYPE, length(f_i_pad), WAVEPAR);

cvx_begin
	cvx_quiet(true);

	variable f(Hsize+PADDINGSIZE*2); % 258 + 42*2 = 342

	% The fit error, get the relevant indices of f
	% Skipping the first PADDINGSIZE indices and removing the last PADDINGSIZE indices
	% TODO: Figure out why f_i_padding is different than this manually indexing.
	minimize(...
		square_pos(norm(H*f(PADDINGSIZE:Hsize+PADDINGSIZE-1)./g'-1, 2)) ... % fit errors
		+ gamma*(norm(W1*f([1:PADDINGSIZE+130  PADDINGSIZE+217:Hsize+PADDINGSIZE*2]),1) + ...
				 norm(W2*f([PADDINGSIZE+131:PADDINGSIZE+258                       ]),1) ...
				 )/mean_g ...
	);

	subject to
		f>=0;
cvx_end

f_final = f(PADDINGSIZE+1:end-PADDINGSIZE);%(PADDINGSIZE:Hsize+PADDINGSIZE-1);
% w1 = norm(W1*f([f_i_pad]),1)/mean_g;
% w2 = norm(W2*f([f_b]),1)/mean_g;
% rn = square_pos(norm(H*f_final./g-1, 2));
% model.sn_w1 = w1;
% model.sn_w2 = w2;
% model.rn = rn;
model.f = f_final;

model.f_i = f_i;
model.f_t = f_t;
model.f_b = f_b;

model.f_b_list = f_b_list;
model.f_t_list = f_t_list;
model.f_i_list = f_i_list;

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


