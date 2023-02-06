function [model] = debug_f_i(model)

global SLIENCE;

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

% Symmlet
WAVETYPEE = 'Symmlet';
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
PADDINGSIZE = 42;
%EDGE_EFFECTS = DOUBLE_MIRROR;

if model.gm == 0
	EDGE_EFFECTS = NOTHING_BASELINE;
else
	EDGE_EFFECTS = DOUBLE_MIRROR;
end

TRIAL = IT_B_SMOOTH;

modeltype = upper(model.modeltype);


g = model.g;
mean_g = mean(g);
H = model.H;
gamma = model.gm;
Hsize = size(H, 2);

% Initial with padding to make it a power of 2
f_i = [];
last = Hsize;
f_i_front = last+[1:1:PADDINGSIZE];
last = last+PADDINGSIZE;
f_i_after = last+[1:1:PADDINGSIZE];
last = last+PADDINGSIZE;
for i = 1:length(model.i_intervals)
	idx = model.i_intervals{i}{2};
	se = model.Hpos{idx};
	f_i = [f_i se(1):1:se(2)];
end
f_i_pad = [f_i_front f_i f_i_after];

% Bottom
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
W1 = getWaveletKernel(WAVETYPE, length(f_i_pad), WAVEPAR);
W2 = getWaveletKernel(WAVETYPE, length(f_b), WAVEPAR);

cvx_begin
	cvx_quiet(true);

	variable f(Hsize+PADDINGSIZE*2);

	% The fit error, get the relevant indices of f
	% Skipping the first PADDINGSIZE indices and removing the last PADDINGSIZE indices
	minimize(...
		square_pos(norm(H*f(PADDINGSIZE:Hsize+PADDINGSIZE-1)./g'-1, 2)) ... % fit errors	
		+ gamma*(norm(W1*f([f_i_pad]),1) + norm(W2*f([f_b]),1))/mean_g ...
	);

	subject to
		f>=0;
cvx_end

f_final = f(PADDINGSIZE:Hsize+PADDINGSIZE-1);
sn = (norm(W1*f([f_i_pad]),1) + norm(W2*f([f_b]),1))/mean_g;
rn = square_pos(norm(H*f_final./g-1, 2));
model.f = f_final;

model.f_i = f_i;
model.f_b = f_b;