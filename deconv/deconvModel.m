function [model, pred_g] = deconvModel(model)

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

DECONV_WAVELET = 1;
DECONV_KERNEL = DECONV_WAVELET;

old_slience = SLIENCE;
if FROM_FINDOPTIMAL
	SLIENCE = 1;
end

%%%%%%%%%%%%%%%%%%%%%%%
% settings
NORMALIZING = 0;
TURN_OFF = 0;
%%%%%%%%%%%%%%%%%%%%%%%

% Symmlet
WAVETYPE = 'Symmlet';
WAVEPAR = 5;

SMALL = 1e-6;

IBT_SMOOTH = 10;
IT_B_SMOOTH = 45;
IB_SMOOTH = 20;
IT_SMOOTH = 30;
I_SMOOTH = 15;

NOTHING_BASELINE = -1;
NOTHING = 0;
PADDING = 1;
MIRROR = 2;
SMALLPADDING = 3;
DOUBLE_MIRROR = 4;
PADDINGSIZE = 30;

if model.gm == 0
	EDGE_EFFECTS = NOTHING_BASELINE;
else
	EDGE_EFFECTS = DOUBLE_MIRROR;
end

TRIAL = IT_B_SMOOTH;

optarg = 1;
if nargin < optarg
	error('...input (model) in deconvModel');
end

g = model.g';
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


if DECONV_KERNEL == DECONV_WAVELET && TRIAL == IT_B_SMOOTH && ~TURN_OFF && EDGE_EFFECTS == DOUBLE_MIRROR
%	if ~SLIENCE 
%		disp('IT+B smooth, DOUBLE MIRROR FOR IT, LINE 510');
%	end
%	disp('line 510');

	Hsize = size(H, 2);

	f_it = [];

	i_intervals = model.intervals.initialPhaseMapping;
	i_list = model.intervals.initialTimepointsList;

	t_intervals = model.intervals.topPhaseMapping;
	t_list = model.intervals.topTimepointsList;

	b_intervals = model.intervals.bottomPhaseMapping;
	b_list = model.intervals.bottomTimepointsList;

	for i = 1:length(i_intervals)
		idx = i_intervals{i}{2};
		se = model.Hpos{idx};
		f_it = [f_it se(1):1:se(2)];
	end

	for i = 1:length(t_intervals)
		idx = t_intervals{i}{2};
		se = model.Hpos{idx};
		f_it = [f_it se(1):1:se(2)];
	end

	f_b = [];
	for i = 1:length(b_intervals)
		idx = b_intervals{i}{2};
		se = model.Hpos{idx};
		f_b = [f_b se(1):1:se(2)];
	end

	f_final = zeros(Hsize,1);

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% right mirroring
	f_b_mirror = [f_b f_b];
	f_it_mirror = [f_it reverse(f_it)];
	factor_fb = 1.5;

	W1 = getWaveletKernel(WAVETYPE, length(f_it_mirror), WAVEPAR);
	W2 = getWaveletKernel(WAVETYPE, length(f_b), WAVEPAR);
	W2pad = zeros(length(f_b));
	W2 = [W2 W2pad; W2pad fliplr(W2)];

	f = zeros(Hsize, 1);

	cvx_begin
		cvx_quiet(true);

		variable f(Hsize);

		minimize(...
			square_pos(norm(H*f ./ g-1, 2)) ... % fit error
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
	f_it_mirror = [reverse(f_it) f_it];

	cvx_begin
		cvx_quiet(true);

		variable f(Hsize);

		minimize(...
			square_pos(norm(H*f./g-1, 2)) ... % fit error
			+ gamma*(norm(W1*f([f_it_mirror]),1) + factor_fb*norm(W2*f([f_b_mirror]),1))/mean_g ... % smooth error
		);

		subject to
			f>=0;
	cvx_end

	f_final(f_it(end/2+1:end)) = f(f_it(end/2+1:end));
	f_b_2 = f(f_b);
	f_it_2 = f(f_it);

	f_final(f_b) = (f_b_1+f_b_2)/2;

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	f = f_final;

	W1 = getWaveletKernel(WAVETYPE, length(f_it), WAVEPAR);
	W2 = getWaveletKernel(WAVETYPE, length(f_b), WAVEPAR);
	sn = ( norm(W1*f([f_it]),1) + norm(W2*f([f_b]),1) )/mean_g;
	rn = square_pos(norm(H*f./g-1, 2));

end

% Note, for baseline computation, we will need to add back the
% NOTHING_BASELINE code from deconv.v2

% ..........................
% sn: solution norm
% rn: residual norm
% ..........................
pred_g = H*f;

model.f = f;
model.sn = sn;
model.rn = rn;
%model.sn0 = sn0;
%model.sn0all = sn0all;
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

SLIENCE = old_slience;
return;
