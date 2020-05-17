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

old_slience = SLIENCE;
if FROM_FINDOPTIMAL
	SLIENCE = 1;
end

%%%%%%%%%%%%%%%%%%%%%%%
% settings
NORMALIZING = 0;
TURN_OFF = 0;
%%%%%%%%%%%%%%%%%%%%%%%

% Daubechies
%WAVETYPE = 'Daubechies';
%WAVEPAR = 8;

% Symmlet
WAVETYPE = 'Symmlet';
WAVEPAR = 5;

% Haar
%WAVETYPE = 'Haar';

% Vaidyanathan
%WAVETYPE = 'Vaidyanathan';
%WAVEPAR = 0;

% Battle
%WAVETYPE = 'Battle';
%WAVEPAR = 3;

% coiflet
%WAVETYPE = 'Coiflet';
%WAVEPAR = 3;

mu0 = model.lengths(DECONV_MU0POS);
lambda = model.lengths(DECONV_LAMBDAPOS);
delta = model.lengths(DECONV_DELTAPOS);
alpha = model.lengths(DECONV_ALPHAPOS);
alpha_avg = mean(alpha);

SMALL = 1e-6;

ibt_models = {'1.2.1' ...
};

it_models = {'1.2.3'...
	'1.2.1.3'...
};

ib_models = {'1.1.1' ...
	'1.2.2' ...
	'1.1.2.1' ...
	'1.2.1.2' ...
};

i_models = {'1.1.1.2' ...
};

ibt_rcg1_sparse_models = {'1.2.1.1' ...
};

ibt_rdg1_sparse_models = {'1.2.1.4' ...
};

ib_cdg1_sparse_models = {'1.1.1.1' ...
};

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
%EDGE_EFFECTS = DOUBLE_MIRROR;

if model.gm == 0
	EDGE_EFFECTS = NOTHING_BASELINE;
else
	EDGE_EFFECTS = DOUBLE_MIRROR;
end

TRIAL = IT_B_SMOOTH;

mom2daughter_ratio = 2;

modeltype = upper(model.modeltype);

optarg = 1;
if nargin < optarg
	error('...input (model) in deconvModel');
end

g = model.g;
mean_g = mean(g);
H = model.H;
%if length(model.gm) == 2
%	gamma = model.gm(1);
%	gamma_2 = model.gm(2);
%else
%	gamma = model.gm;
%	gamma_2 = model.gm;
%end
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
if DECONV_KERNEL == DECONV_WAVELET && TRIAL == IT_SMOOTH && ~TURN_OFF
	if ~SLIENCE 
		disp('I+T smooth');
	end

	f_i = [];
	wf_i = [];
	for i = 1:length(model.i_intervals)
		idx = model.i_intervals{i}{2};
		se = model.Hpos{idx};
		f_i = [f_i se(1):1:se(2)];
		gap = model.iList{i}(2)-model.iList{i}(1);
		wf_i = [wf_i ones(1, se(2)-se(1)+1)*gap];
	end

	f_t = [];
	wf_t = [];
	for i = 1:length(model.t_intervals)
		idx = model.t_intervals{i}{2};
		se = model.Hpos{idx};
		f_t = [f_t se(1):1:se(2)];
		gap = model.tList{i}(2)-model.tList{i}(1);
		wf_t = [wf_t ones(1, se(2)-se(1)+1)*gap];
	end

	Hsize = size(H, 2);
	W1 = getWaveletKernel(WAVETYPE, length(f_i), WAVEPAR);
	W2 = getWaveletKernel(WAVETYPE, length(f_t), WAVEPAR);
	W1 = addWeight(W1, wf_i);
	W2 = addWeight(W2, wf_t);

	cvx_begin
		cvx_quiet(true);
%		cvx_precision high;

		variable f(Hsize);

		minimize(...
			square_pos(norm(H*f./g-1, 2)) ... % fit error
			+ gamma*(norm(W1*f([f_i]),1) + norm(W2*f([f_t]),1))/mean_g ... % smooth error
		);

		subject to
			f>=0;
	cvx_end

	sn = (norm(W1*f([f_i]),1) + norm(W2*f([f_t]),1))/mean_g;
	rn = square_pos(norm(H*f./g-1, 2));

elseif DECONV_KERNEL == DECONV_WAVELET && (TRIAL == IB_SMOOTH || TRIAL == IT_B_SMOOTH) && ~TURN_OFF && EDGE_EFFECTS == NOTHING_BASELINE
	if ~SLIENCE 
		disp('no smooth and gamma=0');
	end

	Hsize = size(H,2);
	cvx_begin
		cvx_quiet(true);
%		cvx_precision high;

		variable f(Hsize);

		minimize(...
			square_pos(norm(H*f./g-1, 2)) ... % fit error
		);

		subject to
			f>=0;
	cvx_end

	sn = Inf;
	rn = square_pos(norm(H*f./g-1,2));

% USING IB
elseif DECONV_KERNEL == DECONV_WAVELET && TRIAL == IB_SMOOTH && ~TURN_OFF && EDGE_EFFECTS == NOTHING
	if ~SLIENCE 
		disp('I+B smooth');
	end

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
%		cvx_precision high;

		variable f(Hsize);

		minimize(...
			square_pos(norm(H*f./g-1, 2)) ... % fit error
			+ gamma*(norm(W1*f([f_i]),1) + norm(W2*f([f_b]),1))/mean_g ... % smooth error
		);

		subject to
			f>=0;
	cvx_end

	sn = (norm(W1*f([f_i]),1) + norm(W2*f([f_b]),1))/mean_g;
	rn = square_pos(norm(H*f./g-1, 2));


elseif DECONV_KERNEL == DECONV_WAVELET && TRIAL == IT_B_SMOOTH && ~TURN_OFF && EDGE_EFFECTS == NOTHING
	if ~SLIENCE 
		disp('IT+B smooth');
	end

	f_it = [];
	wf_it = [];
	it_gap = [];
	for i = 1:length(model.i_intervals)
		idx = model.i_intervals{i}{2};
		se = model.Hpos{idx};
		f_it = [f_it se(1):1:se(2)];
		gap = model.iList{i}(2)-model.iList{i}(1);
		it_gap = [it_gap gap];
		wf_it = [wf_it ones(1, se(2)-se(1)+1)*gap];
	end
	for i = 1:length(model.t_intervals)
		idx = model.t_intervals{i}{2};
		se = model.Hpos{idx};
		f_it = [f_it se(1):1:se(2)];
		gap = model.tList{i}(2)-model.tList{i}(1);
		it_gap = [it_gap gap];
		wf_it = [wf_it ones(1, se(2)-se(1)+1)*gap];
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
	W1 = getWaveletKernel(WAVETYPE, length(f_it), WAVEPAR);
	W2 = getWaveletKernel(WAVETYPE, length(f_b), WAVEPAR);
	W1 = addWeight(W1, wf_it);
	W2 = addWeight(W2, wf_b);

	cvx_begin
		cvx_quiet(true);
%		cvx_precision high;

		variable f(Hsize);

		minimize(...
			square_pos(norm(H*f./g-1, 2)) ... % fit error
			+ gamma*(norm(W1*f([f_it]),1) + norm(W2*f([f_b]),1))/mean_g ... % smooth error
		);

		subject to
			f>=0;
	cvx_end

	sn = (norm(W1*f([f_it]),1) + norm(W2*f([f_b]),1))/mean_g;
	rn = square_pos(norm(H*f./g-1, 2));

%	if ~SLIENCE
%		disp(i_gap);
%		disp(b_gap);
%	end

elseif DECONV_KERNEL == DECONV_WAVELET && TRIAL == IT_B_SMOOTH && ~TURN_OFF && EDGE_EFFECTS == PADDING
	if ~SLIENCE 
		disp('IT+B smooth, PADDING');
	end

	Hsize = size(H, 2);

	f_it = [];
	f_it_front = [];
	f_it_after = [];
	for i = 1:length(model.i_intervals)
		idx = model.i_intervals{i}{2};
		se = model.Hpos{idx};
		f_it = [f_it se(1):1:se(2)];
		mid = (se(2)-se(1)+1)/2;
		f_it_front = [f_it_front se(1)+Hsize:1:se(1)+Hsize+mid];
		f_it_after = [f_it_after se(1)+Hsize+mid+1:1:Hsize+se(2)];
	end
	for i = 1:length(model.t_intervals)
		idx = model.t_intervals{i}{2};
		se = model.Hpos{idx};
		f_it = [f_it se(1):1:se(2)];
		mid = (se(2)-se(1)+1)/2;
		f_it_front = [f_it_front se(1)+Hsize:1:se(1)+Hsize+mid];
		f_it_after = [f_it_after se(1)+Hsize+mid+1:1:Hsize+se(2)];
	end
	f_it_padding = [f_it_front f_it f_it_after];

	f_b = [];
	f_b_front = [];
	f_b_after = [];
	for i = 1:length(model.b_intervals)
		idx = model.b_intervals{i}{2};
		se = model.Hpos{idx};
		f_b = [f_b se(1):1:se(2)];
		mid = (se(2)-se(1)+1)/2;
		f_b_front = [f_b_front se(1)+Hsize:1:se(1)+Hsize+mid];
		f_b_after = [f_b_after se(1)+Hsize+mid+1:1:Hsize+se(2)];
	end
	f_b_padding = [f_b_front f_b f_b_after];

	W1 = getWaveletKernel(WAVETYPE, length(f_it_padding), WAVEPAR);
	W2 = getWaveletKernel(WAVETYPE, length(f_b_padding), WAVEPAR);

	cvx_begin
		cvx_quiet(true);
%		cvx_precision high;

		variable f(Hsize*2);

		minimize(...
			square_pos(norm(H*f([1:Hsize])./g-1, 2)) ... % fit error
			+ gamma*(norm(W1*f([f_it_padding]),1) + norm(W2*f([f_b_padding]),1))/mean_g ... % smooth error
		);

		subject to
			f>=0;
	cvx_end

	f = f([1:Hsize]);
	W1 = getWaveletKernel(WAVETYPE, length(f_it), WAVEPAR);
	W2 = getWaveletKernel(WAVETYPE, length(f_b), WAVEPAR);
	sn = (norm(W1*f([f_it]),1) + norm(W2*f([f_b]),1))/mean_g;
	rn = square_pos(norm(H*f./g-1, 2));

elseif DECONV_KERNEL == DECONV_WAVELET && TRIAL == IT_B_SMOOTH && ~TURN_OFF && EDGE_EFFECTS == MIRROR
	if ~SLIENCE 
		disp('IT+B smooth, MIRROR');
	end

	Hsize = size(H, 2);

	f_it = [];
	for i = 1:length(model.i_intervals)
		idx = model.i_intervals{i}{2};
		se = model.Hpos{idx};
		f_it = [f_it se(1):1:se(2)];
	end
	for i = 1:length(model.t_intervals)
		idx = model.t_intervals{i}{2};
		se = model.Hpos{idx};
		f_it = [f_it se(1):1:se(2)];
	end
	f_it_mirror = [f_it reverse(f_it)];

	f_b = [];
	for i = 1:length(model.b_intervals)
		idx = model.b_intervals{i}{2};
		se = model.Hpos{idx};
		f_b = [f_b se(1):1:se(2)];
	end
	f_b_mirror = [f_b reverse(f_b)];

	W1 = getWaveletKernel(WAVETYPE, length(f_it_mirror), WAVEPAR);
	W2 = getWaveletKernel(WAVETYPE, length(f_b_mirror), WAVEPAR);

	cvx_begin
		cvx_quiet(true);
%		cvx_precision high;

		variable f(Hsize);

		minimize(...
			square_pos(norm(H*f./g-1, 2)) ... % fit error
			+ gamma*(norm(W1*f([f_it_mirror]),1) + norm(W2*f([f_b_mirror]),1))/mean_g ... % smooth error
		);

		subject to
			f>=0;
	cvx_end

	W1 = getWaveletKernel(WAVETYPE, length(f_i), WAVEPAR);
	W2 = getWaveletKernel(WAVETYPE, length(f_b), WAVEPAR);
	sn = (norm(W1*f([f_i]),1) + norm(W2*f([f_b]),1))/mean_g;
	rn = square_pos(norm(H*f./g-1, 2));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif DECONV_KERNEL == DECONV_WAVELET && TRIAL == IT_B_SMOOTH && ~TURN_OFF && EDGE_EFFECTS == MIRROR
	if ~SLIENCE 
		disp('IT+B smooth, MIRROR');
	end

	Hsize = size(H, 2);

	f_i = [];
	for i = 1:length(model.i_intervals)
		idx = model.i_intervals{i}{2};
		se = model.Hpos{idx};
		f_i = [f_i se(1):1:se(2)];
	end
	f_i_mirror = [f_i reverse(f_i)];

	f_b = [];
	for i = 1:length(model.b_intervals)
		idx = model.b_intervals{i}{2};
		se = model.Hpos{idx};
		f_b = [f_b se(1):1:se(2)];
	end
	f_b_mirror = [f_b reverse(f_b)];

	W1 = getWaveletKernel(WAVETYPE, length(f_i_mirror), WAVEPAR);
	W2 = getWaveletKernel(WAVETYPE, length(f_b_mirror), WAVEPAR);

	cvx_begin
		cvx_quiet(true);
%		cvx_precision high;

		variable f(Hsize);

		minimize(...
			square_pos(norm(H*f./g-1, 2)) ... % fit error
			+ gamma*(norm(W1*f([f_i_mirror]),1) + norm(W2*f([f_b_mirror]),1))/mean_g ... % smooth error
		);

		subject to
			f>=0;
	cvx_end

	W1 = getWaveletKernel(WAVETYPE, length(f_i), WAVEPAR);
	W2 = getWaveletKernel(WAVETYPE, length(f_b), WAVEPAR);
	sn = (norm(W1*f([f_i]),1) + norm(W2*f([f_b]),1))/mean_g;
	rn = square_pos(norm(H*f./g-1, 2));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif DECONV_KERNEL == DECONV_WAVELET && TRIAL == IT_B_SMOOTH && ~TURN_OFF && EDGE_EFFECTS == DOUBLE_MIRROR
%	if ~SLIENCE 
%		disp('IT+B smooth, DOUBLE MIRROR FOR IT, LINE 510');
%	end

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
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% right mirroring
	f_b_mirror = [f_b f_b];
	f_it_mirror = [f_it reverse(f_it)];
	factor_fb = 1.5;

	W1 = getWaveletKernel(WAVETYPE, length(f_it_mirror), WAVEPAR);
	W2 = getWaveletKernel(WAVETYPE, length(f_b), WAVEPAR);
	W2pad = zeros(length(f_b));
	W2 = [W2 W2pad; W2pad fliplr(W2)];

	cvx_begin
		cvx_quiet(true);
%		cvx_precision high;

		variable f(Hsize);

		minimize(...
			square_pos(norm(H*f./g-1, 2)) ... % fit error
			+ gamma*(norm(W1*f(f_it_mirror),1) + factor_fb*norm(W2*f(f_b_mirror),1))/mean_g ... % smooth error
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

	c = corr2(f_b_1, f_b_2);

	model.fb_corr = c;
%	if ~SLIENCE
%		figure;
%		plot(f_b_1, 'r');
%		hold on;
%		plot(f_b_2, 'b');
%		figure;
%		plot(f_it_1, 'r');
%		hold on;
%		plot(f_it_2,'r');
%	end

%	if ~SLIENCE
%		disp(sprintf('f_b corr=%0.3g', c));
%	end
	f_final(f_b) = (f_b_1+f_b_2)/2;

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	f = f_final;

	W1 = getWaveletKernel(WAVETYPE, length(f_it), WAVEPAR);
	W2 = getWaveletKernel(WAVETYPE, length(f_b), WAVEPAR);
	sn = ( norm(W1*f([f_it]),1) + norm(W2*f([f_b]),1) )/mean_g;
	rn = square_pos(norm(H*f./g-1, 2));

elseif DECONV_KERNEL == DECONV_WAVELET && TRIAL == IT_B_SMOOTH && ~TURN_OFF && EDGE_EFFECTS == MIRROR
	if ~SLIENCE 
		disp('IT+B smooth, MIRROR');
	end

%	H_mirror = [H fliplr(H)];

	Hsize = size(H, 2);

	f_i = [];
	for i = 1:length(model.i_intervals)
		idx = model.i_intervals{i}{2};
		se = model.Hpos{idx};
		f_i = [f_i se(1):1:se(2)];
	end
%	f_i_mirror = [f_i f_i+Hsize reverse(f_i)+Hsize reverse(f_i)];
	f_i_mirror = [f_i reverse(f_i)];

	f_b = [];
	for i = 1:length(model.b_intervals)
		idx = model.b_intervals{i}{2};
		se = model.Hpos{idx};
		f_b = [f_b se(1):1:se(2)];
	end
%	f_b_mirror = [f_b f_b+Hsize reverse(f_b)+Hsize reverse(f_b)];
	f_b_mirror = [f_b reverse(f_b)];

	W1 = getWaveletKernel(WAVETYPE, length(f_i_mirror), WAVEPAR);
	W2 = getWaveletKernel(WAVETYPE, length(f_b_mirror), WAVEPAR);

	cvx_begin
		cvx_quiet(true);
%		cvx_precision high;

		variable f(Hsize);

		minimize(...
			square_pos(norm(H*f./g-1, 2)) ... % fit error
			+ gamma*(norm(W1*f([f_i_mirror]),1) + norm(W2*f([f_b_mirror]),1))/mean_g ... % smooth error
		);

		subject to
			f>=0;
	cvx_end

	W1 = getWaveletKernel(WAVETYPE, length(f_i), WAVEPAR);
	W2 = getWaveletKernel(WAVETYPE, length(f_b), WAVEPAR);
	sn = (norm(W1*f([f_i]),1) + norm(W2*f([f_b]),1))/mean_g;
	rn = square_pos(norm(H*f./g-1, 2));



elseif DECONV_KERNEL == DECONV_WAVELET && TRIAL == IB_SMOOTH && ~TURN_OFF && EDGE_EFFECTS == DOUBLE_MIRROR
	if ~SLIENCE 
		disp('I+B smooth, DOUBLE MIRROR');
	end

	Hsize = size(H, 2);

	f_i = [];
	for i = 1:length(model.i_intervals)
		idx = model.i_intervals{i}{2};
		se = model.Hpos{idx};
		f_i = [f_i se(1):1:se(2)];
	end

	f_i_last = f_i(end);

	f_b = [];
	for i = 1:length(model.b_intervals)
		idx = model.b_intervals{i}{2};
		se = model.Hpos{idx};
		f_b = [f_b se(1):1:se(2)];
	end

	f_b_first = f_b(1);

	idx = model.t_intervals{1}{2};
	se = model.Hpos{idx};

	f_t_first = se(1);

	f_final = zeros(Hsize,1);

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% right mirroring

	f_i_mirror = [f_i reverse(f_i)];
%	f_b_mirror = [f_b reverse(f_b)];
	f_b_mirror = [f_b];

	W1 = getWaveletKernel(WAVETYPE, length(f_i_mirror), WAVEPAR);
	W2 = getWaveletKernel(WAVETYPE, length(f_b_mirror), WAVEPAR);

	cvx_begin
		cvx_quiet(true);
%		cvx_precision high;

		variable f(Hsize);

		minimize(...
			square_pos(norm(H*f./g-1, 2)) ... % fit error
			+ gamma*(norm(W1*f([f_i_mirror]),1) + norm(W2*f([f_b_mirror]),1))/mean_g ... % smooth error
		);

		subject to
			f>=0;
%			f(f_t_first)+f(f_b_first) >= f(f_i_last);
	cvx_end

	f_final(f_i(1:end/2)) = f(f_i(1:end/2));
	f_final(f_b(1:end/2)) = f(f_b(1:end/2));

	%% left mirroring
	f_i_mirror = [reverse(f_i) f_i];
%	f_b_mirror = [reverse(f_b) f_b];
	f_b_mirror = [f_b];

	cvx_begin
		cvx_quiet(true);
%		cvx_precision high;

		variable f(Hsize);

		minimize(...
			square_pos(norm(H*f./g-1, 2)) ... % fit error
			+ gamma*(norm(W1*f([f_i_mirror]),1) + norm(W2*f([f_b_mirror]),1))/mean_g ... % smooth error
		);

		subject to
			f>=0;
%			f(f_t_first)+f(f_b_first) >= f(f_i_last);
	cvx_end

	f_final(f_i(end/2+1:end)) = f(f_i(end/2+1:end));
	f_final(f_b(end/2+1:end)) = f(f_b(end/2+1:end));
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	f = f_final;

	W1 = getWaveletKernel(WAVETYPE, length(f_i), WAVEPAR);
	W2 = getWaveletKernel(WAVETYPE, length(f_b), WAVEPAR);
	sn = ( norm(W1*f([f_i]),1) + norm(W2*f([f_b]),1) )/mean_g;
	rn = square_pos(norm(H*f./g-1, 2));

elseif DECONV_KERNEL == DECONV_WAVELET && TRIAL == IT_B_SMOOTH && ~TURN_OFF && EDGE_EFFECTS == MIRROR
	if ~SLIENCE 
		disp('IT+B smooth, MIRROR');
	end

%	H_mirror = [H fliplr(H)];

	Hsize = size(H, 2);

	f_i = [];
	for i = 1:length(model.i_intervals)
		idx = model.i_intervals{i}{2};
		se = model.Hpos{idx};
		f_i = [f_i se(1):1:se(2)];
	end
%	f_i_mirror = [f_i f_i+Hsize reverse(f_i)+Hsize reverse(f_i)];
	f_i_mirror = [f_i reverse(f_i)];

	f_b = [];
	for i = 1:length(model.b_intervals)
		idx = model.b_intervals{i}{2};
		se = model.Hpos{idx};
		f_b = [f_b se(1):1:se(2)];
	end
%	f_b_mirror = [f_b f_b+Hsize reverse(f_b)+Hsize reverse(f_b)];
	f_b_mirror = [f_b reverse(f_b)];

	W1 = getWaveletKernel(WAVETYPE, length(f_i_mirror), WAVEPAR);
	W2 = getWaveletKernel(WAVETYPE, length(f_b_mirror), WAVEPAR);

	cvx_begin
		cvx_quiet(true);
%		cvx_precision high;

		variable f(Hsize);

		minimize(...
			square_pos(norm(H*f./g-1, 2)) ... % fit error
			+ gamma*(norm(W1*f([f_i_mirror]),1) + norm(W2*f([f_b_mirror]),1))/mean_g ... % smooth error
		);

		subject to
			f>=0;
	cvx_end

	W1 = getWaveletKernel(WAVETYPE, length(f_i), WAVEPAR);
	W2 = getWaveletKernel(WAVETYPE, length(f_b), WAVEPAR);
	sn = (norm(W1*f([f_i]),1) + norm(W2*f([f_b]),1))/mean_g;
	rn = square_pos(norm(H*f./g-1, 2));


elseif DECONV_KERNEL == DECONV_WAVELET && TRIAL == IB_SMOOTH && ~TURN_OFF && EDGE_EFFECTS == MIRROR
	if ~SLIENCE 
		disp('I+B smooth, MIRROR');
	end

	Hsize = size(H, 2);

	f_i = [];
	for i = 1:length(model.i_intervals)
		idx = model.i_intervals{i}{2};
		se = model.Hpos{idx};
		f_i = [f_i se(1):1:se(2)];
	end
%	f_i_mirror = [f_i reverse(f_i)];
	f_i_mirror = [f_i f_i];

	f_b = [];
	for i = 1:length(model.b_intervals)
		idx = model.b_intervals{i}{2};
		se = model.Hpos{idx};
		f_b = [f_b se(1):1:se(2)];
	end
%	f_b_mirror = [reverse(f_b) f_b];
	f_b_mirror = [f_b f_b];

%	W1 = getWaveletKernel(WAVETYPE, length(f_i_mirror), WAVEPAR);
%	W2 = getWaveletKernel(WAVETYPE, length(f_b_mirror), WAVEPAR);
	W1 = getWaveletKernel(WAVETYPE, length(f_i), WAVEPAR);
	W2 = getWaveletKernel(WAVETYPE, length(f_b), WAVEPAR);
	wpad = zeros(length(f_i));
	W1 = [W1 wpad; wpad fliplr(W1)];
	wpad = zeros(length(f_b));
	W2 = [W2 wpad; wpad fliplr(W2)];

	cvx_begin
		cvx_quiet(true);
%		cvx_precision high;

		variable f(Hsize);

		minimize(...
			square_pos(norm(H*f./g-1, 2)) ... % fit error
			+ gamma*(norm(W1*f([f_i_mirror]),1) + norm(W2*f([f_b_mirror]),1))/mean_g ... % smooth error
		);

		subject to
			f>=0;
	cvx_end

	W1 = getWaveletKernel(WAVETYPE, length(f_i), WAVEPAR);
	W2 = getWaveletKernel(WAVETYPE, length(f_b), WAVEPAR);
	sn = (norm(W1*f([f_i]),1) + norm(W2*f([f_b]),1))/mean_g;
	rn = square_pos(norm(H*f./g-1, 2));

elseif DECONV_KERNEL == DECONV_WAVELET && TRIAL == IT_B_SMOOTH && ~TURN_OFF && EDGE_EFFECTS == MIRROR
	if ~SLIENCE 
		disp('IT+B smooth, MIRROR');
	end

%	H_mirror = [H fliplr(H)];

	Hsize = size(H, 2);

	f_i = [];
	for i = 1:length(model.i_intervals)
		idx = model.i_intervals{i}{2};
		se = model.Hpos{idx};
		f_i = [f_i se(1):1:se(2)];
	end
%	f_i_mirror = [f_i f_i+Hsize reverse(f_i)+Hsize reverse(f_i)];
	f_i_mirror = [f_i reverse(f_i)];

	f_b = [];
	for i = 1:length(model.b_intervals)
		idx = model.b_intervals{i}{2};
		se = model.Hpos{idx};
		f_b = [f_b se(1):1:se(2)];
	end
%	f_b_mirror = [f_b f_b+Hsize reverse(f_b)+Hsize reverse(f_b)];
	f_b_mirror = [f_b reverse(f_b)];

	W1 = getWaveletKernel(WAVETYPE, length(f_i_mirror), WAVEPAR);
	W2 = getWaveletKernel(WAVETYPE, length(f_b_mirror), WAVEPAR);

	cvx_begin
		cvx_quiet(true);
%		cvx_precision high;

		variable f(Hsize);

		minimize(...
			square_pos(norm(H*f./g-1, 2)) ... % fit error
			+ gamma*(norm(W1*f([f_i_mirror]),1) + norm(W2*f([f_b_mirror]),1))/mean_g ... % smooth error
		);

		subject to
			f>=0;
	cvx_end

	W1 = getWaveletKernel(WAVETYPE, length(f_i), WAVEPAR);
	W2 = getWaveletKernel(WAVETYPE, length(f_b), WAVEPAR);
	sn = (norm(W1*f([f_i]),1) + norm(W2*f([f_b]),1))/mean_g;
	rn = square_pos(norm(H*f./g-1, 2));


elseif DECONV_KERNEL == DECONV_WAVELET && TRIAL == IB_SMOOTH && ~TURN_OFF && EDGE_EFFECTS == SMALLPADDING
	if ~SLIENCE 
		disp(sprintf('I+B smooth, LESS PADDING (size = %d)', PADDINGSIZE));
	end

	Hsize = size(H, 2);

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
	
	f_b = [];
	f_b_front = last+[1:1:PADDINGSIZE];
	last = last+PADDINGSIZE;
	f_b_after = last+[1:1:PADDINGSIZE];
	last = last+PADDINGSIZE;
	f_b_pad = [];
	for i = 1:length(model.b_intervals)
		idx = model.b_intervals{i}{2};
		se = model.Hpos{idx};
		f_b = [f_b se(1):1:se(2)];
	end
	f_b_pad = [f_b_front f_b f_b_after];

	W1 = getWaveletKernel(WAVETYPE, length(f_i_pad), WAVEPAR);
	W2 = getWaveletKernel(WAVETYPE, length(f_b_pad), WAVEPAR);

	cvx_begin
		cvx_quiet(true);
%		cvx_precision high;

		variable f(Hsize*2);

		minimize(...
			square_pos(norm(H*f(1:Hsize)./g-1, 2)) ... % fit error
			+ gamma*(norm(W1*f([f_i_pad]),1) + norm(W2*f([f_b_pad]),1))/mean_g ... % smooth error
		);

		subject to
			f>=0;
	cvx_end

%	figure;
%	plot(f(f_i_pad));
%	figure;
%	plot(f(f_b_pad));


	sn = (norm(W1*f([f_i_pad]),1) + norm(W2*f([f_b_pad]),1))/mean_g;
	f = f(1:Hsize);
	rn = square_pos(norm(H*f./g-1, 2));

elseif DECONV_KERNEL == DECONV_WAVELET && TRIAL == IB_SMOOTH && ~TURN_OFF && EDGE_EFFECTS == PADDING
	if ~SLIENCE 
		disp('I+B smooth, PADDING');
	end

	Hsize = size(H, 2);

	f_i = [];
	f_i_front = [];
	f_i_after = [];
%	wf_i = [];
%	i_gap = [];
	for i = 1:length(model.i_intervals)
		idx = model.i_intervals{i}{2};
		se = model.Hpos{idx};
		f_i = [f_i se(1):1:se(2)];
		mid = (se(2)-se(1)+1)/2;
		f_i_front = [f_i_front se(1)+Hsize:1:se(1)+Hsize+mid];
		f_i_after = [f_i_after se(1)+Hsize+mid+1:1:Hsize+se(2)];
%		gap = model.iList{i}(2)-model.iList{i}(1);
%		i_gap = [i_gap gap];
%		wf_i = [wf_i ones(1, se(2)-se(1)+1)*gap];
	end
	f_i_end = f_i(end);
	f_i_pad = [f_i_front f_i f_i_after];
	
	% want to know the first position of top-branch
	idx = model.t_intervals{1}{2};
	se = model.Hpos{idx};
	f_t_first = se(1);

	f_b = [];
	f_b_front = [];
	f_b_after = [];
	f_b_pad = [];
%	wf_b = [];
%	b_gap = [];
	for i = 1:length(model.b_intervals)
		idx = model.b_intervals{i}{2};
		se = model.Hpos{idx};
		f_b = [f_b se(1):1:se(2)];
		mid = (se(2)-se(1)+1)/2;
		f_b_front = [f_b_front se(1)+Hsize:1:se(1)+mid+Hsize];
		f_b_after = [f_b_after se(1)+Hsize+mid+1:1:se(2)+Hsize];
%		gap = model.bList{i}(2)-model.bList{i}(1);
%		b_gap = [b_gap gap];
%		wf_b = [wf_b ones(1, se(2)-se(1)+1)*gap];
	end
	f_b_first = f_b(1);
	f_b_pad = [f_b_front f_b f_b_after];

	W1 = getWaveletKernel(WAVETYPE, length(f_i_pad), WAVEPAR);
	W2 = getWaveletKernel(WAVETYPE, length(f_b_pad), WAVEPAR);
%	W1 = addWeight(W1, wf_i);
%	W2 = addWeight(W2, wf_b);

%	mom_ratio = mom2daughter_ratio/(mom2daughter_ratio+1);
%	daughter_ratio = 1-mom_ratio;

%	disp(sprintf('last i = %0.3g, 1st t = %0.3g, 1st b = %0.3g', f_i_end, f_t_first, f_b_first));

	cvx_begin
		cvx_quiet(true);
%		cvx_precision high;

		variable f(Hsize*2);

		minimize(...
			square_pos(norm(H*f(1:Hsize)./g-1, 2)) ... % fit error
			+ gamma*(norm(W1*f([f_i_pad]),1) + norm(W2*f([f_b_pad]),1))/mean_g ... % smooth error
		);

		subject to
			f>=0;
%			f(f_t_first)+f(f_b_first) >= 2*f(f_i_end);
	%		f(f_i_end) == mom_ratio*f(f_t_first)+daughter_ratio*f(f_b_first);
	cvx_end

	f = f(1:Hsize);
	W1 = getWaveletKernel(WAVETYPE, length(f_i), WAVEPAR);
	W2 = getWaveletKernel(WAVETYPE, length(f_b), WAVEPAR);
	sn = (norm(W1*f([f_i]),1) + norm(W2*f([f_b]),1))/mean_g;
	rn = square_pos(norm(H*f./g-1, 2));

%	sn
%	norm(f(f_t_first)+f(f_b_first)-f(f_i_end),1)

elseif DECONV_KERNEL == DECONV_WAVELET && TRIAL == IB_SMOOTH && TURN_OFF
	if ~SLIENCE
		disp('I+B smooth, turn_off');
	end

	turn_off_name = 'D';
	if ~SLIENCE
		disp(sprintf('turn off %s', turn_off_name));
	end

	seg_idx = -1;
	for idx = 1:length(model.relations)
		name = model.relations{idx}{1};
		if strcmp(name, turn_off_name)
			branch = model.relations{idx}{2};
			branch_idx = model.relations{idx}{3};
			branch_idx = str2num(branch_idx)+1;

			if strcmp(branch, 'i')
				seg_idx = model.i_intervals{branch_idx}{2};
			elseif strcmp(branch, 'b')
				seg_idx = model.b_intervals{branch_idx}{2};
%			elseif strcmp(branch, 't')
%				seg_idx = model.t_intervals{branch_idx}{2};
			end

			seg = model.Hpos{seg_idx};
			break;
		end
	end

	f_i = [];
	wf_i = [];
	for i = 1:length(model.i_intervals)
		idx = model.i_intervals{i}{2};
		if idx ~= seg_idx
			se = model.Hpos{idx};
			if se(1) > seg(end)
				offset = seg(end)-seg(1)+1;
			else
				offset = 0;
			end
			f_i = [f_i se(1)-offset:se(2)-offset];
			gap = model.iList{i}(2)-model.iList{i}(1);
			wf_i = [wf_i ones(1, se(2)-se(1)+1)*gap];
		end
	end

	f_b = [];
	wf_b = [];
	for i = 1:length(model.b_intervals)
		idx = model.b_intervals{i}{2};
		if idx ~= seg_idx
			se = model.Hpos{idx};
			if se(1) > seg(end)
				offset = seg(end)-seg(1)+1;
			else
				offset = 0;
			end
			f_b = [f_b se(1)-offset:se(2)-offset];
			gap = model.bList{i}(2)-model.bList{i}(1);
			wf_b = [wf_b ones(1, se(2)-se(1)+1)*gap];
		end
	end

	H_t = [H(:,1:seg(1)-1) H(:,seg(2)+1:end)];
	Hsize = size(H_t, 2);

	W1 = getWaveletKernel(WAVETYPE, length(f_i), WAVEPAR);
	W2 = getWaveletKernel(WAVETYPE, length(f_b), WAVEPAR);
	W1 = addWeight(W1, wf_i);
	W2 = addWeight(W2, wf_b);

	cvx_begin
		cvx_quiet(true);
%		cvx_precision high;

		variable f(Hsize);

		minimize(...
			square_pos(norm(H_t*f./g-1, 2)) ... % fit error
			+ gamma*(...
			norm(W1*f([f_i]),1)...
			+...
			norm(W2*f([f_b]),1)...
			)/mean_g ... % smooth error
		);

		subject to
			f>=0;
	cvx_end

	sn = gamma*(norm(W1*f([f_i]),1) + norm(W2*f([f_b]),1))/mean_g;
	rn = square_pos(norm(H_t*f./g-1, 2));

	f_add = zeros(seg(2)-seg(1)+1,1);
	f = [f(1:seg(1)-1)' f_add' f(seg(1):end)']';

%%%%%%%%%%%%%%%%%%%%%%%%

% USING, I
elseif DECONV_KERNEL == DECONV_WAVELET && TRIAL == I_SMOOTH && ~TURN_OFF
	if ~SLIENCE
		disp('I smooth');
	end

	f_i = [];
	wf_i = [];
	for i = 1:length(model.i_intervals)
		idx = model.i_intervals{i}{2};
		se = model.Hpos{idx};
		f_i = [f_i se(1):1:se(2)];
		gap = model.iList{i}(2)-model.iList{i}(1);
		wf_i = [wf_i ones(1, se(2)-se(1)+1)*gap];
	end

	Hsize = size(H, 2);
	W1 = getWaveletKernel(WAVETYPE, length(f_i), WAVEPAR);

	W1 = addWeight(W1, wf_i);

	cvx_begin
		cvx_quiet(true);
%		cvx_precision high;

		variable f(Hsize);

		minimize(...
			square_pos(norm(H*f./g-1, 2)) ... % fit error
			+ gamma*( ...
				norm(W1*f([f_i]),1) ...
			)/mean_g ... % smooth error
		);

		subject to
			f>=0;
	cvx_end

	sn = (norm(W1*f([f_i]),1))/mean_g;
	rn = square_pos(norm(H*f./g-1, 2));

% USING, IBT
elseif DECONV_KERNEL == DECONV_WAVELET && TRIAL == IBT_SMOOTH && ~TURN_OFF
	if ~SLIENCE
		disp('I+B+T smooth');
	end

	f_i = [];
	wf_i = [];
	for i = 1:length(model.i_intervals)
		idx = model.i_intervals{i}{2};
		se = model.Hpos{idx};
		f_i = [f_i se(1):1:se(2)];
		gap = model.iList{i}(2)-model.iList{i}(1);
		wf_i = [wf_i ones(1, se(2)-se(1)+1)*gap];
	end

	f_b = [];
	wf_b = [];
	for i = 1:length(model.b_intervals)
		idx = model.b_intervals{i}{2};
		se = model.Hpos{idx};
		f_b = [f_b se(1):1:se(2)];
		gap = model.bList{i}(2)-model.bList{i}(1);
		wf_b = [wf_b ones(1, se(2)-se(1)+1).*gap];
	end

	f_t = [];
	wf_t = [];
	for i = 1:length(model.t_intervals)
		idx = model.t_intervals{i}{2};
		se = model.Hpos{idx};
		f_t = [f_t se(1):1:se(2)];
		gap = model.tList{i}(2)-model.tList{i}(1);
		wf_t = [wf_t ones(1, se(2)-se(1)+1).*gap];
	end

	Hsize = size(H, 2);
	W1 = getWaveletKernel(WAVETYPE, length(f_i), WAVEPAR);
	W2 = getWaveletKernel(WAVETYPE, length(f_b), WAVEPAR);
	W3 = getWaveletKernel(WAVETYPE, length(f_t), WAVEPAR);

	W1 = addWeight(W1, wf_i);
	W2 = addWeight(W2, wf_b);
	W3 = addWeight(W3, wf_t);

	cvx_begin
		cvx_quiet(true);
%		cvx_precision high;

		variable f(Hsize);


		minimize(...
			square_pos(norm(H*f./g-1, 2)) ... % fit error
			+ gamma*( ...
				norm(W1*f([f_i]),1) + ...
				norm(W2*f([f_b]),1) + ...
				norm(W3*f([f_t]),1)...
			)/mean_g ... % smooth error
		);

		subject to
			f>=0;

	cvx_end

	sn = (norm(W1*f([f_i]),1) + norm(W2*f([f_b]),1) + norm(W3*f([f_t]),1))/mean_g;
	rn = square_pos(norm(H*f./g-1, 2));

% USING, ibt_sparse model
elseif DECONV_KERNEL == DECONV_WAVELET && TRIAL == IBT_RCG1_SPARSE_CONDITION && ~TURN_OFF
	if ~SLIENCE
		disp('I+B+T smooth, RG1-CG1 sparse');
	end

	f_i = [];
	wf_i = [];
	f_RG1 = [];
	for i = 1:length(model.i_intervals)
		idx = model.i_intervals{i}{2};
		se = model.Hpos{idx};
		f_i = [f_i se(1):1:se(2)];
		gap = model.iList{i}(2)-model.iList{i}(1);
		wf_i = [wf_i ones(1, se(2)-se(1)+1)*gap];
		% find R
		name = model.i_intervals{i}{1};
		if strcmp(name, 'RG1')
			idx = model.i_intervals{i}{2};
			se = model.Hpos{idx};
			f_RG1 = [se(1):1:se(2)];
		end
	end

	f_b = [];
	wf_b = [];
	for i = 1:length(model.b_intervals)
		idx = model.b_intervals{i}{2};
		se = model.Hpos{idx};
		f_b = [f_b se(1):1:se(2)];
		gap = model.bList{i}(2)-model.bList{i}(1);
		wf_b = [wf_b ones(1, se(2)-se(1)+1).*gap];
	end

	f_t = [];
	wf_t = [];
	for i = 1:length(model.t_intervals)
		idx = model.t_intervals{i}{2};
		se = model.Hpos{idx};
		f_t = [f_t se(1):1:se(2)];
		gap = model.tList{i}(2)-model.tList{i}(1);
		wf_t = [wf_t ones(1, se(2)-se(1)+1).*gap];
		
		name = model.t_intervals{i}{1};
		if strcmp(name, 'CG1')
			idx = model.t_intervals{i}{2};
			se = model.Hpos{idx};
			f_CG1 = [se(1):1:se(2)];
		end
	end

	Hsize = size(H, 2);
	W1 = getWaveletKernel(WAVETYPE, length(f_i), WAVEPAR);
	W2 = getWaveletKernel(WAVETYPE, length(f_b), WAVEPAR);
	W3 = getWaveletKernel(WAVETYPE, length(f_t), WAVEPAR);

	W1 = addWeight(W1, wf_i);
	W2 = addWeight(W2, wf_b);
	W3 = addWeight(W3, wf_t);

	ratio = 1/length(f_RG1);

	cvx_begin
		cvx_quiet(true);
%		cvx_precision high;

		variable f(Hsize);

		minimize(...
			square_pos(norm(H*f./g-1, 2)) ... % fit error
			+ gamma*( ...
				norm(W1*f([f_i]), 1) + ...
				norm(W2*f([f_b]), 1) + ...
				norm(W3*f([f_t]), 1) + ...
				ratio*norm(f([f_RG1])-f([f_CG1]), 1) ...
			)/mean_g ... % smooth error
		);

		subject to
			f>=0;
			f([f_RG1]) >= f([f_CG1]);
	cvx_end

	sn = (norm(W1*f([f_i]),1) ...
		+ norm(W2*f([f_b]),1) ...
		+ norm(W3*f([f_t])) ...
		+ ratio*norm(f([f_RG1])-f([f_CG1]), 1)...
		)/mean_g;
	rn = square_pos(norm(H*f./g-1, 2));

% USING, SCG1DG1 model
elseif DECONV_KERNEL == DECONV_WAVELET && TRIAL == IB_CDG1_SPARSE_CONDITION && ~TURN_OFF
	if ~SLIENCE
		disp('I+B smooth, DG1-CG1 sparse');
	end

	f_i = [];
	wf_i = [];
	f_CG1 = [];
	for i = 1:length(model.i_intervals)
		idx = model.i_intervals{i}{2};
		se = model.Hpos{idx};
		f_i = [f_i se(1):1:se(2)];
		gap = model.iList{i}(2)-model.iList{i}(1);
		wf_i = [wf_i ones(1, se(2)-se(1)+1)*gap];

		name = model.i_intervals{i}{1};
		if strcmp(name, 'CG1')
			idx = model.i_intervals{i}{2};
			se = model.Hpos{idx};
			f_CG1 = [se(1):1:se(2)];
		end
	end

	f_b = [];
	wf_b = [];
	f_DG1 = [];
	for i = 1:length(model.b_intervals)
		idx = model.b_intervals{i}{2};
		se = model.Hpos{idx};
		f_b = [f_b se(1):1:se(2)];
		gap = model.bList{i}(2)-model.bList{i}(1);
		wf_b = [wf_b ones(1, se(2)-se(1)+1).*gap];
		
		name = model.b_intervals{i}{1};
		if strcmp(name, 'DG1')
			idx = model.b_intervals{i}{2};
			se = model.Hpos{idx};
			f_DG1 = [se(1):1:se(2)];
		end
	end

	Hsize = size(H, 2);
	W1 = getWaveletKernel(WAVETYPE, length(f_i), WAVEPAR);
	W2 = getWaveletKernel(WAVETYPE, length(f_b), WAVEPAR);

	W1 = addWeight(W1, wf_i);
	W2 = addWeight(W2, wf_b);

	ratio = 1/length(f_i);

	cvx_begin
		cvx_quiet(true);
%		cvx_precision high;

		variable f(Hsize);

		minimize(...
			square_pos(norm(H*f./g-1, 2)) ... % fit error
			+ gamma*( ...
				norm(W1*f([f_i]), 1) + ...
				norm(W2*f([f_b]), 1) + ...
				ratio*norm(f([f_DG1])-f([f_CG1]), 1) ...
			)/mean_g ... % smooth error
		);

		subject to
			f>=0;
			f([f_DG1]) >= f([f_CG1]);
	cvx_end

	sn = (norm(W1*f([f_i]),1) ...
		+ norm(W2*f([f_b]),1) ...
		+ ratio*norm(f([f_DG1])-f([f_CG1]), 1)...
		)/mean_g;
	rn = square_pos(norm(H*f./g-1, 2));

% USING, SDG1R model
elseif DECONV_KERNEL == DECONV_WAVELET && TRIAL == IBT_RDG1_SPARSE_CONDITION && ~TURN_OFF
	if ~SLIENCE
		disp('I+B+T smooth, RG1-DG1 sparse');
	end

	f_i = [];
	wf_i = [];
	f_RG1 = [];
	for i = 1:length(model.i_intervals)
		idx = model.i_intervals{i}{2};
		se = model.Hpos{idx};
		f_i = [f_i se(1):1:se(2)];
		gap = model.iList{i}(2)-model.iList{i}(1);
		wf_i = [wf_i ones(1, se(2)-se(1)+1)*gap];
		% find RG1
		name = model.i_intervals{i}{1};
		if strcmp(name, 'RG1')
			idx = model.i_intervals{i}{2};
			se = model.Hpos{idx};
			f_RG1 = [se(1):1:se(2)];
		end
	end

	f_b = [];
	wf_b = [];
	f_DG1 = [];
	for i = 1:length(model.b_intervals)
		idx = model.b_intervals{i}{2};
		se = model.Hpos{idx};
		f_b = [f_b se(1):1:se(2)];
		gap = model.bList{i}(2)-model.bList{i}(1);
		wf_b = [wf_b ones(1, se(2)-se(1)+1).*gap];
		
		name = model.b_intervals{i}{1};
		if strcmp(name, 'DG1')
			idx = model.b_intervals{i}{2};
			se = model.Hpos{idx};
			f_DG1 = [se(1):1:se(2)];
		end
	end

	f_t = [];
	wf_t = [];
	for i = 1:length(model.t_intervals)
		idx = model.t_intervals{i}{2};
		se = model.Hpos{idx};
		f_t = [f_t se(1):1:se(2)];
		gap = model.tList{i}(2)-model.tList{i}(1);
		wf_t = [wf_t ones(1, se(2)-se(1)+1).*gap];
		
	end

	Hsize = size(H, 2);
	W1 = getWaveletKernel(WAVETYPE, length(f_i), WAVEPAR);
	W2 = getWaveletKernel(WAVETYPE, length(f_b), WAVEPAR);
	W3 = getWaveletKernel(WAVETYPE, length(f_t), WAVEPAR);

	W1 = addWeight(W1, wf_i);
	W2 = addWeight(W2, wf_b);
	W3 = addWeight(W3, wf_t);

	ratio = 1/length(f_RG1);

	cvx_begin
		cvx_quiet(true);
%		cvx_precision high;

		variable f(Hsize);

		minimize(...
			square_pos(norm(H*f./g-1, 2)) ... % fit error
			+ gamma*( ...
				norm(W1*f([f_i]), 1) + ...
				norm(W2*f([f_b]), 1) + ...
				norm(W3*f([f_t]), 1) + ...
				norm(f([f_RG1])-f([f_DG1]), 1) ...
			)/mean_g ... % smooth error
		);

		subject to
			f>=0;
			f([f_RG1]) >= f([f_DG1]);

	cvx_end

	sn = (norm(W1*f([f_i]),1) ...
		+ norm(W2*f([f_b]),1) ...
		+ norm(W3*f([f_t])) ...
		+ ratio*norm(f([f_RG1])-f([f_DG1]), 1)...
		)/mean_g;
	rn = square_pos(norm(H*f./g-1, 2));

elseif DECONV_KERNEL == DECONV_WAVELET && TRIAL == 2

	f_i = [];
	for i = 1:length(model.i_intervals)
		idx = model.i_intervals{i}{2};
		f_i = [f_i ((idx-1)*len+1):1:idx*len];
	end

	f_b = [];
	for i = 1:length(model.b_intervals)
		idx = model.b_intervals{i}{2};
		f_b = [f_b ((idx-1)*len+1):1:idx*len];
	end

	Hsize = size(H, 2);

	W1 = getWaveletKernel(length(f_i), DAUB);
	W2 = getWaveletKernel(length(f_b), DAUB);

	cvx_begin
		cvx_quiet(true);
		cvx_precision high;

		variable f(Hsize);

		minimize(...
			square_pos(norm(H*f./g-1, 2)) ... % fit error
			+ gamma*(norm(W1*f([f_i]), 1) + norm(W2*f([f_b]), 1)) ... % smooth error
		);

		subject to
			f>=0;

	cvx_end

	sn = (norm(W1*f([f_i]),1) + norm(W2*f([f_b]),1));
	rn = square_pos(norm(H*f./g-1, 2));
	sn0 = sum(W1*f([f_i])>SMALL) + sum(W2*f([f_b])>SMALL);
	sn0all = length(f_i) + length(f_b);


elseif DECONV_KERNEL == DECONV_WAVELET && TRIAL == IBT_SMOOTH && TURN_OFF
	if ~SLIENCE
		disp('I+B+T smooth');
	end

	turn_off_name = 'CG1';
	if ~SLIENCE
		disp(sprintf('turn off %s', turn_off_name));
	end

	seg_idx = -1;
	for idx = 1:length(model.relations)
		name = model.relations{idx}{1};
		if strcmp(name, turn_off_name)
			branch = model.relations{idx}{2};
			branch_idx = model.relations{idx}{3};
			branch_idx = str2num(branch_idx)+1;

			if strcmp(branch, 'i')
				seg_idx = model.i_intervals{branch_idx}{2};
			elseif strcmp(branch, 'b')
				seg_idx = model.b_intervals{branch_idx}{2};
			elseif strcmp(branch, 't')
				seg_idx = model.t_intervals{branch_idx}{2};
			end

			seg = model.Hpos{seg_idx};
			break;
		end
	end

	f_i = [];
	wf_i = [];
	for i = 1:length(model.i_intervals)
		idx = model.i_intervals{i}{2};
		if idx ~= seg_idx
			se = model.Hpos{idx};
			if se(1) > seg(end)
				offset = seg(end)-seg(1)+1;
			else
				offset = 0;
			end
			f_i = [f_i se(1)-offset:se(2)-offset];
			gap = model.iList{i}(2)-model.iList{i}(1);
			wf_i = [wf_i ones(1, se(2)-se(1)+1)*gap];
		end
	end

	f_b = [];
	wf_b = [];
	for i = 1:length(model.b_intervals)
		idx = model.b_intervals{i}{2};
		if idx ~= seg_idx
			se = model.Hpos{idx};
			if se(1) > seg(end)
				offset = seg(end)-seg(1)+1;
			else
				offset = 0;
			end
			f_b = [f_b se(1)-offset:1:se(2)-offset];
			gap = model.bList{i}(2)-model.bList{i}(1);
			wf_b = [wf_b ones(1, se(2)-se(1)+1).*gap];
		end
	end

	f_t = [];
	wf_t = [];
	for i = 1:length(model.t_intervals)
		idx = model.t_intervals{i}{2};
		if idx ~= seg_idx
			se = model.Hpos{idx};
			if se(1) > seg(end)
				offset = seg(end)-seg(1)+1;
			else
				offset = 0;
			end
			f_t = [f_t se(1)-offset:1:se(2)-offset];
			gap = model.tList{i}(2)-model.tList{i}(1);
			wf_t = [wf_t ones(1, se(2)-se(1)+1).*gap];
		end
	end

	H_t = [H(:,1:seg(1)-1) H(:, seg(2)+1:end)];
	Hsize = size(H_t, 2);

	W1 = getWaveletKernel(WAVETYPE, length(f_i), WAVEPAR);
	W2 = getWaveletKernel(WAVETYPE, length(f_b), WAVEPAR);
	W3 = getWaveletKernel(WAVETYPE, length(f_t), WAVEPAR);

	W1 = addWeight(W1, wf_i);
	W2 = addWeight(W2, wf_b);
	W3 = addWeight(W3, wf_t);

	cvx_begin
		cvx_quiet(true);
%		cvx_precision high;

		variable f(Hsize);

		minimize(...
			square_pos(norm(H_t*f./g-1, 2)) ... % fit error
			+ gamma*( ...
				norm(W1*f([f_i]),1) + ...
				norm(W2*f([f_b]),1) + ...
				norm(W3*f([f_t]), 1)...
			)/mean_g ... % smooth error
		);

		subject to
			f>=0;
	cvx_end

	sn = (norm(W1*f([f_i]),1) + norm(W2*f([f_b]),1) + norm(W3*f([f_t])))/mean_g;
	rn = square_pos(norm(H_t*f./g-1, 2));

	f_add = zeros(seg(2)-seg(1)+1,1);
	f = [f(1:seg(1)-1)' f_add' f(seg(1):end)']';


% trial 2: 
elseif DECONV_KERNEL == DECONV_DIFF && TRIAL == 1
	f_i = [];
	for i = 1:length(model.i_intervals)
		idx = model.i_intervals{i}{2};
		len = size(model.Hsegments{idx}, 2);
		start = 0;
		for b = 1:idx-1
			start = start+size(model.Hsegments{b}, 2);
		end
		f_i = [f_i (start+1):1:(start+len)];
	end

	f_b = [];
	for i = 1:length(model.b_intervals)
		idx = model.b_intervals{i}{2};
		len = size(model.Hsegments{idx}, 2);
		start = 0;
		for b = 1:idx-1
			start = start+size(model.Hsegments{b}, 2);
		end
		f_b = [f_b (start+1):1:(start+len)];
	end

	Hsize = size(H, 2);

	W1 = getDiffKernel(length(f_i));
	W2 = getDiffKernel(length(f_b));

	cvx_begin
		cvx_quiet(true);
		cvx_precision high;

		variable f(Hsize);
		minimize(...
			square_pos(norm(H*f./g-1, 2)) ... % fit error
			+ gamma*(norm(W1*f([f_i]), 2) + norm(W2*f([f_b]), 2))/mean(model.g) ... % smooth error (sn)
		);
		
		subject to
			f>=0;
	cvx_end

	% calculate sn, rn, sn0
	rn = square_pos(norm(H*f./g-1, 2));
	sn = (norm(W1*f([f_i]), 2) + norm(W2*f([f_b]), 2))/mean(model.g);
	sn0 = sum(W1*f([f_i])>SMALL) + sum(W2*f([f_b])>SMALL);
	sn0all = length(f([f_i])) + length(f([f_b]));

% trial 3: 
elseif DECONV_KERNEL == DECONV_DIFF && TRIAL == 2
	f_i = [];
	for i = 1:length(model.i_intervals)
		idx = model.i_intervals{i}{2};
		len = size(model.Hsegments{idx}, 2);
		start = 0;
		for b = 1:idx-1
			start = start+size(model.Hsegments{b}, 2);
		end
		f_i = [f_i (start+1):1:(start+len)];
	end

	f_b = [];
	for i = 1:length(model.b_intervals)
		idx = model.b_intervals{i}{2};
		len = size(model.Hsegments{idx}, 2);
		start = 0;
		for b = 1:idx-1
			start = start+size(model.Hsegments{b}, 2);
		end
		f_b = [f_b (start+1):1:(start+len)];
	end

	Hsize = size(H, 2);

	gmean = mean(g);

	gmod = g - gmean;

	W1 = getDiffKernel(length(f_i));
	W2 = getDiffKernel(length(f_b));

	cvx_begin
		cvx_quiet(true);
		cvx_precision high;

		variable fmod(Hsize);
		minimize(...
			square_pos(norm(H*fmod./gmod-1, 2)) ... % fit error
			+ gamma*(norm(W1*fmod([f_i]), 2) + norm(W2*fmod([f_b]), 2))/gmean ... % smooth error (sn)
		);
		
		subject to
			fmod > -gmean;
	cvx_end

	% calculate sn, rn, sn0
	rn = square_pos(norm(H*fmod./g-1, 2));
	sn = (norm(W1*fmod([f_i]), 2) + norm(W2*fmod([f_b]), 2))/gmean;
	sn0 = sum(W1*fmod([f_i])>SMALL) + sum(W2*fmod([f_b])>SMALL);
	sn0all = length(fmod([f_i])) + length(fmod([f_b]));

	f = fmod + gmean;
elseif DECONV_KERNEL == DECONV_WAVELET && TRIAL == 4
	% basis pursuit
	f_i = [];
	for i = 1:length(model.i_intervals)
		idx = model.i_intervals{i}{2};
		f_i = [f_i ((idx-1)*len+1):1:idx*len];
	end

	f_b = [];
	for i = 1:length(model.b_intervals)
		idx = model.b_intervals{i}{2};
		f_b = [f_b ((idx-1)*len+1):1:idx*len];
	end


	Hsize = size(H, 2);
	D = getWaveletKernel(WAVETYPE, Hsize, WAVEPAR);

	cvx_begin
		cvx_quiet(true);

		variable c(size(D,2));

		minimize(...
			square_pos(norm(H*D*c./g-1, 2)) ...
			+ gamma*(norm(c([f_i]), 1) + norm(c([f_b]), 1)) ... % smooth error
		);

		subject to
			D*c>0;

	cvx_end

	sn = norm(c([f_i]),1) + norm(c([f_b]),1);
	rn = square_pos(norm(H*D*c./g-1, 2));
end

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
	R2_wt1 = rsquare(g1, pred_g1);
	R2_wt2 = rsquare(g2, pred_g2);
	model.R2 = [R2_wt1 R2_wt2];
else
	pred_g = model.H*model.f;
	R2 = rsquare(model.g, pred_g);
	model.R2 = [R2];
end

model.pred_g = pred_g;

SLIENCE = old_slience;
return;
