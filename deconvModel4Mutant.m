function [model, pred_g] = deconvModel4Mutant(model)

global SLIENCE;

global DECONV_MU0POS;
global DECONV_LAMBDAPOS;
global DECONV_ALPHAPOS;

global WAVETYPE;
global WAVEPAR;

global DECONV_JOINT;
global DECONV_DATA1;
global DECONV_DATA2;

global NORMALIZING;
global DATASET_MT;
global DATASET_AF;

%%%%%%%%%%%%%%%%%%%%%%%
% settings
NORMALIZING = 0;
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
alpha = model.lengths(DECONV_ALPHAPOS);
alpha_avg = mean(alpha);

SMALL = 1e-6;

optarg = 1;
if nargin < optarg
	error('Needs (model) in deconvModel4Mutant function');
end

g = model.g;
mean_g = mean(g);
H = model.H;
gamma = model.gm;

% model structure so far
% model:
%		- lengths: array
%		- relations: cell array
%		- i/t List: cell arrays
%		- i/t_intervals: string arrays
%		- Hsegments: cell arrays
%		- timepoints: array
% require: i is smooth

% wavelet; actually using
len = size(model.Hsegments{1}, 2);

if ~SLIENCE
	disp('f_t, f_i smooth model');
end

f_i = [];
for i = 1:length(model.i_intervals)
	idx = model.i_intervals{i}{2};
	se = model.Hpos{idx};
	f_i = [f_i se(1):1:se(2)];
	gap = model.iList{i}(2)-model.iList{i}(1);
end

f_t = [];
for i = 1:length(model.t_intervals)
	idx = model.t_intervals{i}{2};
	se = model.Hpos{idx};
	f_t = [f_t se(1):1:se(2)];
	gap = model.tList{i}(2)-model.tList{i}(1);
end

Hsize = size(H, 2);
f_final = zeros(Hsize, 1);

% --------------------------------------------------
TEST = 2;

if ~SLIENCE
	disp(sprintf('DECONV KENERL METHOD = %d', TEST));
end

if TEST == 1
	% right mirroring
	f_i_mirror = [f_i reverse(f_i)];
	f_t_mirror = [f_t reverse(f_t)];

	W1 = getWaveletKernel(WAVETYPE, length(f_i_mirror), WAVEPAR);
	W2 = getWaveletKernel(WAVETYPE, length(f_t_mirror), WAVEPAR);

	cvx_begin
		cvx_quiet(true);
	%		cvx_precision high;

		variable f(Hsize);

		minimize(...
			square_pos(norm(H*f./g-1, 2)) ... % fit error
			+ gamma*(norm(W1*f(f_i_mirror),1) + norm(W2*f(f_t_mirror),1))/mean_g ... % smooth error
		);

		subject to
			f>=0;
	cvx_end

	f_final(f_i(1:end/2)) = f(f_i(1:end/2));
	f_final(f_t(1:end/2)) = f(f_t(1:end/2));

	% left mirroring
	f_i_mirror = [reverse(f_i) f_i];
	f_t_mirror = [reverse(f_t) f_t];

	W1 = getWaveletKernel(WAVETYPE, length(f_i_mirror), WAVEPAR);
	W2 = getWaveletKernel(WAVETYPE, length(f_t_mirror), WAVEPAR);

	cvx_begin
		cvx_quiet(true);
	%		cvx_precision high;

		variable f(Hsize);

		minimize(...
			square_pos(norm(H*f./g-1, 2)) ... % fit error
			+ gamma*(norm(W1*f(f_i_mirror),1) + norm(W2*f(f_t_mirror),1))/mean_g ... % smooth error
		);

		subject to
			f>=0;
	cvx_end

	f_final(f_i(end/2+1:end)) = f(f_i(end/2+1:end));
	f_final(f_t(end/2+1:end)) = f(f_t(end/2+1:end));

	% --------------------------------------------------

	f = f_final;

	W1 = getWaveletKernel(WAVETYPE, length(f_i), WAVEPAR);
	W2 = getWaveletKernel(WAVETYPE, length(f_t), WAVEPAR);

	sn = (norm(W1*f([f_i]),1) + norm(W2*f([f_t])))/mean_g;
	rn = square_pos(norm(H*f./g-1, 2));

elseif TEST == 2
	% right mirroring
	weight = (mu0+lambda-alpha_avg)/lambda;
	%weight = 1;
	f_i_mirror = [f_i reverse(f_i)];
	f_t_mirror = [f_t f_t];

	W1 = getWaveletKernel(WAVETYPE, length(f_i_mirror), WAVEPAR);
	W2 = getWaveletKernel(WAVETYPE, length(f_t_mirror), WAVEPAR);

	cvx_begin
		cvx_quiet(true);
	%		cvx_precision high;

		variable f(Hsize);

		minimize(...
			square_pos(norm(H*f./g-1, 2)) ... % fit error
			+ gamma*(norm(W1*f(f_i_mirror),1) + weight*norm(W2*f(f_t_mirror),1))/mean_g ... % smooth error
		);

		subject to
			f>=0;
	cvx_end

	f_final(f_i(1:end/2)) = f(f_i(1:end/2));
	f_t_1 = f(f_t);

	% left mirroring
	f_i_mirror = [reverse(f_i) f_i];
	f_t_mirror = [f_t f_t];

	cvx_begin
		cvx_quiet(true);
	%		cvx_precision high;

		variable f(Hsize);

		minimize(...
			square_pos(norm(H*f./g-1, 2)) ... % fit error
			+ gamma*(norm(W1*f(f_i_mirror),1) + weight*norm(W2*f(f_t_mirror),1))/mean_g ... % smooth error
		);

		subject to
			f>=0;
	cvx_end

	f_final(f_i(end/2+1:end)) = f(f_i(end/2+1:end));
	f_t_2 = f(f_t);

	c = corr2(f_t_1, f_t_2);

	f_final(f_t) = (f_t_1+f_t_2)/2;

	% --------------------------------------------------

	f = f_final;

	W1 = getWaveletKernel(WAVETYPE, length(f_i), WAVEPAR);
	W2 = getWaveletKernel(WAVETYPE, length(f_t), WAVEPAR);

	sn = (norm(W1*f([f_i]),1) + weight * norm(W2*f([f_t])))/mean_g;
	rn = square_pos(norm(H*f./g-1, 2));

elseif TEST == 3
	% right mirroring
	weight = (mu0+lambda-alpha_avg)/lambda;
	f_i_size = 2^ceil(log2(numel(f_i)));
	f_i_pad = zeros(1, f_i_size-numel(f_i));

	f_i_mirror = [f_i reverse(f_i)];
	f_t_mirror = [f_t f_t];

	W1 = getWaveletKernel(WAVETYPE, f_i_size*2, WAVEPAR);
	W2 = getWaveletKernel(WAVETYPE, length(f_t_mirror), WAVEPAR);


	cvx_begin
		cvx_quiet(true);
	%		cvx_precision high;

		variable f(Hsize);

		minimize(...
			square_pos(norm(H*f./g-1, 2)) ... % fit error
			+ gamma*(norm(W1*[f_i_pad f(f_i_mirror)' f_i_pad]', 1) + weight*norm(W2*f(f_t_mirror),1))/mean_g ... % smooth error
		);

		subject to
			f>=0;
	cvx_end

	f_final(f_i(1:end/2)) = f(f_i(1:end/2));
	f_t_1 = f(f_t);

	% left mirroring
	f_i_mirror = [reverse(f_i) f_i];
	f_t_mirror = [f_t f_t];

	cvx_begin
		cvx_quiet(true);
	%		cvx_precision high;

		variable f(Hsize);

		minimize(...
			square_pos(norm(H*f./g-1, 2)) ... % fit error
			+ gamma*(norm(W1*[f_i_pad f(f_i_mirror)' f_i_pad]',1) + weight*norm(W2*f(f_t_mirror),1))/mean_g ... % smooth error
		);

		subject to
			f>=0;
	cvx_end

	f_final(f_i(end/2+1:end)) = f(f_i(end/2+1:end));
	f_t_2 = f(f_t);

	c = corr2(f_t_1, f_t_2);

	f_final(f_t) = (f_t_1+f_t_2)/2;

	% --------------------------------------------------

	f = f_final;

	W1 = getWaveletKernel(WAVETYPE, f_i_size, WAVEPAR);
	W2 = getWaveletKernel(WAVETYPE, length(f_t), WAVEPAR);

	sn = (norm(W1*[f_i_pad(1:end/2) f(f_i)' f_i_pad(end/2+1:end)]',1) + weight * norm(W2*f([f_t])))/mean_g;
	rn = square_pos(norm(H*f./g-1, 2));

	weight


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
if strcmp(model.datatype, DECONV_JOINT)
	pred_g = model.H*model.f;
	g1 = model.g(1:length(model.timepoints{1}));
	g2 = model.g(length(model.timepoints{1})+1:end);
	pred_g1 = pred_g(1:length(model.timepoints{1}));
	pred_g2 = pred_g(length(model.timepoints{1})+1:end);

	tm1 = model.timepoints{1};
	tm2 = model.timepoints{2};
	R2_wt1 = rsquare(g1, pred_g1);
	R2_wt2 = rsquare(g2, pred_g2);
	model.R2 = [R2_wt1 R2_wt2];
else
	pred_g = model.H*model.f;
	R2 = rsquare(model.g, pred_g);
	model.R2 = [R2];
end

model.pred_g = pred_g;
