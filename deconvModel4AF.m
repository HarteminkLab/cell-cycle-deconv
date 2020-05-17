function [model, pred_g] = deconvModel4AF(model)

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

[H_r, H_c] = size(H);

%if ~SLIENCE
%	disp(sprintf('H size: row = %d, column = %d', H_r, H_c));
%end

len_f_i = length(f_i);
len_f_t = length(f_t);

%if ~SLIENCE
%	disp(sprintf('f_i = %d, f_t = %d', len_f_i, len_f_t));
%end

% --------------------------------------------------

TEST = 6;
%if ~SLIENCE
%	disp(sprintf('DECONV KENERL METHOD = %d', TEST));
%end

switch TEST
case 6
	% right mirroring
%	if ~SLIENCE
%		disp('double mirroring, f_i smooth only, H truncated');
%	end

	t_a = 1;
	t_b = 0;

	if strcmp(model.datatype, DECONV_JOINT)
		l1 = length(model.timepoints{1});
		l2 = length(model.timepoints{2});

		H_t = H([t_a:l1-t_b l1+t_a:end-t_b], :);
		g_t = g([t_a:l1-t_b l1+t_a:end-t_b]);
	else
		H_t = H(t_a:end-t_b,:);
		g_t = g(t_a:end-t_b);
	end

	[H_r, H_c] = size(H_t);

%	if ~SLIENCE
%		disp(sprintf('H truncated size: row = %d, column = %d', H_r, H_c));
%	end

	f_i_mirror = [f_i reverse(f_i)];

	W1 = getWaveletKernel(WAVETYPE, length(f_i_mirror), WAVEPAR);

	cvx_begin
		cvx_quiet(true);
	%	cvx_precision high;

		variable f(Hsize);

		minimize(...
			square_pos(norm(H_t*f./g_t-1, 2)) ... % fit error
			+ gamma*(norm(W1*f(f_i_mirror),1))/mean_g ... % smooth error
		);

		subject to
			f>=0;
	cvx_end

	f_final(f_i(1:end/2)) = f(f_i(1:end/2));

	% left mirroring
	f_i_mirror = [reverse(f_i) f_i];

	W1 = getWaveletKernel(WAVETYPE, length(f_i_mirror), WAVEPAR);

	cvx_begin
		cvx_quiet(true);
	%	cvx_precision high;

		variable f(Hsize);

		minimize(...
			square_pos(norm(H_t*f./g_t-1, 2)) ... % fit error
			+ gamma*(norm(W1*f(f_i_mirror),1))/mean_g ... % smooth error
		);

		subject to
			f>=0;
	cvx_end

	f_final(f_i(end/2+1:end)) = f(f_i(end/2+1:end));

	% --------------------------------------------------

	f = f_final;

	W1 = getWaveletKernel(WAVETYPE, length(f_i), WAVEPAR);

	sn = (norm(W1*f([f_i]),1))/mean_g;
	rn = square_pos(norm(H*f./g-1, 2));

case 8
	if ~SLIENCE
		disp('double mirroring, f_r, f_t smooth');
	end

	f_r = setxor(f_i, f_t);

	% right mirroring
	f_r_mirror = [f_r reverse(f_r)];
	f_t_mirror = [f_t reverse(f_t)];

	W1 = getWaveletKernel(WAVETYPE, length(f_r_mirror), WAVEPAR);
	W2 = getWaveletKernel(WAVETYPE, length(f_t_mirror), WAVEPAR);

	cvx_begin
		cvx_quiet(true);
	%	cvx_precision high;

		variable f(Hsize);

		minimize(...
			square_pos(norm(H*f./g-1, 2)) ... % fit error
			+ gamma*(norm(W1*f(f_r_mirror),1)+norm(W2*f(f_t_mirror),1))/mean_g ... % smooth error
		);

		subject to
			f>=0;
	cvx_end

	f_final(f_r(1:end/2)) = f(f_r(1:end/2));
	f_final(f_t(1:end/2)) = f(f_t(1:end/2));

	% left mirroring
	f_r_mirror = [reverse(f_r) f_r];
	f_t_mirror = [reverse(f_t) f_t];

	cvx_begin
		cvx_quiet(true);
	%	cvx_precision high;

		variable f(Hsize);

		minimize(...
			square_pos(norm(H*f./g-1, 2)) ... % fit error
			+ gamma*(norm(W1*f(f_r_mirror),1)+norm(W2*f(f_t_mirror),1))/mean_g ... % smooth error
		);

		subject to
			f>=0;
	cvx_end

	f_final(f_r(end/2+1:end)) = f(f_r(end/2+1:end));
	f_final(f_t(end/2+1:end)) = f(f_t(end/2+1:end));

	% --------------------------------------------------

	f = f_final;

	W1 = getWaveletKernel(WAVETYPE, length(f_r), WAVEPAR);
	W2 = getWaveletKernel(WAVETYPE, length(f_t), WAVEPAR);

	sn = (norm(W1*f([f_r]),1)+norm(W2*f([f_t]),1))/mean_g;
	rn = square_pos(norm(H*f./g-1, 2));

case 1
	% right mirroring
	
%	disp('double mirroring, f_i smooth');
	f_i_mirror = [f_i reverse(f_i)];

	W1 = getWaveletKernel(WAVETYPE, length(f_i_mirror), WAVEPAR);

	cvx_begin
		cvx_quiet(true);
	%	cvx_precision high;

		variable f(Hsize);

		minimize(...
			square_pos(norm(H*f./g-1, 2)) ... % fit error
			+ gamma*(norm(W1*f(f_i_mirror),1))/mean_g ... % smooth error
		);

		subject to
			f>=0;
	cvx_end

	f_final(f_i(1:end/2)) = f(f_i(1:end/2));

	% left mirroring
	f_i_mirror = [reverse(f_i) f_i];

	W1 = getWaveletKernel(WAVETYPE, length(f_i_mirror), WAVEPAR);

	cvx_begin
		cvx_quiet(true);
	%	cvx_precision high;

		variable f(Hsize);

		minimize(...
			square_pos(norm(H*f./g-1, 2)) ... % fit error
			+ gamma*(norm(W1*f(f_i_mirror),1))/mean_g ... % smooth error
		);

		subject to
			f>=0;
	cvx_end

	f_final(f_i(end/2+1:end)) = f(f_i(end/2+1:end));

	% --------------------------------------------------

	f = f_final;

	W1 = getWaveletKernel(WAVETYPE, length(f_i), WAVEPAR);

	sn = (norm(W1*f([f_i]),1))/mean_g;
	rn = square_pos(norm(H*f./g-1, 2));

case 5
	% right mirroring
	disp('left_right mirroring, f_i smooth');

	f_i_left = reverse(f_i(1:end/2));
	f_i_right = reverse(f_i(end/2+1:end));

	f_i_pad = [f_i_left f_i f_i_right];

	W1 = getWaveletKernel(WAVETYPE, length(f_i_pad), WAVEPAR);

	cvx_begin
		cvx_quiet(true);
	%	cvx_precision high;

		variable f(Hsize);

		minimize(...
			square_pos(norm(H*f./g-1, 2)) ... % fit error
			+ gamma*(norm(W1*f([f_i_pad]),1))/mean_g ... % smooth error
		);

		subject to
			f>=0;
	cvx_end

	W1 = getWaveletKernel(WAVETYPE, length(f_i), WAVEPAR);

	sn = (norm(W1*f([f_i]),1))/mean_g;
	rn = square_pos(norm(H*f./g-1, 2));

case 4
	disp('no mirroring');
	W1 = getWaveletKernel(WAVETYPE, length(f_i), WAVEPAR);

	cvx_begin
		cvx_quiet(true);
	%	cvx_precision high;

		variable f(Hsize);

		minimize(...
			square_pos(norm(H*f./g-1, 2)) ... % fit error
			+ gamma*(norm(W1*f(f_i),1))/mean_g ... % smooth error
		);

		subject to
			f>=0;
	cvx_end

	% --------------------------------------------------

	W1 = getWaveletKernel(WAVETYPE, length(f_i), WAVEPAR);

	sn = (norm(W1*f([f_i]),1))/mean_g;
	rn = square_pos(norm(H*f./g-1, 2));

case 7
	disp('no mirroring, f_t smooth');
	W1 = getWaveletKernel(WAVETYPE, length(f_t), WAVEPAR);

	cvx_begin
		cvx_quiet(true);
	%	cvx_precision high;

		variable f(Hsize);

		minimize(...
			square_pos(norm(H*f./g-1, 2)) ... % fit error
			+ gamma*(norm(W1*f(f_t),1))/mean_g ... % smooth error
		);

		subject to
			f>=0;
	cvx_end

	% --------------------------------------------------

	sn = (norm(W1*f([f_t]),1))/mean_g;
	rn = square_pos(norm(H*f./g-1, 2));

case 2
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

case 3
	disp('f_i, f_t smooth, add weight');
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
