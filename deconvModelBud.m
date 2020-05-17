function [model, pred_g] = deconvModelBud(model, col)

global DECONV_KERNEL;
global DECONV_WAVELET;
global DECONV_DIFF;

global WAVETYPE;
global WAVEPAR;

SMALL = 1e-6;

TRIAL = 1;

optarg = 1;
if nargin < optarg
	error('Needs (model, col(optional) ) in deconvModel function');
end

g = model.g;
H = model.H;
gamma = model.gm;

if exist ('col')
	if col < 1 || col > length(model.g)
		error('col is out of boundary');
	end
	g(col,:) = [];
	H(col,:) = [];
end

% model structure so far
% model:
%		- lengths: array
%		- relations: cell array
%		- i/t/b List: cell arrays
%		- i/t/b_intervals: string arrays
%		- Hsegments: cell arrays
%		- timepoints: array
% require: [R G1] and [S/G2/M] and [DG1 G1] are smooth

% wavelet
% actually using this
if DECONV_KERNEL == DECONV_WAVELET && TRIAL == 1
	% [R G1]
	f_iG1 = [];
  idx = model.i_intervals{1}{2};
  se = model.Hpos{idx};
  f_iG1 = [se(1):1:se(2)];
  idx = model.i_intervals{2}{2};
  se = model.Hpos{idx};
  f_iG1 = [f_iG1 se(1):1:se(2)];

	% [DG1 G1]
	f_bG1 = [];
	idx = model.b_intervals{1}{2};
	se = model.Hpos{idx};
	f_bG1 = [se(1):1:se(2)];

	% [S G2/M]
	f_postG1 = [];
	idx = model.t_intervals{2}{2};
	se = model.Hpos{idx};
	f_postG1 = [se(1):1:se(2)];

	Hsize = size(H, 2);

	W1 = getWaveletKernel(WAVETYPE, length(f_iG1), WAVEPAR);
	W2 = getWaveletKernel(WAVETYPE, length(f_bG1), WAVEPAR);
	W3 = getWaveletKernel(WAVETYPE, length(f_postG1), WAVEPAR);

	cvx_begin
		cvx_quiet(true);
%		cvx_precision high;

		variable f(Hsize);

		minimize(...
			square_pos(norm(H*f-g, 2)) ... % fit error
			+ gamma*( norm(W1*f([f_iG1]), 1) + norm(W2*f([f_bG1]), 1) + norm(W3*f([f_postG1]), 1) )/mean(model.g) ... % smooth error
		);

		subject to
			f>=0;
			f<=1;
	cvx_end
	
	rn = square_pos(norm(H*f-g, 2));
	sn = gamma*( norm(W1*f([f_iG1]), 1) + norm(W2*f([f_bG1]), 1) + norm(W3*f([f_postG1]), 1) )/mean(model.g);
	sn_iG1 = gamma*( norm(W1*f([f_iG1]), 1) )/mean(model.g);
	sn_bG1 = gamma*( norm(W2*f([f_bG1]), 1) )/mean(model.g);
	sn_postG1 = gamma*( norm(W3*f([f_postG1]), 1) )/mean(model.g);
end

% ..........................
% sn: solution norm
% rn: residual norm
% ..........................
if exist('col')
	H = model.H;
	pred_g = H(col,:)*f;
else
	pred_g = H*f;
end

model.f = f;
model.sn = sn;
model.rn = rn;
model.err = sqrt(model.rn/length(model.g));

disp(sprintf('fit error = %0.5g (%0.5g%%), smooth error = %0.5g, gm = %0.5g', model.rn, model.err*100, model.sn, model.gm));
