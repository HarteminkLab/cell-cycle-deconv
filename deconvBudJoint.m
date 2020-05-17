function [model] = deconvBudJoint(dataset, datatype, gamma, learn)

%%% start: some header information %%%
global DECONV_MU0POS;			% mu0
global DECONV_LAMBDAPOS;	% lambda
global DECONV_DELTAPOS;		% delta
global DECONV_ALPHAPOS;		% alpha
global DECONV_SIGMA0POS;	% sigma_0
global DECONV_SIGMAVPOS;	% sigma_v
global DECONV_BETAPOS;		% beta

global DECONV_JOINT;
global DECONV_WT1;
global DECONV_WT2;

global DECONV_KERNEL;
global DECONV_WAVELET;
global DECONV_DIFF;

global DECONV_DATASET;
global DATASET_OLD;
global DATASET_NEW;

global WAVETYPE;
global WAVEPAR;

global MODEL;

global NORMALIZING;

NORMALIZING = 1;

% ===================

DATASET_OLD = '../datasets/';
DATASET_NEW = '../datasets.new/';
DECONV_DATASET = DATASET_NEW;

MODEL_DIR = './models/';

MODEL = 1;

% ===================
%%% end: some header information %%%

DECONV_MU0POS = 1;
DECONV_LAMBDAPOS = 2;
DECONV_DELTAPOS = 3;
DECONV_SIGMA0POS = 4;
DECONV_SIGMAVPOS = 5;
DECONV_ALPHAPOS = 6;
DECONV_BETAPOS = 7;

DECONV_WAVELET = 1;
DECONV_DIFF = 2;

DECONV_WT1 = 'WT1';
DECONV_WT2 = 'WT2';
DECONV_JOINT = 'JOINT';

DATA_FLOW = 'FLOW';
DATA_BUD = 'BUD';
DATA_BOTH = 'BOTH';

OUTPUT_DIR = './deconv_bud_index/';

if ~exist(OUTPUT_DIR, 'dir')
	mkdir(OUTPUT_DIR);
end

%%% start: some settings %%%
dataset = upper(dataset);
if strcmp(dataset, DATA_BUD)
	WT1_folder = 'wt1_bud/';
	WT2_folder = 'wt2_bud/';
elseif strcmp(dataset, DATA_FLOW)
	WT1_folder = 'wt1_flow/';
	WT2_folder = 'wt2_flow/';
elseif strcmp(dataset, DATA_BOTH)
	WT1_folder = 'wt1_budflow/';
	WT2_folder = 'wt2_budflow/';
end


% 1: wavelet (uneven intervals)
% 2: 1st-order differential operator (even intervals)
DECONV_KERNEL = DECONV_WAVELET;

if nargin < 4
	error('... expect 4 inputs: dataset, datatype, gamma, and learn_flag');
end;

%WAVETYPE = 'Daubechies';
%WAVEPAR = 8;
WAVETYPE = 'Haar';
WAVEPAR = 2;
%WAVETYPE = 'Symmlet';
%WAVEPAR = 5;

default_gamma = 1e-2;

if ~exist('gamma', 'var')
	gamma = default_gamma;
end

datatype = upper(datatype);
modeltype = 'BUD';
if MODEL == 1
	modelprefix = 'BUD_MODEL';
elseif MODEL == 2
	modelprefix = 'BUD_MODEL2';
end

modellabel = strcat(modelprefix, '.0.label');

%%% end: some settings %%%

model = {};
model.datatype = datatype;
model.orfname = 'BUD';
model.orig_orfname = 'BUD';
model.orfid = -1;
model.modeltype = modeltype;
model.gm = gamma;

% output files
f_out = sprintf('%sbudres.f.%s.txt', OUTPUT_DIR, datatype);
pred_g_out = sprintf('%sbudres.predg.%s.txt', OUTPUT_DIR, datatype);

if strcmp(datatype, DECONV_WT1)
	buddata = load(strcat(DECONV_DATASET, 'bud_real_wt1.txt'));
	model.g = buddata(:,2);
	model.timepoints = buddata(:,1)';
	model.alpha = [0];

	modelfile = strcat(MODEL_DIR, WT1_folder, modellabel);

	[flag, model] = readModelFormat(modelfile, model);
	if flag == 0
		error('Error in parsing model file');
	end;

	model.H = [];
	[model] = calcH(model);

	if learn == 0;
		[model] = deconvBudModelRes(model);
	else
		[model] = findOptimalBud(model, 1);
	end
	f = [model.f'];
	pred_g = [(model.H*model.f)'];
	R2_wt1 = rsquare(model.g, pred_g);
	model.R2 = [R2_wt1];

	save(f_out, 'f', '-ASCII', '-DOUBLE', '-TABS');
	save(pred_g_out, 'pred_g', '-ASCII', '-DOUBLE', '-TABS');
elseif strcmp(datatype, DECONV_WT2)
	buddata = load(strcat(DECONV_DATASET, 'bud_real_wt2.txt'));
	model.g = buddata(:,2);
	model.timepoints = buddata(:,1)';
	model.alpha = [0];

	modelfile = strcat(MODEL_DIR, WT2_folder, modellabel);

	[flag, model] = readModelFormat(modelfile, model);
	if flag == 0
		error('Error in parsing model file');
	end;

	model.H = [];
	[model] = calcH(model);

	if learn == 0
		[model] = deconvBudModelRes(model);
	else
		[model] = findOptimalBud(model, 1);
	end
	f = [model.f'];
	pred_g = [(model.H*model.f)'];
	R2_wt2 = rsquare(model.g, pred_g);
	model.R2 = [R2_wt2];

	save(f_out, 'f', '-ASCII', '-DOUBLE', '-TABS');
	save(pred_g_out, 'pred_g', '-ASCII', '-DOUBLE', '-TABS');

elseif strcmp(datatype, DECONV_JOINT)
	model.alpha = [0 0];

	% WT1
	buddata1 = load(strcat(DECONV_DATASET, 'bud_real_wt1.txt'));
	g1 = buddata1(:,2);
	model.timepoints = buddata1(:,1)';

	modelfile1 = strcat(MODEL_DIR, WT1_folder, modellabel);

	[flag, model] = readModelFormat(modelfile1, model);
	if flag == 0
		error('Error in parsing model file');
	end;

	model.H = [];
	[model] = calcH(model);

	% WT2
	buddata2 = load(strcat(DECONV_DATASET, 'bud_real_wt2.txt'));
	g2 = buddata2(:,2);
	model.g = [g1' g2']';
	model.timepoints = [model.timepoints' buddata2(:,1)]';

	modelfile2 = strcat(MODEL_DIR, WT2_folder, modellabel);

	[flag, model] = readModelFormat(modelfile2, model);
	if flag == 0
		error('Error in parsing model file');
	end;

	[model] = calcH(model);

	if learn == 0
		[model] = deconvBudModelRes(model);
	else
		[model] = findOptimalBud(model, 0);
	end
	f = [model.f'];
	pred_g = [(model.H*model.f)'];

	R2_wt1 = rsquare(g1, pred_g(1:length(g1)));
	R2_wt2 = rsquare(g2, pred_g(length(g1)+1:end));
	model.R2 = [R2_wt1 R2_wt2];

	save(f_out, 'f', '-ASCII', '-DOUBLE', '-TABS');
	save(pred_g_out, 'pred_g', '-ASCII', '-DOUBLE', '-TABS');
end

% expected f
beta = model.lengths(DECONV_BETAPOS);
lambda = model.lengths(DECONV_LAMBDAPOS);

Hsize = size(model.H, 2);
f_expected = zeros(Hsize,1);
f_expected(1:64) = 1;
model.f_expected = f_expected;

% plot the optimal model we can estimate
R2 = rsquare(model.f, model.f_expected);
disp(sprintf('f vs. expected_f: R^2 is %f\n', R2));

%plotOptimal_general_parts(model, 1);
plotOptimal_general_branch_area(model, 1);
if MODEL == 1
	plotBudRes(model, model.f);
end

if strcmp(datatype, DECONV_JOINT)
	glen = length(model.g);
	best_g = model.H*model.f_expected;
	g1 = model.g(1:glen/2);
	g2 = model.g(glen/2+1:glen);
	best_g1 = best_g(1:glen/2);
	best_g2 = best_g(glen/2+1:glen);

	R2_wt1 = rsquare(g1, best_g1);
	R2_wt2 = rsquare(g2, best_g2);

	disp(sprintf('Principle R^2: wt1=%f, wt2=%f\n', R2_wt1, R2_wt2));
	disp(sprintf('Estimated R^2: wt1=%f, wt2=%f\n', model.R2(1), model.R2(2)));
else
	best_g = model.H*model.f_expected;
	R2_wt = rsquare(model.g, best_g);
	
	disp(sprintf('Principle R^2=%f\n', R2_wt));
	disp(sprintf('Estimated R^2=%f\n', model.R2(1)));
end



% ========================================
% ========================================

function [model] = plotBudRes(model, f)
'here'

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
global DECONV_WT1;
global DECONV_WT2;

global WAVETYPE;
global WAVEPAR;
global MODEL;

mu0 = model.lengths(DECONV_MU0POS);
lambda = model.lengths(DECONV_LAMBDAPOS);
delta = model.lengths(DECONV_DELTAPOS);
alpha = model.lengths(DECONV_ALPHAPOS);
sigma0 = model.lengths(DECONV_SIGMA0POS);
sigmav = model.lengths(DECONV_SIGMAVPOS);
beta = model.lengths(DECONV_BETAPOS);

orfname = model.orfname;

if strcmp(model.datatype, DECONV_WT1)
	s = 0.349;
	beta = 0.153;
elseif strcmp(model.datatype, DECONV_WT2)
	s = 0.391;
	beta = 0.165;
else
	beta = (0.153+0.165)/2;
	s = (0.349+0.391)/2;
end

% get intervals and labels from model structure

idx = model.i_intervals{1}{2};
se = model.Hpos{idx};
f_R = [se(1):1:se(2)];

idx = model.t_intervals{1}{2};
se = model.Hpos{idx};
f_C = [se(1):1:se(2)];
%
idx = model.t_intervals{2}{2};
se = model.Hpos{idx};
f_C = [f_C se(1):1:se(2)];

idx = model.b_intervals{1}{2};
se = model.Hpos{idx};
f_D = [se(1):1:se(2)];
%
idx = model.b_intervals{2}{2};
se = model.Hpos{idx};
f_D = [f_D se(1):1:se(2)];

y_R = f([f_R]);
y_C = f([f_C]);
y_D = f([f_D]);

x_R = [model.iList{1}(1:96)+mu0 ...
];

xticks_R = [0 ...
	mu0/2 mu0 ...
];

x_C = [model.tList{1}(1:32) ...
	model.tList{2}(1:65) ...
];

xticks_C = [0 ...
	beta*lambda/2 beta*lambda ...
	(s+beta)/2*lambda s*lambda ...
	(1+s)/2*lambda lambda ...
];

x_D = [model.bList{1}(1:64)+delta ...
	model.bList{2}(1:64)+delta ...
];

xticks_D = [0 ...
	delta/2 delta ...
	delta+beta*lambda/2 delta+beta*lambda ...
	delta+(beta*lambda)+lambda*(s-beta)/2 delta+s*lambda ...
	delta+(1+s)/2*lambda delta+lambda ...
];

xticks_R_label = {'|', 'R', '|'};
xticks_C_label = {'|', 'G1', '|', 'S', '|', 'G2/M', '|'};
xticks_D_label = {'|', 'DG1', '|', 'G1', '|', 'S', '|', 'G2/M', '|'};

% drawing
figure;

subplot(3, 1, 1);
plot(x_R, y_R, 'color', [217 150 148]/255, 'LineWidth', 2);
set(gca, 'XTick', xticks_R);
set(gca, 'XTickLabel', xticks_R_label);
ylim([0 1.03]);
xlim([0 mu0]);
box off;

subplot(3, 1, 2);
plot(x_C, [y_C' 0], 'color', [147 205 221]./255, 'LineWidth', 2);
set(gca, 'XTick', xticks_C);
set(gca, 'XTickLabel', xticks_C_label);
ylim([0 1.03]);
xlim([0 lambda]);
box off;

subplot(3, 1, 3);
mid = ceil(64*delta/(delta+beta*lambda));
plot(x_D(1:mid), y_D(1:mid), 'color', [250 192 144]./255, 'LineWidth', 2);
hold on;
plot(x_D(mid:end), y_D(mid:end), 'color', [147 205 221]./255, 'LineWidth', 2);
set(gca, 'XTick', xticks_D);
set(gca, 'XTickLabel', xticks_D_label);
ylim([0 1.03]);
xlim([0 delta+lambda]);
box off;

return;

% ====================================
% ====================================

function [model] = deconvBudModelRes(model)

global DECONV_KERNEL;
global DECONV_WAVELET;
global DECONV_DIFF;
global WAVETYPE;
global WAVEPAR;

global DECONV_JOINT;
global DECONV_WT1;
global DECONV_WT2;
global MODEL;

SMALL = 1e-8;

optarg = 1;
if nargin < optarg
	error('Needs (model, col(optional) ) in deconvBudModelRes function');
end

g = model.g;
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
% require: [R G1] and [S/G2/M] and [DG1] are smooth

% wavelet

if DECONV_KERNEL == DECONV_WAVELET && MODEL == 1
	% [R G1]
	f_iG1 = [];
	idx = model.i_intervals{1}{2};
	se = model.Hpos{idx};
	f_iG1 = [se(1):1:se(2)];
	idx = model.i_intervals{2}{2};
	se = model.Hpos{idx};
	f_iG1 = [f_iG1 se(1):1:se(2)];

	% [DG1]
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

	disp('f_iG1, f_bG1 and f_postG1 version');

	cvx_begin
		cvx_quiet(true);
	%	cvx_precision high;

		variable f(Hsize);

		minimize(...
			square_pos(norm(H*f-g, 2)) ... % fit error
			+ gamma*( norm(W1*f([f_iG1]), 1) + norm(W2*f([f_bG1]), 1) + norm(W3*f([f_postG1]), 1) )/mean(model.g) ... % smooth error
		);

		subject to
			f >= 0;
			f <= 1;
	cvx_end

	rn = square_pos(norm(H*f-g, 2));
	sn = gamma*( norm(W1*f([f_iG1]), 1) + norm(W2*f([f_bG1]), 1) + norm(W3*f([f_postG1]), 1) )/mean(model.g);
	sn_iG1 = gamma*( norm(W1*f([f_iG1]), 1) )/mean(model.g);
	sn_bG1 = gamma*( norm(W2*f([f_bG1]), 1) )/mean(model.g);
	sn_postG1 = gamma*( norm(W3*f([f_postG1]), 1) )/mean(model.g);
	disp(sprintf('rn = %f, sn = %f', rn, sn));
	disp(sprintf('sn_iG1 = %f, sn_bG1 = %f, sn_postG1 = %f', sn_iG1, sn_bG1, sn_postG1));


	% ..........................
	% sn: solution norm
	% rn: residual norm
	% ..........................
	pred_g = H*f;

	model.f = f;
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

elseif DECONV_KERNEL == DECONV_WAVELET && MODEL == 2
	% [R C]
	f_i = [];
	wf_i = [];
	for i = 1:length(model.i_intervals)
		idx = model.i_intervals{i}{2};
		se = model.Hpos{idx};
		f_i = [f_i se(1):1:se(2)];
		gap = model.iList{i}(2) - model.iList{i}(1);
		wf_i = [wf_i ones(1, se(2)-se(1)+1)*gap];
	end

	% [DG1 C]
	f_b = [];
	wf_b = [];
	for i = 1:length(model.b_intervals)
		idx = model.b_intervals{i}{2};
		se = model.Hpos{idx};
		f_b = [f_b se(1):1:se(2)];
		gap = model.bList{i}(2) - model.bList{i}(1);
		wf_b = [wf_b ones(1, se(2)-se(1)+1)*gap];
	end

	Hsize = size(H, 2);

	W1 = getWaveletKernel(WAVETYPE, length(f_i), WAVEPAR);
	W2 = getWaveletKernel(WAVETYPE, length(f_b), WAVEPAR);
	W1 = addWeight(W1, wf_i);
	W2 = addWeight(W2, wf_b);

	disp('f_i, and f_b version');

	cvx_begin
		cvx_quiet(true);
	%	cvx_precision high;

		variable f(Hsize);

		minimize(...
			square_pos(norm(H*f-g, 2)) ... % fit error
			+ gamma*( norm(W1*f([f_i]), 1) + norm(W2*f([f_b]), 1))/mean(model.g) ... % smooth error
		);

		subject to
			f >= 0;
			f <= 1;
	cvx_end

	rn = square_pos(norm(H*f-g, 2));
	sn = gamma*(norm(W1*f([f_i]), 1) + norm(W2*f([f_b]), 1))/mean(model.g);
	disp(sprintf('rn = %f, sn = %f', rn, sn));

	% ..........................
	% sn: solution norm
	% rn: residual norm
	% ..........................
	pred_g = H*f;

	model.f = f;
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

end

return;
