function [model] = deconvCLOCCS(orfname, modeltype, alpha, gamma)

global SLIENCE;

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

global DECONV_DATASET;
global DATASET_OLD;
global DATASET_NEW;

DATASET_OLD = '../datasets/';
DATASET_NEW = '../datasets.new/';
DECONV_DATASET = DATASET_NEW;


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

SLIENCE = 0;

SHOW_DETAILS = 0;

DATA_WT1 = strcat(DECONV_DATASET, 'wt1.txt');
DATA_WT2 = strcat(DECONV_DATASET, 'wt2.txt');

WT1_TP = 30:16:254;
WT2_TP = 38:16:262;

modeldir1 = './budres/WT1/';
modeldir2 = './budres/WT2/';
runs = 100;

DECONV_WT1 = 'WT1';
DECONV_WT2 = 'WT2';
DECONV_JOINT = 'JOINT';

model = {};

if DECONV_KERNEL == DECONV_WAVELET
	MODEL_DIR = 'models/';
elseif DECONV_KERNEL == DECONV_DIFF
	MODEL_DIR = 'models_even/';
end

% deal with orfname and datatype
orig_orfname = orfname;
[orfname, orfid] = map2SystemNames(orig_orfname);
model.orig_orfname = orig_orfname;
model.orfname = orfname;
model.orfid = orfid;

model.datatype = DECONV_JOINT;
model.alpha = alpha;

output_f = strcat(orig_orfname, '.100runs.f');
f_fid = fopen(output_f, 'w');

% deal with modeltype
model.modeltype = upper(modeltype);
[model] = parseModelType(model);
modelprefix = model.modelprefix;

% get model.g
ymax = 0;
if length(orfid) == 0
	error('I guess your orfname is wrong.');
else
	dataset = load(DATA_WT1, 'ascii');
	g1 = dataset(orfid,:)';
	dataset = load(DATA_WT2, 'ascii');
	g2 = dataset(orfid,:)';
end
model.g = [g1' g2']';
model.gm = gamma;

% start runs
ALLF = [];
for r = 1:runs
	model.H = [];
	disp(sprintf('runs %d', r));

	% ==========================================
	% WT1
	modelfile = strcat(modeldir1, modelprefix, '.', int2str(model.alpha(1)), '.', int2str(r), '.label');

	[flag, model] = readModelFormat(modelfile, model);
	if flag == 0
		error('Error in parsing model file');
	end

	if SHOW_DETAILS
		mu0_1 = model.lengths(DECONV_MU0POS);
		lambda_1 = model.lengths(DECONV_LAMBDAPOS);
		delta_1 = model.lengths(DECONV_DELTAPOS);
		alpha_1 = model.alpha(1);
		beta_1 = model.lengths(DECONV_BETAPOS);

		disp(sprintf('\twt1 %s', modelfile));
		disp(sprintf('\tWT1, mu_0=%0.3f, lambda=%0.3f, delta=%0.3f, beta=%0.3f, alpha=%d', mu0_1, lambda_1, delta_1, beta_1, alpha_1));
	end

	% calculate H
	model.timepoints = WT1_TP;
	[model] = calcH(model);

	% ==========================================
	% WT2
	modelfile = strcat(modeldir2, modelprefix, '.', int2str(model.alpha(2)), '.', int2str(r), '.label');

	[flag, model] = readModelFormat(modelfile, model);
	if flag == 0
		error('Error in parsing model file');
	end

	if SHOW_DETAILS
		mu0_2 = model.lengths(DECONV_MU0POS);
		lambda_2 = model.lengths(DECONV_LAMBDAPOS);
		delta_2 = model.lengths(DECONV_DELTAPOS);
		alpha_2 = model.alpha(2);
		beta_2 = model.lengths(DECONV_BETAPOS);

		disp(sprintf('\twt2 %s',modelfile));
		disp(sprintf('\tWT2, mu_0=%0.3f, lambda=%0.3f, delta=%0.3f, beta=%0.3f, alpha=%d', mu0_2, lambda_2, delta_2, beta_2, alpha_2));
	end

	% calculate H
	model.timepoints = [model.timepoints' WT2_TP']';
	[model] = calcH(model);

	% ==========================================
	% DECONV
	[model] = deconvModel(model);

	y = model.f;
	ALLF = [ALLF; y];

	fprintf(f_fid, '%0.5g\t', y(1:end-1));
	fprintf(f_fid, '%0.5g\n', y(end));
end

fclose(f_fid);
