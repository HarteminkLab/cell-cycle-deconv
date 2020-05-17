function [model] = deconv_with_error(orfname, gamma, noise_lv)

% =====================
alpha = [26 27];
modeltype = '1.1.1';
% =====================

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

WT1_DIR = 'wt1_budflow/';
WT2_DIR = 'wt2_budflow/';

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

output_f = strcat(orig_orfname, '.error', int2str(noise_lv), '.f');
noise_lv = noise_lv/100;
display(output_f);
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
	% WT1
	dataset = load(DATA_WT1, 'ascii');
	g1 = dataset(orfid,:)';
	model.timepoints = WT1_TP;

	modelfile = strcat(MODEL_DIR, WT1_DIR, modelprefix, '.', int2str(alpha(1)), '.label');
	[flag, model] = readModelFormat(modelfile, model);
	if flag == 0
		error('Error in parsing model file');
	end

	[model] = calcH(model);

	% WT2
	dataset = load(DATA_WT2, 'ascii');
	g2 = dataset(orfid,:)';

	modelfile = strcat(MODEL_DIR, WT2_DIR, modelprefix, '.', int2str(alpha(2)), '.label');
	[flag, model] = readModelFormat(modelfile, model);
	if flag == 0
		error('Error in parsing model file');
	end
	model.timepoints = [model.timepoints' WT2_TP']';

	[model] = calcH(model);

	model.g1 = g1;
	model.g2 = g2;
end
model.gm = gamma;

% start runs
ALLF = [];
for r = 1:runs
	disp(sprintf('runs %d', r));

	g1 = model.g1;
	noise_ratio = normrnd(1, noise_lv, [length(g1) 1]);
	g1_error = max(1, g1.*noise_ratio);

	g2 = model.g2;
	noise_ratio = normrnd(1, noise_lv, [length(g2) 1]);
	g2_error = max(1, g2.*noise_ratio);

	model.g = [g1_error' g2_error']';

	% DECONV
	[model] = deconvModel(model);

	y = model.f;
	ALLF = [ALLF; y];

	fprintf(f_fid, '%0.5g\t', y(1:end-1));
	fprintf(f_fid, '%0.5g\n', y(end));
end

fclose(f_fid);
