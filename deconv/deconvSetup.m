function [model] = deconvSetup(orfname, modeltype, alpha)

global SILENCE;

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

DECONV_DATASET = 'datasets/';

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

SILENCE = 0;

SHOW_DETAILS = 1;

DATA_WT1 = strcat(DECONV_DATASET, 'wt1.txt');
DATA_WT2 = strcat(DECONV_DATASET, 'wt2.txt');

WT1_TP = 30:16:254;
WT2_TP = 38:16:262;

outdir = 'output/';
modeldir1 = 'models/wt1_budflow/';
modelfile = '1.1.1.0.label';

runs = 1;

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
fprintf("ORF ID: %s, %d\n", orfname, orfid);

model.orig_orfname = orig_orfname;
model.orfname = orfname;
model.orfid = orfid;

model.datatype = DECONV_WT1;
model.alpha = alpha;

% deal with modeltype
model.modeltype = upper(modeltype);
[model] = parseModelType(model);
modelprefix = model.modelprefix;

dataset = load(DATA_WT1, 'ascii');
g1 = dataset(orfid,:)';

model.g = [g1'];

model.H = [];

modelpath = strcat(modeldir1, modelfile);

[flag, model] = readModelFormat(modelpath, model);

% calculate H
model.timepoints = WT1_TP;
[model] = calcH(model);


