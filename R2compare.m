function R2compare(genefile, output, modeltype, alpha)

global SLIENCE;

global DECONV_MU0POS;
global DECONV_LAMBDAPOS;
global DECONV_DELTAPOS;
global DECONV_ALPHAPOS;
global DECONV_SIGMA0POS;
global DECONV_SIGMAVPOS;
global DECONV_BETAPOS;

global DECONV_LEARN_FLAG;

global DECONV_JOINT;
global DECONV_WT1;
global DECONV_WT2;

global DECONV_KERNEL;
global DECONV_WAVELET;
global DECONV_DIFF;

global DECONV_DATASET;
global DATASET_OLD;
global DATASET_NEW;

DECONV_WAVELET = 1;
DECONV_DIFF = 2;

DECONV_MU0POS = 1;
DECONV_LAMBDAPOS = 2;
DECONV_DELTAPOS = 3;
DECONV_SIGMA0POS = 4;
DECONV_SIGMAVPOS = 5;
DECONV_ALPHAPOS = 6;
DECONV_BETAPOS = 7;

SLIENCE = 1;

DATASET_OLD = '../datasets/';
DATASET_NEW = '../datasets.new/';

DECONV_DATASET = DATASET_NEW;

DATA_WT1 = strcat(DECONV_DATASET, 'wt1.txt');
DATA_WT2 = strcat(DECONV_DATASET, 'wt2.txt');

DECONV_WT1 = 'WT1';
DECONV_WT2 = 'WT2';
DECONV_JOINT = 'JOINT';

WT1_TP = 30:16:254;
WT2_TP = 38:16:262;
WT1_DIR = 'wt1_budflow/';
WT2_DIR = 'wt2_budflow/';

ON = 1;
OFF = 0;

IMAGE = OFF;

DECONV_KERNEL = DECONV_WAVELET;

if DECONV_KERNEL == DECONV_WAVELET
	MODEL_DIR = 'models/';
elseif DECONV_KERNEL == DECONV_DIFF
	MODEL_DIR = 'models_even/';
end

if nargin < 4
	error('... 4 inputs: gene file, output file, modeltype, alpha');
end

% ===========================================
model = {};

model.modeltype = upper(modeltype);
model = parseModelType(model);
modelprefix = model.modelprefix;
datatype = 'JOINT';
model.datatype = datatype;
model.alpha = alpha;

if strcmp(datatype, DECONV_JOINT)
	prefix = strcat(genefile, '.', modeltype, '.', datatype, '.', int2str(alpha(1)), '-', int2str(alpha(2)));
else
	prefix = strcat(genefile, '.', modeltype, '.', datatype, '.', int2str(alpha(1)));
end

%output_r2 = strcat(genefile, '.R2');
output_fid = fopen(output, 'w');

% ==========================================

[orig_orfnames] = textread(genefile, '%s');
[orfnames, orfids] = map2SystemNames(orig_orfnames, 1);

% ==========================================

% = WT1 = %
modelfile1 = strcat(MODEL_DIR, WT1_DIR, modelprefix, '.', int2str(alpha(1)), '.label');
if exist(modelfile1, 'file') ~= 2
	error(sprintf('%s does not exist', modelfile1));
end

dataset1 = load(DATA_WT1, 'ascii');
model.timepoints = WT1_TP;

% read model file
[flag, model] = readModelFormat(modelfile1, model);
if flag == 0
	error('Error in parsing model file');
end

% calculate H
[model] = calcH(model);

% ================ %
% deal with 111305 %
% ================ %
modelfile2 = strcat(MODEL_DIR, WT2_DIR, modelprefix, '.', int2str(alpha(2)), '.label');
if exist(modelfile2, 'file') ~= 2
	error(sprintf('%s does not exist', modelfile2));
end

dataset2 = load(DATA_WT2, 'ascii');
model.timepoints = [model.timepoints' WT2_TP']';

% read model file
[flag, model] = readModelFormat(modelfile2, model);
if flag == 0
	error('Error in parsing model file');
end

% calculate H
[model] = calcH(model);


len_orfs = length(orfnames);
% ========= %
% each gene %
% ========= %
for idx = 1:1:len_orfs
	model.orig_orfname = orig_orfnames{idx};
	model.orfname = orfnames{idx};
	model.orfid = orfids{idx};

	g1 = dataset1(model.orfid,:);
	g2 = dataset2(model.orfid,:);

	wt_r2 = rsquare(g1, g2);

	model.g = [g1 g2]';

	model.gm = 0;
	[model] = deconvModel(model);

	pred_g1 = model.pred_g(1:end/2);
	pred_g2 = model.pred_g(end/2+1:end);

	pred_r2 = (rsquare(g1, pred_g1)+rsquare(g2, pred_g2))/2;

	disp(sprintf('%d : %s\t%s\t%0.3g\t%0.3g', idx, model.orig_orfname, model.orfname, wt_r2, pred_r2));
	fprintf(output_fid, '%s\t%s\t%0.3g\t%0.3g\n', model.orig_orfname, model.orfname, wt_r2, pred_r2);
end

fclose(output_fid);

return;
