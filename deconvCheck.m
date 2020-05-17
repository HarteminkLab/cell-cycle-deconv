function deconvCheck(genelist, f_file, modeltype, datatype, alpha)

% ---------- %

CUTOFF = 2;
%CUTOFF = 1;

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
	error('... Expect at least 5 inputs: genelist, f_file, modeltype, datatype, alpha');
end

% ===========================================
model = {};

model.modeltype = upper(modeltype);
model = parseModelType(model);
modelprefix = model.modelprefix;
datatype = upper(datatype);
model.datatype = datatype;
model.alpha = alpha;

if strcmp(datatype, DECONV_JOINT)
	prefix = strcat(genelist, '.', modeltype, '.', datatype, '.', int2str(alpha(1)), '-', int2str(alpha(2)));
else
	prefix = strcat(genelist, '.', modeltype, '.', datatype, '.', int2str(alpha(1)));
end

% ==========================================
[orig_orfnames, std_names, gammas] = textread(genelist, '%s\t%s\t%f');
[orfnames, orfids] = map2SystemNames(orig_orfnames, 1);
% ==========================================

allF = load(f_file);

% datatype = WT1
if strcmp(datatype, DECONV_WT1)
	modelfile = strcat(MODEL_DIR, WT1_DIR, modelprefix, '.', int2str(alpha(1)), '.label'); 
	if exist(modelfile, 'file') ~= 2
		error(sprintf('%s does not exist', modelfile));
	end

	dataset = load(DATA_WT1, 'ascii');
	model.timepoints = WT1_TP;

	% read model file
	[flag, model] = readModelFormat(modelfile, model);
	if flag == 0
		error('Error in parsing model file');
	end

	% calculate H
	[model] = calcH(model);

% datatype = WT2
elseif strcmp(datatype, DECONV_WT2)
	modelfile = strcat(MODEL_DIR, WT2_DIR, modelprefix, '.', int2str(alpha(1)), '.label'); 
	if exist(modelfile, 'file') ~= 2
		error(sprintf('%s does not exist', modelfile));
	end

	dataset = load(DATA_WT2, 'ascii');
	model.timepoints = WT2_TP;

	% read model file
	[flag, model] = readModelFormat(modelfile, model);
	if flag == 0
		error('Error in parsing model file');
	end

	% calculate H
	[model] = calcH(model);

% datatype = JOINT
elseif strcmp(datatype, DECONV_JOINT)
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

	% deal with WT2 
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
else
	error('wrong data type (expect WT1/WT2/JOINT)');
end

out_fid = fopen('Final_check.lst', 'w');

% ============ %
% deconvlution %
% ============ %
len_orfs = length(orfnames);
for idx = 1:1:len_orfs
	name = orig_orfnames{idx};
	orfid = orfids{idx};
	g = [dataset1(orfid,:) dataset2(orfid,:)]';

	f = allF(orfid, :);

	pred_g = model.H*f';
	rn = square_pos(norm(pred_g./g-1, 2));

	disp(sprintf('%d\t:\t%0.3f (%s)', idx, rn, name));

	if rn > CUTOFF
		fprintf(out_fid, '%s\t%0.3f\n', name, rn);
	end
end

fclose(out_fid);

return;
