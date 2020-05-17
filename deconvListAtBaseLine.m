function deconvListAtBaseline(genefile, modeltype, datatype, alpha)

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

GM = 0;

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

DECONV_KERNEL = DECONV_WAVELET;

if DECONV_KERNEL == DECONV_WAVELET
	MODEL_DIR = 'models/';
elseif DECONV_KERNEL == DECONV_DIFF
	MODEL_DIR = 'models_even/';
end

if nargin < 4
	error('... Expect at least inputs: genefile, modeltype, datatype, alpha');
end

model = {};
datatype = upper(datatype);
model.datatype = datatype;
model.modeltype = upper(modeltype);
% ===========================================
model = parseModelType(model);
modelprefix = model.modelprefix;
% ==========================================
model.alpha = alpha;

if strcmp(datatype, DECONV_JOINT)
	prefix = strcat(genefile, '.', modeltype, '.', datatype, '.', int2str(alpha(1)), '-', int2str(alpha(2)));
else
	prefix = strcat(genefile, '.', modeltype, '.', datatype, '.', int2str(alpha(1)));
end

%output_f = strcat(prefix, '.baseline.f');
output_err = strcat(prefix, '.baseline.err');

err_fid = fopen(output_err, 'w');
%f_fid = fopen(output_f, 'w');

% ==========================================

[orig_orfnames] = textread(genefile, '%s');
[orfnames, orfids] = map2SystemNames(orig_orfnames, 1);

% ==========================================

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

else
	error('wrong data type (expect WT1/WT2/JOINT)');
end

% ============ %
% deconvlution %
% ============ %
len_orfs = length(orfnames);
for idx = 1:1:len_orfs
	model.orig_orfname = orig_orfnames{idx};
	model.orfname = orfnames{idx};
	model.orfid = orfids{idx};
	model.g = [dataset1(model.orfid,:) dataset2(model.orfid,:)]';
	c = corr2(dataset1(model.orfid,:),dataset2(model.orfid,:));
	model.gm = GM;
	disp(sprintf('%d : %s', idx, model.orig_orfname));
	[model] = deconvModel(model);

%	plotOptimal(model, 1);

%	fprintf(f_fid, '%0.5g\t', model.f(1:length(model.f)-1));
%	fprintf(f_fid, '%0.5g\n', model.f(length(model.f)));
	fprintf(err_fid, '%s\t%0.5g\t%0.5g\n', model.orig_orfname, c, model.rn);
end

%fclose(f_fid);
fclose(err_fid);

return;
