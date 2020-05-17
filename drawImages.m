function [model] = drawImages(imagedir, datatype, modeltype, alpha, f_file, gm_file)

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

% 1: wavelet (uneven intervals)
% 2: 1st-order different operator (even intervals)
DECONV_WAVELET = 1;
DECONV_DIFF = 2;
DECONV_KERNEL = DECONV_WAVELET;

DATA_WT1 = strcat(DECONV_DATASET, 'wt1.txt');
DATA_WT2 = strcat(DECONV_DATASET, 'wt2.txt');

WT1_DIR = 'wt1_budflow/';
WT2_DIR = 'wt2_budflow/';

WT1_TP = 30:16:254;
WT2_TP = 38:16:262;

SMALL = 1e-8;

DECONV_WT1 = 'WT1';
DECONV_WT2 = 'WT2';
DECONV_JOINT = 'JOINT';

if DECONV_KERNEL == DECONV_WAVELET
	MODEL_DIR = 'models/';
elseif DECONV_KERNEL == DECONV_DIFF
	MODEL_DIR = 'models_even/';
end

S_partion = 0.2;

% ===================================
model = {};
model.modeltype = upper(modeltype);
[model] = parseModelType(model);
modelprefix = model.modelprefix;

datatype = upper(datatype);
model.datatype = datatype;
model.alpha = alpha;

%%% datatype = WT1
if strcmp(datatype, DECONV_WT1)
	modelfile = strcat(MODEL_DIR, WT1_DIR, modelprefix, '.', int2str(alpha(1)), '.label'); 
	if exist(modelfile, 'file') ~= 2
		error(sprintf('%s does not exist', modelfile));
	end

	model.timepoints = WT1_TP;

	% read model file
	[flag, model] = readModelFormat(modelfile, model);
	if flag == 0
		error('Error in parsing model file');
	end

	% calculate H
	[model] = calcH(model);

	mu0 = model.lengths(DECONV_MU0POS);
	lambda = model.lengths(DECONV_LAMBDAPOS);
	delta = model.lengths(DECONV_DELTAPOS);
	beta = model.lengths(DECONV_BETAPOS);
	g1_len = lambda*beta;
	dataset = load(DATA_WT1, 'ascii');

%%% datatype = WT2
elseif strcmp(datatype, DECONV_WT2)
	modelfile = strcat(MODEL_DIR, WT2_DIR, modelprefix, '.', int2str(alpha(1)), '.label');
	if exist(modelfile, 'file') ~= 2
		error(sprintf('%s does not exist', modelfile));
	end

	model.timepoints = WT2_TP;

	% read model file
	[flag, model] = readModelFormat(modelfile, model);
	if flag == 0
		error('Error in parsing model file');
	end

	% calculate H
	[model] = calcH(model);

	mu0 = model.lengths(DECONV_MU0POS);
	lambda = model.lengths(DECONV_LAMBDAPOS);
	delta = model.lengths(DECONV_DELTAPOS);
	beta = model.lengths(DECONV_BETAPOS);
	g1_len = lambda*beta;
	dataset = load(DATA_WT2, 'ascii');

%%% datatype = JOINT
elseif strcmp(datatype, DECONV_JOINT)
	% deal with WT1
	modelfile1 = strcat(MODEL_DIR, WT1_DIR, modelprefix, '.', int2str(alpha(1)), '.label');
	if exist(modelfile1, 'file') ~= 2
		error(sprintf('%s does not exist', modelfile1));
	end

	model.timepoints = WT1_TP;

	% read model file
	[flag, model] = readModelFormat(modelfile1, model);
	if flag == 0
		error('Error in parsing model file');
	end

	% calculate H
	[model] = calcH(model);

	mu0_1 = model.lengths(DECONV_MU0POS);
	lambda_1 = model.lengths(DECONV_LAMBDAPOS);
	delta_1 = model.lengths(DECONV_DELTAPOS);
	beta_1 = model.lengths(DECONV_BETAPOS);
	g1_1 = beta_1*lambda_1;

	% ============= %
	% deal with WT2 %
	% ============= %
	modelfile2 = strcat(MODEL_DIR, WT2_DIR, modelprefix, '.', int2str(alpha(2)), '.label');
	if exist(modelfile2, 'file') ~= 2
		error(sprintf('%s does not exist', modelfile2));
	end

	model.timepoints = [model.timepoints' WT2_TP']';

	% read model file
	[flag, model] = readModelFormat(modelfile2, model);
	if flag == 0
		error('Error in parsing model file');
	end

	% calculate H
	[model] = calcH(model);

	mu0_2 = model.lengths(DECONV_MU0POS);
	lambda_2 = model.lengths(DECONV_LAMBDAPOS);
	delta_2 = model.lengths(DECONV_DELTAPOS);
	beta_2 = model.lengths(DECONV_BETAPOS);
	g1_2 = beta_2*lambda_2;

	mu0 = (mu0_1+mu0_2)/2;
	lambda = (lambda_1+lambda_2)/2;
	delta = (delta_1+delta_2)/2;
	g1_len = (g1_1+g1_2)/2;
	beta = (beta_1+beta_2)/2;
	dataset1 = load(DATA_WT1, 'ascii');
	dataset2 = load(DATA_WT2, 'ascii');

else
	error('wrong data type (expect WT1/WT2/JOINT)');
end

g1_len = g1_len+alpha;
s_len = lambda*S_partion;
g2m_len = lambda-g1_len-s_len;

[genes, names, gm] = textread(gm_file, '%s\t%s\t%f');
num_genes = length(names);
f = load(f_file, 'ascii');

model.mu0 = mu0;
model.lambda = lambda;
model.delta = delta;
model.g1_len = g1_len;
model.s_len = s_len;
model.g2m_len = g2m_len;
model.beta = beta;

for i=1:num_genes
%i = 1;
	model.orfname = names{i};
	[orfname, orfid] = map2SystemNames(names{i});
	if strcmp(datatype, DECONV_JOINT)
		g1 = dataset1(orfid,:)';
		g2 = dataset2(orfid,:)';
		model.g = [g1' g2']';
	else
		g = dataset(orfid,:)';
		model.g = g;
	end

	model.f = f(i,:)';
	model.orig_orfname = genes{i};
	model.gm = gm(i);

	plotOptimal_general_branch_area(model, 1, imagedir);
%	plotOptimal_general_branch(model, 1);
	disp(sprintf('%d, %s', i, genes{i}));
end

return;
