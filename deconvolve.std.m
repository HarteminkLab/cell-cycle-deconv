function [model] = deconvolve(orfname, modeltype, datatype, alpha, gamma, learn_flag, fig_flag)

global SLIENCE;

global DECONV_MU0POS;			% mu0
global DECONV_LAMBDAPOS;	% lambda
global DECONV_DELTAPOS;		% delta
global DECONV_ALPHAPOS;		% alpha
global DECONV_SIGMA0POS;	% sigma_0
global DECONV_SIGMAVPOS;	% sigma_v
global DECONV_BETAPOS;		% beta

global DECONV_KERNEL;
global DECONV_WAVELET;
global DECONV_DIFF;

global DECONV_JOINT;
global DECONV_WT1;
global DECONV_WT2;

global DECONV_DATASET;
global DATASET_OLD;
global DATASET_NEW;

global FROM_FINDOPTIMAL;

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

FROM_FINDOPTIMAL = 0;
SLIENCE = 0;

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

if nargin < 7
	error('... need 7 inputs: orfname, modeltype, datatype, alpha, gamma, learn_flag, fig_flag');
end

if DECONV_KERNEL == DECONV_WAVELET
	MODEL_DIR = 'models/';
elseif DECONV_KERNEL == DECONV_DIFF
	MODEL_DIR = 'models_even/';
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% deal with modeltype
model = {};
model.modeltype = upper(modeltype);
[model] = parseModelType(model);
modelprefix = model.modelprefix;

% deal with orfname and datatype
orig_orfname = orfname;
[orfname, orfid] = map2SystemNames(orig_orfname);
model.orig_orfname = orig_orfname;
model.orfname = orfname;
model.orfid = orfid;

datatype = upper(datatype);
model.datatype = datatype;
model.alpha = alpha;

%%% datatype = WT1
if strcmp(datatype, DECONV_WT1)
	modelfile = strcat(MODEL_DIR, WT1_DIR, modelprefix, '.', int2str(alpha(1)), '.label'); 
	if exist(modelfile, 'file') ~= 2
		error(sprintf('%s does not exist', modelfile));
	end

	dataset = load(DATA_WT1, 'ascii');
	g = dataset(orfid,:)';
	model.g = g;
	model.timepoints = WT1_TP;

	% read model file
	[flag, model] = readModelFormat(modelfile, model);
	if flag == 0
		error('Error in parsing model file');
	end

	% calculate H
	[model] = calcH(model);

%%% datatype = WT2
elseif strcmp(datatype, DECONV_WT2)
	modelfile = strcat(MODEL_DIR, WT2_DIR, modelprefix, '.', int2str(alpha(1)), '.label');
	if exist(modelfile, 'file') ~= 2
		error(sprintf('%s does not exist', modelfile));
	end

	dataset = load(DATA_WT2, 'ascii');
	g = dataset(orfid,:)';
	model.g = g;
	model.timepoints = WT2_TP;

	% read model file
	[flag, model] = readModelFormat(modelfile, model);
	if flag == 0
		error('Error in parsing model file');
	end

	% calculate H
	[model] = calcH(model);


%%% datatype = JOINT
elseif strcmp(datatype, DECONV_JOINT)
	% deal with WT1
	modelfile1 = strcat(MODEL_DIR, WT1_DIR, modelprefix, '.', int2str(alpha(1)), '.label');
	if exist(modelfile1, 'file') ~= 2
		error(sprintf('%s does not exist', modelfile1));
	end

	dataset1 = load(DATA_WT1, 'ascii');
	g1 = dataset1(orfid,:)';
	model.timepoints = WT1_TP;

	% read model file
	[flag, model] = readModelFormat(modelfile1, model);
	if flag == 0
		error('Error in parsing model file');
	end

	% calculate H
	[model] = calcH(model);

	% ============= %
	% deal with WT2 %
	% ============= %
	modelfile2 = strcat(MODEL_DIR, WT2_DIR, modelprefix, '.', int2str(alpha(2)), '.label');
	if exist(modelfile2, 'file') ~= 2
		error(sprintf('%s does not exist', modelfile2));
	end

	dataset2 = load(DATA_WT2, 'ascii');
	g2 = dataset2(orfid,:)';
	model.timepoints = [model.timepoints' WT2_TP']';
	model.g = [g1' g2']';

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
if learn_flag == 1
	[model] = findOptimal(model, fig_flag);
else
	model.gm = gamma;
	paramstr=['Parameters - gamma: %0.5g'];
	disp(sprintf(paramstr, gamma));
	[model] = deconvModel(model);

%	figure_name = strcat('model_comparison_images/', upper(modeltype),'_images/', orig_orfname, '_', modeltype);
	[model] = plotOptimal_general_branch(model, fig_flag);
end

return;
