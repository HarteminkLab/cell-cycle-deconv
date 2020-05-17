function [model] = deconvolve_mutant(orfname, dataset, datatype, alpha, gamma, learn_flag, fig_flag)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
global SLIENCE;

global DECONV_MU0POS;			% mu0
global DECONV_LAMBDAPOS;	% lambda
global DECONV_ALPHAPOS;		% alpha
global DECONV_SIGMA0POS;	% sigma_0
global DECONV_SIGMAVPOS;	% sigma_v
global DECONV_BETAPOS;		% beta

global DECONV_DATA1;
global DECONV_DATA2;
global DECONV_JOINT;

global DATASET_MT;
global DATASET_AF;

global DECONV_DATASET;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

DATASET_MT = 'MT';
DATASET_AF = 'AF';

DECONV_DATASET = '../datasets.new/';

DECONV_MU0POS = 1;
DECONV_LAMBDAPOS = 2;
DECONV_SIGMA0POS = 3;
DECONV_SIGMAVPOS = 4;
DECONV_ALPHAPOS = 5;
DECONV_BETAPOS = 6;

DECONV_DATA1 = 'DATA1';
DECONV_DATA2 = 'DATA2';
DECONV_JOINT = 'JOINT';

if nargin < 7
	error('... need 7 inputs: orfname, dataset, datatype, alpha, gamma, learn_flag, fig_flag');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dataset = upper(dataset);
if strcmp(dataset, DATASET_MT)
	DATA1 = strcat(DECONV_DATASET, 'mt1.txt');
	DATA2 = strcat(DECONV_DATASET, 'mt2.txt');
	DIR1 = 'mt1/';
	DIR2 = 'mt2/';
	TP1 = 38:16:262;
	TP2 = 30:16:254;
elseif strcmp(dataset, DATASET_AF)
	DATA1 = strcat(DECONV_DATASET, 'af1.txt');
	DATA2 = strcat(DECONV_DATASET, 'af2.txt');
	DIR1 = 'af1/';
	DIR2 = 'af2/';
	TP1 = [0:10:40 60:20:180 220:20:300];
	TP2 = [0:10:40 60:20:340];
else
	error('expect MT/AF for dataset');
end

MODEL_DIR = 'models/';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model = {};

modeltype = 'NORMAL';
model.modeltype = upper(modeltype);
model = parseModelType4Mutant(model);
modelprefix = model.modelprefix;

model.dataset = dataset;

% deal with orfname and datatype
orig_orfname = orfname;
[orfname, orfid] = map2SystemNames(orig_orfname);
model.orig_orfname = orig_orfname;
model.orfname = orfname;
model.orfid = orfid;

datatype = upper(datatype);
model.datatype = datatype;
model.alpha = alpha;

%%% datatype = DATA1
if strcmp(datatype, DECONV_DATA1)
	modelfile = strcat(MODEL_DIR, DIR1, modelprefix, '.', int2str(alpha(1)), '.label');
	if exist(modelfile, 'file') ~= 2
		error(sprintf('%s does not exist', modelfile));
	end

	dataset = load(DATA1, 'ascii');
	g = dataset(orfid,:)';
	model.g = g;
	model.timepoints{1} = TP1;

	% read model file
	[flag, model] = readModelFormat4Mutant(modelfile, model);
	if flag == 0
		error('Error in parsing model file');
	end

	% calculate H
	[model] = calcH4Mutant(model);

%%% datatype = DATA2
elseif strcmp(datatype, DECONV_DATA2)
	modelfile = strcat(MODEL_DIR, DIR2, modelprefix, '.', int2str(alpha(1)), '.label');
	if exist(modelfile, 'file') ~= 2
		error(sprintf('%s does not exist', modelfile));
	end

	dataset = load(DATA2, 'ascii');
	g = dataset(orfid,:)';
	model.g = g;
	model.timepoints{1} = TP2;

	% read model file
	[flag, model] = readModelFormat4Mutant(modelfile, model);
	if flag == 0
		error('Error in parsing model file');
	end

	% calculate H
	[model] = calcH4Mutant(model);

%%% datatype = JOINT
elseif strcmp(datatype, DECONV_JOINT)
	% deal with DATA1
	modelfile1 = strcat(MODEL_DIR, DIR1, modelprefix, '.', int2str(alpha(1)), '.label');
	if exist(modelfile1, 'file') ~= 2
		error(sprintf('%s does not exist', modelfile1));
	end

	dataset1 = load(DATA1, 'ascii');
	g1 = dataset1(orfid,:)';
	model.timepoints{1} = TP1;

	% read model file
	[flag, model] = readModelFormat4Mutant(modelfile1, model);
	if flag == 0
		error('Error in parsing model file');
	end

	% calculate H
	[model] = calcH4Mutant(model);

	% deal with MT2 %
	modelfile2 = strcat(MODEL_DIR, DIR2, modelprefix, '.', int2str(alpha(2)), '.label');
	if exist(modelfile2, 'file') ~= 2
		error(sprintf('%s does not exist', modelfile2));
	end

	dataset2 = load(DATA2, 'ascii');
	g2 = dataset2(orfid,:)';
	model.timepoints{2} = TP2;
	model.g = [g1' g2']';

	% read model file
	[flag, model] = readModelFormat4Mutant(modelfile2, model);
	if flag == 0
		error('Error in parsing model file');
	end

	% calculate H
	[model] = calcH4Mutant(model);

else
	error('wrong data type (expect DATA1/DATA2/JOINT)');
end

% ============ %
% deconvlution %
% ============ %
if learn_flag == 1
	SLIENCE = 1;
	[model] = findOptimal4Mutant(model, fig_flag);
else
	SLIENCE = 0;
	model.gm = gamma;
	paramstr=['Parameters - gamma: %0.5g'];
	disp(sprintf(paramstr, gamma));
	[model] = deconvModel4Mutant(model);

%	dir = '/home/home5/xinguo/public_html/cc/website/mutant_plots/';
%	image_name = strcat(dir, model.orfname);
	[model] = plotOptimal_general_branch4Mutant(model, fig_flag);
end

return;
