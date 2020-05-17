function drawH(modeltype, datatype, alpha)

% note:
% normal model: DCR

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
global DECONV_NEW;
global DECONV_OLD;

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

DATASET_OLD = '../datasets/';
DATASET_NEW = '../datasets.new/';
DECONV_DATASET = DATASET_NEW;

DATA_WT1 = strcat(DECONV_DATASET, 'wt1.txt');
DATA_WT2 = strcat(DECONV_DATASET, 'wt2.txt');

SMALL = 1e-8;

if nargin < 2
	error('... Expect 2 inputs: datatype, alpha');
end

WT1_DIR = 'wt1_budflow/';
WT2_DIR = 'wt2_budflow/';

WT1_TP = 30:16:254;
%WT1_TP = 0:32:224;
WT2_TP = 38:16:262;
%WT2_TP = 0:16:224;

SMALL = 1e-8;

DECONV_WT1 = 'WT1';
DECONV_WT2 = 'WT2';
DECONV_JOINT = 'JOINT';

if DECONV_KERNEL == DECONV_WAVELET
	MODEL_DIR = 'models/';
elseif DECONV_KERNEL == DECONV_DIFF
	MODEL_DIR = 'models_even/';
end

% deal with modeltype
model = {};
model.modeltype = upper(modeltype);
model = parseModelType(model);
modelprefix = model.modelprefix;

% ===================================

% deal with orfname and datatype
orfname = 'ACT1';
orig_orfname = orfname;
[orfname, orfid] = map2SystemNames(orig_orfname);
model.orig_orfname = orig_orfname;
model.orfname = orfname;
model.orfid = orfid;

datatype = upper(datatype);
model.datatype = datatype;
model.alpha = alpha;
model.modeltype = modeltype;

% WT1
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
% == draw H == %
% ============ %
H = model.H;

%fig1 = figure;
%set(fig1, 'OuterPosition', [400 50 800 1100]);

maxH = max(max(H));

alpha = mean(alpha);
n = size(H,2);
lengths = model.lengths;
mu0 = lengths(DECONV_MU0POS);
delta = lengths(DECONV_DELTAPOS);
lambda = lengths(DECONV_LAMBDAPOS);
beta = lengths(DECONV_BETAPOS);

frames = 4;
colors = getLineColors();
for i=1:n
	if mod(i, frames) == 1
		figure;
	end
	y = H(:, i);

	set(gca, 'ytick', 0);
	set(gca, 'xtick', []);

	subplot(frames,1, mod(i-1, frames)+1);
	plot(y);

	xlim([0 numel(y)]);
	ylim([0 maxH*1.2]);
	set(gca, 'yticklabel', i);
end

return;
