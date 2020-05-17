function [model] = drawImages4AF(imagedir, f_file, gm_file)

alpha = [0 0];
datatype = 'JOINT';
modeltype = 'NORMAL';

% -------------------------

global SLIENCE;

global DECONV_MU0POS;
global DECONV_LAMBDAPOS;
global DECONV_ALPHAPOS;
global DECONV_SIGMA0POS;
global DECONV_SIGMAVPOS;
global DECONV_BETAPOS;

global DECONV_JOINT;
global DECONV_DATA1;
global DECONV_DATA2;

global DECONV_DATASET;

% -------------------------

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

% -------------------------

if nargin < 3
	error('... Expect 3 inputs: imagedir, f_file, gm_file');
end

% -------------------------

f = load(f_file, 'ascii');
[genes, names, gammas] = textread(gm_file, '%s\t%s\t%f');
num_genes = length(names);

DATA1 = strcat(DECONV_DATASET, 'af1.txt');
DATA2 = strcat(DECONV_DATASET, 'af2.txt');
DIR1 = 'af1/';
DIR2 = 'af2/';
TP1 = [0:10:40 60:20:180 220:20:300];
TP2 = [0:10:40 60:20:340];

MODEL_DIR = 'models/';

% ==========================================

model = {};

modeltype = 'NORMAL';
model.modeltype = upper(modeltype);
model = parseModelType4Mutant(model);
modelprefix = model.modelprefix;

model.alpha = alpha;
datatype = upper(datatype);
model.datatype = datatype;

ON = 1;
OFF = 0;

% ==========================================

% datatype = DATA1
if strcmp(datatype, DECONV_DATA1)
	modelfile = strcat(MODEL_DIR, DIR1, modelprefix, '.', int2str(alpha(1)), '.label'); 
	if exist(modelfile, 'file') ~= 2
		error(sprintf('%s does not exist', modelfile));
	end

	dataset = load(DATA1, 'ascii');
	model.timepoints{1} = TP1;

	% read model file
	[flag, model] = readModelFormat4Mutant(modelfile, model);
	if flag == 0
		error('Error in parsing model file');
	end

	% calculate H
	[model] = calcH4Mutant(model);

% datatype = DATA2
elseif strcmp(datatype, DECONV_DATA2)
	modelfile = strcat(MODEL_DIR, DIR2, modelprefix, '.', int2str(alpha(1)), '.label'); 
	if exist(modelfile, 'file') ~= 2
		error(sprintf('%s does not exist', modelfile));
	end

	dataset = load(DATA2, 'ascii');
	model.timepoints{1} = TP2;

	% read model file
	[flag, model] = readModelFormat4Mutant(modelfile, model);
	if flag == 0
		error('Error in parsing model file');
	end

	% calculate H
	[model] = calcH4Mutant(model);

% datatype = JOINT
elseif strcmp(datatype, DECONV_JOINT)
	% = DATA1 = %
	modelfile1 = strcat(MODEL_DIR, DIR1, modelprefix, '.', int2str(alpha(1)), '.label');
	if exist(modelfile1, 'file') ~= 2
		error(sprintf('%s does not exist', modelfile1));
	end

	dataset1 = load(DATA1, 'ascii');
	model.timepoints{1} = TP1;

	% read model file
	[flag, model] = readModelFormat4Mutant(modelfile1, model);
	if flag == 0
		error('Error in parsing model file');
	end

	% calculate H
	[model] = calcH4Mutant(model);

	% = DATA2 = %
	modelfile2 = strcat(MODEL_DIR, DIR2, modelprefix, '.', int2str(alpha(2)), '.label');
	if exist(modelfile2, 'file') ~= 2
		error(sprintf('%s does not exist', modelfile2));
	end

	dataset2 = load(DATA2, 'ascii');
	model.timepoints{2} = TP2;

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
% ==========================================
[orfnames, orfids] = map2SystemNames(names, 1);
% ==========================================
for idx = 1:1:num_genes;
	model.orfname = names{idx};
	disp(sprintf('%d : %s', idx, genes{idx}));
%	[orfname, orfid] = map2SystemNames(names{i});
	orfname = orfnames{idx};
	orfid = orfids{idx};

	if strcmp(datatype, DECONV_JOINT)
		g1 = dataset1(orfid, :)';
		g2 = dataset2(orfid, :)';
		model.g = [g1' g2'];
	else
		g = dataset(orfid, :);
	end

	model.f = f(idx,:)';
	model.orig_orfname = genes{idx};
	model.gm = gammas(idx);

	plotOptimal_general_branch4AF(model, 1, imagedir);
end

return;
