function deconvList4Mutant(genefile, dataset, imagedir, gamma)

alpha = [0 0];
datatype = 'JOINT';

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

global DATASET_MT;
global DATASET_AF;

global DECONV_DATASET;

% -------------------------

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

% -------------------------

if nargin < 4
	error('... Expect 4 inputs: genefile, dataset, imagedir, gamma(optional, a numerical value; if 0, learn gamma; if -1, gnenfile specify gamma for each gene)');
end

% -------------------------

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

if strcmp(imagedir, '')
	IMAGE = OFF;
else
	IMAGE = ON;
end

% ==========================================

prefix = strcat(genefile, '.', dataset, '.', datatype);

output_f = strcat(prefix, '.f');
output_gm = strcat(prefix, '.gm');

gm_fid = fopen(output_gm, 'w');
f_fid = fopen(output_f, 'w');

% ==========================================

if gamma ~= -1
	[orig_orfnames] = textread(genefile, '%s');
else
	[orig_orfnames, gammas] = textread(genefile, '%s\t%f');
end
[orfnames, orfids] = map2SystemNames(orig_orfnames, 1);

% ==========================================

model.dataset = dataset;
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
len_orfs = length(orfnames);
for idx = 1:1:len_orfs
	model.orig_orfname = orig_orfnames{idx};
	model.orfname = orfnames{idx};
	model.orfid = orfids{idx};
	model.g = [dataset1(model.orfid,:) dataset2(model.orfid,:)]';
	disp(sprintf('%d : %s', idx, model.orig_orfname));
	if gamma > 0 % fix gamma
		SLIENCE = 1;
		model.gm = gamma;
		[model] = deconvModel4Mutant(model);
	elseif gamma == 0
		SLIENCE = 0;
		[model] = findOptimal4Mutant(model, 0);
	else
		gm = gammas(idx);
		SLIENCE = 1;
		model.gm = gm;
		[model] = deconvModel4Mutant(model);
	end
	fprintf(f_fid, '%0.5g\t', model.f(1:length(model.f)-1));
	fprintf(f_fid, '%0.5g\n', model.f(length(model.f)));
	fprintf(gm_fid, '%s\t%s\t%0.5g\n', model.orig_orfname, model.orfname, model.gm);

	if IMAGE == ON
		imagename = strcat(imagedir, '/', model.orfname);
		plotOptimal_general_branch4Mutant(model, 1, imagename);
	end
end

fclose(f_fid);
fclose(gm_fid);

return;
