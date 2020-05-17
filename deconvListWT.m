function deconvList(input)

% ---------- %
gammafile = 'allgenes/allgenes.gm';
modeltype = '1.1.1';
alpha = [26 27];

OUTDIR = 'wt/';
input = strcat(OUTDIR, input);
regulatedfile = input;

[reg_orfs] = textread(regulatedfile, '%s');

% ---------- %

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

% ===========================================
[orfnames, orfids] = map2SystemNames(reg_orfs, 1);
model1 = {};
model2 = {};

% model 1
model1.modeltype = upper(modeltype);
model1 = parseModelType(model1);
modelprefix1 = model1.modelprefix;
model1.datatype = DECONV_WT1;
model1.alpha = alpha(1);

% model 2
model2.modeltype = upper(modeltype);
model2 = parseModelType(model2);
modelprefix2 = model2.modelprefix;
model2.datatype = DECONV_WT2;
model2.alpha = alpha(2);

output_f1 = strcat(input, '.wt1.f');
output_gm1 = strcat(input, '.wt1.gm');
output_f2 = strcat(input, '.wt2.f');
output_gm2 = strcat(input, '.wt2.gm');

f_fid1 = fopen(output_f1, 'w');
gm_fid1 = fopen(output_gm1, 'w');
f_fid2 = fopen(output_f2, 'w');
gm_fid2 = fopen(output_gm2, 'w');
% ==========================================

[all_orig_orfnames, orig_orfs, all_gammas] = textread(gammafile, '%s\t%s\t%f');
[all_orfnames, all_orfids] = map2SystemNames(all_orig_orfnames, 1);

orig_orfnames = {};
gammas = [];
for idx=1:length(orfids)
	cur_orf = orfnames{idx};
	pos = strmatch(cur_orf, all_orfnames, 'exact');

	orig_orfnames{idx} = all_orig_orfnames{pos};
	gammas(idx) = all_gammas(pos);
end


% ==========================================

% datatype = WT1
modelfile1 = strcat(MODEL_DIR, WT1_DIR, modelprefix1, '.', int2str(alpha(1)), '.label'); 
if exist(modelfile1, 'file') ~= 2
	error(sprintf('%s does not exist', modelfile1));
end

dataset1 = load(DATA_WT1, 'ascii');
model1.timepoints = WT1_TP;

% read model file
[flag, model1] = readModelFormat(modelfile1, model1);
if flag == 0
	error('Error in parsing model1 file');
end

% calculate H
[model1] = calcH(model1);


% datatype = WT2
modelfile2 = strcat(MODEL_DIR, WT2_DIR, modelprefix2, '.', int2str(alpha(2)), '.label'); 
if exist(modelfile2, 'file') ~= 2
	error(sprintf('%s does not exist', modelfile2));
end

dataset2 = load(DATA_WT2, 'ascii');
model2.timepoints = WT2_TP;

% read model file
[flag, model2] = readModelFormat(modelfile2, model2);
if flag == 0
	error('Error in parsing model2 file');
end

% calculate H
[model2] = calcH(model2);


% ============ %
% deconvlution %
% ============ %
len_orfs = length(orfnames);
for idx = 1:1:len_orfs
	disp(sprintf('%d : %s', idx, orig_orfnames{idx}));
	% WT1
	model1.orig_orfname = orig_orfnames{idx};
	model1.orfname = orfnames{idx};
	model1.orfid = orfids{idx};
	model1.g = [dataset1(model1.orfid,:)]';
	model1.gm = gammas(idx);
	[model1] = deconvModel(model1);

	fprintf(f_fid1, '%0.5g\t', model1.f(1:length(model1.f)-1));
	fprintf(f_fid1, '%0.5g\n', model1.f(length(model1.f)));
	fprintf(gm_fid1, '%s\t%s\t%0.5g\n', orig_orfnames{idx}, orfnames{idx}, gammas(idx));

	% WT2
	model2.orig_orfname = orig_orfnames{idx};
	model2.orfname = orfnames{idx};
	model2.orfid = orfids{idx};
	model2.g = [dataset2(model2.orfid,:)]';
	model2.gm = gammas(idx);
	[model2] = deconvModel(model2);

	fprintf(f_fid2, '%0.5g\t', model2.f(1:length(model2.f)-1));
	fprintf(f_fid2, '%0.5g\n', model2.f(length(model2.f)));
	fprintf(gm_fid2, '%s\t%s\t%0.5g\n', orig_orfnames{idx}, orfnames{idx}, gammas(idx));

	if IMAGE == ON
		imagename = strcat(imagedir, '/', model.orfname, '.png');
		plotOptimal(model, 1, imagename);
	end
end

fclose(f_fid1);
fclose(f_fid2);
fclose(gm_fid1);
fclose(gm_fid2);

return;

