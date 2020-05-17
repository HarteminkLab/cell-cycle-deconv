function [std_g] = deconvNoiseAddedList(gmfile, noise_lv, runs, modeltype, datatype, alpha)

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

DECONV_KERNEL = DECONV_WAVELET;

if DECONV_KERNEL == DECONV_WAVELET
	MODEL_DIR = 'models/';
elseif DECONV_KERNEL == DECONV_DIFF
	MODEL_DIR = 'models_even/';
end

if nargin < 6
	error('... Expect 6 inputs: gm_file, noise_level, sample_runs, modeltype, datatype, alpha');
end

output = strcat(gmfile, '.', int2str(noise_lv*100), '.noise.added');
out_fid = fopen(output, 'w');

% ===========================================
model = {};

model.modeltype = upper(modeltype);
[model] = parseModelType(model);
modelprefix = model.modelprefix;

datatype = upper(datatype);
model.datatype = datatype;
model.alpha = alpha;

% ==========================================

[orig_orfnames, gms] = textread(gmfile, '%s\t%f');
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

	lambda = model.lengths(DECONV_LAMBDAPOS);
	delta = model.lengths(DECONV_DELTAPOS);

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

	lambda = model.lengths(DECONV_LAMBDAPOS);
	delta = model.lengths(DECONV_DELTAPOS);

% datatype = JOINT
elseif strcmp(datatype, DECONV_JOINT)
	% ============= %
	% deal with WT1 %
	% ============= %
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

	lambda_1 = model.lengths(DECONV_LAMBDAPOS);
	delta_1 = model.lengths(DECONV_DELTAPOS);

	% ============= %
	% deal with WT2 %
	% ============= %
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

	lambda_2 = model.lengths(DECONV_LAMBDAPOS);
	delta_2 = model.lengths(DECONV_DELTAPOS);

	lambda = (lambda_1+lambda_2)/2;
	delta = (delta_1+delta_2)/2;
else
	error('wrong data type (expect WT1/WT2/JOINT)');
end

model = getRDCidx(model);

% deconvlution 
len_orfs = length(orfnames);

alldiff = [];
diff_idx = 1;

bin = 100;

for idx = 1:1:len_orfs
	model.orig_orfname = orig_orfnames{idx};
	model.orfname = orfnames{idx};
	model.orfid = orfids{idx};
	model.gm = gms(idx);

	model.g = [dataset1(model.orfid,:) dataset2(model.orfid,:)]';
	disp(sprintf('%d : %s %0.3f', idx, model.orig_orfname, gms(idx)));
	[model] = deconvModel(model);
	
	if ~SLIENCE
		figure;
		plot(model.f, 'color', 'black', 'linewidth', 3);
		hold on;
	end

	[cd_flag, std_peak_idx] = judgePeak(model, bin);

	std_g = model.g;
	for run = 1:1:runs
		noise_ratio = normrnd(1, noise_lv, [length(std_g) 1]);
		noise_g = max(1, std_g.*noise_ratio);
		model.g = noise_g;
		[model] = deconvModel(model);

		if ~SLIENCE
			plot(model.f, '--', 'color', [128 128 128]./256, 'linewidth', 1);
			hold on;
		end

%		plotOptimal(model, 1);
		if cd_flag == 1
			noise_R = lambda;
			noise_x = model.t_x;
			noise_y = model.f(model.t_y_idx);
		else
			noise_R = lambda+delta;
			noise_x = model.b_x;
			noise_y = model.f(model.b_y_idx);
		end
		[noise_yr, noise_xr] = rescale(noise_x, noise_y, (noise_x(end)-noise_x(1))/(bin-1));
		[noise_v, noise_idx] = max(noise_yr);

		if noise_idx >= std_peak_idx
			peak_diff = min(noise_idx-std_peak_idx, std_peak_idx+bin-noise_idx)*noise_R/bin;
		else
			peak_diff = min(std_peak_idx-noise_idx, noise_idx+bin-std_peak_idx)*noise_R/bin;
		end

		disp(sprintf('...run %d : %f', run, peak_diff));
		alldiff(diff_idx) = peak_diff;
		diff_idx = diff_idx+1;
		fprintf(out_fid, '%0.3f\n', peak_diff);
	end
end

if ~SLIENCE
	hold off;
end

diff_mean = mean(alldiff);
diff_std = std(alldiff);

disp(sprintf('mean = %f, std = %f', diff_mean, diff_std));

fclose(out_fid);

return;
