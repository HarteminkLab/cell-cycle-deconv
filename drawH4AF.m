function drawH4AF(modeltype, datatype, alpha)

global DECONV_MU0POS;
global DECONV_LAMBDAPOS;
global DECONV_ALPHAPOS;
global DECONV_SIGMA0POS;
global DECONV_SIGMAVPOS;
global DECONV_BETAPOS;

global DECONV_KERNEL;
global DECONV_WAVELET;
global DECONV_DIFF;

global DECONV_JOINT;
global DECONV_AF1;
global DECONV_AF2;

global DECONV_DATASET;

DECONV_MU0POS = 1;
DECONV_LAMBDAPOS = 2;
DECONV_SIGMA0POS = 3;
DECONV_SIGMAVPOS = 4;
DECONV_ALPHAPOS = 5;
DECONV_BETAPOS = 6;

% 1: wavelet (uneven intervals)
% 2: 1st-order different operator (even intervals)
DECONV_WAVELET = 1;
DECONV_DIFF = 2;
DECONV_KERNEL = DECONV_WAVELET;

DECONV_DATASET = '../datasets.new/';

DATA_AF1 = strcat(DECONV_DATASET, 'af1.txt');
DATA_AF2 = strcat(DECONV_DATASET, 'af2.txt');

SMALL = 1e-8;

if nargin < 1
	error('... Expect one input: datatype(AF1/AF2/JOINT)');
end

AF1_DIR = 'af1/';
AF2_DIR = 'af2/';

AF1_TP = [0:10:40 60:20:180 220:20:300];
AF2_TP = [0:10:40 60:20:340];

SMALL = 1e-8;

DECONV_AF1 = 'AF1';
DECONV_AF2 = 'AF2';
DECONV_JOINT = 'JOINT';

if DECONV_KERNEL == DECONV_WAVELET
	MODEL_DIR = 'models/';
elseif DECONV_KERNEL == DECONV_DIFF
	MODEL_DIR = 'models_even/';
end

% ===================================
model = {};
model.modeltype = upper(modeltype);
model = parseModelType4Mutant(model);
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

% AF1
if strcmp(datatype, DECONV_AF1)
	modelfile = strcat(MODEL_DIR, AF1_DIR, modelprefix, '.', int2str(alpha(1)), '.label'); 
	if exist(modelfile, 'file') ~= 2
		error(sprintf('%s does not exist', modelfile));
	end

	dataset = load(DATA_AF1, 'ascii');
	g = dataset(orfid,:)';
	model.g{1} = g;
	model.timepoints{1} = AF1_TP;

	% read model file
	[flag, model] = readModelFormat4Mutant(modelfile, model);
	if flag == 0
		error('Error in parsing model file');
	end

	% calculate H
	[model] = calcH4Mutant(model);

%%% datatype = AF2
elseif strcmp(datatype, DECONV_AF2)
	modelfile = strcat(MODEL_DIR, AF2_DIR, modelprefix, '.', int2str(alpha(1)), '.label');
	if exist(modelfile, 'file') ~= 2
		error(sprintf('%s does not exist', modelfile));
	end

	dataset = load(DATA_AF2, 'ascii');
	g = dataset(orfid,:)';
	model.g{1} = g;
	model.timepoints{1} = AF2_TP;

	% read model file
	[flag, model] = readModelFormat4Mutant(modelfile, model);
	if flag == 0
		error('Error in parsing model file');
	end

	% calculate H
	[model] = calcH4Mutant(model);

%%% datatype = JOINT
elseif strcmp(datatype, DECONV_JOINT)
	% deal with AF1
	modelfile1 = strcat(MODEL_DIR, AF1_DIR, modelprefix, '.', int2str(alpha(1)), '.label');
	if exist(modelfile1, 'file') ~= 2
		error(sprintf('%s does not exist', modelfile1));
	end

	dataset1 = load(DATA_AF1, 'ascii');
	g1 = dataset1(orfid,:)';
	model.timepoints{1} = AF1_TP;
	model.g{1} = g1;

	% read model file
	[flag, model] = readModelFormat4Mutant(modelfile1, model);
	if flag == 0
		error('Error in parsing model file');
	end

	% calculate H
	[model] = calcH4Mutant(model);

	% ============= %
	% deal with AF2 %
	% ============= %
	modelfile2 = strcat(MODEL_DIR, AF2_DIR, modelprefix, '.', int2str(alpha(2)), '.label');
	if exist(modelfile2, 'file') ~= 2
		error(sprintf('%s does not exist', modelfile2));
	end

	dataset2 = load(DATA_AF2, 'ascii');
	g2 = dataset2(orfid,:)';
	model.timepoints{2} = AF2_TP;
	model.g{2} = g2;

	% read model file
	[flag, model] = readModelFormat4Mutant(modelfile2, model);
	if flag == 0
		error('Error in parsing model file');
	end

	% calculate H
	[model] = calcH4Mutant(model);
else
	error('wrong data type (expect AF1/AF2/JOINT)');
end

% ============ %
% == draw H == %
% ============ %
H = model.H;

fig1 = figure;
set(fig1, 'OuterPosition', [400 50 800 1100]);

alpha = mean(alpha);
n = size(H,1);
lengths = model.lengths;
mu0 = lengths(DECONV_MU0POS);
lambda = lengths(DECONV_LAMBDAPOS);
beta = lengths(DECONV_BETAPOS);

names = {};
intervals = {};
range = 0;
mid_range = [];

for idx = 1:length(model.relations)
	names{idx} = model.relations{idx}{1};
	branch = model.relations{idx}{2};
	branch_idx = model.relations{idx}{3};
	branch_idx = str2num(branch_idx)+1;
	
	if strcmp(branch, 'i')
		seg_idx = model.i_intervals{branch_idx}{2};
		list = model.iList{branch_idx};
	elseif strcmp(branch, 't')
		seg_idx = model.t_intervals{branch_idx}{2};
		list = model.tList{branch_idx};
	end
	seg = model.Hpos{seg_idx};
	seg_start = seg(1);
	seg_end = seg(2);
	% intervals
	% (1) segment range
	% (2) gap
	% (3) range
	% (4) x-axis
	intervals{idx}{1} = [seg_start:1:seg_end];
	intervals{idx}{2} = list(2)-list(1);
	intervals{idx}{3} = [range range+list(end-1)-list(1)];
	intervals{idx}{4} = range-list(1)+list(1:end-1);
	mid_range = [mid_range range+(list(end)-list(1))/2];
	range = range+list(end)-list(1);
end

colors = getLineColors();
for i=1:n
	y = H(i,:);

	if strcmp(datatype, DECONV_AF1)
		fprintf('Time = %d: ', AF1_TP(i));
	elseif strcmp(datatype, DECONV_AF2)
		fprintf('Time = %d: ', AF2_TP(i));
	else
		if i<=n/2
			fprintf('Rep1 Time = %d: ', AF1_TP(i));
		else
			fprintf('Rep2 Time = %d: ', AF2_TP(i-n/2));
		end
	end

	subplot(n,1,i);
	weight = 0;
	for idx = 1:length(model.relations)
		plot(intervals{idx}{4}, y(intervals{idx}{1})./intervals{idx}{2}, 'color', colors{idx}, 'linewidth', 2);
		hold on;
		cur_weight = sum(y(intervals{idx}{1}));
		fprintf('%s = %0.3g, ', names{idx}, cur_weight);
		weight = weight+cur_weight;
	end
	xlim([0 range]);
%	ymax = 0.020;
%	ylim([0 ymax]);

	fprintf('total weight = %0.3g\n', weight);

	max_y = max(y(intervals{idx}{1}));
%	set(gca, 'ytick', max_y/2);
%	set(gca, 'ytick', ymax/2);
	set(gca, 'ytick', 0);

	if strcmp(datatype, DECONV_AF1)
		str = sprintf('%d min', AF1_TP(i));
	elseif strcmp(datatype, DECONV_AF2)
		str = sprintf('%d min', AF2_TP(i));
	else
		if i<=n/2
			str = sprintf('Rep1: %d min', AF1_TP(i));
		else
			str = sprintf('Rep2: %d min', AF2_TP(i-n/2));
		end
	end
	set(gca, 'yticklabel', str);

	if (i~=n)
		set(gca, 'xtick', []);
	else
		set(gca, 'xtick', mid_range);
		set(gca, 'xticklabel', names);
	end
end

return;
