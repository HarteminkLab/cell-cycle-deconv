function drawArea(modeltype, datatype, alpha)

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

figure_flag = 0;

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

WT1_G = 30:16:254;
WT1_TP = 0:1:max(WT1_G);
WT2_G = 38:16:262;
WT2_TP = 0:1:max(WT2_G);

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
[model] = parseModelType(model);
modelprefix = model.modelprefix;

% ===================================
% ===================================

% deal with orfnames and datatype

turn_on_genes = 1;

if turn_on_genes
	orfnames = {'DSE1', 'DSE2', 'DSE3', 'DSE4'};
else
	orfnames = {};
end

datatype = upper(datatype);
model.datatype = datatype;
model.alpha = alpha;
model.modeltype = upper(modeltype);

g_matrix = [];

% WT1
if strcmp(datatype, DECONV_WT1)
	modelfile = strcat(MODEL_DIR, WT1_DIR, modelprefix, '.', int2str(alpha(1)), '.label'); 
	if exist(modelfile, 'file') ~= 2
		error(sprintf('%s does not exist', modelfile));
	end

	dataset = load(DATA_WT1, 'ascii');
	for idx = 1:length(orfnames)
		[orfname, orfid] = map2SystemNames(orfnames{idx});
		g_matrix = [g_matrix; dataset(orfid, :)];
	end

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
	for idx = 1:length(orfnames)
		[orfname, orfid] = map2SystemNames(orfnames{idx});
		g_matrix = [g_matrix; dataset(orfid, :)];
	end

	model.timepoints = WT2_TP;

	% read model file
	[flag, model] = readModelFormat(modelfile, model);
	if flag == 0
		error('Error in parsing model file');
	end

	% calculate H
	[model] = calcH(model);
else
	error('wrong data type (expect WT1/WT2)');
end

% ============ %
% == draw H == %
% ============ %
H = model.H;

if figure_flag
	fig1 = figure;
	set(fig1, 'OuterPosition', [400 50 800 1100]);
end

alpha = mean(alpha);
n = size(H,1);
lengths = model.lengths;
mu0 = lengths(DECONV_MU0POS);
delta = lengths(DECONV_DELTAPOS);
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
	elseif strcmp(branch, 'b')
		seg_idx = model.b_intervals{branch_idx}{2};
		list = model.bList{branch_idx};
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

linecolors = getLineColors();

y_area = [];

for i=1:n
	y = H(i,:);

	if strcmp(datatype, DECONV_WT1)
		fprintf('Time = %d: ', WT1_TP(i));
	elseif strcmp(datatype, DECONV_WT2)
		fprintf('Time = %d: ', WT2_TP(i));
	else
		if i<=n/2
			fprintf('Rep1 Time = %d: ', WT1_TP(i));
		else
			fprintf('Rep2 Time = %d: ', WT2_TP(i-n/2));
		end
	end

	if figure_flag
		subplot(n,1,i);
	end
	weight = 0;
	for idx = 1:length(model.relations)
		if figure_flag
			plot(intervals{idx}{4}, y(intervals{idx}{1})./intervals{idx}{2}, 'color', colors{idx}, 'linewidth', 2);
			hold on;
		end
		cur_weight = sum(y(intervals{idx}{1}));
		fprintf('%s = %0.3g, ', names{idx}, cur_weight);
		y_area(i,idx) = cur_weight;
			
		weight = weight+cur_weight;
	end

	fprintf('total weight = %0.3g\n', weight);

	if figure_flag
		xlim([0 range]);
	%	ymax = 0.020;
	%	ylim([0 ymax]);

		max_y = max(y(intervals{idx}{1}));
	%	set(gca, 'ytick', max_y/2);
	%	set(gca, 'ytick', ymax/2);
		set(gca, 'ytick', 0);
	end

	if strcmp(datatype, DECONV_WT1)
		str = sprintf('%d min', WT1_TP(i));
	elseif strcmp(datatype, DECONV_WT2)
		str = sprintf('%d min', WT2_TP(i));
	else
		if i<=n/2
			str = sprintf('Rep1: %d min', WT1_TP(i));
		else
			str = sprintf('Rep2: %d min', WT2_TP(i-n/2));
		end
	end

	if figure_flag
		set(gca, 'yticklabel', str);

		if (i~=n)
			set(gca, 'xtick', []);
		else
			set(gca, 'xtick', mid_range);
			set(gca, 'xticklabel', names);
		end
	end
end

linecolors = gray(6);

[y_area, names] = rowSwitch(y_area, names, 1, 2);
[y_area, names] = rowSwitch(y_area, names, 2, 3);
[y_area, names] = rowSwitch(y_area, names, 3, 4);

if figure_flag == 0
	fig1 = figure;
%	h1 = axes('Position', [0.05 0.25, 0.9, 0.7]);
	if strcmp(datatype, DECONV_WT1)
		area(WT1_TP, y_area);
	else
		area(WT2_TP, y_area);
	end
	ylim([0 1]);
	title(sprintf('%s, %s, %s, %d', regexprep(model.modeltype, '_', '-'), regexprep(model.modeltype, '_', '-'), datatype, alpha));
	grid on;
	set(gca,'Layer','top');
	colormap summer;
	xlabel('Time(min)');

	hold on;
	factor = 0.4;
%	factor = 1;
	if strcmp(datatype, DECONV_WT1)
		for idx = 1:length(orfnames)
			plot(WT1_G, g_matrix(idx,:)./max(g_matrix(idx,:))*factor, '-o', 'lineWidth', 4, 'color', linecolors(idx,:));
		end
	else
		for idx = 1:length(orfnames)
			plot(WT2_G, g_matrix(idx,:)./max(g_matrix(idx,:))*factor, '-o', 'lineWidth', 4, 'color', linecolors(idx,:));
		end
	end
	legend({names{:},orfnames{:}});
end

return;

function [y, names] = rowSwitch(y, names, a, b)
temp = y(:,a);
y(:,a) = y(:,b);
y(:,b) = temp;

temp_n = names{a};
names{a} = names{b};
names{b} = temp_n;
return;
