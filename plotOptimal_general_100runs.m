function [model] = plotOptimal_general_100runs(input)

% =======================

modeltype = '1.1.1';
datatype = 'JOINT';
alpha = [26 27];
gamma = 0;

seg = strsplit('.', input);
orfname = seg{1};

% =======================

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

DECONV_WT1 = 'WT1';
DECONV_WT2 = 'WT2';
DECONV_JOINT = 'JOINT';

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

datatype = upper(datatype);
model.datatype = datatype;
model.alpha = alpha;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% deal with WT1
modelfile1 = strcat(MODEL_DIR, WT1_DIR, modelprefix, '.', int2str(alpha(1)), '.label');
if exist(modelfile1, 'file') ~= 2
	error(sprintf('%s does not exist', modelfile1));
end

%dataset1 = load(DATA_WT1, 'ascii');
%g1 = dataset1(orfid,:)';
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

%dataset2 = load(DATA_WT2, 'ascii');
%g2 = dataset2(orfid,:)';
model.timepoints = [model.timepoints' WT2_TP']';
%model.g = [g1' g2']';

% read model file
[flag, model] = readModelFormat(modelfile2, model);
if flag == 0
	error('Error in parsing model file');
end

% calculate H
[model] = calcH(model);



%%%%%%%%%%%%%%%%
% plot optimal %
%%%%%%%%%%%%%%%%

use_standard = 0;

if use_standard
	mu0 = (94.387+101.904)/2;
	lambda = (79.487+82.014)/2;
	delta = (44.318+37.436)/2;
	beta = (0.153+0.165)/2;
	alpha = [26 27];
else
	mu0 = model.lengths(DECONV_MU0POS);
	lambda = model.lengths(DECONV_LAMBDAPOS);
	delta = model.lengths(DECONV_DELTAPOS);
	alpha = model.alpha;
	beta = model.lengths(DECONV_BETAPOS);
end

% get intervals and labels from model structure
t_x = [];
t_y_idx = [];

b_x = [];
b_y_idx = [];

% b_y_idx = DC
for i = 1:length(model.b_intervals)
	idx = model.b_intervals{i}{2};
	se = model.Hpos{idx};
	b_y_idx = [b_y_idx se(1):1:se(2)];
end

% t_y_idx = C
for i = 1:length(model.t_intervals)
	idx = model.t_intervals{i}{2};
	se = model.Hpos{idx};
	t_y_idx = [t_y_idx se(1):1:se(2)];
end

% b_x = [0 delta+lambda]
offset_b_min = inf;
b_segments = {};
for i = 1:length(model.bList)
	offset_b_min = min(offset_b_min, min(model.bList{i}'));
end
for i = 1:length(model.bList)
	list = model.bList{i}';
	b_x = [b_x; list(1:length(list)-1)-offset_b_min];
	b_segments{i}{1} = list(1)-offset_b_min;
	b_segments{i}{2} = list(end-1)-offset_b_min;
	b_segments{i}{3} = model.b_intervals{i}{1};
end

% t_x = [0 lambda]
t_segments = {};
offset_t_min = inf;
for i = 1:length(model.tList)
	offset_t_min = min(offset_t_min, min(model.tList{i}'));
end
for i = 1:length(model.tList)
	list = model.tList{i}';
	t_x = [t_x; list(1:length(list)-1)-offset_t_min];
	t_segments{i}{1} = list(1)-offset_t_min;
	t_segments{i}{2} = list(end-1)-offset_t_min;
	t_segments{i}{3} = model.t_intervals{i}{1};
end

S_phase = 0.2;
avg_alpha = mean(alpha);

s_interval = S_phase*lambda;

% -------------------
xticks_bottom = [0];
xticks_label_bottom = {'|'};
for i = 1:length(b_segments)
	name = b_segments{i}{3};
	if strcmp(name, 'postG1')
		smid = b_segments{i}{1} + s_interval/2;
		xticks_bottom = [xticks_bottom smid b_segments{i}{1}+s_interval];

		g2mmid = (b_segments{i}{1}+s_interval+b_segments{i}{2})/2;
		xticks_bottom = [xticks_bottom g2mmid b_segments{i}{2}];
		xticks_label_bottom = {xticks_label_bottom{:}, 'S', '|', 'G2/M', '|'};
		i=i+1;
	else
		mid = (b_segments{i}{1}+b_segments{i}{2})/2;
		xticks_bottom = [xticks_bottom mid b_segments{i}{2}];
		xticks_label_bottom = {xticks_label_bottom{:}, b_segments{i}{3}, '|'};
	end
end

xticks_top = [0];
xticks_label_top = {''};
for i = 1:length(t_segments)
	name = t_segments{i}{3};
	if strcmp(name, 'postG1')
		smid = t_segments{i}{1} + s_interval/2;
		xticks_top = [xticks_top smid t_segments{i}{1}+s_interval];

		g2mmid = (t_segments{i}{1}+s_interval+t_segments{i}{2})/2;
		xticks_top = [xticks_top g2mmid t_segments{i}{2}];
		xticks_label_top = {xticks_label_top{:}, 'S', '|', 'G2/M', '|'};
		i=i+1;
	else
		mid = (t_segments{i}{1}+t_segments{i}{2})/2;
		xticks_top = [xticks_top mid t_segments{i}{2}];
		xticks_label_top = {xticks_label_top{:}, t_segments{i}{3}, '|'};
	end
end

allF = load(input);
nrow = size(allF, 1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

y_lim = 1.2;

ymax = max(max(allF(:,[t_y_idx b_y_idx])));
allF = allF./ymax;

grey = [141 141 141]./255;

if strcmp(DECONV_JOINT, model.datatype)
	titlename = sprintf('%s ', orfname);
else
	titlename = sprintf('%s ', orfname);
end
	
fig1 = figure;
set(fig1, 'OuterPosition', [400 600 600 600]);
heigth = 0.38;
w = 0.88;
maxl = b_x(end);
top_base = 0.54;
bottom_base = 0.07;
left = 0.05;

% top branch
curw = lambda/maxl*w;
h2 = axes('Position', [left, top_base, curw, heigth]);
for i = 1:nrow
	t_y = allF(i, t_y_idx);
	plot(t_x, t_y, 'color', grey, 'linewidth', 0.5);
	hold on;
end

set(h2, 'XTick', xticks_top);
set(h2, 'XTickLabel', xticks_label_top);
set(h2, 'Ytick', []);
set(h2, 'YTickLabel', {});
xlim([0 t_x(end)]);
ylim([0 y_lim]);
title(strcat(titlename, ' (mother)'));

% bottom branch
curw = w;
h3 = axes('Position', [left, bottom_base, curw, heigth]);
for i = 1:nrow
	b_y = allF(i, b_y_idx);
	plot(b_x, b_y, 'color', grey, 'linewidth', 0.5);
	hold on;
end

set(h3, 'XTick', xticks_bottom);
set(h3, 'XTickLabel', xticks_label_bottom);
set(h3, 'Ytick', []);
set(h3, 'YTickLabel', {});
xlim([0 b_x(end)]);
ylim([0 y_lim]);
title(strcat(titlename, ' (daughter)'));

return;
