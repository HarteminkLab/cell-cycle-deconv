function drawBranch(modeltype, datatype, alpha)

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
WT2_TP = 38:16:262;

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
% ===================================

% deal with orfname and datatype
datatype = upper(datatype);
model.datatype = datatype;
model.alpha = alpha;
model.modeltype = upper(modeltype);

% WT1
if strcmp(datatype, DECONV_WT1)
	modelfile = strcat(MODEL_DIR, WT1_DIR, modelprefix, '.', int2str(alpha(1)), '.label'); 
	if exist(modelfile, 'file') ~= 2
		error(sprintf('%s does not exist', modelfile));
	end

	% read model file
	[flag, model] = readModelFormat(modelfile, model);
	if flag == 0
		error('Error in parsing model file');
	end

%%% datatype = WT2
elseif strcmp(datatype, DECONV_WT2)
	modelfile = strcat(MODEL_DIR, WT2_DIR, modelprefix, '.', int2str(alpha(1)), '.label');
	if exist(modelfile, 'file') ~= 2
		error(sprintf('%s does not exist', modelfile));
	end

	% read model file
	[flag, model] = readModelFormat(modelfile, model);
	if flag == 0
		error('Error in parsing model file');
	end

%%% datatype = JOINT
elseif strcmp(datatype, DECONV_JOINT)
	% deal with WT1
	modelfile1 = strcat(MODEL_DIR, WT1_DIR, modelprefix, '.', int2str(alpha(1)), '.label');
	if exist(modelfile1, 'file') ~= 2
		error(sprintf('%s does not exist', modelfile1));
	end

	% read model file
	[flag, model] = readModelFormat(modelfile1, model);
	if flag == 0
		error('Error in parsing model file');
	end

	% ============= %
	% deal with WT2 %
	% ============= %
	modelfile2 = strcat(MODEL_DIR, WT2_DIR, modelprefix, '.', int2str(alpha(2)), '.label');
	if exist(modelfile2, 'file') ~= 2
		error(sprintf('%s does not exist', modelfile2));
	end

	% read model file
	[flag, model] = readModelFormat(modelfile2, model);
	if flag == 0
		error('Error in parsing model file');
	end

else
	error('wrong data type (expect WT1/WT2/JOINT)');
end

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

i_segments = {}; 
offset_i_min = inf;
%offset_i_max = -inf;
for i = 1:length(model.iList)
  offset_i_min = min(offset_i_min, min(model.iList{i}'));
end
for i = 1:length(model.iList)
	list = model.iList{i}';
	i_segments{i}{1} = list(1)-offset_i_min;
	i_segments{i}{2} = list(end-1)-offset_i_min;
	i_segments{i}{3} = model.i_intervals{i}{1};
end

t_segments = {};
offset_t_min = inf;
for i = 1:length(model.tList)
	offset_t_min = min(offset_t_min, min(model.tList{i}'));
end
for i = 1:length(model.tList)
	list = model.tList{i}';
	t_segments{i}{1} = list(1)-offset_t_min;
	t_segments{i}{2} = list(end-1)-offset_t_min;
	t_segments{i}{3} = model.t_intervals{i}{1};
end

offset_b_min = inf;
b_segments = {};
for i = 1:length(model.bList)
	offset_b_min = min(offset_b_min, min(model.bList{i}'));
end
for i = 1:length(model.bList)
	list = model.bList{i}';
	b_segments{i}{1} = list(1)-offset_b_min;
	b_segments{i}{2} = list(end-1)-offset_b_min;
	b_segments{i}{3} = model.b_intervals{i}{1};
end

ticks_rc = [0];
ticks_label_rc = {'|'};
for i=1:length(i_segments)
	mid = (i_segments{i}{1}+i_segments{i}{2})/2;
	ticks_rc = [ticks_rc mid i_segments{i}{2}];
	ticks_label_rc{2*i} = i_segments{i}{3};
	ticks_label_rc{2*i+1} = '|';
end
offset = max(ticks_rc);
offset_idx = length(i_segments);
for i=1:length(t_segments)
	mid = (t_segments{i}{1}+t_segments{i}{2})/2 + offset;
	ticks_rc = [ticks_rc mid t_segments{i}{2}+offset];
	ticks_label_rc{2*(i+offset_idx)} = t_segments{i}{3};
	ticks_label_rc{2*(i+offset_idx)+1} = '|';
end

ticks_d = [0];
ticks_label_d = {'|'};
for i=1:length(b_segments)
	mid = (b_segments{i}{1}+b_segments{i}{2})/2;
	ticks_d = [ticks_d mid b_segments{i}{2}];
	ticks_label_d{2*i} = b_segments{i}{3};
	ticks_label_d{2*i+1} = '|';
end

% ===================
% get colors
%colors = getStructureColors();

branch_names = {'postG1', ...
	'R', 'C', 'D', ...
	'RG1', 'CG1', 'DG1' ...
};

colors = {[166 206 227], ...
[217 150 148], [147 205 221], [250 192 144], ...
[251 180 174], [198 219 239], [254 217 166], ...
};
for i=1:size(colors, 2)
	colors{i} = colors{i}./255;
end

linecolor = [127 127 127]./255;

% drawing

fig1 = figure;
height = 0.2;
set(fig1, 'OuterPosition', [1, 1, 900, 300]);
w = 0.8;
maxw = i_segments{end}{2}+b_segments{end}{2};

% RCC
baseline = 0.2;
curw = w/maxw*(i_segments{end}{2}+t_segments{end}{2});
left = 0.1;
h1 = axes('Position', [left, 0.5+baseline, curw, height]);

% ===================


for i=1:length(i_segments)
	if i==1
		s = i_segments{i}{1};
	else
		s = i_segments{i-1}{2};
	end
	e = i_segments{i}{2};
	X = [s s e e];
	Y = [0 1 1 0];
	idx = strmatch(i_segments{i}{3}, branch_names);
	c = colors{idx};
	fill(X, Y, c, 'linewidth', 3, 'edgecolor', linecolor);
	hold on;
end
offset = i_segments{end}{2};
for i=1:length(t_segments)
	if i==1
		s = t_segments{i}{1}+offset;
	else
		s = t_segments{i-1}{2}+offset;
	end
	e = t_segments{i}{2}+offset;
	X = [s s e e];
	Y = [0 1 1 0];
	idx = strmatch(t_segments{i}{3}, branch_names);
	c = colors{idx};
	fill(X, Y, c, 'linewidth', 3, 'edgecolor', linecolor);
	hold on;
end
ylim([0 1]);
xlim([i_segments{1}{1} t_segments{end}{2}+offset]);
set(h1, 'ytick', []);
set(h1, 'xtick', []);

%set(h1, 'xtick', ticks_rc);
set(h1, 'xtick', []);
%set(h1, 'xtickLabel', ticks_label_rc);
for i=2:2:length(ticks_label_rc)
	l = ticks_label_rc{i};
	x = ticks_rc(i);
	text(x, 0.5, l, 'VerticalAlignment', 'middle', 'HorizontalAlignment','center', 'color', 'black', 'fontsize', 18, 'Fontname', 'Times', 'FontWeight', 'Bold');
end

% DC
curw = w/maxw*(b_segments{end}{2});
left = 0.9-curw;
h2 = axes('Position', [left, baseline+0.025, curw, height]);

for i=1:length(b_segments)
	if i==1
		s = b_segments{i}{1};
	else
		s = b_segments{i-1}{2};
	end
	e = b_segments{i}{2};
	X = [s s e e];
	Y = [0 1 1 0];
	idx = strmatch(b_segments{i}{3}, branch_names);
	c = colors{idx};
	fill(X, Y, c, 'linewidth', 3, 'edgecolor', linecolor);
	hold on;
end
ylim([0 1]);
xlim([b_segments{1}{1} b_segments{end}{2}]);
set(h2, 'ytick', []);
set(h2, 'xtick', []);

for i=2:2:length(ticks_label_d)
	l = ticks_label_d{i};
	x = ticks_d(i);
	text(x, 0.5, l, 'VerticalAlignment', 'middle', 'HorizontalAlignment','center', 'color', 'black', 'fontsize', 18, 'Fontname', 'Times', 'FontWeight', 'Bold');
end

%title(model.modeltype, 'Interpreter', 'none');

return;

