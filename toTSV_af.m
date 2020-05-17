function [model] = toTSV_af(f_in, f_out)

% ============================= %

if nargin == 0
	f_in = 'af_allgenes/allgenes.AF.f';
	f_out = 'af_allgenes/deconvolved_profiles.AF.tsv';
end

% ============================= %

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

% -------------------------

DECONV_DATASET = '../datasets.new/';

DECONV_MU0POS = 1;
DECONV_LAMBDAPOS = 2;
DECONV_SIGMA0POS = 3;
DECONV_SIGMAVPOS = 4;
DECONV_ALPHAPOS = 5;
DECONV_BETAPOS = 6;

NAME_MAPPING = strcat(DECONV_DATASET, 'std2full.txt');

DIR1 = 'af1/';
DIR2 = 'af2/';

% ============================= %

MODEL_DIR = 'models/';
TP1 = [0:10:40 60:20:180 220:20:300];
TP2 = [0:10:40 60:20:340];

% ==========================================

model = {};

modeltype = 'NORMAL';
model.modeltype = upper(modeltype);
model = parseModelType4Mutant(model);
modelprefix = model.modelprefix;

alpha = [0, 0];
model.alpha = alpha;

datatype = 'JOINT';
model.datatype = datatype;

% ==========================================

% ---------- DATA1 ---------- %
modelfile1 = strcat(MODEL_DIR, DIR1, modelprefix, '.', int2str(alpha(1)), '.label');
if exist(modelfile1, 'file') ~= 2
	error(sprintf('%s does not exist', modelfile1));
end

model.timepoints{1} = TP1;
% read model file
[flag, model] = readModelFormat4Mutant(modelfile1, model);
if flag == 0
	error('Error in parsing model file');
end

mu0_1 = model.lengths(DECONV_MU0POS);
lambda_1 = model.lengths(DECONV_LAMBDAPOS);
beta_1 = model.lengths(DECONV_BETAPOS);

% calculate H
[model] = calcH4Mutant(model);

% ---------- DATA2 ---------- %
modelfile2 = strcat(MODEL_DIR, DIR2, modelprefix, '.', int2str(alpha(2)), '.label');
if exist(modelfile2, 'file') ~= 2
	error(sprintf('%s does not exist', modelfile2));
end

model.timepoints{2} = TP2;
% read model file
[flag, model] = readModelFormat4Mutant(modelfile2, model);
if flag == 0
	error('Error in parsing model file');
end

mu0_2 = model.lengths(DECONV_MU0POS);
lambda_2 = model.lengths(DECONV_LAMBDAPOS);
beta_2 = model.lengths(DECONV_BETAPOS);

% calculate H
[model] = calcH4Mutant(model);

% ======================== %
mu0 = (mu0_1+mu0_2)/2;
lambda = (lambda_1+lambda_2)/2;
alpha = (alpha(1)+alpha(2))/2;

g1_len = (lambda_1*beta_1+lambda_2*beta_2)/2+alpha;
s_len = 0.2*lambda;
g2m_len = lambda-g1_len-s_len;

% ================

% get index of f_i and f_t
f_i = [];
for i = 1:length(model.i_intervals)
	idx = model.i_intervals{i}{2};
	se = model.Hpos{idx};
	f_i = [f_i se(1):1:se(2)];
end

f_t = [];
for i = 1:length(model.t_intervals)
	idx = model.t_intervals{i}{2};
	se = model.Hpos{idx};
	f_t = [f_t se(1):1:se(2)];
end

f_r = setxor(f_i, f_t);

% ================
% i_x = [0, mu0]
offset_i_min = inf;
for i = 1:length(model.iList)
  offset_i_min = min(offset_i_min, min(model.iList{i}'));
end
i_x = [];
for i = 1:length(model.iList)
  list = model.iList{i}';
  i_x = [i_x; list(1:length(list)-1)-offset_i_min];
end

% t_x = [0 lambda]
offset_t_min = inf;
for i = 1:length(model.tList)
  offset_t_min = min(offset_t_min, min(model.tList{i}'));
end
t_x = [];
for i = 1:length(model.tList)
  list = model.tList{i}';
  t_x = [t_x; list(1:length(list)-1)-offset_t_min];
end

% ================

H = load(f_in);
[Sys,Std,Ali]  = textread(NAME_MAPPING, '%s\t%s\t%s\n');

% ================

% headings
out_fid = fopen(f_out, 'w');
fprintf(out_fid, 'SystematicName\tStandardName\tAliases');

for i=1:length(t_x);
	fprintf(out_fid, '\tC(TP%d)', i);
end
fprintf(out_fid, '\n');

fprintf(out_fid, 'clock(min)\t\t');
for i=1:length(t_x);
	fprintf(out_fid, '\t%0.3f', t_x(i));
end
fprintf(out_fid, '\n');

% timing heading

for idx = 1:length(Sys)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	curf = H(idx, :);

	C = curf(f_t);

	fprintf(out_fid, '%s\t%s\t%s', Sys{idx}, Std{idx}, Ali{idx});

	for i=1:length(C)
		fprintf(out_fid, '\t%0.3f', C(i));
	end
	fprintf(out_fid, '\n');
end
fclose(out_fid);
% -----------------------------------
