function analysis_all()

f_in = 'allgenes/allgenes.f';
short_in = 'allgenes/commongenelist.txt';

fprintf("Common gene list: %s\n", short_in);
fprintf("All genes: %s\n", f_in);

% script to cluster genes into different categories

% ============================= %
write_score = 0;
R_mass_cutoff = 0.5;
R_peak_ratio = 2;

% ============================= %

outdir = './output/';
postfix = '';

genes_path = strcat(outdir, 'R_only', postfix, '.genes');
orf_genes_path = strcat(outdir, 'R_only', postfix, '.orf.genes');

R_only_fid = fopen(genes_path, 'w');
R_only_orf_fid = fopen(orf_genes_path, 'w');

% % ============================= %

R_genes = {};
R_orf_genes = {};
R_score = [];

% ============================= %

global DECONV_MU0POS;
global DECONV_LAMBDAPOS;
global DECONV_DELTAPOS;
global DECONV_ALPHAPOS;
global DECONV_SIGMA0POS;
global DECONV_SIGMAVPOS;
global DECONV_BETAPOS;

global DECONV_JOINT;
global DECONV_WT1;
global DECONV_WT2;

global DECONV_KERNEL;
global DECONV_WAVELET;
global DECONV_DIFF;

global DECONV_DATASET;
global DATASET_OLD;
global DATASET_NEW;

DECONV_MU0POS = 1;
DECONV_LAMBDAPOS = 2;
DECONV_DELTAPOS = 3;
DECONV_SIGMA0POS = 4;
DECONV_SIGMAVPOS = 5;
DECONV_ALPHAPOS = 6;
DECONV_BETAPOS = 7;
DECONV_DATASET = 'datasets/';

DECONV_WT1 = 'WT1';
DECONV_WT2 = 'WT2';
DECONV_JOINT = 'JOINT';

WT1_TP = 30:16:254;
WT2_TP = 38:16:262;

WT1_DIR = 'wt1_budflow/';
WT2_DIR = 'wt2_budflow/';

% ============================= %

MODEL_DIR = 'models/';

model = {};
model.modeltype = 'CDG1';
model.modelprefix = '1.1.1';
model.modeltype = 'JOINT';
alpha = [26 27];
model.alpha = alpha;

% ============= %
% deal with WT1 %
% ============= %
modelfile = strcat(MODEL_DIR, WT1_DIR, model.modelprefix, '.', int2str(alpha(1)), '.label');
fprintf("Reading model file: %s\n", modelfile);
if exist(modelfile, 'file') ~= 2
	error(sprintf('%s does not exist', modelfile));
end

% read model file
[flag, model] = readModelFormat(modelfile, model);
if flag == 0
	error('Error in parsing model file');
end

mu0_1 = model.lengths(DECONV_MU0POS);
lambda_1 = model.lengths(DECONV_LAMBDAPOS);
delta_1 = model.lengths(DECONV_DELTAPOS);
beta_1 = model.lengths(DECONV_BETAPOS);

model.timepoints = WT1_TP;

[model] = calcH(model);

% ============= %
% deal with WT2 %
% ============= %
modelfile = strcat(MODEL_DIR, WT2_DIR, model.modelprefix, '.', int2str(alpha(2)), '.label');
if exist(modelfile, 'file') ~= 2
	error(sprintf('%s does not exist', modelfile));
end

% read model file
[flag, model] = readModelFormat(modelfile, model);
if flag == 0
	error('Error in parsing model file');
end

mu0_2 = model.lengths(DECONV_MU0POS);
lambda_2 = model.lengths(DECONV_LAMBDAPOS);
delta_2 = model.lengths(DECONV_DELTAPOS);
beta_2 = model.lengths(DECONV_BETAPOS);

model.timepoints = [WT1_TP' WT2_TP']';
[model] = calcH(model);

% ======================== %
mu0 = (mu0_1+mu0_2)/2;
lambda = (lambda_1+lambda_2)/2;
delta = (delta_1+delta_2)/2;
alpha = (alpha(1)+alpha(2))/2;

g1_len = (lambda_1*beta_1+lambda_2*beta_2)/2+alpha;
s_len = 0.2*lambda;
g2m_len = lambda-g1_len-s_len;

% ================
[orig_orfnames] = textread(short_in, '%s');

[shortnames, shortids] = map2SystemNames(orig_orfnames, 1);
H = load(f_in);
% ================

% ================
model = getRDCidx(model);
% ================

t_x = model.t_x;
t_y_idx = model.t_y_idx;

b_x = model.b_x;
b_y_idx = model.b_y_idx;

r_x = model.r_x;
r_y_idx = model.r_y_idx;

bin = 100;

for idx = 1:length(shortnames)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	gene = orig_orfnames{idx};
	gene_orf = shortnames{idx};
	Hidx = shortids{idx};

	curf = H(Hidx, :);

	r_y = curf(r_y_idx);
	t_y = curf(t_y_idx);
	b_y = curf(b_y_idx);

	% --------------------------------
	rmass = myMass(r_y, r_x);
	tmass = myMass(t_y, t_x);
	bmass = myMass(b_y, b_x);
	allmass = rmass+tmass+bmass;

	tbravg = (rmass+tmass+bmass)/(t_x(end)-t_x(1) + b_x(end)-b_x(1) + r_x(end)-r_x(1));

	% rescale mother and daughter cells programs
	[t_yr, t_xr] = rescale(t_x, t_y, (t_x(end)-t_x(1))/bin);
	[b_yr, b_xr] = rescale(b_x, b_y, (b_x(end)-b_x(1))/bin);
	[r_yr, r_xr] = rescale(r_x, r_y, (r_x(end)-r_x(1))/bin);

	t_max = max(t_yr);
	b_max = max(b_yr);
	r_max = max(r_yr);

	% R only genes
	perR = rmass/allmass;
	if perR >= R_mass_cutoff && r_max/max(t_max, b_max) >= R_peak_ratio
		R_genes = {R_genes{:} gene};
		R_orf_genes = {R_orf_genes{:} gene_orf};
		R_score = [R_score perR];
	end

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end

% ----------------------------------

[R_score, order] = sort(R_score, 'descend');
for i=1:numel(R_genes)
	n = order(i);
	if write_score
		fprintf(R_only_fid, '%s\t%0.2f\n', R_genes{n}, R_score(i));
	else
		fprintf(R_only_fid, '%s\n', R_genes{n});
		fprintf(R_only_orf_fid, '%s\n', R_orf_genes{n});
	end
end

fclose(R_only_fid);
fclose(R_only_orf_fid);
% -----------------------------------
