function [s_list, p_list, o_list, i_list] = peak2through(Weight)

disp(sprintf('Weight = %0.3f', Weight));

f_in = 'allgenes.f';
genename_in = 'std2full.txt';

output = 'gene_associated';
alpha = [26 27];
modeltype = '1.1.1';

out_fid = fopen(strcat(output, '_', num2str(sprintf('%0.0f', Weight*100)), '.txt'), 'w');

global DECONV_MU0POS;			% mu0
global DECONV_LAMBDAPOS;	% lambda
global DECONV_DELTAPOS;		% delta
global DECONV_ALPHAPOS;		% alpha
global DECONV_SIGMA0POS;	% sigma_0
global DECONV_SIGMAVPOS;	% sigma_v
global DECONV_BETAPOS;		% beta

global DECONV_DATASET;
global DATASET_OLD;
global DATASET_NEW;

DATASET_OLD = '../../datasets/';
DATASET_NEW = '../../datasets.new/';
DECONV_DATASET = DATASET_NEW;

DECONV_MU0POS = 1;
DECONV_LAMBDAPOS = 2;
DECONV_DELTAPOS = 3;
DECONV_SIGMA0POS = 4;
DECONV_SIGMAVPOS = 5;
DECONV_ALPHAPOS = 6;
DECONV_BETAPOS = 7;

DATA_WT1 = strcat(DECONV_DATASET, 'wt1.txt');
DATA_WT2 = strcat(DECONV_DATASET, 'wt2.txt');

WT1_DIR = 'wt1_budflow/';
WT2_DIR = 'wt2_budflow/';

WT1_TP = 30:16:254;
WT2_TP = 38:16:262;

SMALL = 1e-8;

MODEL_DIR = '../../deconv/models/';

model = {};
model.modeltype = upper(modeltype);
[model] = parseModelType(model);
modelprefix = model.modelprefix;
model.alpha = alpha;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[genes, orfs, alias] = textread(genename_in, '%s\t%s\t%s');
[orfnames, orfids] = map2SystemNames(genes, 1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% data WT1
modelfile1 = strcat(MODEL_DIR, WT1_DIR, modelprefix, '.', int2str(alpha(1)), '.label');
if exist(modelfile1, 'file') ~= 2
	error(sprintf('%s does not exist', modelfile1));
end

model.timepoints = WT1_TP;
[flag, model] = readModelFormat(modelfile1, model);
if flag == 0
	error('Error in parsing model file');
end

model = calcH(model);

dataset1 = load(DATA_WT1, 'ascii');
mu0_1 = model.lengths(DECONV_MU0POS);
lambda_1 = model.lengths(DECONV_LAMBDAPOS);
delta_1 = model.lengths(DECONV_DELTAPOS);
beta_1 = model.lengths(DECONV_BETAPOS);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% data WT2
modelfile2 = strcat(MODEL_DIR, WT2_DIR, modelprefix, '.', int2str(alpha(2)), '.label');
if exist(modelfile2, 'file') ~= 2
	error(sprintf('%s does not exist', modelfile2));
end

[flag, model] = readModelFormat(modelfile2, model);
if flag == 0
	error('Error in parsing model file');
end

model.timepoints = [model.timepoints' WT2_TP']';

model = calcH(model);

dataset2 = load(DATA_WT2, 'ascii');
mu0_2 = model.lengths(DECONV_MU0POS);
lambda_2 = model.lengths(DECONV_LAMBDAPOS);
delta_2 = model.lengths(DECONV_DELTAPOS);
beta_2 = model.lengths(DECONV_BETAPOS);

alpha_mean = (alpha(1)+alpha(2))/2;
mu0 = (mu0_1+mu0_2)/2;
delta = (delta_1+delta_2)/2;
lambda = (lambda_1+lambda_2)/2;
beta = (beta_1+beta_2)/2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model = getRDCidx(model);
t_x = model.t_x;
t_y_idx = model.t_y_idx;
b_x = model.b_x;
b_y_idx = model.b_y_idx;

allF = load(f_in);

g1_range = find(WT1_TP>=mu0_1 & WT1_TP<=mu0_1+lambda_1);
g2_range = find(WT2_TP>=mu0_2 & WT2_TP<=mu0_2+lambda_2);


orig_array = [];
%ptr5_array = [];
Cr10_array = [];
Dr10_array = [];
ptr10_array = [];
%ptr20_array = [];


for i = 1:numel(orfnames)
	orfid = orfids{i};

	curf = allF(orfid,:);

	g1 = dataset1(orfid, :)';
	g2 = dataset2(orfid, :)';

	g1 = g1(g1_range);
	g2 = g2(g2_range);

	orig_p2t = (max(g1)/min(g1) + max(g2)/min(g2))/2;

	orig_array = [orig_array orig_p2t];

	curC = curf(t_y_idx);
	curD = curf(b_y_idx);

	curC_r = rescale(t_x, curC);
	curD_r = rescale(b_x, curD);
	
	ptrC = quantile(curC_r, [5 10 20 80 90 95]./100);
	ptrD = quantile(curD_r, [5 10 20 80 90 95]./100);

	Cr5 = ptrC(6)/ptrC(1);
	Dr5 = ptrD(6)/ptrD(1);
	m5 = ptrScore(Cr5, Dr5, Weight);

	Cr10 = ptrC(5)/ptrC(2);
	Cr10_array = [Cr10_array Cr10];
	Dr10 = ptrD(5)/ptrD(2);
	Dr10_array = [Dr10_array Dr10];
	m10 = ptrScore(Cr10, Dr10, Weight);

	Cr20 = ptrC(4)/ptrC(3);
	Dr20 = ptrD(4)/ptrD(3);
	m20 = ptrScore(Cr20, Dr20, Weight);


%	disp(sprintf('%s\t%0.3f\t%0.3f\t%0.3f', orfs{i}, Cr10, Dr10, m10));

	ptr5_array = [ptr5_array m5];
	ptr10_array = [ptr10_array m10];
	ptr20_array = [ptr20_array m20];

	% 
%	fprintf(out_fid, '%s\t%s\t%s\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f\n', gene, orfs{i}, alias{i}, orig_p2t, Cr5, Dr5, m5, Cr10, Dr10, m10, Cr20, Dr20, m20);
%	fprintf(out_fid, '%s\t%s\t%s\t%0.3f\t%0.3f\t%0.3f\t%0.3f\n', gene, orfs{i}, alias{i}, orig_p2t, m5, m10, m20);

%	disp(sprintf('%s\t%0.3f', orfs{i}, m10));

end

% deal with previous CCR
spellman = 'spellman.txt';
orlando = 'orlando.txt';
premila = 'premila.txt';
inter = 'intersection.txt';

[s_name] = textread(spellman, '%s');
[o_name, o_rank] = textread(orlando, '%s\t%d');
[p_name, p_rank] = textread(premila, '%s\t%d');
[i_name] = textread(inter, '%s');

[values, order] = sort(ptr10_array, 'descend');

s_idx = [];
p_idx = [];
o_idx = [];
i_idx = [];

for i = 1:numel(orfnames)
	n = order(i);
	orf = orfnames{n};

	% spellman
	s_match = strmatch(orf, s_name, 'exact');
	if numel(s_match) > 0
		s_idx = [s_idx 1];
	else
		s_idx = [s_idx 0];
	end

	% premila
	p_match = strmatch(orf, p_name, 'exact');
	if numel(p_match) > 0
		r = p_match(1);
		p_idx = [p_idx r];
	else
		p_idx = [p_idx 0];
	end

	% orlando
	o_match = strmatch(orf, o_name, 'exact');
	if numel(o_match) > 0
		r = min(o_match);
		o_idx = [o_idx r];
	else
		o_idx = [o_idx 0];
	end

	% intersection
	i_match = strmatch(orf, i_name, 'exact');
	if numel(i_match) > 0
		i_idx = [i_idx 1];
	else
		i_idx = [i_idx 0];
	end
end

% writing file
fprintf(out_fid, 'SystematicName\tStandardName\tAliases\tOrig_PTR\tCptr(10)\tDptr(10)PTR(10)\tspellman\tpremila\torlando\tintersection\n');

step = 500;
s_count = 0;
p_count = 0;
o_count = 0;
i_count = 0;
s_all = numel(s_name);
p_all = numel(p_name);
o_all = numel(o_name);
i_all = numel(i_name);

s_list = [];
p_list = [];
o_list = [];
i_list = [];

rank_out = 'Compare.txt';
rank_fid = fopen(rank_out, 'a+');

fprintf(rank_fid, 'Weight=%0.3f\n', Weight);

for i = 1:numel(orfnames)
	n = order(i);
	fprintf(out_fid, '%s\t%s\t%s\t%0.3f\t%0.3f\t%0.3f\t%0.3f', orfnames{n}, orfs{n}, alias{n}, orig_array(n), Cr10_array(n), Dr10_array(n), ptr10_array(n));

	fprintf(out_fid, '\t%d\t%d\t%d\t%d\n', s_idx(i), p_idx(i), o_idx(i), i_idx(i));

	if s_idx(i) > 0
		s_count = s_count+1;
	end

	if p_idx(i) > 0
		p_count = p_count+1;
	end

	if o_idx(i) > 0
		o_count = o_count+1;
	end

	if i_idx(i) > 0
		i_count = i_count+1;
	end

	if rem(i, step) == 0
		fprintf(rank_fid, 'R=%d\ts=%0.0f\tp=%0.0f\to=%0.0f\ti=%0.0f\n', i, s_count/s_all*100, p_count/p_all*100, o_count/o_all*100, i_count/i_all*100);
	end

	s_list = [s_list s_count];
	p_list = [p_list p_count];
	o_list = [o_list o_count];
	i_list = [i_list i_count];
end

s_list = s_list./s_all;
p_list = p_list./p_all;
o_list = o_list./o_all;
i_list = i_list./i_all;

fclose(out_fid);
fclose(rank_fid);
return;

function score = ptrScore(C, D, Weight)
score = power(C, Weight)*power(D, 1-Weight);
%score = C*Weight+D*(1-Weight);
return;
