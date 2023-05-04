
addpath(genpath('lib/YAMLMatlab'));
addpath(genpath('deconv')) 
addpath(genpath('analysis'))

outdir = 'output/genes';

% ----------- Find Optimal Rewrite / Refactor -----------

% Parameters
gamma_val = 0.01;
genename = "CLB2";

fprintf("%s...\n", genename);
model = Model(genename, gamma_val);

tic;

[model, rns, gammas, lruns, rruns] = findOptimal(model);

toc;

%model.gm = 0.0074;

% ------------------------------------------------------------------------------------

WT1_TP = [0	20	30	40	50	60	70	80	90	100	120	130	140];
WT2_TP = [0	20	30	40	50	60	70	80	90	100	120	130	140];


model = Model(genename, gamma_val);

model.g1 = (clb2_chrom(:, 1, 3, 9)' + 1)*10;
model.g2 = (clb2_chrom(:, 2, 3, 9)' + 1)*10;
model.g = [model.g1 model.g2];

model.H = [];

% calculate H for WT1
model.timepoints = WT1_TP;
[H1, Hsegments, Hpos] = calcH(model);
model.Hsegments = Hsegments;
model.Hpos = Hpos;

% calculate H for WT2 and combine into a joint H
model.timepoints = WT2_TP;
[H2, Hsegments, Hpos] = calcH(model);
model.H = [H1' H2']';

model.timepoints1 = WT1_TP;
model.timepoints2 = WT2_TP;
model.timepoints = [WT1_TP WT2_TP];

model.gm = 0.02;
model = deconvolve(model);
drawDeconvolved(model);

clb2_chrom = h5read('/Users/trung/Research/chromatin-transformers/output/deconvolution/clb2_3x16.h5', '/data');
% permute the dimensions to match the original order in the NumPy array
clb2_chrom = permute(clb2_chrom, [4 3 2 1]);

