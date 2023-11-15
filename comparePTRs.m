
addpath(genpath('lib/YAMLMatlab'));
addpath(genpath('deconv'))
addpath(genpath('analysis'))

% This script will be used to validate that the peak to trough function
% I am using is computing the PTR values properly..

fprintf("First we will compare the PTR values for a single gene. (1) published PTR value, (2) the PTR value computed from the published deconvolved gene expression, and finally (3) the calculated PTR value for my deconvolved gene expression \n");

% ---- 1. Load the gene association data from disk and the gene expression ----
gene_table_path = '/Users/trung/Research/cell-cycle/data/dataset_from_deconv_web/gene_associated.tsv';
gene_table = readtable(gene_table_path, "FileType","text",'Delimiter', '\t');

deconvolved_ge_path = '/Users/trung/Research/cell-cycle/data/dataset_from_deconv_web/deconvolved_profiles.tsv';
deconvolved_ge = readtable(deconvolved_ge_path, "FileType", "text",'Delimiter', '\t');

% ---- 2. Choose a gene, CLN2 to start ----

gene_name = 'CLN2';

% --- Load the gene expression data for CLN2, then the C and D rows ---

cln2_ge_row = deconvolved_ge(strcmp(deconvolved_ge.StandardName, gene_name), :);

% ---- 3. Separate out the common (mother) and daughter values, f ----

% Get gene expression data for just the common/mother columns
common_columns = deconvolved_ge.Properties.VariableNames;
c_columns = startsWith(common_columns, 'C_');
cln2_c_ge = cell2mat(table2cell(cln2_ge_row(:, c_columns)));

d_columns = startsWith(common_columns, 'D_');
cln2_d_ge = cell2mat(table2cell(cln2_ge_row(:, d_columns)));

% ---- 4. Compute and compare the published PTR ----

% Print the published deconvolved PTR
cln2_row = gene_table(strcmp(gene_table.StandardName, gene_name), :);

cln2_deconvolved_ptr = cln2_row.DeconvolvedPTR_80pt_20pt_;
fprintf("  (1) The published deconvolved PTR (80/20) for CLN2 is: %.3f\n", cln2_deconvolved_ptr);

computedPtr = peak2trough(cln2_c_ge, cln2_d_ge);
fprintf("  (2) The computed PTR (80/20) for the published ge: %.3f\n", computedPtr);

% ---- 5. Deconvolve the gene expression for CLN2 -----

% Deconvolve the gene expression, (fixed gamma, precomputed)
config = DeconvolutionConfig.xg_gene_expression_config();
gene = 'CLN2';
model = Model(config, gene, 0.00429);
model = deconvolve(model);
my_computed_ptr = peak2troughModel(model);
fprintf("  (3) The computed PTR (80/20) for my deconvolution is: %.3f\n", my_computed_ptr);

fprintf("  ** Note: The biggest difference seems to be that I didn't include the rescale step in the PTR calculation for (2) **\n");

% ===========================================================================

% Next we will compare all the PTR values that we've computed with the
% published values.

% 1. Load the gene expression values for all the of deconvolved genes
% 2. Load the gene names
% 4. Compute the PTRs for all genes
% 5. Merge with the published dataset
%. Scatter plot published vs my deconvolution ptr values.

% Load the f matrix for all genes that were run
all_genes_f = cell2mat(table2cell(readtable('saved_output/2023-11-10_xg_gammas/all_genes_f.csv')));
genenames = readtable('saved_output/2023-11-10_xg_gammas/all_genes.csv', 'ReadVariableNames', false);

n = size(genenames, 2);

p2ts = zeros(n, 1);

for idx = 1:n
    model.f = all_genes_f(idx, :);
    p2t = peak2troughModel(model);
    p2ts(idx) = p2t;
end

writematrix(p2ts, "output/2023-11-10_xg_gammas/peak2troughs.csv");


