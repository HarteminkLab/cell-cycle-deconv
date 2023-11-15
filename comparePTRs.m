
addpath(genpath('lib/YAMLMatlab'));
addpath(genpath('deconv'))
addpath(genpath('analysis'))

% This script will be used to validate that the peak to trough function
% I am using is computing the PTR values properly..


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
fprintf("The published deconvolved PTR (80/20) for CLN2 is: %.3f\n", cln2_deconvolved_ptr);

computedPtr = peak2trough(cln2_c_ge, cln2_d_ge);
fprintf("The computed PTR (80/20) for CLN2 is: %.3f\n", computedPtr);







