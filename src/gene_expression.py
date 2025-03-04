
import numpy as np
import pandas as pd


def load_gene_expression(gene_name, replicate):

	from src.sgd import get_orfname

	gene_expression_data = read_yl_vst_data_rep(replicate)
	orf_name = get_orfname(gene_name)
	gene_expression = gene_expression_data.loc[orf_name]
	gene_expressions_tpm = pd.read_csv(
		f'datasets/yl_cell_cycle/replicate{replicate}_gene_expression_TPM.csv')
	gene_expressions_tpm = gene_expressions_tpm.set_index('orf_name')

	gene_expression_tpm = gene_expressions_tpm.loc[orf_name]
	g = np.log2(gene_expression_tpm.values+1)

	return g


def read_yl_vst_data_rep(replicate, drop_rep2_70=True):
	wt_data = pd.read_csv(f'datasets/yl_cell_cycle/replicate{replicate}_deseq2_vst_counts.csv')
	wt_data = wt_data.rename(columns={"Unnamed: 0": "orf_name"}).set_index('orf_name')
	wt_data.columns = [int(s.replace('X', '')) for s in wt_data.columns.values]

	# Replicate 2, timepoint 70 appears to be low quality
	# Checking if removing this timepoint improves the fit quality.
	if replicate == 2 and drop_rep2_70:
		# Remove timepoint 70 for replicate 2
		wt_data = wt_data[wt_data.columns[(wt_data.columns != 70)]]

	# Normalize such that all timepoints are equal
	target_read_counts = 60000 # (approximate read counts prior to normalization)
	wt_data.loc[:] = wt_data.values / wt_data.values.sum(axis=0).reshape((1, -1)) * target_read_counts

	return wt_data
