
import pandas as pd
import numpy as np


def cyclin_genes():

	genes = [
		# three G1 cyclins:
		'CLN1', 'CLN2', 'CLN3',

		# Two S-phase cyclins:
		'CLB5', 'CLB6',

		# And four mitotic cyclins:
		'CLB1', 'CLB2', 'CLB3', 'CLB4']

	return genes


def positive_control_genes():

	genes = [

		# B-type cyclins
		"CLB2", "CLN2",

		# RNR complex
		"RNR1", "RNR3",

		# Alpha factor genes
		"FIG1", "FIG2",

		"RAD51", "RAD53",
		
		# Daughter-specific expression genes
		"DSE1", "DSE2", "DSE3", "DSE4"
	]

	return genes

def guo_ds_genes():
	xin_dg1_genes = pd.read_csv('datasets/datasets_from_web_deconvolution.cs.duke.edu/guo_daughter_specific_genes.csv',
		header=None)[0].values
	return xin_dg1_genes

def get_missing_geneset():
	"""48 genes missing from failed gene expression deconvolution."""
	geneset = pd.read_csv('data/reference_data/geneset_nondub_w_prom_genebodies_missing_temp_2024-02-29.csv').set_index('orf_name')
	return geneset


def get_deconvolved_geneset():
	"""Get the list of genes to deconvolve first"""

	from src.reference_data import load_analysis_genes

	# Deconvolve genes we have filtered for coverage
	return load_analysis_genes()


def create_windows_to_deconvolve(round_window=2000):

	print(f"Defining test windows to deconvolve, rounded to the nearest {round_window}")

	# todo: contingent on a completed replication data set
	chr4_replicate_data_Fr = pd.read_csv(
		'output/prototype_pipeline_subset/combined_replication/combined_chr4_F.csv')
	chr4_replicate_data_Fr = chr4_replicate_data_Fr.T.iloc[1:]
	chr4_replicate_data_Fr.index = chr4_replicate_data_Fr.index.astype(int)

	early_set = chr4_replicate_data_Fr.idxmax(1).sort_values().head(200)
	late_set = chr4_replicate_data_Fr.idxmax(1).sort_values().tail(100)

	np.random.seed(123)

	size = 5
	chr4_early_random_indices = np.random.choice(early_set.index, size=size)
	chr4_late_random_indices = np.random.choice(late_set.index, size=size)

	def create_df_for_set(chr4_early_random_indices, chr, group):
		early_set_df = pd.DataFrame(chr4_early_random_indices, columns=['start'])
		early_set_df['group'] = group
		early_set_df['chr'] = 4
		early_set_df['end'] = early_set_df.start + 10000
		early_set_df['name'] = group
		return early_set_df

	def create_window_for_gene(gene_name, group):
		from src.sgd import get_orfname
		from src.geneset import get_deconvolved_geneset

		genes = get_deconvolved_geneset()
		orfname = get_orfname(gene_name)
		
		# Plot the 10kb window
		window = 10000
		win_2 = 5000

		gene = genes.loc[orfname]
		rounded_position = (gene.TSS//round_window)*round_window
		rounded_start = rounded_position-win_2
		rounded_start = max(rounded_position, 0)
		mnase_span = rounded_start, rounded_start+window
		
		return pd.DataFrame({
			'start': mnase_span[0], 
			'end': mnase_span[1], 
			'chr': gene.chr,
			'group': group,
			'name': gene_name
		}, index=[orfname])
		
	early_set_df = create_df_for_set(chr4_early_random_indices, 4, 'early_chr4')
	late_set_df = create_df_for_set(chr4_late_random_indices, 4, 'late_chr4')

	window_set = pd.concat([early_set_df, late_set_df])

	genes_df = pd.DataFrame()
	for gene_name in ['CLB2', 'CLN2', 'MCM6', "CLB3", "CLB6"]:
		df = create_window_for_gene(gene_name, 'cell_cycle_genes')
		genes_df = pd.concat([genes_df, df])
		
	for gene_name in ['DSE1', 'DSE2', 'DSE3', 'DSE4', 'SIC1']:
		df = create_window_for_gene(gene_name, 'daughter-specific_genes')
		genes_df = pd.concat([genes_df, df])
		
	for gene_name in ['SSK22', 'RPL2B', 'FLO9']:
		df = create_window_for_gene(gene_name, 'control_genes')
		genes_df = pd.concat([genes_df, df])

	from src.gene_expression import load_expression_data_summary, select_low_tx_genes
	expression_summary = load_expression_data_summary()
	selected_off_genes = select_low_tx_genes(expression_summary)

	deconv_genes = get_deconvolved_geneset()
	off_genes = selected_off_genes.join(deconv_genes[[]], how='inner')

	for gene_name in off_genes['gene']:
		df = create_window_for_gene(gene_name, 'off_genes')
		genes_df = pd.concat([genes_df, df])

	window_set = pd.concat([window_set, genes_df]).reset_index(drop=True)

	return window_set
