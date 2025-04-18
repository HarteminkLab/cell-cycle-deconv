
import numpy as np
import pandas as pd


def load_gene_expression(gene_name, replicate):

	from src.sgd import get_orfname
	orf_name = get_orfname(gene_name)
	gene_expressions_logtpm = load_gene_expression_data(replicate)
	gene_expression_logtpm = gene_expressions_logtpm.loc[orf_name]

	return gene_expression_tpm

def load_gene_expression_data(replicate, log_transform=True):

	from src.sgd import get_orfname

	gene_expressions_tpm = pd.read_csv(
		f'datasets/yl_cell_cycle/replicate{replicate}_gene_expression_TPM.csv')
	gene_expressions_tpm = gene_expressions_tpm.set_index('orf_name')

	if log_transform:
		gene_expressions_tpm.loc[:] = np.log2(gene_expressions_tpm.values+1)

	return gene_expressions_tpm


def load_expression_data_summary():
	# Let's add to the geneset to select genes that are pretty much off like FLO9

	from src.gene_expression import load_gene_expression_data

	repl1_expression = load_gene_expression_data(1)
	repl2_expression = load_gene_expression_data(2)

	expression_summary = repl1_expression[[]].copy()
	expression_summary['mean_repl1'] = repl1_expression.mean(axis=1)
	expression_summary['mean_repl2'] = repl2_expression.mean(axis=1)

	expression_summary['std_repl1'] = repl2_expression.std(axis=1)
	expression_summary['std_repl2'] = repl2_expression.std(axis=1)

	expression_summary['combined_mean'] = (expression_summary.mean_repl1 + \
		expression_summary.mean_repl2)/2.
	expression_summary['combined_std'] = (expression_summary.std_repl1 + \
		expression_summary.std_repl2)/2.

	return expression_summary

def select_low_tx_genes(expression_summary, min_length=500, cutoffs=(0.1, 0.5),
	plot=False):
	"""Select verified genes with a minimum length with no expression"""

	from src.sgd import read_nondubious_genes_dataset
	import matplotlib.pyplot as plt

	genes = read_nondubious_genes_dataset()

	if plot:
		plt.figure(figsize=(5, 3))
		plt.scatter(expression_summary.combined_std, expression_summary.combined_mean,
			s=2, alpha=0.25)
		plt.title("Variation and Mean of Raw Expression")
		plt.xlabel("$\\sigma$")
		plt.ylabel("$\\mu$")

		plt.axvline(cutoffs[0], c='red')
		plt.axhline(cutoffs[1], c='red')

	# Select the lowest expressed and lowest variance genes
	low_tx_genes = expression_summary[(expression_summary.combined_mean < cutoffs[1]) & 
					   (expression_summary.combined_std < cutoffs[0])]

	selected_genes = genes[['gene', 'length', 'classification']].join(low_tx_genes, how='inner')
	selected_genes = selected_genes[(selected_genes['length'] > min_length) & 
	  (selected_genes['classification'] == 'Verified')]

	print(f"Criteria for gene selection: Verified genes greater than {min_length} long," 
		  f"less than {cutoffs[1]} mean expression, less than {cutoffs[0]} std expression")
	print("Number of verified genes with low expression:", len(selected_genes))

	return selected_genes

