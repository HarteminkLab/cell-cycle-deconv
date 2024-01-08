# Usage:
# python src/deconvolution_plotting_runner.py output/deconvolve_plots_2024_01_08/ 1 1

import sys
sys.path.append('.')

import pandas as pd
import matplotlib.pyplot as plt

from src.timer import Timer
from src.model import Model
from src.find_gamma import FindOptimalGamma
from src.config import load_yl_replicate2_rg1_config


GENESET = pd.read_csv('data/reference_data/geneset_nondub_w_prom_genebodies.csv')
CHROMATIN_DIRECTORY_PATH = 'output/deconvolve_chromatin_2023_12_15/'
GENE_EXPRESSION_DIRECTORY_PATH = 'output/deconvolve_gene_expression_2024_01_04/'


def gene_from_index_batch_idx(batch_idx, index):


	# Each batch will run 1000 genes, second argument in ARGS is the batch index that 
	# will be multiplied against the array index
	gene_index = int(batch_idx)*1000 + int(index)

	print(f"Running batch: {batch_idx}, array index: {index}, or gene_index: {gene_index}...")
	sys.stdout.flush()
	gene = GENESET.iloc[gene_index]

	return gene, gene_index

def main():
	"""
	Run the deconvolution on a gene, indexed by the command-line argument
	"""

	# ---------- Parse the arguments ----------------

	(_, out_dir, batch_idx, index) = tuple(sys.argv)
	gene, gene_index = gene_from_index_batch_idx(batch_idx, index)

	# -------------------------------------------------

	print(f"Index: [{gene_index}/{len(GENESET)}] Creating deconvolution plot for gene: {gene.gene}/{gene.orf_name}...")
	sys.stdout.flush()

	timer = Timer()

	from src.deconvolution_plotter import load_deconvolution_models_from_disk

	gene_name = gene['gene']

	try:
		deconv_plotter = load_deconvolution_models_from_disk(gene_name, CHROMATIN_DIRECTORY_PATH, 
											GENE_EXPRESSION_DIRECTORY_PATH)
	except IndexError:
		print(f"Incomplete deconvolution files for gene: {gene_name}. Exiting.")
		return

	deconv_plotter.chromatin_gridder.create_bins_per_all_sample()
	gene_title = f"{gene['gene']}_{gene['orf_name']}"

	deconv_plotter.chromatin_gridder.plot_prediction_comparison(deconv_plotter.chrom_model)
	plt.savefig(f"{out_dir}/{gene_index}_{gene_title}_data.png", dpi=250)

	deconv_plotter.plot_deconvolved_models()
	plt.savefig(f"{out_dir}/{gene_index}_{gene_title}_deconvolution.png", dpi=250)

	print(f"Finished finding the optimal gamma in : {timer.get_time()}")
	sys.stdout.flush()

	# Save f, g, meta, ptr values
	model.save_deconvolved_outputs(gene_index, out_dir)


if __name__ == '__main__':
	main()
