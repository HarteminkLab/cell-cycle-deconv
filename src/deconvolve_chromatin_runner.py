
import sys
sys.path.append('.')

import pandas as pd
from src.utils import print_fl, mkdirs_safe

from src.timer import Timer
from src.model import Model
from cc_src.chromatin_model import ChromatinModel
from src.config import load_yl_rg1_vst_config
from matplotlib import pyplot as plt

def main():
	"""
	Run the deconvolution on a gene, indexed by the command-line argument

	Usage:

		<output> <replicate> <gamma_value> <gene_index>

	or

		<output> <replicate> <gene_index>

	Will run the find optimal gamma procedure on the chromatin


	Example script command for replicate 1, gene index 10, and gamma value of 0.006

		python src/deconvolve_chromatin_runner.py output/deconvolution_results_006_2024-02-20/ 1 0.006 10


	"""

	system_args = tuple(sys.argv)

	from cc_src.geneset import get_sorted_geneset

	geneset = get_sorted_geneset()

	# Specify replicate and gamma value
	if len(system_args) == 5:
		(_, out_dir, replicate, gamma, gene_index) = system_args
		gene_index = int(gene_index)
		replicate = int(replicate)
		gamma = float(gamma)

	# No gamma value, so find optimal gamma
	elif len(system_args) == 4:
		(_, out_dir, replicate, gene_index) = system_args
		gene_index = int(gene_index)
		replicate = int(replicate)
		gamma = None

	sys.stdout.flush()
	gene = geneset.iloc[gene_index]

	print_fl(f"Index: [{gene_index}/{len(geneset)}] Deconvolving replicate={replicate}, gene: {gene['gene']}/{gene.name}...")

	# -------------------------

	plot_dir = f'{out_dir}/plots'
	chromatin_out_dir = f'{out_dir}/chromatin'
	geneexpression_out_dir = f'{out_dir}/gene_expression'
	mkdirs_safe([chromatin_out_dir, geneexpression_out_dir, plot_dir])

	# ----------------------

	config = load_yl_rg1_vst_config(replicate=replicate)
	chromatin_model = ChromatinModel(config)
	chromatin_model.load_mnase_gene(gene['gene'])
	chromatin_model.create_deconvolution_bins()

	if gamma is not None:
		chromatin_model.deconvolve(verbose=True)
	else:
		chromatin_model.deconvolve_find_optimal_gamma()

	# ----------------------

	ge_model = Model(config, gene['gene'])
	ge_model.deconvolve_find_optimal_gamma()

	fig = chromatin_model.create_deconvolution_plots_abbreviated_flipped(vmax=10, ge_model=ge_model)
	save_path = f"{plot_dir}/{gene_index}_{gene['gene']}_{gene.name}_deconvolution.png"
	plt.savefig(save_path, dpi=200)
	plt.close(fig)

	fig = chromatin_model.plot_prediction_comparison()
	save_path = f"{plot_dir}/{gene_index}_{gene['gene']}_{gene.name}_data.png"
	plt.savefig(save_path, dpi=200)
	plt.close(fig)

	# -------------- Save the output ------------------

	chromatin_model.save_deconvolved_outputs(chromatin_out_dir, gene_index, (not chromatin_model.found_optimal_success))
	ge_model.save_deconvolved_outputs(geneexpression_out_dir, gene_index)



if __name__ == '__main__':
	main()
