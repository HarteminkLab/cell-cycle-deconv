
import sys
sys.path.append('.')

import pandas as pd
from src.utils import print_fl, mkdirs_safe

from src.timer import Timer
from src.model import Model
from src.config import load_yl_rg1_vst_config
from matplotlib import pyplot as plt
from cc_src.combined_chromatin_model import CombinedChromatinModel


def main():
	"""
	Run the deconvolution on a gene, indexed by the command-line argument

	Usage:

		<output> <replicate> <gamma_value> <gene_index>

	or

		<output> <replicate> <gene_index>

	Will run the find optimal gamma procedure on the chromatin


	Example script command for replicate 1, gene index 10, and gamma value of 0.006

	python src/deconvolve_combined_runner.py output/deconvolution_results_006_2024-03-08/ 0.006 10


	"""

	system_args = tuple(sys.argv)

	from cc_src.geneset import get_deconvolved_geneset

	geneset = get_deconvolved_geneset()

	# Specify output directory, gamma value and gene index
	if len(system_args) == 4:
		(_, out_dir, gamma, gene_index) = system_args
		gene_index = int(gene_index)
		gamma = float(gamma)

	sys.stdout.flush()
	gene = geneset.iloc[gene_index]

	print_fl(f"Index: [{gene_index}/{len(geneset)}] Deconvolving combined model, gene: {gene['gene']}/{gene.name}...")
	print_fl(f"Output to: {out_dir}")

	# -------------------------

	plot_dir = f'{out_dir}/plots'
	chromatin_out_dir = f'{out_dir}/chromatin'
	geneexpression_out_dir = f'{out_dir}/gene_expression'
	mkdirs_safe([chromatin_out_dir, geneexpression_out_dir, plot_dir])

	# ----------------------

	config1 = load_yl_rg1_vst_config(1)
	config2 = load_yl_rg1_vst_config(2)


	combined_model = CombinedChromatinModel(config1, config2)
	combined_model.load_combined_mnase_gene(gene['gene'])
	combined_model.deconvolve(verbose=True, gamma=gamma)

	# ----------------------

	# Deconvolve the combined gene expression model
	from src.config import load_combined_yl_alpha_vst_gene_expression_config
	from src.model import Model

	combined_ge_config = load_combined_yl_alpha_vst_gene_expression_config()
	ge_model = Model(combined_ge_config, gene['gene'])
	ge_model.deconvolve_find_optimal_gamma()

	# -----------------------

	fig = ge_model.plot_deconvolved_gene()
	save_path = f"{plot_dir}/{gene_index}_{gene['gene']}_{gene.name}_gene_expression_deconvolution.png"
	plt.savefig(save_path, dpi=200)
	plt.close(fig)

	fig = combined_model.create_deconvolution_plots_abbreviated_flipped(vmax=50, ge_model=ge_model)
	save_path = f"{plot_dir}/{gene_index}_{gene['gene']}_{gene.name}_chromatin_deconvolution.png"
	plt.savefig(save_path, dpi=200)
	plt.close(fig)

	fig = combined_model.plot_raw_prediction(1, vmax=50)
	save_path = f"{plot_dir}/{gene_index}_rep1_{gene['gene']}_{gene.name}_data.png"
	plt.savefig(save_path, dpi=200)
	plt.close(fig)

	fig = combined_model.plot_raw_prediction(2, vmax=50)
	save_path = f"{plot_dir}/{gene_index}_rep2_{gene['gene']}_{gene.name}_data.png"
	plt.savefig(save_path, dpi=200)
	plt.close(fig)

	# -------------- Save the output ------------------

	combined_model.save_deconvolved_outputs(chromatin_out_dir, gene_index, 
		(not combined_model.found_optimal_success))
	ge_model.save_deconvolved_outputs(geneexpression_out_dir, gene_index)



if __name__ == '__main__':
	main()
