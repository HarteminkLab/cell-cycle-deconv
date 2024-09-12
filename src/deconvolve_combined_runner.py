
import sys
sys.path.append('.')

import pandas as pd
from src.utils import print_fl, mkdirs_safe

from src.timer import Timer
from src.model import Model
from src.config import load_yl_rg1_vst_config
from src.config import load_combined_yl_alpha_vst_gene_expression_config
from src.delta_config import load_delta_combined_gene_expression_config, load_yl_delta_config

from matplotlib import pyplot as plt
from src.combined_chromatin_model import CombinedChromatinModel


def main():
	"""
	Run the deconvolution on a gene, indexed by the command-line argument

	Usage:

		<output> <gamma_value/-1> <config_type: shared/distinct/delta> <0/1 for PAS deconvolution> <gene_index>

	Will run the find optimal gamma procedure on the chromatin


	Script for shared G1 config model with a fixed gamma value and copy number correction

	python src/deconvolve_combined_runner.py output/deconvolve_sharedg1_0066_cc 0.0066 shared 1 10
	python src/deconvolve_combined_runner.py output/deconvolve_distinctg1_0066_cc 0.0066 distinct 1 10

	"""

	system_args = tuple(sys.argv)

	from src.geneset import get_deconvolved_geneset
	geneset = get_deconvolved_geneset()

	print_fl(f"System arguments:\t{system_args}")

	# Specify output directory, gamma value and gene index
	(_, out_dir, gamma, config_type, deconvolve_PAS, gene_index) = system_args
	gene_index = int(gene_index)
	gamma = float(gamma)
	should_copy_correct = True#should_copy_correct == '1'
	should_deconvolve_PAS = deconvolve_PAS != '0'

	print("Config type: ", config_type)
	print("Will copy correct: ", should_copy_correct)
	print("Deconvolve PAS: ", should_deconvolve_PAS)

	# Find optimal gamma
	if gamma < 0: gamma = None

	sys.stdout.flush()
	gene = geneset.iloc[gene_index]

	print_fl(f"Index: [{gene_index}/{len(geneset)}] Deconvolving combined model, gene: {gene['gene']}/{gene.name}...")

	if gamma is None:
		print_fl(f"No gamma specified, finding optimal gamma value for chromatin.")

	print_fl(f"Output to: {out_dir}")

	# -------------------------

	plot_dir = f'{out_dir}/plots'
	chromatin_out_dir = f'{out_dir}/chromatin'
	geneexpression_out_dir = f'{out_dir}/gene_expression'
	mkdirs_safe([chromatin_out_dir, geneexpression_out_dir, plot_dir])

	# ----------------------

	# Load the configuration for the combined chromatin and gene expression models
	from src.config import load_configs_by_config_type
	from src.config import load_combined_gene_expression_by_config_type 

	config1, config2 = load_configs_by_config_type(config_type, with_copy_correction=should_copy_correct)
	combined_ge_config = load_combined_gene_expression_by_config_type(config_type, 
		with_copy_correction=should_copy_correct)

	# ----------------------

	combined_model = CombinedChromatinModel(config1, config2)
	combined_model.chrom1_model.center_on_TSS = not should_deconvolve_PAS
	combined_model.chrom2_model.center_on_TSS = not should_deconvolve_PAS
	combined_model.load_combined_mnase_gene(gene['gene'])

	if gamma is not None:
		combined_model.deconvolve(verbose=False, gamma=gamma)
	else:
		combined_model.deconvolve_find_optimal_gamma()

	# ----------------------

	# Deconvolve the combined gene expression model
	from src.model import Model

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

	combined_model.save_deconvolved_outputs(chromatin_out_dir, gene_index)
	ge_model.save_deconvolved_outputs(geneexpression_out_dir, gene_index)


if __name__ == '__main__':
	main()
