
import sys
sys.path.append('.')

import pandas as pd

from src.timer import Timer
from src.model import Model
from src.find_gamma_chromatin import FindOptimalGammaChromatin
from src.config import load_yl_replicate2_rg1_chromatin_config
from cc_src.chromatin_grid_compute import ChromatinGrid


def main():
	"""
	Run the deconvolution on a gene, indexed by the command-line argument
	"""

	(_, out_dir, batch_idx, index) = tuple(sys.argv)

	geneset = pd.read_csv('data/reference_data/geneset_nondub_w_prom_genebodies.csv')

	# Each batch will run 1000 genes, second argument in ARGS is the batch index that 
	# will be multiplied against the array index
	gene_index = int(batch_idx)*1000 + int(index)

	print(f"Running batch: {batch_idx}, array index: {index}, or gene_index: {gene_index}...")
	sys.stdout.flush()

	gene = geneset.iloc[gene_index]

	config = load_yl_replicate2_rg1_chromatin_config()
	model = Model(config, gene.orf_name, 0.001)

	print(f"Index: [{gene_index}/{len(geneset)}] Deconvolving gene: {model.gene_name}/{model.orf_name}...")
	sys.stdout.flush()

	# Initialize the chromatin grid
	timer = Timer()

	chromatin_gridder = ChromatinGrid()

	if model.gene_name is None: chromatin_gridder.set_gene(model.orf_name)
	else: chromatin_gridder.set_gene(model.gene_name)

	chromatin_gridder.create_bins_per_all_sample()
	chromatin_gridder.create_deconvolution_matrices()

	find_gamma_chromatin = FindOptimalGammaChromatin(model, chromatin_gridder)
	found_optimal_success = find_gamma_chromatin.find_optimal()
	using_default_flag = not found_optimal_success

	f = find_gamma_chromatin.f

	print(f"Finished finding the optimal gamma in : {timer.get_time()}")
	sys.stdout.flush()

	chromatin_gridder.save_deconvolved_outputs(out_dir, index, model, f, using_default_flag)


if __name__ == '__main__':
	main()
