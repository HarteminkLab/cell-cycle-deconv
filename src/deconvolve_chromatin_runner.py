


import sys
sys.path.append('.')

from src.timer import Timer
from src.config import load_yl_replicate2_rg1_chromatin_config, \
	load_xg_gene_expression_config
from src.model import Model
from cc_src.chromatin_grid_compute import ChromatinGrid
from src.find_gamma_chromatin import FindOptimalGammaChromatin

import sys

def main():

	(_, out_dir, orf_name) = tuple(sys.argv)


	config = load_yl_replicate2_rg1_chromatin_config()
	model = Model(config, orf_name, 0.001)

	print(f"Deconvolving gene: {model.gene_name}/{model.orf_name}...")
	sys.stdout.flush()

	# Initialize the chromatin grid
	timer = Timer()

	chromatin_gridder = ChromatinGrid()
	chromatin_gridder.set_gene(model.gene_name)
	chromatin_gridder.create_bins_per_all_sample()
	chromatin_gridder.create_deconvolution_matrices()

	find_gamma_chromatin = FindOptimalGammaChromatin(model, chromatin_gridder)
	find_gamma_chromatin.find_optimal()

	f = find_gamma_chromatin.f

	print(f"Finished finding the optimal gamma in : {timer.get_time()}")
	sys.stdout.flush()

	chromatin_gridder.save_deconvolved_outputs(out_dir, model, f)


if __name__ == '__main__':
	main()
