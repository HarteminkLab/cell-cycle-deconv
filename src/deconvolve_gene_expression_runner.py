
import sys
sys.path.append('.')

import pandas as pd

from src.timer import Timer
from src.model import Model
from src.find_gamma import FindOptimalGamma
from src.config import load_yl_replicate2_rg1_config


def main():
	"""
	Run the deconvolution on a gene, indexed by the command-line argument
	"""

	(_, out_dir, index, batch_idx) = tuple(sys.argv)

	geneset = pd.read_csv('data/reference_data/geneset_nondub_w_prom_genebodies.csv')

	# Each batch will run 1000 genes, second argument in ARGS is the batch index that 
	# will be multiplied against the array index
	gene_index = int(batch_idx)*1000 + int(index)

	print(f"Running batch: {batch_idx}, array index: {index}, or gene_index: {gene_index}...")
	sys.stdout.flush()

	gene = geneset.iloc[gene_index]

	print(f"Index: [{gene_index}/{len(geneset)}] Deconvolving gene: {gene.gene}/{gene.name}...")
	sys.stdout.flush()

	config = load_yl_replicate2_rg1_config()
	model = Model(config, gene.orf_name, 0.001)

	find_gamma = FindOptimalGamma(model)
	find_gamma.find_optimal()

	# Initialize the chromatin grid
	timer = Timer()

	print(f"Finished finding the optimal gamma in : {timer.get_time()}")
	sys.stdout.flush()

	# Save f, g, meta, ptr values
	model.save_deconvolved_outputs(gene_index, out_dir)


if __name__ == '__main__':
	main()
