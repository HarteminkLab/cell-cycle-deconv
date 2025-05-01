
import numpy as np


class CombinedDeconvolveGeneExpressionRunner:

	def __init__(self, output_directory, should_load_cloccs_configs=False):

		from src.config import load_default_expression_configs, load_cloccs_configs

		# Load the config parameters from disk, for gene expression, the copy
		# correction information will not be used, so we can just use the learned cell cycle
		# parameters
		if should_load_cloccs_configs:
			config1, config2 = load_cloccs_configs(mode='expression', shift_CLOCCS=False)
		else:
			config1, config2 = load_default_expression_configs()

		self.config1 = config1
		self.config2 = config2

	def deconvolve_gene(self, gene_name, kappa=0.0):

		from src.gene_expression import load_gene_expression
		from src.sgd import get_orfname
		from src.CombinedReplicationDeconvolution import concatenate_H_G
		from src.expression_gamma_search import GeneExpressionFindOptimalGamma

		self.gene_name = gene_name
		self.orf_name = get_orfname(gene_name)
		gene_expression_replicate1 = load_gene_expression(gene_name, 1, log_transform=False)
		gene_expression_replicate2 = load_gene_expression(gene_name, 2, log_transform=False)

		H1 = self.config1.H
		H2 = self.config2.H

		H, G = concatenate_H_G(H1, H2, gene_expression_replicate1, gene_expression_replicate2)

		expression_find_gamma = GeneExpressionFindOptimalGamma(config=self.config1, H=H, 
			gene_expression=G)
		expression_find_gamma.find_optimal_gamma(plot=False, verbose=False, kappa=kappa)
		self.expression_find_gamma = expression_find_gamma

		return expression_find_gamma


	def save_to_disk(self, output_directory):

		from matplotlib import pyplot as plt

		expression_F = self.expression_find_gamma.retrieve_solution()
		optimal_gamma = self.expression_find_gamma.gamma_optimizer.optimal_gamma

		F_savepath = f"{output_directory}/{self.orf_name}_{self.gene_name}_{optimal_gamma:.6f}.npy"
		fig_savepath = f"{output_directory}/{self.orf_name}_{self.gene_name}.png"

		np.save(F_savepath, expression_F)

		fig = self.expression_find_gamma.plot_gamma_sweep()
		plt.savefig(fig_savepath)
		plt.close(fig)
