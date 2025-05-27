
import numpy as np
from src.global_config import GlobalConstants


class CombinedDeconvolveGeneExpressionRunner:

	def __init__(self, output_directory):

		from src.config import load_default_expression_configs, load_cloccs_configs

		# Load the config parameters from disk, for gene expression, the copy
		# correction information will not be used, so we can just use the learned cell cycle
		# parameters
		config1, config2 = load_cloccs_configs(mode='expression')
		combined_config, _ = load_cloccs_configs(mode='expression')

		self.config1 = config1
		self.config2 = config2

		# Create another config for the combined model, 
		# in case we want to modify alpha if either single replicate configs
		self.combined_config = combined_config 
		self.combined_config.timepoints = np.concatenate([GlobalConstants.EXPRESSION_WT1_TIMEPOINTS, 
				GlobalConstants.EXPRESSION_WT2_TIMEPOINTS])

	def deconvolve_gene_find_gamma(self, gene_name, replicate='combined', kappa=0.0, verbose=False,
		alphas=None, num_g1_indices=None):

		from src.gene_expression import load_gene_expression
		from src.sgd import get_orfname
		from src.CombinedReplicationDeconvolution import concatenate_H_G
		from src.expression_gamma_search import GeneExpressionFindOptimalGamma

		self.gene_name = gene_name
		self.orf_name = get_orfname(gene_name)

		gene_expression_replicate1 = load_gene_expression(gene_name, 1, log_transform=True)
		gene_expression_replicate2 = load_gene_expression(gene_name, 2, log_transform=True)

		# Modify alphas if specified
		if alphas is not None:
			self.config1.modify_alpha(alphas[0], num_g1_indices)
			self.config2.modify_alpha(alphas[1], num_g1_indices)

			# Use for indices lookup and timepoints
			self.combined_config.modify_alpha(alphas[0], num_g1_indices)

		H1 = self.config1.H
		H2 = self.config2.H

		from src.global_config import GlobalConstants

		if replicate == 'combined':
			H, G = concatenate_H_G(H1, H2, gene_expression_replicate1, gene_expression_replicate2)
			self.combined_config.H = H
			config = self.combined_config

		elif replicate == 1:
			H = H1
			G = gene_expression_replicate1
			config = self.config1
		elif replicate == 2:
			H = H2
			G = gene_expression_replicate2
			config = self.config2
		else:
			raise ValueError()

		self.expression_find_gamma = GeneExpressionFindOptimalGamma(config=config, H=H, 
			gene_expression=G)
		self.expression_find_gamma.find_optimal_gamma(plot=False, verbose=verbose, kappa=kappa)

		return self.expression_find_gamma


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
