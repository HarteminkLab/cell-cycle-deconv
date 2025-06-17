
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

	def deconvolve_G_optimal_gamma(self, G, kappa=0.0, eta=0.0,  alphas=None, 
		verbose=False, replicate='combined'):
		"""
		General function to deconvolve any expression data G to find optimal gamma.
		"""
		from src.expression_gamma_search import GeneExpressionFindOptimalGamma

		# Modify alphas if specified
		if alphas is not None:
			self.config1.modify_alpha(alphas[0], num_g1_indices)
			self.config2.modify_alpha(alphas[1], num_g1_indices)
			# Use for indices lookup and timepoints
			self.combined_config.modify_alpha(alphas[0], num_g1_indices)

		H1 = self.config1.H
		H2 = self.config2.H

		from src.CombinedReplicationDeconvolution import concatenate_H, concatenate_G

		if replicate == 'combined':
			# Concatenate H and G for combined replicate
			H = concatenate_H(H1, H2)
			self.combined_config.H = H
			config = self.combined_config

		elif replicate == 1:
			H = H1
			config = self.config1
			
		elif replicate == 2:
			H = H2
			config = self.config2
			
		else:
			raise ValueError("replicate must be 'combined', 1, or 2")

		self.expression_find_gamma = GeneExpressionFindOptimalGamma(config=config, H=H, 
																   gene_expression=G)
		self.expression_find_gamma.find_optimal_gamma(plot=False, verbose=verbose, kappa=kappa,
													 eta=eta)

		return self.expression_find_gamma


	def deconvolve_gene_optimal_gamma(self, gene_name, replicate='combined', kappa=0.0, 
									 eta=0.0, verbose=False, alphas=None, num_g1_indices=None):
		"""
		Deconvolve a specific gene's expression to find optimal gamma.
		"""
		from src.gene_expression import load_transcription_for_name
		from src.sgd import get_orfname
		from src.CombinedReplicationDeconvolution import concatenate_G

		# Set gene-specific instance variables
		self.gene_name = gene_name
		self.orf_name = get_orfname(gene_name)
		self.save_name = f"{self.orf_name}_{self.gene_name}"

		return self.deconvolve_transcript_optimal_gamma(self.orf_name, replicate=replicate, kappa=kappa,
			eta=eta, verbose=verbose, alphas=alphas, num_g1_indices=num_g1_indices)

	def deconvolve_transcript_optimal_gamma(self, orf_or_transcript_name, replicate='combined', kappa=0.0, 
									 eta=0.0, verbose=False, alphas=None, num_g1_indices=None):

		from src.gene_expression import load_transcription_for_name
		from src.CombinedReplicationDeconvolution import concatenate_G

		# Load gene expression data if not overridden
		gene_expression_replicate1 = load_transcription_for_name(orf_or_transcript_name, 1, log_transform=True)
		gene_expression_replicate2 = load_transcription_for_name(orf_or_transcript_name, 2, log_transform=True)

		from src.sgd import get_gene_name

		self.save_name = f"{orf_or_transcript_name}"

		# If deconvolving a gene update save name
		try:
			self.gene_name = get_gene_name(orf_or_transcript_name)
			self.save_name = f"{orf_or_transcript_name}_{self.gene_name}"
		except:
			self.gene_name = None

		if replicate == 'combined':
			# Concatenate H and G for combined replicate
			G = concatenate_G(gene_expression_replicate1, gene_expression_replicate2)
		elif replicate == 1:
			G = gene_expression_replicate1
		elif replicate == 2:
			G = gene_expression_replicate2
		else:
			raise ValueError("replicate must be 'combined', 1, or 2")

		# Call the general deconvolution function
		return self.deconvolve_G_optimal_gamma(G, kappa=kappa, eta=eta, 
			alphas=alphas, verbose=verbose, replicate=replicate)

	def plot_solution(self):
		import matplotlib.pyplot as plt

		solver = self.expression_find_gamma.deconvolution_solver

		G = solver.g
		F = self.expression_find_gamma.retrieve_solution()

		config = solver.config

		i_indices = config.get_Hpositions_for_branch('i')
		t_indices = config.get_Hpositions_for_branch('t')
		b_indices = config.get_Hpositions_for_branch('b')
		h_indices = config.get_Hpositions_for_phase('Halted')

		i_tps = config.get_timepoints_for_branch('i')
		t_tps = config.get_timepoints_for_branch('t')
		b_tps = config.get_timepoints_for_branch('b')

		num_cols = 4

		fig, axs = plt.subplots(1, num_cols, figsize=(16, 3))

		predicted_G = config.H@F

		g_vmax = G.max()
		vmax = max(G.max(), F.max())
		ylims = (vmax*-0.05, vmax*1.2)
		g_ylims = (g_vmax*-0.05, g_vmax*1.2)

		if G.max() == 0:
			ylims = -0.1, 1

		cmap = plt.cm.viridis

		ax_row = axs

		ax = ax_row[0]

		timepoints = config.timepoints
		tps1 = GlobalConstants.EXPRESSION_WT1_TIMEPOINTS
		tps2 = GlobalConstants.EXPRESSION_WT2_TIMEPOINTS
		n_tps1 = len(tps1)
		n_tps2 = len(tps2)

		ax.plot(tps1, G[:n_tps1], c='black', lw=3, label="Raw data")
		ax.plot(tps1, predicted_G[:n_tps1], c='red',
			   lw=3, label="Optimal $\\gamma$ solution")

		ax.plot(tps2, G[n_tps1:], c='black', lw=3, label="Raw data")
		ax.plot(tps2, predicted_G[n_tps1:], c='red',
			   lw=3, label="Optimal $\\gamma$ solution")

		ax.set_ylim(*g_ylims)
		ax.set_title("Data vs Fit")
		ax.legend()

		ax = ax_row[1]
		ax.plot(i_tps, F[i_indices].T, c='red',
				lw=3)
		ax.set_title("Initial branch")
		ax.set_ylim(*ylims)

		halted_tx = F[h_indices[0]]
		ax.axhline(halted_tx, ls='dotted', color='#555', lw=1)

		ax = ax_row[2]
		ax.plot(t_tps, F[t_indices].T, c='red',
				lw=3)
		ax.set_ylim(*ylims)
		ax.set_title("Top branch")

		ax = ax_row[3]
		ax.plot(b_tps, F[b_indices].T, c='red',
				lw=3)
		ax.set_ylim(*ylims)
		ax.set_title("Bottom branch")

	def retrieve_solution(self):
		return self.expression_find_gamma.retrieve_solution()

	def save_to_disk(self, output_directory):

		from matplotlib import pyplot as plt

		expression_F = self.expression_find_gamma.retrieve_solution()
		optimal_gamma = self.expression_find_gamma.gamma_optimizer.optimal_gamma

		F_savepath = f"{output_directory}/{self.save_name}_{optimal_gamma:.6f}.npy"
		fig_savepath = f"{output_directory}/{self.save_name}.png"

		np.save(F_savepath, expression_F)

		fig = self.expression_find_gamma.plot_gamma_sweep()
		plt.savefig(fig_savepath)
		plt.close(fig)
