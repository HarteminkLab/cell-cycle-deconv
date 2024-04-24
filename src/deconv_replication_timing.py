


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.config import load_yl_replicate1_rg1_alpha_vst_config
from src.dynamic_config_alpha import create_dynamic_alpha_config
from src.model import Model
from src.chromatin_model import ChromatinModel


class DeconvReplicationProfileAnalysis:
	"""Class to handle deconvolution of mnase seq occupancy 10k bins for 
	computing the replication timing

	Goals:
	1. Identify optimal alpha value from chr4 profiles for replicate 1 and 2
	2. Compute the profiles for all chromosomes and save to disk
	"""

	def __init__(self, mnase_analysis_rep1, mnase_analysis_rep2):
		self.mnase_analysis_rep1 = mnase_analysis_rep1
		self.mnase_analysis_rep2 = mnase_analysis_rep2

	def compare_early_late_deconv_bin_curves(self, alpha, replicate):
		"""Show the deconvolution of two example bin curves, early and late"""

		self.replicate = replicate

		if replicate == 1:
			mnase_analysis = self.mnase_analysis_rep1
		else:
			mnase_analysis = self.mnase_analysis_rep2

		# An early and late replicating gene
		from src.sgd import get_orfname
		orf_early = get_orfname('VPS8')
		orf_late = get_orfname('SSK22')

		early_bin_curve = mnase_analysis.normalized_bin_curves.loc[orf_early]
		late_bin_curve = mnase_analysis.normalized_bin_curves.loc[orf_late]

		dynamic_config = create_dynamic_alpha_config(alpha, replicate)

		gene_name = "CLN1"
		model = Model(dynamic_config, gene_name)

		model.g = early_bin_curve+1.
		model.g1 = early_bin_curve+1.
		model.deconvolve_find_optimal_gamma()
		_ = model.plot_deconvolved_gene()
		plt.suptitle(f"Early replicating 10k bin, $\\alpha$={alpha}\nsn={model.sn:.2f}, rn={model.rn:.2f}"
			f", gamma={model.gamma:.3f}, Replicate 1", fontsize=32)

		model.g = late_bin_curve+1.
		model.g1 = late_bin_curve+1.
		model.deconvolve_find_optimal_gamma()

		_ = model.plot_deconvolved_gene()
		plt.suptitle(f"Late replicating 10k bin, $\\alpha$={alpha}\nsn={model.sn:.2f}, rn={model.rn:.2f}"
			f"\ngamma={model.gamma:.3f}, "
			f"Replicate {self.replicate}", fontsize=32)

	def create_chr_bin_curves(self, replicate=1, chrom=4):
		"""Select the chromosome for the bin curves we will deconvolve"""

		print(f"Setting up bin curves for replicate {replicate}, chromosome {chrom}")

		if replicate == 1:
			mnase_analysis = self.mnase_analysis_rep1
		else:
			mnase_analysis = self.mnase_analysis_rep2
		self.replicate = replicate

		from src.geneset import get_deconvolved_geneset
		geneset = get_deconvolved_geneset()
		self.bin_curves_G_df = mnase_analysis.normalized_bin_curves.dropna()
		self.chr_bin_curves = self.bin_curves_G_df.join(geneset[geneset['chr'] == chrom][[]], how='inner')

		# Deconvolve the normalized bin curves, will this allow for us 
		# to create a more high resolution replication timing profile?
		print("Setting up model with temporary gene. (Actual deconvolution will use chrom bin curves, not gene data)...")
		self.config = create_dynamic_alpha_config(0, self.replicate)
		self.chrom_model = ChromatinModel(self.config)
		self.chrom_model.load_mnase_gene("CLN1") # todo: dummy gene, we will not actually be deconvolving gene


	def get_f_for_alpha(self, alpha):
		"""Deconvolve the bin curves for a given alpha"""

		config = create_dynamic_alpha_config(alpha, self.replicate)
		self.config = config
		
		# Set the config to the updated alpha
		chrom_model = self.chrom_model
		chrom_model.config = config
		chrom_model.create_deconvolution_bins()
		
		# Set G to the 10k occupancy bin curves for the chromosome
		chrom_model.G = self.chr_bin_curves.T.values+1.
		
		chrom_model.setup_deconv_model()
		chrom_model.gamma = 0.001
		chrom_model.deconvolve()

		f = chrom_model.deconvolved_f()

		return f

	def alpha_search(self, alphas = np.arange(0, 30, 1)):
		"""Deconvolve the set of alpha values"""

		# Setup to get H dimensions
		chrom_model = self.chrom_model
		chrom_model.setup_deconv_model()

		# Dimensions of the returned alpha search f images
		u, m = self.chr_bin_curves.shape[0], chrom_model.deconv_model.H.shape[1]
		
		alpha_fs = np.zeros((len(alphas), m, u))

		from src.timer import Timer

		timer = Timer()

		for i in range(len(alphas)):
			alpha = alphas[i]
			print(f"{i}/{len(alphas)}, alpha={alpha}")
			deconvolved_f = self.get_f_for_alpha(alpha)
			alpha_fs[i] = deconvolved_f
			timer.print_time()

		self.alphas = alphas
		self.alpha_fs = alpha_fs


	def plot_alpha_search_results(self):

		config = self.config
		alphas = self.alphas
		alpha_fs = self.alpha_fs

		# For indexing the result
		i_indices = config.get_Hpositions_for_branch('i')
		t_indices = config.get_Hpositions_for_branch('t')
		b_indices = config.get_Hpositions_for_branch('b')
		cg1_ind = config.get_Hpositions_for_phase('CG1')
		dg1_ind = config.get_Hpositions_for_phase('DG1')
		postG1_ind = config.get_Hpositions_for_phase('postG1')

		def plot_alpha_avg(g1_indices, alpha_fs, i, alphas):
			color = plt.get_cmap('Spectral')(i/len(alphas))
			alpha = alphas[i]
			dg1 = alpha_fs[i][g1_indices].mean(axis=1)
			postg1 = alpha_fs[i][postG1_ind].mean(axis=1)
			
			plt.plot(np.arange(len(dg1)), dg1, c='black', lw=2, zorder=1)
			plt.plot(np.arange(len(postg1)) + len(dg1), postg1, c='gray', lw=3, zorder=1)
			
			plt.plot(np.arange(len(dg1)), dg1, c=color)
			plt.plot(np.arange(len(postg1)) + len(dg1), postg1, c=color, label=alpha)
			plt.ylim(0.5, 3.75)
			plt.legend(ncol=3)

		plt.figure(figsize=(17, 4))
		plt.subplot(1, 4, 1)
		plt.subplots_adjust(wspace=0.35, top=0.75)

		for i in range(len(alphas)):
			plot_alpha_avg(cg1_ind, alpha_fs, i, alphas)
		plt.title("Mother")

		plt.subplot(1, 4, 2)
		for i in range(len(alphas)):
			plot_alpha_avg(dg1_ind, alpha_fs, i, alphas)
		plt.title("Daughter")

		mean_dg1 = np.median(alpha_fs[:, dg1_ind[0], :], axis=1)
		mean_cg1 = np.median(alpha_fs[:, cg1_ind[0], :], axis=1)
		avg_cg1_dg1 = (mean_dg1+mean_cg1)/2

		plt.subplot(1, 4, 3)
		plt.plot(alphas, avg_cg1_dg1, label="CG1/DG1 mean")

		min_ind = np.argmin(mean_cg1)
		min_alpha = alphas[min_ind]
		plt.axvline(min_alpha, c='black', lw=1, ls='dotted')
		plt.axvline(14, c='black', lw=1, ls='dotted')
		plt.xlabel("Alpha")
		plt.ylabel("First time point occupancy")
		plt.title("Determining Alpha from CG1/DG1")

		# The continuity from the end of CG1 to the start of PostG1
		plt.subplot(1, 4, 4)
		last_dg1 = np.median(alpha_fs[:, dg1_ind[-1], :], axis=1)
		last_cg1 = np.median(alpha_fs[:, cg1_ind[-1], :], axis=1)
		start_postg1 = np.median(alpha_fs[:, postG1_ind[0], :], axis=1)
		cg1_dg1_post_g1_diff = (np.abs(last_dg1-start_postg1) + np.abs(last_cg1-start_postg1))/2.
		plt.plot(alphas, cg1_dg1_post_g1_diff, label="DG1/CG1 and postG1")
		plt.axvline(14, c='black', lw=1, ls='dotted')
		plt.title("Continuity\nbetween G1 and PostG1")
		plt.suptitle("Replicate 1, chromosome 4", fontsize=32)

	def plot_example_alpha_repl_timing(self, ind):
		
		config = self.config
		alpha_fs = self.alpha_fs
		t_indices = config.get_Hpositions_for_branch('t')
		b_indices = config.get_Hpositions_for_branch('b')

		t_tps = config.get_timepoints_for_branch('t')
		b_tps = config.get_timepoints_for_branch('b')
		
		plt.figure(figsize=(13, 4))

		alpha = self.alphas[ind]

		plt.subplot(2, 1, 1)
		plt.imshow(alpha_fs[ind][t_indices], origin='lower', 
				   vmin=1, vmax=2, aspect='auto',
				  extent=[0, 1000, t_tps[0], t_tps[-1]])
		plt.title("Mother")
		plt.xticks([])

		plt.subplot(2, 1, 2)
		plt.imshow(alpha_fs[ind][b_indices], origin='lower', vmin=1, vmax=2, aspect='auto',
				  extent=[0, 1000, b_tps[0], b_tps[-1]])
		plt.title("Daughter")
		plt.xticks([])

		plt.suptitle(f"Deconvolved gene replication profile, chr4, replicate 1, alpha={alpha}")
