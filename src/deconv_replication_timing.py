


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
		self.chrom = chrom

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

		# cg1_tps = config.get_timepoints_for_branch('t')[:len(cg1_ind)]
		# postg1_tps = config.get_timepoints_for_branch('t')[len(cg1_ind):]
		# dg1_tps = config.get_timepoints_for_branch('t')[:len(dg1_ind)]


		def plot_alpha_avg(g1_indices, alpha_fs, i, alphas):
			color = plt.get_cmap('Spectral')(i/len(alphas))
			alpha = alphas[i]
			dg1 = np.median(alpha_fs[i][g1_indices], axis=1)
			postg1 = np.median(alpha_fs[i][postG1_ind], axis=1)
			
			plt.plot(np.arange(len(dg1)), dg1, c='black', lw=2, zorder=1)
			plt.plot(np.arange(len(postg1)) + len(dg1), postg1, c='gray', lw=3, zorder=1)
			
			plt.plot(np.arange(len(dg1)), dg1, c=color)
			plt.plot(np.arange(len(postg1)) + len(dg1), postg1, c=color, label=alpha)
			plt.ylim(0.5, 3.75)
			plt.legend(ncol=3)

		plt.figure(figsize=(31, 9))
		plt.subplot(1, 4, 1)
		plt.subplots_adjust(wspace=0.35, top=0.75)

		for i in range(len(alphas)):
			plot_alpha_avg(cg1_ind, alpha_fs, i, alphas)
		plt.title("Mother median curve", fontsize=32, pad=20)
		plt.ylabel("Occupancy", fontsize=16)

		plt.subplot(1, 4, 2)
		for i in range(len(alphas)):
			plot_alpha_avg(dg1_ind, alpha_fs, i, alphas)
		plt.title("Daughter median curve", fontsize=32, pad=20)
		plt.ylabel("Occupancy", fontsize=16)

		median_dg1 = np.median(alpha_fs[:, dg1_ind[0], :], axis=1)
		median_cg1 = np.median(alpha_fs[:, cg1_ind[0], :], axis=1)
		avg_cg1_dg1 = (median_dg1+median_cg1)/2

		plt.subplot(1, 4, 3)
		plt.plot(alphas, avg_cg1_dg1, label="CG1/DG1 median")
		plt.xlabel("Alpha", fontsize=16)
		plt.ylabel("Occupancy", fontsize=16)
		plt.title("First timepoint (CG1+DG1)/2", fontsize=32, pad=20)

		# The continuity from the end of CG1 to the start of PostG1
		plt.subplot(1, 4, 4)
		last_dg1 = np.median(alpha_fs[:, dg1_ind[-1], :], axis=1)
		last_cg1 = np.median(alpha_fs[:, cg1_ind[-1], :], axis=1)
		start_postg1 = np.median(alpha_fs[:, postG1_ind[0], :], axis=1)
		cg1_dg1_post_g1_diff = (np.abs(last_dg1-start_postg1) + np.abs(last_cg1-start_postg1))/2.
		plt.plot(alphas, cg1_dg1_post_g1_diff, label="DG1/CG1 and postG1")
		plt.title("Continuity\nbetween G1 and PostG1", fontsize=32, pad=20)
		plt.suptitle(f"Replicate {self.replicate}, chromosome {self.chrom}", fontsize=64)

	def plot_example_alpha_repl_timing(self, ind):

		self.compute_replication_timing(ind)
		
		config = self.config
		alpha_fs = self.alpha_fs
		t_indices = config.get_Hpositions_for_branch('t')
		b_indices = config.get_Hpositions_for_branch('b')

		t_tps = config.get_timepoints_for_branch('t')
		b_tps = config.get_timepoints_for_branch('b')
		
		plt.figure(figsize=(13, 4))

		self.alpha = self.alphas[ind]

		plt.subplot(2, 1, 1)
		plt.imshow(alpha_fs[ind][t_indices], origin='lower', 
				   vmin=1, vmax=2.5, aspect='auto',
				  extent=[0, alpha_fs[ind].shape[1], t_tps[0], t_tps[-1]])
		plt.plot(self.mother_timing_df.tp, c='red')
		plt.title("Mother")
		plt.xticks([])

		plt.subplot(2, 1, 2)
		plt.imshow(alpha_fs[ind][b_indices], origin='lower', vmin=1, vmax=2.5, aspect='auto',
				  extent=[0, alpha_fs[ind].shape[1], b_tps[0], b_tps[-1]])
		plt.plot(self.daughter_timing_df.tp, c='red')

		plt.title("Daughter")
		plt.xticks([])

		plt.suptitle(f"Deconvolved gene replication profile, chr{self.chrom}, replicate {self.replicate}, alpha={self.alpha}")


	def compute_replication_timing(self, alpha_ind):
		self.config = create_dynamic_alpha_config(self.alpha, self.replicate)
		selected_alpha_data = self.alpha_fs[alpha_ind]
		self.alpha = self.alphas[alpha_ind]
		gene_index = self.chr_bin_curves.index
		self.mother_timing_df, self.daughter_timing_df = compute_replication_timing(selected_alpha_data, 
			self.config, gene_index)


	def plot_replicate_timing_distribution(self):
		bins = np.arange(-20, 20, 1)

		plt.figure(figsize=(11, 3))
		plt.subplots_adjust(top=0.75, wspace=0.35)
		plt.subplot(1, 3, 1)

		plt.hist(self.mother_timing_df.tp, bins=bins)
		plt.title("Mother\nreplicate timing")
		plt.xlabel("Mother replicate timing")

		plt.subplot(1, 3, 2)
		plt.hist(self.daughter_timing_df.tp, bins=bins)
		plt.title("Daughter\nreplicate timing")
		plt.xlabel("Daughter replicate timing")

		plt.subplot(1, 3, 3)
		plt.scatter(self.mother_timing_df.tp, 
			self.daughter_timing_df.tp, s=1)
		plt.xlim(-5, 20)
		plt.ylim(-5, 20)
		plt.ylabel("Mother replicate timing")
		plt.xlabel("Daughter replicate timing")

		plt.title("Mother, daughter replicate\ntiming concordance")
		plt.suptitle(f"Replication timing profile replicate {self.replicate}, "
					 f"chromosome {self.chrom}, alpha={self.alpha}")


def get_index_threshold(data, indices, threshold):
	"""Get the index of the first index in which the threshold is met"""

	from src.TracerPlotter import normalize_max_min

	dat = data[indices].copy()
	norm_dat = normalize_max_min(dat)
	idx_threshold = (norm_dat > threshold).argmax()
	if (dat.max() - dat.min()) < .05: return -1
	return idx_threshold


def get_replication_timing(dat, indices, tps, threshold=0.75):
	"""Get the replication timing from a set of deconvolved 10k F bins"""

	# Create array of indices to store the replication indices
	ncols = dat.shape[1]
	repl_indices = np.zeros(ncols).astype(int)
	
	# Loop over the get index function
	for column in np.arange(ncols):
		repl_index = get_index_threshold(dat[:, column], indices, threshold)
		repl_indices[column] = repl_index
		
	# Compute the mapping from index to timing
	repl_timing = [tps[i] if i >= 0 else np.nan for i in repl_indices]
	
	return repl_indices, repl_timing

def compute_replication_timing(f_data, config, gene_index, threshold=0.75):
	"""Compute the replication timing for mother and daughter branch"""

	# Let's see if can compute replicating timing now
	# Same logic as before: 75% of way from the min to the max will be designated as the
	# replication time.

	from src.TracerPlotter import normalize_max_min

	t_indices = config.get_Hpositions_for_branch('t')
	t_tps =  config.get_timepoints_for_branch('t')

	b_indices = config.get_Hpositions_for_branch('b')
	b_tps =  config.get_timepoints_for_branch('b')

	mother_repl_indices, mother_repl_timing = get_replication_timing(f_data, t_indices, t_tps, threshold)
	daughter_repl_indices, daughter_repl_timing = get_replication_timing(f_data, b_indices, b_tps, threshold)

	from src.geneset import get_deconvolved_geneset

	mother_timing_df = pd.DataFrame(index=gene_index, data={
		'f_index': mother_repl_indices,
		'tp': mother_repl_timing})

	daughter_timing_df = pd.DataFrame(index=gene_index, data={
		'f_index': daughter_repl_indices,
		'tp': daughter_repl_timing})

	return mother_timing_df, daughter_timing_df
