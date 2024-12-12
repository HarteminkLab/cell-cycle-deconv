
import numpy as np
import cvxpy as cp
from src.timer import Timer
import pandas as pd
from matplotlib import pyplot as plt
from src.replication_deconvolution_solver import deconvolve_replication


class RealDataReplicationDeconvolution():
	"""This model deconvolve the replication timing.
	"""
	def __init__(self, config=None, chr=10, replicate=None, configs=None,
		deconvolve_combined=False):

		if deconvolve_combined:

			self.load_replicate_data(chr, 1)
			self.setup_deconvolution(configs[0])
			self.H1 = self.H
			self.raw1_data = self.normalized_occupancy
			self.G1 = self.normalized_occupancy
			self.config1 = configs[0]
			self.thresh1 = self.selected_threshold_region

			self.load_replicate_data(chr, 2)
			self.setup_deconvolution(configs[1])
			self.raw2_data = self.normalized_occupancy
			self.H2 = self.H
			self.G2 = self.normalized_occupancy
			self.config2 = configs[0]
			self.thresh2 = self.selected_threshold_region

			# Construct combined data
			self.normalized_occupancy = np.concatenate([
				self.raw1_data, self.raw2_data
			], axis=1)
			self.H = np.concatenate([self.H1, self.H2], axis=0)
			self.normalized_occupancy = np.concatenate([self.G1, self.G2], 
				axis=1)
			self.deconvolve_combined = True
			self.selected_threshold_region = self.thresh1 | self.thresh2

		else:
			self.load_replicate_data(chr, replicate)
			self.setup_deconvolution(config)


	def load_replicate_data(self, chr, replicate):

		from src.mnase_replication_timing_analysis import MNaseOriginAnalysis

		mnase_analysis_rep = MNaseOriginAnalysis()
		mnase_analysis_rep.load_mnase_data(replicate=replicate, chromosome=chr)

		mnase_analysis_rep.compute_sliding_window_counts_all_times()
		unnormalized_total_occupancy = mnase_analysis_rep.all_counts_unnormalized_df
		normalized_total_occ = unnormalized_total_occupancy / \
			unnormalized_total_occupancy.mean(axis=0).values.reshape((1, -1))

		self.mnase_analysis_rep = mnase_analysis_rep
		self.unnormalized_total_occupancy = unnormalized_total_occupancy
		self.normalized_occupancy = normalized_total_occ
	
		# Setup regions to threshold, 
		# regions with low occupancy will be omitted when needed
		self.selected_threshold_region = self.normalized_occupancy.T.mean(axis=0) > 0.75


	def setup_deconvolution(self, config):
		"""Setup the deconvolution:

		1. The config and H
		"""

		self.config = config
		self.H, _ = config.calcH_function(config.intervals_wt1, 
			config.WT1_TIMEPOINTS)



def threshold_selection(dat, threshold_selection, fill=1.,
	renormalize=False):
	new_dat = dat.copy()
	new_dat[:, ~threshold_selection] = fill

	# If thresholded to fill with nans, we can renormalize such
	# that the non-thresholded out regions mean to 1
	if renormalize:

		# Get the current mean of the good rows
		row_means = np.nanmean(new_dat, axis=1)

		# Divide the data by these mean
		new_dat = new_dat / row_means.reshape((-1, 1))

	return new_dat


def plot_comparison_occupancies_G():
	# Def plot of chromosome 4 values
	# # Comparison of chrIV t=0 and t=30
	# plt.figure(figsize=(24, 6))
	# plt.subplot(3, 1, 1)
	# plt.plot(real_deconv_chr4.normalized_occupancy.T.loc[0])
	# plt.xticks([])
	# plt.axhline(1, c='black', lw=1)
	# plt.title("T=0")

	# plt.subplot(3, 1, 2)
	# plt.plot(real_deconv_chr4.normalized_occupancy.T.loc[30])
	# plt.axhline(1, c='black', lw=1)
	# plt.xticks([])
	# plt.title("T=30")

	# plt.subplot(3, 1, 3)
	# plt.plot(np.log2(real_deconv_chr4.normalized_occupancy.T.loc[30] / 
	#                  real_deconv_chr4.normalized_occupancy.T.loc[0])
	#         )
	# plt.axhline(0, c='black', lw=1)
	# plt.ylim(-1, 1)
	# plt.title("log2 ratio difference")
	# plt.suptitle("ChrIV t=0 vs t=30 occupancy")
	pass

def plot_histogram_occupancies_G(config, G):
	fig, axs = plt.subplots(3, 6, figsize=(13, 6))

	tps = config.WT1_TIMEPOINTS
	axs = np.array(axs).T.flatten()
	plot_G = G.T

	for ax in axs:
	    ax.set_xticks([])
	    ax.set_yticks([])
	for i in range(plot_G.shape[1]):
	    ax = axs[i]
	    
	    ax.hist(plot_G[:, i], bins=20, facecolor=plt.get_cmap('Spectral')(i/plot_G.shape[1]),
	           edgecolor='gray', lw=1)
	    ax.set_xlim(0, 2)
	    ax.set_ylim([0, 120])
	    ax.axvline(1, c='black', lw=1, ls='dotted')
	    ax.set_title(f"{tps[i]} min")
	    if i % 3 == 2:
	        ax.set_xticks([0, 1, 2])
	    
	plt.suptitle("Distribution of normalized G data per timepoint")
	plt.subplots_adjust(hspace=0.5)

	# This shoes the data for G is normal-ish and centered around 1.

def plot_heatmaps(N, F, H, B, G):

    inv_B = np.linalg.inv(B)
    transformed_G = (np.linalg.inv(N)@G@inv_B)

    plt.figure(figsize=(13, 11))

    plt.subplot(6, 1, 1)
    plt.imshow(F, cmap='RdBu_r', vmin=0, vmax=2, aspect='auto')
    plt.xticks([])
    plt.colorbar()
    plt.title("$F$")

    plt.subplot(6, 1, 2)
    plt.imshow(H@F, cmap='RdBu_r', vmin=0, vmax=2, aspect='auto')
    plt.xticks([])
    plt.colorbar()
    plt.title("$HF$")

    plt.subplot(6, 1, 3)
    plt.imshow(transformed_G, cmap='RdBu_r', vmin=0, vmax=2, aspect='auto')
    plt.xticks([])
    plt.colorbar()
    plt.title("$(N^{-1})G(B^{-1})$")

    plt.subplot(6, 1, 4)
    predicted_G = N@H@F@B
    plt.imshow(predicted_G, cmap='RdBu_r', vmin=0, vmax=2, aspect='auto')
    plt.xticks([])
    plt.colorbar()
    plt.title("Predicted G: $NHFB$")

    plt.subplot(6, 1, 5)
    plt.imshow(G, cmap='RdBu_r', vmin=0, vmax=2, aspect='auto')
    plt.colorbar()
    plt.xticks([])
    plt.title("$G$")

    plt.subplot(6, 1, 6)
    plt.imshow((N@H@F@B)/(G)-1, vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto')
    plt.colorbar()
    plt.title("$\\frac{NHFB}{G} -1$")
    plt.subplots_adjust(hspace=0.5)

def compute_N(config):
	from src.model import color_for_key

	H = config.calculate_H()
	tps = config.WT1_TIMEPOINTS

	cg1_mass = H[:, config.get_Hpositions_for_phase('RG1')].sum(axis=1)
	rg1_mass = H[:, config.get_Hpositions_for_phase('CG1')].sum(axis=1)
	s_mass = H[:, config.get_Hpositions_for_phase('S')].sum(axis=1)
	g2m_mass = H[:, config.get_Hpositions_for_phase('G2M')].sum(axis=1)
	h_mass = H[:, config.get_Hpositions_for_phase('H')].sum(axis=1)

	# Assume linear transition of S-phase
	s_indices = config.get_Hpositions_for_phase('S')
	s_masses = np.linspace(1, 2, len(s_indices))

	g1_mass = cg1_mass+rg1_mass
	s_dna_content = H[:, config.get_Hpositions_for_phase('S')] @ np.diag(s_masses).sum(axis=1)
	replicating_mass = s_dna_content+g2m_mass*2

	average_DNA = h_mass+g1_mass+replicating_mass
	N = np.linalg.inv(np.diag(average_DNA))

	plt.figure(figsize=(6, 3))

	plt.fill_between(tps, h_mass, 0, label="H mass", color=color_for_key('H'))

	plt.fill_between(tps, g1_mass+h_mass, h_mass, label="G1 mass", color=color_for_key('CG1'))

	# ---------

	plt.fill_between(tps, g1_mass+s_dna_content+h_mass, 
	                      g1_mass+h_mass, label="S mass", color=color_for_key('S'))

	plt.fill_between(tps, g1_mass+s_dna_content+g2m_mass*2+h_mass, 
	                     g1_mass+s_dna_content+h_mass, 
	                     label="G2M mass", color=color_for_key('G2M'))


	plt.legend()

	plt.title("Estimation of average DNA content, CLOCCS")
	plt.xlabel("Clock time")
	plt.ylabel("Average DNA content")

	return average_DNA, N
