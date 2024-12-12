
import numpy as np
import cvxpy as cp
from src.timer import Timer
import pandas as pd
from matplotlib import pyplot as plt
from src.replication_deconvolution_solver import deconvolve_replication
from src.utils import print_fl

class RealDataReplicationDeconvolution():
	"""This model deconvolve the replication timing.
	"""
	def __init__(self, config=None, chr=10, replicate=None, configs=None,
		deconvolve_combined=False):

		self.chrom = chr

		if deconvolve_combined:

			print_fl("Setting up combined deconvolution model")

			self.load_replicate_data(chr, 1)
			self.setup_deconvolution(configs[0])
			self.H1 = self.H
			self.average_DNA_1 = self.average_DNA
			self.N1 = self.N
			self.G1 = self.G
			self.B1 = self.B
			self.config1 = configs[0]
			self.thresh1 = self.selected_threshold_region

			self.load_replicate_data(chr, 2)
			self.setup_deconvolution(configs[1])
			self.average_DNA_2 = self.average_DNA
			self.H2 = self.H
			self.B2 = self.B
			self.G2 = self.G
			self.config2 = configs[0]
			self.thresh2 = self.selected_threshold_region

			self.H = np.concatenate([self.H1, self.H2], axis=0)
			self.G = np.concatenate([self.G1, self.G2], 
				axis=0)

			# Construct N as average DNA from replicate 1 and 2 concatenated
			self.average_DNA = np.concatenate([self.average_DNA_1, self.average_DNA_2])
			self.N = np.linalg.inv(np.diag(self.average_DNA))

			# Construct B as the average of replicate 1 and 2
			self.B = (self.B1 + self.B2)/2.

			self.deconvolve_combined = True
			self.selected_threshold_region = self.thresh1 | self.thresh2

		else:
			self.config = config
			self.load_replicate_data(chr, replicate)
			self.setup_deconvolution(config)


	def load_replicate_data(self, chr, replicate):

		from src.mnase_10kb_loader import MNase10kbLoader

		mnase_loader = MNase10kbLoader()
		mnase_loader.load_mnase_data(replicate=replicate, chromosome=chr)
		mnase_loader.compute_sliding_window_counts_all_times()

		self.mnase_loader = mnase_loader
		self.unnormalized_total_occupancy = mnase_loader.all_counts_unnormalized_df
		self.normalized_occupancy = mnase_loader.normalized_total_occupancy_df
	
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

		print_fl("Initializing N using config H.")
		self.G = self.normalized_occupancy.T.values
		self.average_DNA, self.N = compute_N(config)

		print_fl("Initializing B using timepoint 0")
		self.B = np.diag(self.G[0])


	def deconvolve(self):
		self.F, self.rn, self.sn = deconvolve_replication(self.config, self.H, self.G, self.N, self.B)


	def plot_heatmaps(self):
		plot_heatmaps(self.N, self.F, self.H, self.B, self.G)
		plt.suptitle(f"Replication deconvolution,\nChromosome {self.chrom}")


	def save_replication_deconvolution(self):
		from src.utils import mkdirs_safe

		copy_correction_save_dir = 'data/copy_correction/'
		mkdirs_safe([copy_correction_save_dir])

		N_save_path = f'{copy_correction_save_dir}/N_combined.npy'
		B_save_path = f'{copy_correction_save_dir}/B_chr{self.chrom}_combined.csv'
		F_save_path = f'{copy_correction_save_dir}/F_chr{self.chrom}_combined.csv'

		np.save(N_save_path, self.N)
		np.save(B_save_path, self.B)

		B_df = pd.DataFrame(self.B,
		            index=self.normalized_occupancy.index,
		            columns=self.normalized_occupancy.index)
		B_df.index.name = 'start'
		B_df.to_csv(B_save_path)

		F_df = pd.DataFrame(self.F, 
		    columns=self.normalized_occupancy.index)
		F_df.to_csv(F_save_path, index=False)

		print(f"Saved to: {N_save_path}")
		print(f"Saved to: {B_save_path}")
		print(f"Saved to: {F_save_path}")


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
	plt.subplots_adjust(hspace=0.5, top=0.9)

def compute_N(config, plot=False):
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

	if plot:
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


def read_n_fr_b(chrom, deconv_span):
    from src.mnase_10kb_loader import get_bin_for_position

    N = np.load('data/copy_correction/N_combined.npy')
    B_df = pd.read_csv(f'data/copy_correction/B_chr{chrom}_combined.csv').set_index('start')
    Fr_df = pd.read_csv(f'data/copy_correction/F_chr{chrom}_combined.csv')
    B_df.columns = B_df.columns.astype(int)
    Fr_df.columns = Fr_df.columns.astype(int)

    start_indices = B_df.index

    mid_span = (deconv_span[0]+deconv_span[1])/2
    bin_idx, start = get_bin_for_position(mid_span, start_indices)

    # Load the b and f_replication for the span to be deconvolved.
    b = B_df.loc[start].loc[start]
    fr = Fr_df[start]

    return N, fr, b


def read_no_copy_correction_n_fr_b(H):

	n, m = H.shape
	N = np.eye(n)
	b = 1
	fr = np.ones(m)

	return N, fr, b

