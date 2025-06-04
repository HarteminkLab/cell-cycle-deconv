 
import numpy as np
import cvxpy as cp
import pandas as pd

from src.timer import Timer
from src.utils import print_fl
from matplotlib import pyplot as plt
from src.replication_deconvolution_solver import deconvolve_replication_brute_force
from typing import Tuple, Optional, NamedTuple
from src.expression_chromatin_plots import draw_phase_label_annotations
from src.global_config import fragment_lengths_definitions
from src.mnase_10kb_loader import get_bin_for_position


early_color = plt.get_cmap('Oranges')(0.75)
late_color = plt.get_cmap('Purples')(0.75)


class RealDataReplicationDeconvolution():
	"""This model deconvolve the replication timing.
	"""
	def __init__(self, config, chr, replicate, len_span=None):

		self.chrom = chr
		self.replicate = replicate
		self.config = config
		self.load_replicate_data(chr, replicate, len_span)
		self.std_q_threshold = 0.75
		self.enable_std_thresholding = False
		self.deconvolve_stage = 1


	def load_replicate_data(self, chr, replicate, len_span=None):

		from src.mnase_10kb_loader import MNase10kbLoader
		small_span, med_span, nuc_span = fragment_lengths_definitions()

		if len_span is None:
			len_span = nuc_span

		mnase_loader = MNase10kbLoader()
		mnase_loader.load_mnase_data(replicate=replicate, chromosome=chr,
			fragment_lengths_span=len_span)
		mnase_loader.compute_sliding_window_counts_all_times()

		self.mnase_loader = mnase_loader
		self.unnormalized_total_occupancy = mnase_loader.all_counts_unnormalized_df
		self.normalized_occupancy = mnase_loader.normalized_total_occupancy_df
		self.G_df = self.normalized_occupancy.T
	
		# Setup regions to threshold, 
		# regions with low occupancy will be omitted when needed
		print_fl("Omit windows with less than 75% read coverage.")
		self.selected_threshold_region = self.normalized_occupancy.T.mean(axis=0) > 0.75

	def setup_deconvolution(self, config=None, initial_N=None, initial_B=None,
		warm_start_output_directory=None, warm_start_chrom=None):
		"""Setup the deconvolution:
		1. H from the config parameters
		2. G from the normalized data, masked out for low coverage regions
		3. N from the cell cycle parameters as defined in the config
		4. B from the first timepoint in G

		If loading from a warm start, we will load N, F, and the config parameters
		from disk. From the output directory: warm_start_output_directory

		"""

		if config is None and self.config is None:
			raise ValueError("Config has not been initialized")

		elif config is not None:
			self.config = config

		if warm_start_output_directory is not None:
			print_fl(f"Warm start config, N, F, and B from directory: {warm_start_output_directory}")
			self.config = modify_config_from_run(self.config, warm_start_output_directory, self.replicate,
				warm_start_chrom)
		else:
			self.config.calculate_H()

		# Mask out the low coverage regions
		all_indices = self.selected_threshold_region.index

		# Two stages of masking

		# ---- 1. Mask indices of low coverage based on threshold region -------

		# Selected on adequate occupancy and non-alpha affected regions
		# Mask out high occupancy regions during alpha-release (RG1)
		keep_column_indices = all_indices[self.selected_threshold_region]

		self.masked_G_df = self.G_df[keep_column_indices]
		self.masked_start_indices = keep_column_indices
		self.full_start_indices = all_indices

		# ---- 2. Mask indices of high variation, and randomly subsample to increase convergence speed ----

		if self.enable_std_thresholding:
			std_cutoff_start_indices = self.randomly_subset_windows_to_deconvolve(self.masked_G_df, 
				quantile=self.std_q_threshold,
				k=None, plot_std=False)
		else:
			std_cutoff_start_indices = keep_column_indices

		self.G = self.G_df[std_cutoff_start_indices].values
		self.deconvolve_start_indices = std_cutoff_start_indices

		# ---------------------------------------------------------

		print(f"Deconvolving {len(self.deconvolve_start_indices)} regions")

		if warm_start_output_directory is not None:
			# N is the most important to load from disk, F, and B will converge properly on the
			# first set of N, F, B iterations
			F, N, B = load_N_F_B_for_replication_deconv_from_save(warm_start_output_directory, self.replicate, warm_start_chrom)
			initial_N = N

		if initial_N is None:
			self.average_DNA, self.initial_N = compute_N(self.config)
		else:
			self.initial_N = initial_N

		if initial_B is None:
			print_fl("Initializing B using timepoint 0")
			self.initial_B = np.diag(self.G[0])
		else:
			self.initial_B = initial_B

	def randomly_subset_windows_to_deconvolve(self, G_df, quantile=0.75, k=None, plot_std=False):
		
		std_df, under_cutoff_windows, \
		quantile, cutoff = self.compute_std_cutoffs_G(G_df, quantile=quantile, plot=plot_std)

		under_cutoff_start_indices = sorted(under_cutoff_windows.index.values)

		# Randomly subset to try and speed up convergence
		if k is not None:
			np.random.seed(123)
			ret_indices = np.random.choice(under_cutoff_start_indices, k)
		else:
			ret_indices = under_cutoff_start_indices

		return ret_indices
		

	def iterative_deconvolution_updates(self, total_iterations, timer=None,
		initial_N=None, initial_B=None, verbose=True):
		"""Iteratively deconvolve for the replication curve F."""

		# If first run, use the initalized N and B from setup
		if initial_N is None:
			initial_N = self.initial_N
		if initial_B is None:
			initial_B = self.initial_B

		result = iterative_deconvolution_updates(
			config=self.config, H=self.config.H, G=self.G, initial_N=initial_N, 
			initial_B=initial_B, total_iterations=total_iterations, 
			timer=timer, verbose=verbose)

		self.N = result.Ns[-1]
		self.F = result.Fs[-1]
		self.rn = result.iterative_update_rns[-1]
		self.B = result.Bs[-1]
		self.result = result

		self.F_df = pd.DataFrame(self.F, columns=self.deconvolve_start_indices,
			index=range(self.F.shape[0]))
		self.b_df = pd.DataFrame(np.diag(self.B), index=self.deconvolve_start_indices,
						   columns=['b']).T

	def compute_rn(self):
		N, H, F, B = self.N, self.config.H, self.F, self.B
		G = self.G
		return compute_rn(N, H, F, B, G)


	def save_to_disk(self, out_directory):
		"""Save the deconvolved replication profiles to disk"""
		from src.utils import mkdirs_safe

		self.save_dir = out_directory

		mkdirs_safe([out_directory])

		N_save_path = f'{self.save_dir}/rep{self.replicate}_chr{self.chrom}_N.npy'
		B_save_path = f'{self.save_dir}/rep{self.replicate}_chr{self.chrom}_B.csv'
		F_save_path = f'{self.save_dir}/rep{self.replicate}_chr{self.chrom}_F.csv'
		H_save_path = f'{self.save_dir}/rep{self.replicate}_chr{self.chrom}_H.npy'
		G_save_path = f'{self.save_dir}/rep{self.replicate}_chr{self.chrom}_G.csv'

		if self.deconvolve_stage == 1:
			parameters_save_path = f'{self.save_dir}/rep{self.replicate}_chr{self.chrom}_parameters.csv'
		else:
			parameters_save_path = f'{self.save_dir}/rep{self.replicate}_chr{self.chrom}_parameters_stage2.csv'
			configs

		# Save latest config to json file
		self.config.save_to_path(f"{self.save_dir}/rep{self.config.replicate}.json")

		fig_path = f'{self.save_dir}/rep{self.replicate}_chr{self.chrom}.png'

		np.save(N_save_path, self.N)
		np.save(H_save_path, self.config.H)
		self.F_df.to_csv(F_save_path)
		self.G_df.to_csv(G_save_path)
		self.b_df.to_csv(B_save_path)
		pd.DataFrame(self.config.params_dic, index=[0]).to_csv(parameters_save_path)

		fig = self.plot_heatmaps()
		plt.savefig(fig_path, dpi=200)
		plt.close(fig)

		print_fl(f"Saved to: {N_save_path}")
		print_fl(f"Saved to: {B_save_path}")
		print_fl(f"Saved to: {H_save_path}")
		print_fl(f"Saved to: {F_save_path}")
		print_fl(f"Saved to: {G_save_path}")
		print_fl(f"Saved to: {parameters_save_path}")
		print_fl(f"Saved to: {fig_path}")


	def update_N_B(self, N, H, G, B, F):
		"""Using the solution from the last run, update N and B"""

		# Update N based on G, B, H, and F
		GBinv_HF_div = np.divide((G@np.linalg.inv(B)), (H@F))
		updated_N = np.diag(GBinv_HF_div.mean(axis=1))

		# Update B based on N H F and G
		num_rows = G.shape[0]
		G_sums = G.T @ np.ones((num_rows, 1))
		NHF_sums = (updated_N@H@F).T @ np.ones((num_rows, 1))
		updated_b_diag = (G_sums / NHF_sums).flatten()
		updated_B = np.diag(updated_b_diag)

		return updated_N, updated_B


	def plot_heatmaps(self, figsize=(13, 11)):
		N, F, G, B, H = self.N, self.F, self.G, self.B, self.config.H

		fig = plot_heatmaps(N, F, H, B, G, column_names=self.deconvolve_start_indices,
			full_column_names=self.G_df.columns, figsize=figsize)
		plt.suptitle(f"Replication {self.replicate}"
			f" deconvolution,\nChromosome {self.chrom}")

		return fig

	def plot_B(self):

		from src.sgd import get_chromosome_length

		chrom = self.chrom
		B = self.B
		chrom_len = get_chromosome_length(chrom)

		ys = np.diag(B)
		xs = np.linspace(0, chrom_len, len(ys))

		plt.figure(figsize=(9, 3))
		plt.plot(xs, ys, c=plt.get_cmap('Spectral')(0.3), lw=3)
		plt.title(f"Average 10 kb occupancy, chr{chrom}", pad=11)
		plt.xlim(0, chrom_len)
		plt.xlabel("Genomic position, bp")
		plt.ylabel("Average occupancy")
		plt.subplots_adjust(bottom=0.3)


	def plot_G(self):

		tps = self.config.timepoints
		start_indices = self.unnormalized_total_occupancy.index.values
		extent = [0, start_indices[-1], -5, tps[-1]+5]


		# predicted_G = self.N@self.config.H@self.F@self.B
		G = self.G# @ self.initial_B

		plt.figure(figsize=(13, 4))
		plt.imshow(G, cmap='RdBu_r', vmin=0, vmax=2, 
			interpolation='none', aspect='auto',
			extent=extent, origin='lower')
		plt.colorbar()
		plt.title(f"Experiment 10 kb MNase-seq reads, replicate {self.config.replicate}, chr{self.chrom}")
		plt.xlabel("Genomic position, bp")
		plt.ylabel("Experimental time")

		plt.ylim(tps[-1]+5, -5)
		plt.yticks(tps)

		plt.subplots_adjust(bottom=0.2)


	def plot_predicted_G(self):

		predicted_G = self.N@self.config.H@self.F@self.B

		tps = self.config.timepoints
		start_indices = self.unnormalized_total_occupancy.index.values
		extent = [0, start_indices[-1], 0, tps[-1]]

		plt.figure(figsize=(13, 3))
		plt.imshow(predicted_G, cmap='RdBu_r', vmin=0, vmax=2, 
			interpolation='none', aspect='auto',
			extent=extent)
		plt.colorbar()
		plt.title(f"Predicted $G$, replicate 1, chr{self.chrom}")
		plt.xlabel("Genomic position, bp")
		plt.ylabel("Experimental time")

		plt.subplots_adjust(bottom=0.2)

	def plot_residual(self):

		predicted_G = self.N@self.config.H@self.F@self.B
		residual = predicted_G - self.G

		tps = self.config.timepoints
		start_indices = self.unnormalized_total_occupancy.index.values
		extent = [0, start_indices[-1], 0, tps[-1]]

		plt.figure(figsize=(13, 3))
		plt.imshow(residual, cmap='RdBu_r', vmin=-1, vmax=1, 
			interpolation='none', aspect='auto',
			extent=extent)
		plt.colorbar()
		plt.title(f"Residual, replicate 1, chr{self.chrom}")
		plt.xlabel("Genomic position, bp")
		plt.ylabel("Experimental time")

		plt.subplots_adjust(bottom=0.2)

	def plot_F(self):
		from src.sgd import get_chromosome_length

		chrom = self.chrom
		chrom_len = get_chromosome_length(chrom)

		F = self.top_F_df

		fig = plt.figure(figsize=(13, 3))

		config = self.config
		t_indices = config.get_Hpositions_for_branch('t')
		t_tps = config.get_timepoints_for_branch('t')

		extent = [0, F.index[-1],
			F.columns[0], F.columns[-1]]

		plt.imshow(F.T, cmap='RdBu_r', vmin=0, vmax=2, 
			interpolation='none', aspect='auto',
				  extent=extent, origin='lower')

		cbar = plt.colorbar()
		cbar.ax.set_ylim(1, 2)
		cbar.ax.set_yticks([1, 2])
		cbar.ax.set_ylabel("Copy number")

		plt.title(f"Chr{chrom} replication profile, $F_r$")

		ax = plt.gca()

		draw_phase_label_annotations(ax, config, 
			flip=False, annotations_x=-11000)

		plt.xlim(extent[0]-25000, extent[1])
		plt.ylim(extent[3]-1, extent[2])
		plt.xlabel("Genomic position, bp")
		plt.ylabel("Average single\ncell time, min")
		plt.subplots_adjust(bottom=0.2)

	def compute_replication_profile(self):

		config = self.config
		t_indices = config.get_Hpositions_for_branch('t')
		t_tps = config.get_timepoints_for_branch('t')

		t_index_tp_mapping = pd.DataFrame({'tp': config.get_timepoints_for_branch('t')},
			index=config.get_Hpositions_for_branch('t'))

		timing_dict = t_index_tp_mapping.iloc[:, 0].to_dict()

		start_indices = self.unnormalized_total_occupancy.index.values

		F = self.F
		replication_profile = pd.DataFrame(np.argmax(F, axis=0), index=start_indices, 
			columns=['replication_index'])
		replication_profile['replication_timing'] = replication_profile['replication_index'].map(timing_dict)

		self.replication_profile = replication_profile

		# Create a dataframe of the top branch replication profile
		self.top_F_df = pd.DataFrame(F.T[:, t_indices], 
			index=start_indices, columns=t_tps)

	def plot_example_f_curves(self):

		config = self.config
		t_indices = config.get_Hpositions_for_branch('t')
		t_tps = config.get_timepoints_for_branch('t')

		start_indices = self.unnormalized_total_occupancy.index.values
		sorted_replication_profile = self.replication_profile.sort_values('replication_index')

		early_row = sorted_replication_profile.iloc[10]
		late_row = sorted_replication_profile.loc[570000]
		self.early_row = early_row
		self.late_row = late_row

		F = self.F

		# Create a dataframe of the top branch replication profile
		top_F_df = pd.DataFrame(F.T[:, t_indices], 
			index=start_indices, columns=t_tps)

		plt.figure(figsize=(8, 4))

		plt.plot(top_F_df.loc[early_row.name]+0.005, 
				 label=f"Early, {early_row.name}",
				c=early_color, lw=4)
		plt.plot(top_F_df.loc[late_row.name]-0.005, 
				 label=f"Late, {late_row.name}",
				 c=late_color, lw=4)
		plt.legend(loc='upper left')
		plt.ylabel("Copy number")
		plt.xlabel("Average single cell time, min")
		draw_phase_label_annotations(plt.gca(), config, flip=True, annotations_x=0.9)
		plt.xlim(t_tps[0], t_tps[-1])
		plt.yticks([1, 2])
		plt.title("Example replication curves", pad=11)


	def update_parameters(self, updated_parameters):
		self.config.params_dic.update(updated_parameters)
		self.config.update_timepoints()
		self.config.calculate_H()


	def compute_std_cutoffs_G(self, G_df, quantile=0.9, plot=True):
		"""Compute the quantile cutoff for standard deviation of G. High standard deviation
		windows typically are more difficult to deconvolve"""

		start_indices = G_df.columns

		std_df = pd.DataFrame({'std': np.std(G_df, axis=0), 
			'window_index': np.arange(G_df.shape[1])},
			index=start_indices)

		std_df = std_df.sort_values('std')
		reordering = std_df.window_index

		cutoff = np.quantile(std_df['std'], q=quantile)

		if plot:
			plt.figure(figsize=(6, 2))
			plt.subplot(1, 2, 1)
			plt.plot(std_df['std'], np.arange(G_df.shape[1]))
			plt.axvline(cutoff, c='red', alpha=0.2)
			plt.ylabel("Sorted window index")
			plt.xlabel("Window standard deviation, $\\sigma$")

			plt.subplot(1, 2, 2)
			plt.hist(std_df['std'], bins=20)
			plt.axvline(cutoff, c='red', alpha=0.2)
			plt.xlabel("Window standard deviation, $\\sigma$")
			plt.ylabel("Number of windows")

			plt.suptitle("Distribution of G standard deviation")
			plt.subplots_adjust(top=0.85, wspace=0.3)

		under_cutoff_windows = std_df[std_df['std'] < cutoff]
		sorted_window_number_cutoff = len(under_cutoff_windows)

		print(f"Quantile for cutoff: {quantile:.2f}")
		print(f"Cutoff for threshold: {cutoff:.2f}")
		print(f"Number of windows to include: ", sorted_window_number_cutoff)

		return std_df, under_cutoff_windows, quantile, cutoff


def plot_histogram_occupancies_G(config, G):
	fig, axs = plt.subplots(3, 6, figsize=(13, 6))

	tps = config.timepoints
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

def plot_heatmap(dat, column_names, full_column_names, ax=None, cmap=plt.cm.RdBu_r,
	vmin=0, vmax=2, plot_cbar=True, **kwargs):

	if ax is None:
		ax = plt.gca()

	cmap.set_bad('#c0c0c0')  # Set the color for NaN values
	def create_df_and_full_cols(dat, column_names, full_column_names):
		"""Insert back in the nan columns for plotting using reindex"""
		# Create the initial dataframe with existing data
		existing_df = pd.DataFrame(dat, columns=column_names)
		
		# Use reindex to add all missing columns at once
		# This automatically fills new columns with NaN values
		complete_dat = existing_df.reindex(columns=sorted(full_column_names))
		
		return complete_dat

	masked_dat = create_df_and_full_cols(dat, column_names, full_column_names)
	im = ax.imshow(masked_dat, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='none', aspect='auto',
		**kwargs)

	if plot_cbar: plt.colorbar(im)


def plot_heatmaps(N, F, H, B, G, column_names, full_column_names,
		figsize=(13, 11)):

	inv_B = np.linalg.inv(B)

	HF = H@F
	Ninv_G_B_inv = (np.linalg.inv(N)@G@inv_B)
	predicted_G = N@H@F@B
	residual_diff = G-(N@H@F@B)

	fig = plt.figure(figsize=figsize)

	plt.subplot(6, 1, 1)
	plot_heatmap(F, column_names, full_column_names)
	plt.xticks([])
	plt.title("$F$")
	plt.ylim(F.shape[0], 0)

	plt.subplot(6, 1, 2)
	plot_heatmap(HF, column_names, full_column_names)
	plt.xticks([])
	plt.title("$HF$")

	plt.subplot(6, 1, 3)
	plot_heatmap(Ninv_G_B_inv, column_names, full_column_names)
	plt.title("$(N^{-1})G(B^{-1})$")
	plt.xticks([])

	plt.subplot(6, 1, 4)
	plot_heatmap(predicted_G, column_names, full_column_names)
	plt.xticks([])
	plt.title("Predicted G: $NHFB$")

	plt.subplot(6, 1, 5)
	plot_heatmap(G, column_names, full_column_names)
	plt.xticks([])
	plt.title("$G$")

	plt.subplot(6, 1, 6)
	plot_heatmap(residual_diff, column_names, full_column_names, vmin=-1, vmax=1)
	plt.title("$G - NHFB$")
	plt.subplots_adjust(hspace=0.5, top=0.9)

	return fig


def compute_N(config, plot=False):
	from src.plot_helpers import color_for_key

	config.calculate_H()
	H = config.H
	tps = config.timepoints

	cg1_mass = H[:, config.get_Hpositions_for_phase('CG1')].sum(axis=1)
	dg1_mass = H[:, config.get_Hpositions_for_phase('DG1')].sum(axis=1)
	rg1_mass = H[:, config.get_Hpositions_for_phase('RG1')].sum(axis=1)
	s_mass = H[:, config.get_Hpositions_for_phase('S')].sum(axis=1)
	g2m_mass = H[:, config.get_Hpositions_for_phase('G2M')].sum(axis=1)
	h_mass = H[:, config.get_Hpositions_for_phase('H')].sum(axis=1)

	# Assume linear transition of S-phase
	s_indices = config.get_Hpositions_for_phase('S')
	s_masses = np.linspace(1, 2, len(s_indices))

	g1_mass = dg1_mass+cg1_mass+rg1_mass
	s_dna_content = H[:, config.get_Hpositions_for_phase('S')] @ np.diag(s_masses).sum(axis=1)
	replicating_mass = s_dna_content+g2m_mass*2

	average_DNA = h_mass+g1_mass+replicating_mass
	N = np.linalg.inv(np.diag(average_DNA))

	if plot:
		plt.figure(figsize=(6, 4))

		plt.fill_between(tps, h_mass, 0, label="H mass", color=color_for_key('H'))

		plt.fill_between(tps, g1_mass+h_mass, h_mass, label="G1 mass", color=color_for_key('CG1'))

		# ---------

		inv_n = g1_mass+s_dna_content+g2m_mass*2+h_mass

		plt.fill_between(tps, g1_mass+s_dna_content+h_mass, 
							  g1_mass+h_mass, label="S mass", color=color_for_key('S'))

		plt.fill_between(tps, inv_n, 
							 g1_mass+s_dna_content+h_mass, 
							 label="G2M mass", color=color_for_key('G2M'))

		plt.plot(tps, inv_n, c='red', lw=4, 
			label="Avg. DNA content")
		plt.ylim(0, 2.2)

		plt.legend(ncol=3, loc='upper right')

		plt.title("Estimation of average DNA content, CLOCCS")
		plt.xlabel("Clock time")
		plt.ylabel("Average DNA content")
		plt.xlim(0, tps[-1])

	return average_DNA, N


def read_g(chrom, deconv_span, replicate, verbose=True):

	if verbose:
		print_fl(f"Loading G data from combined replication run, 3/10/25")
		print_fl(f"Refactor to use the output directory, of the replication deconvolution run")

	directory = 'output/prototype_pipeline_subset/combined_replication'

	G_df = pd.read_csv(f'{directory}/rep{replicate}_chr{chrom}_G.csv')
	G_df = G_df[G_df.columns[1:]]	
	G_df.columns = G_df.columns.astype(int)
	
	start_indices = G_df.columns

	mid_span = (deconv_span[0]+deconv_span[1])/2
	bin_idx, start = get_bin_for_position(mid_span, start_indices)

	g = G_df[start]

	return g


def get_estimated_S_phase_end_index(config, output_directory, plot=False):
	# Get the estimated replication timings for chromosome 4 to estimate
	# a length of S for plotting

	pg1_indices = config.get_Hpositions_for_phase('postG1')
	Fr_df, repl_indices = load_replication_Fr_df(output_directory, 4)

	average_pg1_copy = Fr_df.mean(1)[pg1_indices]

	threshold = 1.9
	threshold_index = average_pg1_copy[average_pg1_copy > threshold].index[0]

	if plot:
		plt.figure(figsize=(4, 3))
		plt.plot(average_pg1_copy)
		plt.scatter(threshold_index, average_pg1_copy.loc[threshold_index], c='red',
				   label="90% replicated")
		plt.title("Chr4 average copy number in S/G2/M")
		plt.legend()
		plt.xlabel("PostG1 index")
		plt.ylabel("Average copy number")

	return threshold_index


def load_replication_Fr_df(output_dir, chrom, with_replication_timing=False):

	from src.config import load_default_chrom_configs, retrieve_replication_timing

	config1, config2 = load_default_chrom_configs()

	combined_directory = f'{output_dir}/combined_replication'
	Fr_df = pd.read_csv(f'{combined_directory}/combined_chr{chrom}_F.csv')
	Fr_df = Fr_df[Fr_df.columns[1:]]
	Fr_df.columns = Fr_df.columns.astype(int)

	# Use the top branch indices to identify the replication timing
	# in case the timing is within G1
	# To take the average of the top and bottom branch's timing (in case
	# replication 'occurs' in late G1)
	replication_indices_b = Fr_df.loc[config1.b_indices()].idxmax(0)
	replication_indices_t = Fr_df.loc[config1.t_indices()].idxmax(0)

	if with_replication_timing:

		replication_timings_t = retrieve_replication_timing(config1, config2, 
			replication_indices_t.values)
		replication_timings_b = retrieve_replication_timing(config1, config2, 
			replication_indices_b.values)
		replication_timings = (replication_timings_t.values + replication_timings_b.values)/2

		repl_df = pd.DataFrame({
			'replication_index_t': replication_indices_t,
			'replication_index_b': replication_indices_b,
			'replication_time': replication_timings
		}, index=replication_indices_t.index)
		repl_df.index.name = 'start'

		return Fr_df, repl_df

	# Use the top branch replication indices as default, we should work with the
	# timing though, which handles the top and bottom branch replication timing together
	return Fr_df, replication_indices_t

def load_B_df(output_dir, chrom, starts=None):
	B = np.load(f'{output_dir}/combined_replication/combined_chr{chrom}_B.npy')

	if starts is None:
		starts = np.arange(B.shape[0])

	b_df = pd.DataFrame(np.diag(B), columns=['b'], index=starts)
	return B, b_df['b']


def read_n_fr_b(chrom, deconv_span, replicate,
	parent_directory, log=True):
	"""Load the N, replication timing, and b from disk"""

	single_directory = f'{parent_directory}/single_replication'
	combined_directory = f'{parent_directory}/combined_replication'
	combined_N = np.load(f'{combined_directory}/N.npy')

	Fr_df, _ = load_replication_Fr_df(parent_directory, chrom)
	start_indices = Fr_df.columns

	B, B_df = load_B_df(parent_directory, chrom, start_indices)

	mid_span = (deconv_span[0]+deconv_span[1])/2
	bin_idx, start = get_bin_for_position(mid_span, start_indices)

	fr = Fr_df[start].values
	b = B_df.loc[start]

	return start, combined_N, fr, b


def read_no_copy_correction_n_fr_b(H):

	n, m = H.shape
	N = np.eye(n)
	b = 1
	fr = np.ones(m)

	return N, fr, b

def compute_rn(N, H, F, B, G):
	NHFB = N @ H @ F @ B
	loss = np.mean((NHFB - G)**2)
	return loss


def update_N_B(N: np.ndarray, H: np.ndarray, G: np.ndarray,
			   B: np.ndarray, F: np.ndarray):
	"""Update N and B matrices based on current F solution.
		
	Returns:
		Tuple of (updated_N, updated_B)
	"""
	# Update N based on G, B, H, and F
	GBinv_HF_div = np.divide((G @ np.linalg.inv(B)), (H @ F))
	updated_N = np.diag(GBinv_HF_div.mean(axis=1))

	# Update B based on N H F and G
	num_rows = G.shape[0]
	G_sums = G.T @ np.ones((num_rows, 1))
	NHF_sums = (updated_N @ H @ F).T @ np.ones((num_rows, 1))
	updated_b_diag = (G_sums / NHF_sums).flatten()
	updated_B = np.diag(updated_b_diag)

	return updated_N, updated_B


class DeconvolutionResult(NamedTuple):
	"""Container for all results from the iterative deconvolution process"""
	Ns: np.ndarray  # History of N values for each iteration 
	Bs: np.ndarray  # History of B values for each iteration
	Fs: np.ndarray  # History of F values for each iteration
	iterative_update_rns: np.ndarray  # Residual norms for each iteration


def iterative_deconvolution_updates(
	config: dict,
	H: np.ndarray,
	G: np.ndarray, 
	initial_N: np.ndarray,
	initial_B: np.ndarray,
	total_iterations: int,
	timer=None,
	verbose: bool = False
) -> DeconvolutionResult:
	"""Iteratively deconvolve for the replication curve F and update N and B.
		DeconvolutionResult containing iteration history and final values
	"""

	if timer is None:
		timer = Timer()

	# Initialize arrays to store iteration history
	Ns = np.zeros((total_iterations, *initial_N.shape))
	Bs = np.zeros((total_iterations, *initial_B.shape))
	Ns[0] = initial_N
	Bs[0] = initial_B
	
	# Initialize arrays for F solutions and residual norms
	n, m = H.shape
	num_sites = initial_B.shape[0]
	Fs = np.zeros((total_iterations, m, num_sites))
	iterative_update_rns = np.zeros(total_iterations)

	# Current working values
	current_N = initial_N
	current_B = initial_B
	
	for iteration in range(total_iterations):

		# Perform deconvolution step
		F, rn = deconvolve_replication_brute_force(
			config, H, G, current_N, current_B,
			timer=timer, verbose=False
		)		

		# Store the solutions
		Fs[iteration] = F
		iterative_update_rns[iteration] = rn
		
		# Update N and B for next iteration
		if iteration < total_iterations - 1:
			current_N, current_B = update_N_B(current_N, H, G, current_B, F)
			Ns[iteration + 1] = current_N
			Bs[iteration + 1] = current_B
			
		if verbose:
			print_fl(f"[{iteration}] {timer.get_time()}, rn={rn}")

	return DeconvolutionResult(
		Ns=Ns,
		Bs=Bs, 
		Fs=Fs,
		iterative_update_rns=iterative_update_rns,
	)

def plot_average_replication_time(real_deconv1):

	config = real_deconv1.config

	fig = plt.figure(figsize=(4, 3))
	t_indices = config.get_Hpositions_for_branch('t')
	t_tps = config.get_timepoints_for_branch('t')
	plt.plot(t_tps, real_deconv1.F.mean(axis=1)[t_indices])
	draw_phase_label_annotations(plt.gca(), config, flip=True, annotations_x=0.85)
	plt.xlim(t_tps[0], t_tps[-1])
	plt.ylim(0.8, 2.1)
	plt.suptitle("Average single cell copy number, replicate 1, chrIV")

	# Plot when 95% of the genome has been replicated
	x, y = t_tps, real_deconv1.F.mean(axis=1)[t_indices]
	g2m_start = t_tps[y > 1.95][0]
	plt.axvline(g2m_start)
	config.params_dic['lambda'] - g2m_start


def plot_masks():
	"""todo: Plot the mask for low coverage and regions with high occupancy during alpha-factor
	these regions appear to be difficult to converge with. 
	
	Show that the occupancy at these regions to justify the masking.
	"""
	plt.figure(figsize=(13, 4))
	plt.subplot(2, 1, 1)

	masked_alpha_G_df = masked_G_df[masked_G_df.columns[mask_alpha]]

	plt.imshow(masked_alpha_G_df,
			  vmin=0, vmax=2, cmap='RdBu_r', aspect='auto')
	plt.title("Masked Regions")
	plt.xticks([])
	plt.subplot(2, 1, 2)
	plt.imshow(masked_G_df[masked_G_df.columns[~mask_alpha]],
			  vmin=0, vmax=2, cmap='RdBu_r', aspect='auto')
	plt.title("Unmasked Regions")
	plt.xticks([])
	plt.subplots_adjust(hspace=0.45)

	num_masked = mask_alpha.sum()
	num_total = masked_G_df.shape[1]

	print(f"There are {num_masked}/{num_total} ({num_masked/num_total*100:.1f}%)"
		  f"windows with >{proportion_above*100}% occupancy "
		  f"during alpha factor release compared to any other timepoint")


def load_N_F_B_for_replication_deconv_from_save(output_dir, replicate, chrom):
	"""Load the F, N, and B from disk from a previous run."""

	F = pd.read_csv(f'{output_dir}/rep{replicate}_chr{chrom}_F.csv')
	F_values = F[F.columns[1:]].values
	F_values.shape

	N = np.load(f'{output_dir}/rep{replicate}_chr{chrom}_N.npy')
	B = pd.read_csv(f'{output_dir}/rep{replicate}_chr{chrom}_B.csv')
	B = np.diag(B[B.columns[1:]].values[0])
	
	return F_values, N, B


def modify_config_from_run(config, output_directory, replicate, chrom):
	"""To warm start, modify a config using the output directory parameters from
	a previously run instance, the last item in the parameters csv will be used to update
	the config parameters"""

	# Load the replicate 1 and 2 parameters and deconvolve combined
	parameters_df = pd.read_csv(f'{output_directory}/parameter_updates_rep{replicate}_chr{chrom}.csv')
	parameters = parameters_df.iloc[-1]

	mu0, gamma2, sigma0 = parameters.mu0, parameters.gamma2, parameters.sigma0

	# todo: load all parameters in the parameters csv, when we deconvolve all parameters
	config.params_dic['mu0'] = mu0
	config.params_dic['gamma2'] = gamma2
	config.params_dic['sigma0'] = sigma0
	config.update_timepoints()
	config.calculate_H()

	return config
