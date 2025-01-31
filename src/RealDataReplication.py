 
import numpy as np
import cvxpy as cp
import pandas as pd

from src.timer import Timer
from src.utils import print_fl
from matplotlib import pyplot as plt
from src.replication_deconvolution_solver import deconvolve_replication_brute_force
from typing import Tuple, Optional, NamedTuple

early_color = plt.get_cmap('Oranges')(0.75)
late_color = plt.get_cmap('Purples')(0.75)


class RealDataReplicationDeconvolution():
	"""This model deconvolve the replication timing.
	"""
	def __init__(self, config, chr, replicate):

		self.chrom = chr
		self.replicate = replicate
		self.config = config
		self.load_replicate_data(chr, replicate)


	def load_replicate_data(self, chr, replicate):

		from src.mnase_10kb_loader import MNase10kbLoader
		from src.chromatin_metrics import fragment_lengths_definitions
		small_span, med_span, nuc_span = fragment_lengths_definitions()

		mnase_loader = MNase10kbLoader()
		mnase_loader.load_mnase_data(replicate=replicate, chromosome=chr,
			fragment_lengths_span=nuc_span)
		mnase_loader.compute_sliding_window_counts_all_times()

		self.mnase_loader = mnase_loader
		self.unnormalized_total_occupancy = mnase_loader.all_counts_unnormalized_df
		self.normalized_occupancy = mnase_loader.normalized_total_occupancy_df
		self.G_df = self.normalized_occupancy.T
	
		# Setup regions to threshold, 
		# regions with low occupancy will be omitted when needed
		print_fl("Threshold windows with less than 75% read coverage.")
		self.selected_threshold_region = self.normalized_occupancy.T.mean(axis=0) > 0.75

	def setup_deconvolution(self, config=None, initial_N=None, initial_B=None):
		"""Setup the deconvolution:
		1. H from the config parameters
		2. G from the normalized data, masked out for low coverage regions
		3. N from the cell cycle parameters as defined in the config
		4. B from the first timepoint in G
		"""

		if config is None and self.config is None:
			raise ValueError("Config has not been initialized")

		elif config is not None:
			self.config = config

		print_fl("Initializing N using config H.")
		self.config.calculate_H()
		self.H = self.config.H

		# Mask out the low coverage regions
		keep_column_indices = self.selected_threshold_region[self.selected_threshold_region].index
		self.masked_G_df = self.G_df[keep_column_indices]

		self.masked_start_indices = keep_column_indices
		self.G = self.masked_G_df.values

		if initial_N is None:
			self.average_DNA, self.initial_N = compute_N(self.config)
		else:
			self.initial_N = initial_N

		print_fl("Initializing B using timepoint 0")
		if initial_B is None:
			self.initial_B = np.diag(self.G[0])
		else:
			self.initial_B = initial_B


	def deconvolve(self):
		self.setup_deconvolution(self.config)
		self.F, self.rn = deconvolve_replication_brute_force(self.config, 
			self.H, self.G, self.N, self.B)


	def iterative_deconvolution_updates(self, total_iterations, timer=None,
		initial_N=None, initial_B=None, verbose=True):
		"""Iteratively deconvolve for the replication curve F."""

		result = iterative_deconvolution_updates(
		    config=self.config, H=self.H, G=self.G, initial_N=self.initial_N, 
		    initial_B=self.initial_B, total_iterations=total_iterations, 
		    timer=timer, verbose=verbose)

        self.N = result.Ns[-1]
        self.F = result.Fs[-1]
        self.rn = result.iterative_update_rns[-1]
        self.B = result.Bs[-1]

		return result

	def compute_rn(self):
		N, H, F, B = self.N, self.H, self.F, self.B
		G = self.G
		return compute_rn(N, H, F, B, G)


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


	def plot_heatmaps(self):
		N, F, G, B, H = self.N, self.F, self.G, self.B, self.H

		fig = plot_heatmaps(N, F, H, B, G, column_names=self.masked_G_df.columns,
			full_column_names=self.G_df.columns)
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
		extent = [0, start_indices[-1], 0, tps[-1]]

		plt.figure(figsize=(13, 3))
		plt.imshow(self.G, cmap='RdBu_r', vmin=0, vmax=2, 
			interpolation='none', aspect='auto',
			extent=extent)
		plt.colorbar()
		plt.title(f"Experiment 10 kb MNase-seq reads, replicate 1, chr{self.chrom}")
		plt.xlabel("Genomic position, bp")
		plt.ylabel("Experimental time")

		plt.subplots_adjust(bottom=0.2)


	def plot_predicted_G(self):

		predicted_G = self.N@self.H@self.F@self.B

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

		predicted_G = self.N@self.H@self.F@self.B
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
		from src.chromatin_model import draw_phase_label_annotations

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

		from src.chromatin_model import draw_phase_label_annotations

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

		from src.chromatin_model import draw_phase_label_annotations

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
		self.H = self.config.H

# def threshold_selection(dat, threshold_selection, fill=1.,
# 	renormalize=False):
# 	new_dat = dat.copy()
# 	new_dat[:, ~threshold_selection] = fill

# 	# If thresholded to fill with nans, we can renormalize such
# 	# that the non-thresholded out regions mean to 1
# 	if renormalize:

# 		# Get the current mean of the good rows
# 		row_means = np.nanmean(new_dat, axis=1)

# 		# Divide the data by these mean
# 		new_dat = new_dat / row_means.reshape((-1, 1))

# 	return new_dat


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

	# This shoes the data for G is normal-ish and centered around 1.

def plot_heatmaps(N, F, H, B, G, column_names, full_column_names):

	RdBu_cmap = plt.cm.RdBu_r
	RdBu_cmap.set_bad('#aaaaaa')  # Set the color for NaN values

	inv_B = np.linalg.inv(B)

	HF = H@F
	Ninv_G_B_inv = (np.linalg.inv(N)@G@inv_B)
	predicted_G = N@H@F@B
	residual_diff = (N@H@F@B) - (G)

	def create_df_and_full_cols(dat, column_names, full_column_names):
		"""Insert back in the nan columns for plotting"""
		complete_dat = pd.DataFrame(dat, columns=column_names)
		for c in full_column_names:
			if c not in column_names:
				complete_dat[c] = np.nan
		complete_dat = complete_dat[sorted(complete_dat.columns)]
		return complete_dat

	F = create_df_and_full_cols(F, column_names, full_column_names)
	G = create_df_and_full_cols(G, column_names, full_column_names)
	HF = create_df_and_full_cols(HF, column_names, full_column_names)
	Ninv_G_B_inv = create_df_and_full_cols(Ninv_G_B_inv, column_names, full_column_names)
	predicted_G = create_df_and_full_cols(predicted_G, column_names, full_column_names)
	residual_diff = create_df_and_full_cols(residual_diff, column_names, full_column_names)

	fig = plt.figure(figsize=(13, 11))

	plt.subplot(6, 1, 1)
	plt.imshow(F, cmap=RdBu_cmap, vmin=0, vmax=2, interpolation='none', aspect='auto')
	plt.xticks([])
	plt.colorbar()
	plt.title("$F$")

	plt.subplot(6, 1, 2)
	plt.imshow(HF, cmap=RdBu_cmap, vmin=0, vmax=2, interpolation='none', aspect='auto')
	plt.colorbar()
	plt.xticks([])
	plt.title("$HF$")

	plt.subplot(6, 1, 3)
	plt.imshow(Ninv_G_B_inv, cmap=RdBu_cmap, vmin=0, vmax=2, interpolation='none', aspect='auto')
	plt.colorbar()
	plt.title("$(N^{-1})G(B^{-1})$")

	plt.subplot(6, 1, 4)
	plt.imshow(predicted_G, cmap=RdBu_cmap, vmin=0, vmax=2, interpolation='none', aspect='auto')
	plt.xticks([])
	plt.colorbar()
	plt.title("Predicted G: $NHFB$")

	plt.subplot(6, 1, 5)
	plt.imshow(G, cmap=RdBu_cmap, vmin=0, vmax=2, interpolation='none', aspect='auto')
	plt.colorbar()
	plt.xticks([])
	plt.title("$G$")

	plt.subplot(6, 1, 6)
	plt.imshow(residual_diff, vmin=-1, vmax=1, cmap=RdBu_cmap, interpolation='none', aspect='auto')
	plt.colorbar()
	plt.title("$NHFB - G$")
	plt.subplots_adjust(hspace=0.5, top=0.9)

	return fig


def compute_N(config, plot=False):
	from src.model import color_for_key

	config.calculate_H()
	H = config.H
	tps = config.timepoints

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
    verbose: bool = True
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
        if verbose:
            print_fl(f"Iteration {iteration}")
            
        # Perform deconvolution step
        F, rn = deconvolve_replication_brute_force(
            config, H, G, current_N, current_B,
            timer=timer, verbose=verbose
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
            print_fl(f"Iteration completed {timer.get_time()}, rn={rn}")

    return DeconvolutionResult(
        Ns=Ns,
        Bs=Bs, 
        Fs=Fs,
        iterative_update_rns=iterative_update_rns,
    )
