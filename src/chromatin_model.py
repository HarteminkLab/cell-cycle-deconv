
import cvxpy
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from src.sgd import get_gene_name_orf_name
from src.mnase_plotting import plot_mnase_density
from src.figure_configs import FiguresConfig

from src.timer import Timer
from src.utils import print_fl
from src.global_config import GlobalConstants
from src.geneset import get_deconvolved_geneset


class ChromatinModel:
	"""
	In this class, we will be taking mnase-seq reads for a gene, and computing a grid of occupancy 
	values for the gene's locus.

	That we can then deconvolve.

	We will be able to plot the gene's locus (raw data) as well as the histogram of the grid for 
	verification that the grid values
	are created properly.

	Notes: We will eventually need to normalize or scale (by copy number and/or by sample depth)
	"""


	def __init__(self, config):

		self.impute_50_rep2 = False

		# Padding defines the window around the TSS to retrieve MNase data
		self.padding = 5000
		self.geneset = get_deconvolved_geneset()
		self.config = config
		self.gamma = 0.006 # default gamma value

		self.bin_width = GlobalConstants.BIN_WIDTH
		self.bin_height = GlobalConstants.BIN_HEIGHT
		self.prom_len = GlobalConstants.PROM_LEN
		self.gb_len = GlobalConstants.GB_LEN
		self.chr = None
		self.chr_reads = None

		self.max_y_len = GlobalConstants.MAX_Y_LEN
		self.timepoints = self.config.timepoints

		# Normalize such that the mean G values center to 1
		# note: works best with large windows of G
		self.normalize_mean_1 = True

		# todo: default mnase output directory
		# is stored directly in the output
		self.mnase_output_directory = 'output/mnase'

	def load_mnase_span(self, chrom, mnase_span, verbose=True, impute_50_rep2=False,
		mnase_output_directory=None):
		"""Load the MNase for an arbitrary genomic span"""

		replicate = self.config.replicate
		self.impute_50_rep2 = impute_50_rep2

		# convert to integers
		self.mnase_span = int(mnase_span[0]), int(mnase_span[1])

		if mnase_output_directory is None:
			mnase_output_directory = self.mnase_output_directory

		if not self.chr == chrom:

			del self.chr_reads

			if verbose:
				print_fl(f"Loading chromosome reads: {chrom}")

			self.chr_reads = read_chromosome_mnase_reads(mnase_output_directory, 
				replicate, chrom)
			self.chr = chrom

		else:
			if verbose:
				print_fl(f"Already loaded chromosome reads for {self.chr}. Using cache.")

		self.locus_reads = self.chr_reads[(self.chr_reads.mid > self.mnase_span[0]) & 
			(self.chr_reads.mid < self.mnase_span[1])]

		self.timepoints = GlobalConstants.CHROM_WT1_TIMEPOINTS if replicate == 1 else GlobalConstants.CHROM_WT2_TIMEPOINTS
		self.config.WT1_TIMEPOINTS = self.timepoints

		new_span = self.mnase_span

		# Create the bins for the reads
		exact_bins = create_exact_bins(self.locus_reads, new_span, self.timepoints)

		if self.impute_50_rep2 and self.config.replicate == 2:
			print("Imputing timepoint 50 as average of 40 and 60")
			exact_bins[5] = (exact_bins[4]+exact_bins[6])/2.

		# Load target length distribution
		from src.mnase_normalization import load_target_distribution
		target_length_distribution = load_target_distribution(verbose=verbose)

		# Load target total sums g from replication profile
		from src.RealDataReplication import read_g
		window_10kb_g_curve = read_g(chrom, mnase_span, replicate, verbose)

		# Preserve the length distribution and 10kb total curve
		self.window_10kb_g_curve = window_10kb_g_curve
		self.target_length_distribution = target_length_distribution

		# Normalize to mean 1, to target length distribution, 
		# to expected sums as defined from the 10kb windows from replication deconvolution
		# then downsample
		if verbose:
			print("Normalizing to mean 1, to target length distribution, to target 10kb occupancy sums.")

		# Perform the normalization and downsampling steps
		from src.mnase_normalization import normalize_and_downsample
		exact_bins_normalized, length_normalized, length_normalized_target_sums, downsampled_bins = \
			normalize_and_downsample(exact_bins, target_length_distribution, window_10kb_g_curve)

		if verbose:
			print(f"And downsampling from {exact_bins_normalized.shape} to {downsampled_bins.shape}")

		self.length_normalized = length_normalized
		self.length_normalized_target_sums = length_normalized_target_sums
		self.downsampled_bins = downsampled_bins

		from src.helpers import downsample_bins

		self.new_span = new_span

		exact_extent = [self.mnase_span[0], self.mnase_span[1],
					0, GlobalConstants.MAX_Y_LEN]
		
		self.exact_bins_unnormalized = exact_bins
		self.exact_bins = exact_bins_normalized
		self.exact_extent = exact_extent
		self.bin_extents = exact_extent

		self.G_imgs = downsampled_bins
		self.image_shape = self.G_imgs.shape[1:]
		self.original_shape = self.G_imgs.shape
		self.G = downsampled_bins.reshape(downsampled_bins.shape[0], -1)

		if self.normalize_mean_1 == True:
			eps = 1e-5
			self.non_mean_centered_G = self.G
			self.G = self.G / (eps+self.G.mean())

	
	def compute_bin_counts_sample(self, sample, x_bins, y_bins):
		return compute_bin_counts_sample(self.locus_reads, sample, x_bins, y_bins)


	def deconvolved_f(self):

		# TODO, there are multiple places where f value can be set and used
		# fix this setter and getter to be clear
		if self.deconvolved_f_value is None:
			self.deconvolved_f_value = self.solver.f.value

		return self.deconvolved_f_value


	def find_max_plus_one_location(self):
		"""
		Find the position of the +1 by finding the max number of nucleosome reads in
		a 200bp window around the TSS.

		For the currently selected gene
		"""

		# From the MNase-seq it appears the TSS may need to move a bit to match
		# the expected +1 nucloeosome location
		hardcoded_TSS = {
		}

		if self.center_on_TSS:
			gene_center = self.gene.TSS
		else:
			gene_center = self.gene.PAS

		from src.global_config import fragment_lengths_definitions
		small_lens, med_lens, nuc_lens = fragment_lengths_definitions()

		# Next, we will align at the +1
		# from the TSS/PAS, stack up all timepoints, then look up and dowstream (200 bp window) for the
		# position with the highest number of reads, and we will use that position as our +1 position
		# We will put that position into our gene data set and use that as our reference data set

		# We can get all of the  nucleosome length fragments for the gene, and stack them up by time

		cur_reads = self.locus_reads.copy()

		# Search around the TSS with a 200bp window
		window = 200
		search_peak_span = gene_center-window//2, \
			gene_center+window//2 

		# Larger length span for nucleosome reads search
		nuc_lens = 120, 200

		cur_nuc_reads = cur_reads[(cur_reads['length'] >= nuc_lens[0]) & 
								  (cur_reads['length'] < nuc_lens[1]) & 
								  (cur_reads['mid'] >= search_peak_span[0]) &
								  (cur_reads['mid'] < search_peak_span[1])]

		counts_per_pos_search = cur_nuc_reads.groupby('mid').count()
		counts_per_pos_search = counts_per_pos_search[['start']].rename({'start': 'count'})

		pos_max = counts_per_pos_search.idxmax().start
		self.computed_plus_one = pos_max

		return pos_max

	def setup_deconv_model(self):
		"""Set up the deconvolution model from the config, the model class was originally for gene expression
		but has built-in functions that will be useful for chromatin deconvolution

		Currently a lot of deconvolution parameters and configuration
		are built into the Model class used for gene expression.

		TODO: We will want to generalize those so that we can subclass or have a parent
		class that the chromatin and gene expression can use separately
		for now we will just instantiate this model class for use for the chromatin 
		deconvolution.

		We will try to not use the deconv_model object externally, such that it will be easier to 
		refactor in the future
		"""

		self.deconv_model = Model(self.config, None, self.gamma, for_chromatin_deconv=True)

		# The config for MNase and RNA-seq have a different number of timepoints, so 
		# we need to recalculate H with the chromatin number of timepoints
		calcH_function = self.config.calcH_function

		self.deconv_model.H, self.deconv_model.Hpos = calcH_function(self.config.intervals_wt1, self.timepoints)

	def setup_solver(self, wavelet="Symmlet"):
		from src.chromatin_deconvolution_solver import ChromatinDeconvolveSolver
		from src.replication_deconvolution_solver import ReplicationChromatinDeconvolveSolver

		self.solver = ChromatinDeconvolveSolver(self.deconv_model.config, self.deconv_model.H, self.G, 
			wavelet=wavelet)

		self.found_optimal_success = None
		self.deconvolved_f_value = None


	def deconvolve(self, gamma, kappa, verbose=True):
		"""
		Deconvolve the chromatin for a single gamma value
		"""
		F = self.solver.deconvolve_G_iteratively(gamma=gamma, kappa=kappa, verbose=verbose)
		self.F = F


	def compute_ptr(self, quantiles=[0.2, 0.8]):

		# -------- Compute the PTR ---------

		f = self.deconvolved_f()
		from src.peak_to_trough import compute_ptr_f

		f_ptrs = compute_ptr_f(self.config, f)
		self.f_ptrs = f_ptrs

		# ------- Compute ranks (highest ptr bin ranks) ------------

		m = len(self.f_ptrs)
		
		# Get the argsort of the array, argsort again
		# will get the ranks of the values in the original data
		# inverse with m-1 to get reverse the order of the list
		f_argsort = np.argsort(self.f_ptrs)
		f_rank = np.argsort(f_argsort)
		self.f_ranks = m-1 - f_rank


	def create_exact_bins(self):
		xbins = np.arange(*self.mnase_span)
		ybins = np.arange(0, 252)

		n = len(self.timepoints)

		# Bin histogram is one less than the bin definitions because the bins include the outer edges
		# of the bins
		exact_bins = np.zeros((n, len(ybins)-1, len(xbins)-1))

		for time_idx in range(n):

			sample = self.timepoints[time_idx]

			plotting_reads, hist, \
				x_edges, y_edges = self.compute_bin_counts_sample(sample, xbins, ybins)
			hist = hist.T
			exact_bins[time_idx] = hist
		return exact_bins


	def plot_bin_comparison(self):
		cols = 3
		rows = len(self.timepoints)

		plt.figure(figsize=(6, 12))

		for row in range(rows):

			plt.subplot(rows, cols, row*cols+1)
			plt.imshow(self.exact_bins[row], cmap='magma_r', vmax=0.25, origin='lower', aspect='auto',
					  extent=self.exact_extent)
			plt.xlim(self.bin_extents[0], self.bin_extents[1])
			plt.yticks([])
			plt.xticks([])
			plt.axvline(self.computed_plus_one, c='gray', lw=1)

			plt.subplot(rows, cols, row*cols+2)
			plt.imshow(self.G_imgs[row], cmap='magma_r', vmax=20, origin='lower', aspect='auto',
					  extent=self.bin_extents)
			
			plt.xlim(self.bin_extents[0], self.bin_extents[1])
			plt.yticks([])
			plt.xticks([])
			plt.axvline(self.computed_plus_one, c='gray', lw=1)

			plt.subplot(rows, cols, row*cols+3)
			plt.imshow(self.smooth_bins[row], cmap='magma_r', vmax=20, origin='lower', aspect='auto',
					  extent=self.bin_extents)
			plt.axvline(self.computed_plus_one, c='gray', lw=1)
			plt.yticks([])
			plt.xticks([])


	def get_f_images(self):
		f = self.deconvolved_f()
		f_imgs = f.reshape((f.shape[0], *self.image_shape))
		return f_imgs


	def save_deconvolved_outputs(self, out_dir, index, using_default_flag):

		orf_name = self.gene.name
		gene_name = self.gene['gene']

		f = self.deconvolved_f()
		g_save_path = f'{out_dir}/{index}_g_{orf_name}_{gene_name}.npy'
		f_save_path = f'{out_dir}/{index}_f_{orf_name}_{gene_name}.npy'
		ptr_save_path = f'{out_dir}/{index}_ptr_{orf_name}_{gene_name}.npy'
		meta_save_path = f'{out_dir}/{index}_meta_{orf_name}_{gene_name}.csv'

		# ------- Reshape f ---------

		shape = self.G_imgs[0].shape
		reshaped_f = f.reshape((-1, shape[0], shape[1]))
		reshaped_ptrs = self.f_ptrs.reshape(*shape)

		#---------- Save to disk -------------

		# Save the g to disk
		np.save(g_save_path, self.G_imgs)

		# Save the f to disk
		np.save(f_save_path, reshaped_f)

		# Save the ptr to disk
		np.save(ptr_save_path, reshaped_ptrs)

		# Save meta information
		from datetime import datetime
		run_date = datetime.now().strftime("%D")

		df = pd.DataFrame({
			'rn': self.rn, 'sn': self.sn, 'gm': self.gamma,
			'config': self.config.name,
			'model_path': self.config.model_wt1_file,
			'run_date': run_date,
			'replicate': self.config.replicate
			},
			index=[orf_name])
		df.to_csv(meta_save_path, float_format="%.4f")

		print_fl(f"Saved to {g_save_path}...")
		print_fl(f"Saved to {f_save_path}...")
		print_fl(f"Saved to {ptr_save_path}...")
		print_fl(f"Saved to {meta_save_path}...")

	def plot_normalization_sanity_check(self, axs=None):
		from src.mnase_normalization import plot_normalization_sanity, load_target_distribution

		plot_normalization_sanity(self.exact_bins, 
			self.length_normalized, self.length_normalized_target_sums, 
			self.downsampled_bins, self.target_length_distribution,
			self.window_10kb_g_curve, axs=axs)

	def plot_raw_data(self, ax=None, figsize=(2, 11), vmax=40, timepoint=None):

		if timepoint is None:
			return plot_raw(self, figsize, vmax=vmax)
		else:
			G = self.G
			G_imgs = G.reshape((G.shape[0], 26, -1))
			timepoints = self.config.timepoints
			t_index = timepoints.index(timepoint)
			G_img = G_imgs[t_index]

			if ax is None:
				fig = plt.figure(figsize=figsize)
				ax = plt.gca()

			extent = [self.mnase_span[0], self.mnase_span[1], 0, 260]
			plot_G_img(ax, G_img, vmax=vmax, extent=extent, vmin=0, cmap='magma_r')

def read_chromosome_mnase_reads(output_directory, replicate, chr):
	chr_reads = pd.read_hdf(f'{output_directory}/yl_rep{replicate}_mnase_reads/yl_rep{replicate}_mnase_reads_chr{chr}.h5', 
							 'mnase_data')

	return chr_reads


def create_exact_bins(locus_reads, mnase_span, timepoints):
	xbins = np.arange(*mnase_span)
	ybins = np.arange(0, 252)

	n = len(timepoints)

	# Bin histogram is one less than the bin definitions because the bins include the outer edges
	# of the bins
	exact_bins = np.zeros((n, len(ybins)-1, len(xbins)-1))

	for time_idx in range(n):

		sample = timepoints[time_idx]

		plotting_reads, hist, \
			x_edges, y_edges = compute_bin_counts_sample(locus_reads, sample, xbins, ybins)
		hist = hist.T
		exact_bins[time_idx] = hist
	return exact_bins


def compute_bin_counts_sample(locus_reads, sample, x_bins, y_bins):

	plotting_reads = locus_reads[locus_reads['sample'] == sample]
	hist, x_edges, y_edges = np.histogram2d(plotting_reads['mid'], 
		plotting_reads['length'], bins=[x_bins, y_bins])

	return plotting_reads, hist, x_edges, y_edges


def plot_raw(chromatin_model, figsize=(2, 7), vmax=40):
	config = chromatin_model.config
	G = chromatin_model.G
	chrom, mnase_span = chromatin_model.chr, chromatin_model.mnase_span
	return plot_raw_G(G, config, chrom, mnase_span, figsize=figsize,
		title=f"Raw data, replicate {config.replicate}", vmax=vmax)


def plot_G_img(ax, img, cmap, extent, vmax, vmin=0):
	ax.imshow(img, origin='lower', aspect='auto', vmin=vmin, vmax=vmax,
			  cmap=cmap, extent=extent)
	ax.set_xticks([])
	ax.set_yticks([])


def plot_raw_G(G, config, chrom, mnase_span, figsize=(2, 7),
	vmin=0, vmax=40, cmap='magma_r', title=""):

	G_imgs = G.reshape((G.shape[0], 26, -1))
	timepoints = config.timepoints

	num_rows = len(timepoints)+1
	num_cols = 1

	fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize)

	from src.orf_plotter import load_default_orf_plotter
	from src.sgd import read_nondubious_genes_dataset


	orf_plotter = load_default_orf_plotter()
	orf_plotter.set_chrom_span(chrom, mnase_span)

	orf_plotter.plot_orf_annotations(axs[0])

	extent = [mnase_span[0], mnase_span[1], 0, 260]

	for i in range(1, num_rows):

		ax = axs[i]
		plot_G_img(ax, G_imgs[i-1], cmap=cmap, vmax=vmax, extent=extent, vmin=vmin)
		ax.set_ylabel(f"{timepoints[i-1]}'")

	plt.suptitle(title, fontweight='demi', fontsize=18)
	plt.subplots_adjust(top=0.957)

	return fig

def plot_prediction(chromatin_model, G, N, F, F_replicate, b):

	config = chromatin_model.config
	predicted_G = N@config.H@(np.multiply(F, F_replicate[:, None])*b)
	predicted_G_imgs = predicted_G.reshape((predicted_G.shape[0], 26, -1))
	G_imgs = G.reshape((G.shape[0], 26, -1))
	timepoints = config.timepoints

	num_rows = len(timepoints)+1
	num_cols = 3

	fig, axs_rows = plt.subplots(num_rows, num_cols, figsize=(13, 13))

	from src.orf_plotter import load_default_orf_plotter
	from src.sgd import read_nondubious_genes_dataset

	chrom, mnase_span = chromatin_model.chr, chromatin_model.mnase_span

	orf_plotter = load_default_orf_plotter()
	orf_plotter.set_chrom_span(chrom, mnase_span)

	for ax in axs_rows[0]:
		orf_plotter.plot_orf_annotations(ax)

	axs_rows[0][0].set_title("Raw")
	axs_rows[0][1].set_title("Prediction")

	for i in range(1, num_rows):

		ax_row = axs_rows[i]
		ax = ax_row[0]
		ax.imshow(G_imgs[i-1], origin='lower', aspect='auto', vmin=0, vmax=50,
				  cmap='magma_r')
		ax.set_xticks([])
		ax.set_yticks([])

		ax = ax_row[1]
		ax.imshow(predicted_G_imgs[i-1], origin='lower', aspect='auto', vmin=0, vmax=50,
				  cmap='magma_r')
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_ylabel(f"{timepoints[i-1]}'")


		ax = ax_row[2]
		ax.imshow(G_imgs[i-1]-predicted_G_imgs[i-1], origin='lower', aspect='auto',
				  cmap='RdBu_r', vmin=-50, vmax=50)
		ax.set_xticks([])
		ax.set_yticks([])

	plt.suptitle("Predicted vs Raw data bins")
	plt.subplots_adjust(top=0.95)

	return fig


def plot_img(ax, img, vmax=None, extent=None, zorder=1):
	ax.imshow(img, origin='lower', aspect='auto', vmin=0, vmax=vmax,
					  cmap='magma_r', extent=extent, zorder=zorder)

