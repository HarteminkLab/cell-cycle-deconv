
import cvxpy
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from src.sgd import get_gene_name_orf_name
from src.mnase_plotting import plot_mnase_density
from src.origins import load_origins_w_replication
from src.figure_configs import FiguresConfig

from src.model import Model
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

		self.max_y_len = GlobalConstants.MAX_Y_LEN
		from src.global_config import load_chrom_timepoints
		self.timepoints = load_chrom_timepoints(self.config.replicate)


	def load_mnase_span(self, chrom, mnase_span, log=True, downsample=True):
		"""Load the MNase for an arbitrary genomic span"""

		replicate = self.config.replicate
		self.mnase_span = mnase_span

		if not self.chr == chrom:

			if log:
				print_fl(f"Loading chromosome reads: {chrom}")

			self.chr_reads = read_chromosome_mnase_reads(replicate, chrom)
			self.chr = chrom

		else:
			if log:
				print_fl(f"Already loaded chromosome reads for {self.chr}. Using cache.")

		self.locus_reads = self.chr_reads[(self.chr_reads.mid > self.mnase_span[0]) & 
			(self.chr_reads.mid < self.mnase_span[1])]

		self.timepoints = GlobalConstants.CHROM_WT1_TIMEPOINTS if replicate == 1 else GlobalConstants.CHROM_WT2_TIMEPOINTS
		self.config.WT1_TIMEPOINTS = self.timepoints

		new_span = self.mnase_span

		# Create the bins for the reads
		exact_bins = self.create_exact_bins()
		normalized_bins = self.normalize_bins(exact_bins, log=log)
		self.normalized_bins = normalized_bins

		# Apply copy correction and renormalization
		self.apply_copy_correction(log=log)

		if downsample:
			downsampled_bins = self.downsample_bins(self.corrected_normalized_bins, new_span)
			self.new_span = new_span

		exact_extent = [self.mnase_span[0], self.mnase_span[1],
					0, GlobalConstants.MAX_Y_LEN]
		
		self.exact_bins = exact_bins
		self.exact_extent = exact_extent
		self.bin_extents = exact_extent

		if downsample:
			self.deconv_hist_unflattened = downsampled_bins
			self.image_shape = self.deconv_hist_unflattened.shape[1:]
			self.G = downsampled_bins.reshape(downsampled_bins.shape[0], -1)

	
	def compute_bin_counts_sample(self, sample, x_bins, y_bins):

		plotting_reads = self.locus_reads[self.locus_reads['sample'] == sample]
		hist, x_edges, y_edges = np.histogram2d(plotting_reads['mid'], 
			plotting_reads['length'], bins=[x_bins, y_bins])

		return plotting_reads, hist, x_edges, y_edges


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

		from src.chromatin_metrics import fragment_lengths_definitions
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


	def deconvolve(self, solver=cvxpy.MOSEK, verbose=False, 
			wavelet="Symmlet", verbose_progress=True):
		"""
		Deconvolve the chromatin for a single gamma value
		"""

		timer = Timer()
		self.setup_deconv_model()
		self.setup_solver(wavelet)

		print_fl(f"Deconvolving with gamma={self.gamma}")
		self.deconvolved_f_value = self.solver.deconvolve_G_iteratively(self.gamma,
			verbose=verbose, verbose_progress=verbose_progress)
		self.rn = self.solver.rn
		self.sn = self.solver.sn

		print_fl(f"Deconvolved in : {timer.get_time()}")
		print_fl(f"The fitting norm is {self.rn:.2f}, "
			  f"the smoothing norm is: {self.sn:.2f}")


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


	def deconvolve_find_optimal_gamma(self):
		"""
		Find the optimal gamma value using a binary search as defined by Xin, 2011
		"""

		from src.find_gamma_chromatin import FindOptimalGammaChromatin

		timer = Timer()

		# Let's stick to no spatial smoothing for now
		self.setup_deconv_model()
		self.setup_solver()
		self.find_gamma_chromatin = FindOptimalGammaChromatin(self.solver)
		self.found_optimal_success = self.find_gamma_chromatin.find_optimal(silence=False)
		self.deconvolved_f_value = self.find_gamma_chromatin.f
		self.gamma = self.find_gamma_chromatin.gamma
		self.rn = self.find_gamma_chromatin.rn
		self.sn = self.find_gamma_chromatin.sn

		print_fl(f"Found optimal gamma in: {timer.get_time()}")
		print_fl(f"Find optimal success: {self.found_optimal_success}")
		print_fl(f"The fitting norm is {self.rn:.2f}, "
			  f"the smoothing norm is: {self.sn:.2f}")

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
		

	def normalize_bins(self, exact_bins, log=True, scaling_mat=None):
		"""Normalize the histogram of exact length, position counts"""

		# Todo: refactoring

		if scaling_mat is None:
			from src.preprocessing import load_scaling_mat
			scaling_mat = load_scaling_mat(self.config.replicate)

		timepoints = self.timepoints
		normalized_bins = exact_bins.copy()

		# Normalization that matches
		# the length distribution across all timepoints and replicates
		if log: print_fl("Applying a normalization for length distribution")
		for i in range(len(timepoints)):
			time = timepoints[i]
			cur_normalized_bins = (scaling_mat[time].values.reshape((-1, 1)) * exact_bins[i])
			normalized_bins[i] = cur_normalized_bins

		return normalized_bins


	def downsample_bins(self, bin_data, new_span):

		bin_width = self.bin_width
		bin_height = self.bin_height
		self.new_span = new_span

		# Define the bin positions and the fragment lengths, these will define
		# the lower bound of the bin (the last bin will be truncated)
		x_bins = np.arange(new_span[0], new_span[1]+bin_width, bin_width)
		y_bins = np.arange(0, self.max_y_len+bin_height, bin_height)

		# Now we will loop through each x and y bin to aggregate the counts to 
		# create our new downsampled histogram
		downscaled_bins = np.zeros((bin_data.shape[0], len(y_bins), len(x_bins)))

		from src.coordinate_translator import CoordinateTranslator

		# Translate from the selected mnase span to np array space
		coord_translator = CoordinateTranslator(self.mnase_span)

		for t_index in range(len(self.timepoints)):
			for x_ind in range(1, len(x_bins)):
				for y_ind in range(1, len(y_bins)):
					x_start = coord_translator.translate(x_bins[x_ind-1])
					x_end = coord_translator.translate(x_bins[x_ind])
					y_start = y_bins[y_ind-1]
					y_end = y_bins[y_ind]
					
					bin_counts = bin_data[t_index][y_start:y_end, x_start:x_end].sum()
					downscaled_bins[t_index][y_ind-1][x_ind-1] = bin_counts

		# Bins are filled up until the last one row and column, so subset
		downscaled_bins = downscaled_bins[:, :-1, :-1]

		return downscaled_bins


	def create_deconvolution_bins(self, log=False):
		
		exact_bins = self.create_exact_bins()
		normalized_bins = self.normalize_bins(exact_bins, log=log)
		downsampled_bins = self.downsample_bins_gene(normalized_bins)
		
		exact_extent = [self.mnase_span[0], self.mnase_span[1],
					0, GlobalConstants.MAX_Y_LEN]
		gene_extent = [self.new_span[0], self.new_span[1],
						0, GlobalConstants.MAX_Y_LEN]
		
		self.exact_extent = exact_extent
		self.bin_extents = gene_extent
		self.deconv_hist_unflattened = downsampled_bins
		self.image_shape = self.deconv_hist_unflattened.shape[1:]
		self.normalized_bins = normalized_bins

		self.exact_bins = exact_bins
		self.G = downsampled_bins.reshape(downsampled_bins.shape[0], -1)

		self.apply_copy_correction(log=log)


	def apply_copy_correction(self, log=True):
		"""Copy correct using the precomputed correction and normalization scalar vector.
			See 0_Copy_Correction_Procedure notebook for details
		"""

		from src.copy_correction_reanalysis import perform_precomputed_correction_normalisation

		corrected_normalized_bins = perform_precomputed_correction_normalisation(
		    self.normalized_bins, self.chr, self.mnase_span, self.config.replicate)

		self.corrected_normalized_bins = corrected_normalized_bins
		

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
			plt.imshow(self.deconv_hist_unflattened[row], cmap='magma_r', vmax=20, origin='lower', aspect='auto',
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


	def find_origin_p1_and_m1_nucleosome_position(self, find_p1=True):
		"""
		Identify the +1 and -1 nucleosome and track its position
		"""

		from src.chrom_img_segment_selector import translate_span_for_bins
		from src.chromatin_metric_tracking import ChromatinMetricTracking
		from src.chromatin_metrics import fragment_lengths_definitions
		from src.global_config import GlobalConstants

		# Center on the middle of the window (centered on the origin site)
		center_pos = self.origin.pos

		img_data = self.get_f_images()

		# Fragment lengths
		sm_lens, med_lens, nuc_lens = fragment_lengths_definitions()
		print("Nucleosomal fragment length range: ", nuc_lens)

		# Select a largish window of fragments for origins, such that we can retrieve the entirety of what
		# appears to be origin fragments
		origin_frag_lens = sm_lens[0]+GlobalConstants.BIN_HEIGHT, sm_lens[1]+GlobalConstants.BIN_HEIGHT
		print("Origin fragment length range: ", origin_frag_lens)

		# todo: Add an additional step here, in which we expand the search range for each
		# metric, and automate narrowing the search tighter based peak occupancy and a window around the peak
		nucleosome_movement_span = 143
		origin_occ_span = 99

		# Wide search span that will automatically be narrowed down
		p1_search_span = (0, 352)
		m1_search_span = (-352, 0)
		origin_span = (-121, 121)

		# Override to refine search window for selected origins
		if self.origin.ars_name == 'ARS423':
			p1_search_span = (0, 242)
			m1_search_span = (-242, 0)
		elif self.origin.ars_name == 'ARS1623':
			p1_search_span = (0, 242)
			m1_search_span = (-142, 0)
			print(m1_search_span)

		# Currently we allow the search span to be any genomic position, but the bins restrict us
		# to the bin width (24 bp), so we need to round to the nearest 24 bp bin
		is_crick = (self.origin.strand == '-')
		updated_p1_span = translate_span_for_bins(center_pos, p1_search_span, 
			flip=is_crick)
		updated_m1_span = translate_span_for_bins(center_pos, m1_search_span, 
			flip=is_crick)
		updated_origin_span = translate_span_for_bins(center_pos, origin_span, 
			flip=is_crick)

		print(f"The +1 span for tracking is:", updated_p1_span, " length: ", updated_p1_span[1]-updated_p1_span[0])
		print(f"The -1 span for tracking is:", updated_m1_span, " length: ", updated_m1_span[1]-updated_m1_span[0])
		print(f"The span for origin occupancy is:", updated_origin_span, " length: ", 
			updated_origin_span[1]-updated_origin_span[0])

		# Track the +1 nucleosome position. Tracker selects the nucleosome positions
		# of the +1 search range
		def create_tracker(genomic_span, frag_lens, window, tracker_type='nuc_movement'):
			tracker = ChromatinMetricTracking(chrom_model=self)
			tracker.select_range(genomic_span, frag_lens)
			tracker.find_peak_and_update_genomic_positions(window=window)
			if tracker_type == 'nuc_movement': tracker.track_genomic_movement()
			elif tracker_type == 'occupancy': tracker.track_occupancy()
			else: raise ValueError("Unknown parameter: ", tracker_type)
			return tracker

		self.p1_tracker = create_tracker(updated_p1_span, nuc_lens, window=198)
		self.m1_tracker = create_tracker(updated_m1_span, nuc_lens, window=198)
		self.origin_tracker = create_tracker(updated_origin_span, origin_frag_lens, 
			window=198, tracker_type='occupancy')


	def plot_nfr_origin_occ_comparision(self, t_tps=None):
		from src.helpers import normalize_max_min

		t_indices = self.config.get_Hpositions_for_branch('t')

		if t_tps is None:
			t_tps = self.config.get_timepoints_for_branch('t')

		# Compute the NFR size per time
		nfr_size = self.p1_tracker.called_peak_weighted_mean -\
			self.m1_tracker.called_peak_weighted_mean

		# Get the top branch values for NFR length and origin occupancy
		origin_occ = self.origin_tracker.total_occupancy[t_indices]
		nfr_size_t = nfr_size[t_indices]

		normalized_origin_occ_t = normalize_max_min(origin_occ.values)
		normalized_nfr_size_t = normalize_max_min(nfr_size_t.values)

		fig = plt.figure(figsize=(12, 3))

		def plot_comparison(origin_occ, nfr_size_t):
			cmap = plt.get_cmap('Spectral')
			plt.plot(t_tps, origin_occ, label="Origin occupancy", color=cmap(0.9))
			plt.plot(t_tps, nfr_size_t, label="NFR length", color=cmap(0.1))
			plt.legend()

			ax = plt.gca()
			draw_phase_label_annotations(ax, self.config, flip=True, annotations_x=-0.13)
			plt.xlim(t_tps[0], t_tps[-1])
			plt.xticks([])
			plt.yticks([])

		plt.subplot(1, 2, 1)
		plot_comparison(origin_occ.values-origin_occ.values.min() + 20, 
			nfr_size_t.values - nfr_size_t.values.min() + 20)
		plt.ylabel("Occupancy and length")
		plt.ylim(-15, 200)

		plt.subplot(1, 2, 2)
		plot_comparison(normalized_origin_occ_t, normalized_nfr_size_t)
		plt.ylabel("Normalized occupancy and length")
		plt.ylim(-0.25, 1.6)

		if self.origin.strand == '-':
			# flip the xlims
			xlim = plt.xlims()
			plt.xlim(xlim[1], xlim[0])

		return fig


	def plot_nfr_shift_origin_occupancy(self):
		"""Show the +1 nucleosome shifts"""

		# Show that we can track the +1 nucleosome shift per each phase on a high resolution 
		# timescale

		t_indices = self.config.get_Hpositions_for_branch('t')
		s_indices = self.config.get_Hpositions_for_phase('S')
		start_of_s = s_indices[0]
		t_tps = self.config.get_timepoints_for_branch('t')

		plus_position = self.p1_tracker.called_peak_weighted_mean
		plus_position_movement = plus_position - self.center_origin

		minus_position = self.m1_tracker.called_peak_weighted_mean
		minus_position_movement = minus_position - self.center_origin

		m = len(plus_position)

		fig = plt.figure(figsize=(5, 3))

		ax = plt.gca()

		m1_movement = minus_position_movement[t_indices]
		ax.plot(m1_movement, t_tps, lw=4, color='#555')

		p1_movement = plus_position_movement[t_indices]
		ax.plot(p1_movement, t_tps, lw=4, color='#555')

		ylim = t_tps[0], t_tps[-1]
		ax.set_ylim(ylim)

		# Specific xlims for selected genes
		if self.origin.ars_name == 'ARS1212.5':
			translation = -50
			draw_phase_label_annotations(ax, self.config, annotations_x=-120+translation)
			ax.set_xlim(-133+translation, 250+translation)
		else:
			draw_phase_label_annotations(ax, self.config, annotations_x=-120)
			ax.set_xlim(-133, 250)

		from src.plot_helpers import hide_spines

		# Convert H index to t indices for plotting replication location
		def get_t_tp_from_H_index(config, H_index):
			t_indices = config.get_Hpositions_for_branch('t')
			t_tps = config.get_timepoints_for_branch('t')
			index_t = np.where(t_indices == H_index)[0][0]
			tp = t_tps[index_t]
			return tp

		# Plot the replication time
		repl_tp = get_t_tp_from_H_index(self.config, self.origin.replication_index)
		plt.axhline(repl_tp, c='black', ls='dotted', lw=0.5, zorder=0)


		# Plot the boundaries of S
		s_start_tp = get_t_tp_from_H_index(self.config, s_indices[0])
		plt.axhline(s_start_tp, c='black', ls='solid', lw=0.5, zorder=0)
		s_end_tp = get_t_tp_from_H_index(self.config, s_indices[-1])
		plt.axhline(s_end_tp, c='black', ls='solid', lw=0.5, zorder=0)

		ax.set_yticks([])
		plt.suptitle(f"{self.origin.ars_name}", fontsize=FiguresConfig.FIG_SUPTITLE_FONTSIZE)
		plt.subplots_adjust(top=0.8)

		origin_occupancy = self.origin_tracker.total_occupancy

		color_values = origin_occupancy[t_indices].values

		# Plot the origin occupancy as a colormap scatter plot on the 
		# center of the tracked origin location
		x = self.origin_tracker.get_center_selected_bp()
		origin_pos = x - self.center_origin
		origin_pos_extent = origin_pos-10, origin_pos+10
		plt.imshow(color_values.reshape((-1, 1)), extent=[origin_pos_extent[0], origin_pos_extent[1], 
			t_tps[0], t_tps[-1]], cmap='Oranges', aspect='auto', origin='lower', interpolation='none')
		plt.axvline(origin_pos_extent[0], c='#666', lw=0.75,)
		plt.axvline(origin_pos_extent[1], c='#666', lw=0.75,)

		return fig

	def plot_origin_trackers(self):

		fig, ax = self.p1_tracker.plot_selected_region()
		self.m1_tracker.plot_selected_range_rect(ax)
		self.origin_tracker.plot_selected_range_rect(ax)
		plt.title(f"{self.origin.ars_name}\nOrigin nucleosome and subnucleosome tracking regions")
		return fig


	def get_origin_tracking_df(self):
		p1 = self.p1_tracker.called_peak_weighted_mean
		m1 = self.m1_tracker.called_peak_weighted_mean
		origin_occupancy = self.origin_tracker.total_occupancy

		df = pd.DataFrame({
			'+1': p1, '-1': m1, 'origin_occupancy': origin_occupancy
		})
		return df


	def plot_orf_annotation(self, ax1):
		from src.orf_plotter import ORFAnnotationPlotter, plot_rect

		gene = self.gene
		gene_window = self.bin_extents[0],\
		    self.bin_extents[1]

		geneset = get_deconvolved_geneset()
		orf_plotter = ORFAnnotationPlotter(geneset)
		orf_plotter.set_span_chrom(gene_window, gene.chr)
		orf_plotter.plot_orf_annotations(ax1)

		if gene.strand == '-':
			ax1.set_xlim(gene_window[1], gene_window[0])
		else:
			ax1.set_xlim(*gene_window)

		ax1.set_ylim(-120, 120)


	def save_deconvolved_outputs(self, out_dir, index, using_default_flag):

		orf_name = self.gene.name
		gene_name = self.gene['gene']

		f = self.deconvolved_f()
		g_save_path = f'{out_dir}/{index}_g_{orf_name}_{gene_name}.npy'
		f_save_path = f'{out_dir}/{index}_f_{orf_name}_{gene_name}.npy'
		ptr_save_path = f'{out_dir}/{index}_ptr_{orf_name}_{gene_name}.npy'
		meta_save_path = f'{out_dir}/{index}_meta_{orf_name}_{gene_name}.csv'

		# ------- Reshape f ---------

		shape = self.deconv_hist_unflattened[0].shape
		reshaped_f = f.reshape((-1, shape[0], shape[1]))
		reshaped_ptrs = self.f_ptrs.reshape(*shape)

		#---------- Save to disk -------------

		# Save the g to disk
		np.save(g_save_path, self.deconv_hist_unflattened)

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


def read_chromosome_mnase_reads(replicate, chr):
	chr_reads = pd.read_hdf(f'output/mnase/yl_rep{replicate}_mnase_reads/yl_rep{replicate}_mnase_reads_chr{chr}.h5', 
							 'mnase_data')

	return chr_reads


def draw_phase_label_annotations(ax, config=None, phases = ['CG1', 'S', 'G2M'], 
		flip=False, annotations_x=0, offset=False):

	from src.model import color_for_key

	tp_set = []
	for phase in phases:
		tps = config.get_phase_timepoints_for_phase(phase)
		tp_set.append(tps)

	if offset:
		offset_by = -tp_set[0][0]

	last_tp = None
	for i in range(len(tp_set)):
		tps = tp_set[i]

		if offset: tps = tps + offset_by

		phase = phases[i]

		if last_tp is None:
			last_tp = tps[0]
		tp_start, tp_end = last_tp, tps[-1]
		last_tp = tps[-1]

		tp_mid = (tp_start + tp_end)/2

		xs = [annotations_x, annotations_x]
		ys = [tp_start, tp_end]

		text_x = annotations_x
		text_y = tp_mid
		rotation = 90

		if flip:
			xs, ys = ys, xs
			text_x, text_y = text_y, text_x
			rotation = 0

		ax.plot(xs, ys, c=color_for_key(phase), lw=20, solid_capstyle='butt')
		ax.text(text_x, text_y, phase, va='center', ha='center', fontsize=10,
			color='white', rotation=rotation)
