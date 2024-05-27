
import cvxpy
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from src.sgd import get_gene_name_orf_name
from src.mnase_plotting import plot_mnase_density

from src.model import Model
from src.timer import Timer
from src.utils import print_fl


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

		from src.global_config import GlobalConstants

		# Padding defines the window around the TSS to retrieve MNase data
		self.padding = 1000
		self.geneset = pd.read_csv('data/reference_data/geneset_nondub_w_prom_genebodies.csv').set_index('orf_name')
		self.config = config
		self.gamma = 0.006 # default gamma value

		self.bin_width = GlobalConstants.BIN_WIDTH
		self.bin_height = GlobalConstants.BIN_HEIGHT
		self.prom_len = GlobalConstants.PROM_LEN
		self.gb_len = GlobalConstants.GB_LEN
		self.chr = None

		self.max_y_len = GlobalConstants.MAX_Y_LEN

		# For computing the image shape
		self.num_bins_x = GlobalConstants.NUM_BINS_X
		self.num_bins_y = GlobalConstants.NUM_BINS_Y
		self.image_shape = GlobalConstants.IMAGE_SHAPE

	def load_deconvolution_results(self, gene_name):
		from src.sgd import get_gene_name_orf_name, get_gene
		gene = get_gene(gene_name)

	def set_gene(self, gene_or_orfname):
		self.orf_name, self.gene_name = get_gene_name_orf_name(gene_or_orfname)
		if self.gene_name is None: self.gene_name = self.orf_name
		self.gene = self.geneset.loc[self.orf_name]

	def load_mnase_gene(self, gene_or_orfname, log=True):

		self.set_gene(gene_or_orfname)

		# Get some gene information
		replicate = self.config.replicate
		self.computed_plus_one = None
		self.mnase_span = self.gene.TSS-self.padding, self.gene.TSS+self.padding

		if log:
			print_fl(f"Loading MNase reads for {self.orf_name}/{self.gene_name}...", end='')
		# TODO: This may take a little while, when we've deconvolved already we may want to skip this step,
		# But that will mean needing to save the +1 location to disk.

		if not self.chr == self.gene.chr:

			if log:
				print_fl(f"Loading chromosome reads: {self.gene.chr}")
			self.chr_reads = read_chromosome_mnase_reads(replicate, self.gene.chr)
			self.chr = self.gene.chr
		else:

			if log:
				print_fl(f"Already loaded chromosome reads for {self.chr}. Using cache.")

		self.gene_reads = self.chr_reads[(self.chr_reads.mid > self.mnase_span[0]) & 
			(self.chr_reads.mid < self.mnase_span[1])]

		try:
			self.find_max_plus_one_location()
		except ValueError:
			self.computed_plus_one = self.gene.TSS

			if log:
				print_fl(f"Error finding plus one location, possibly not enough read coverage. Setting plus one to TSS by default")

		self.times = self.gene_reads['sample'].unique()

		# Now that we have the +1 position defined, let's realign on this position
		#
		# TODO: Note, that we may run into some weird behavior if the read counts are very low for nucleosome fragments around
		# the TSS, in that case, we will need a back up plan... maybe a minimum threshold for this procedure...
		self.mnase_span = self.computed_plus_one-self.padding, self.computed_plus_one+self.padding
		self.gene_reads = self.chr_reads[(self.chr_reads.mid > self.mnase_span[0]) & 
			(self.chr_reads.mid < self.mnase_span[1])]

		if log:
			print_fl("Done.")

		# This will work for the single replicate model
		timepoints = self.chr_reads['sample'].unique()
		self.timepoints = timepoints
		self.config.WT1_TIMEPOINTS = self.timepoints

		
	def compute_bin_counts_sample(self, sample, x_bins, y_bins):

		plotting_reads = self.gene_reads[self.gene_reads['sample'] == sample]
		hist, x_edges, y_edges = np.histogram2d(plotting_reads['mid'], 
			plotting_reads['length'], bins=[x_bins, y_bins])

		return plotting_reads, hist, x_edges, y_edges


	def deconvolved_f(self):

		# TODO, there are multiple places where f value can be set and used
		# fix this setter and getter to be clear
		if self.deconvolved_f_value is None:
			self.deconvolved_f_value = self.solver.f.value

		return self.deconvolved_f_value


	def create_deconvolution_plots_abbreviated_flipped(self, ax_cols=None, num_rows=5, ge_model=None, 
		vmin=0, vmax=30, smooth=False, f=None, mask=None):

		if f is None:
			f = self.deconvolved_f().copy()

		f_imgs = f.reshape((-1, self.deconv_hist_unflattened.shape[1], self.deconv_hist_unflattened.shape[2]))

		if ax_cols is None:

			# We will add the first row as the deconvolved gene expression
			if ge_model is not None:
				num_rows = num_rows+1

			fig, ax_cols = plt.subplots(num_rows, 4, figsize=(11, 8))
			plt.subplots_adjust(hspace=0.5, top=0.77)

		from src.model import color_for_key

		shape = self.deconv_hist_unflattened[0].shape
		reshaped_f = f.reshape(-1, shape[0], shape[1])
		phase_cols = self.config.phase_columns

		phases = []
		indices = []

		for key, values in phase_cols.items():
			phases = phases + [key] * len(values)
			indices = indices + list(values)

		phase_col_df = pd.DataFrame({'phase': phases, 'column': indices})
		phase_col_df = phase_col_df.set_index('column')
		phase_col_df.head()

		def color_for_index(phase_col_df, index):
			phase = phase_col_df.loc[index].phase
			color = color_for_key(phase)
			return color

		plotting_index = 0
		last_phase = None

		column_titles = ["Recovery G1", "Mother G1", "Daughter G1", "Post G1"]
		phase_keys = ['RG1', 'CG1', 'DG1', 'postG1']

		# Flip
		ax_cols = np.array(ax_cols).T

		# Plot for each deconvolved cell phase
		for index in range(len(ax_cols)):

			col = index
			phase_axs = ax_cols[col]

			if ge_model is None:
				num_chromatin_rows = len(phase_axs)
			else:
				num_chromatin_rows = len(phase_axs)-1

			phase = phase_keys[col]

			# Set the title as the phase
			phase_axs[0].set_title(column_titles[col], fontsize=16)

			for row in range(num_chromatin_rows):

				# Our first row is for the gene expression, so +1
				if ge_model is None:
					ax = phase_axs[row]
				else:
					ax = phase_axs[row+1]

				shape = self.deconv_hist_unflattened[0].shape
				reshaped_f = f.reshape(-1, shape[0], shape[1])

				self.plot_f_img_phase(ax, reshaped_f, phase, row, num_chromatin_rows, show_title=False, vmax=vmax, vmin=vmin,
					mask=mask)

				if col == 0:
					ax.set_ylabel(f"{row+1}", rotation=0, ha='right', labelpad=10, fontsize=16)


		# Add some xtick and xtick labels to the first column last row
		first_col_last_row = ax_cols[0][-1]

		xticks = self.bin_extents[0], \
				 self.computed_plus_one, \
				 self.bin_extents[1]
		xtick_labels = [str(x-self.computed_plus_one) for x in xticks]
		xtick_labels[1] = 'TSS'
		xtick_labels[2] = '+'+xtick_labels[2]

		first_col_last_row.set_xticks(xticks)
		first_col_last_row.set_xticklabels(xtick_labels)

		# ---------------------

		# If we have deconvolved gene expression, add it to the last column
		if ge_model is not None:

			from src.model import color_for_key

			# f data minus the halted f
			# TODO: How do we plot the halted data if it is off the chart?
			# Do we keep the ylim names?
			ge_f = ge_model.f[:-1]
			ge_f_diff = ge_f.max() - ge_f.min()
			ylim = [ge_f.min()-ge_f_diff*.1, ge_f.min()+ge_f_diff*1.3]

			ylim[0] = max(ylim[0], 0)

			for col in range(len(ax_cols)):

				phase = phase_keys[col]
				hindices = self.config.get_Hpositions_for_phase(phase)

				# The last subplot in the row
				ax = ax_cols[col][0]
				y = ge_f[hindices]
				x = np.arange(len(y))

				ax.fill_between(x, 0, y, color=color_for_key(phase))
				ax.set_xlim(x.min(), x.max())

				ax.set_ylim(*ylim)

				if col < len(ax_cols)-1:
					ax.set_yticks([])
				else:
					ax.yaxis.tick_right()
					ax.yaxis.set_tick_params(pad=3, length=3)

				# Add some grid lines to help show where the chromatin images map to
				xgridlines = np.linspace(0, x.max(), num_chromatin_rows)
				for xval in xgridlines:
					ax.axvline(xval, c='black', alpha=0.15, lw=1, linestyle='solid')
				ax.set_xticks([])

				# This is in the for loop so we can get the grid lines as they
				# will be different per row
				# Label the last column first row
				ax.set_xticks(xgridlines)
				ax.set_xticklabels([f"{i+1}" for i in np.arange(len(xgridlines))], fontsize=9)
				ax.xaxis.set_tick_params(pad=3, length=0)

				if col == 0:
					ax.set_ylabel("Expression", fontsize=16, labelpad=10, 
						ha='right', rotation=0, va='center')

		title = self.define_title()
		plt.suptitle(title, fontsize=24)
		return fig


	def gene_title(self):
		if self.gene.gene == self.orf_name:
			gene_title = ("$\\it{" + self.orf_name + "}$")
		else:
			gene_title = ("$\\it{" + self.gene.gene + "}$ / $\\it{" + self.orf_name + "}$")
		return gene_title

	def define_title(self):

		gene_title = self.gene_title()
		title = (f"{gene_title}\n" +
				self.config.name + ", " +
				f"$\\gamma$={self.solver.gamma.value:.3f}\nrn={self.rn:.2f}, sn={self.sn:.2f}")
		return title


	def plot_f_img_phase(self, ax, reshaped_f, phase, column, num_columns, show_title=True, x_padding=0, y_padding=0,
		vmin=0, vmax=200, mask=None):
		"""Plot the f image of a phase and column for the grid of f images progressing through each phase
		compute the proper index to plot from the num_columns parameter for the phase"""


		# Get the index within the f matrix of the appropriate image
		# by phase and column, num_columns signifies how many subsets of the phase
		# we are going to plot, the other returned items are for logging and for the title
		f_index, index, len_sub_f = self.f_index_for_column(phase, column, num_columns)

		# Plot the deconvolved chromatin for the appropriate column
		img = reshaped_f[f_index]

		if show_title:
			ax.set_title(f"{index+1}/{len_sub_f} ({(index/(len_sub_f-1))*100:.0f}%)", fontsize=9)

		self.plot_f_img(ax, img, x_padding=x_padding, y_padding=y_padding, vmin=vmin, vmax=vmax, mask=mask)


	def plot_f_img(self, ax, img, show_title=True, x_padding=0, y_padding=0,
		vmin=0, vmax=200, mask=None):
		"""Plot the f image of a phase and column for the grid of f images progressing through each phase
		compute the proper index to plot from the num_columns parameter for the phase"""

		is_crick = self.gene.strand == '-'

		bin_extents = self.bin_extents

		# pad the extents using xlim and ylim
		xlims = bin_extents[0]-x_padding, \
			bin_extents[1]+x_padding
		ylims = bin_extents[2]-y_padding, \
			bin_extents[3]+y_padding

		ax.set_xlim(*xlims)
		ax.set_ylim(*ylims)
		ax.set_xticks([])
		ax.set_yticks([])

		if mask is not None:
			img = img * mask

		im = ax.imshow(img, origin='lower', cmap='magma_r', aspect='auto', vmax=vmax,
			extent=self.bin_extents, zorder=1)
		ax.axvline(self.computed_plus_one, c='gray', linewidth=1.25, linestyle='solid', alpha=0.5)

		if is_crick:
			# flip the xlims
			xlims = ax.get_xlim()
			ax.set_xlim(xlims[1], xlims[0])


	def f_index_for_column(self, phase, column, columns):
		"""If we are plotting only a subset of all of the images for a phase, we can
		subdivide the number of f by some step determined by the number of columns.
		
		For example if we have 20 f values for a phase and we only have 3 columns, we will
		return 0, 10, 20 as the indices we are interested in.
		
		Also note, that the indices in H (and f) have their own indices per phase so 
		we will need to map into those values as well (phase_indices[phase_indices_index])
		"""
		phase_indices = self.config.get_Hpositions_for_phase(phase)

		phase_indices_index = len(phase_indices) / (columns-1) * column
		phase_indices_index = round(phase_indices_index)
		phase_indices_index = min(phase_indices_index, len(phase_indices)-1)
		return  phase_indices[phase_indices_index], phase_indices_index, len(phase_indices)


	def create_deconvolution_plots_full(self, model):

		f = model.f
		from src.model import color_for_key

		shape = self.deconv_hist_unflattened[0].shape
		reshaped_f = f.reshape(-1, shape[0], shape[1])
		phase_cols = model.config.phase_columns

		phases = []
		indices = []

		for key, values in phase_cols.items():
			phases = phases + [key] * len(values)
			indices = indices + list(values)

		phase_col_df = pd.DataFrame({'phase': phases, 'column': indices})
		phase_col_df = phase_col_df.set_index('column')
		phase_col_df.head()

		def color_for_index(phase_col_df, index):
			phase = phase_col_df.loc[index].phase
			color = color_for_key(phase)
			return color

		fig, axs = plt.subplots(26, 10, figsize=(13, 9))
		axs = np.array(axs).T.flatten()

		x_bins, y_bins = self.x_bins, self.y_bins

		plotting_index = 0
		last_phase = None
		for i in range(len(axs)):
			
			ax = axs[plotting_index]

			if plotting_index >= reshaped_f.shape[0]: 
				ax.set_xticks([])
				ax.set_yticks([])
				continue

			phase = phase_col_df.loc[i].phase
			color = color_for_index(phase_col_df, i)

			im = ax.imshow(reshaped_f[i], origin='lower', cmap='magma_r', aspect='auto', vmax=500,
				extent=self.bin_extents)
			ax.set_xticks([])
			ax.set_yticks([])
			ax.axvline(self.computed_plus_one, c='gray', lw=1, linestyle='dotted')

			if last_phase is not None and phase != last_phase:
				plotting_index += 2
			else:
				plotting_index += 1

			last_phase = phase


	def plot_im_gene(self, ax, dat):

		im = ax.imshow(dat, origin='lower', cmap='magma_r', aspect='auto',
				extent=self.bin_extents)
		# ax.set_xticks([])
		#ax.set_yticks([])
		ax.axvline(self.computed_plus_one, c='gray', lw=1, linestyle='dotted')
		return im


	def plot_prediction_comparison(self, predicted_g=None, title=None, vmax=15):

		times = self.times

		if predicted_g is None:
			f = self.deconvolved_f()
			predicted_g = np.matmul(self.deconv_model.H, f)

		shape = self.deconv_hist_unflattened[0].shape
		n = self.deconv_hist_unflattened.shape[0]

		predicted_g_reshaped = predicted_g.reshape(n, shape[0], shape[1])

		fig, axs = plt.subplots(n, 5, figsize=(13, 13))
		plt.subplots_adjust(top=0.82)

		axs = np.array(axs).T
		raw_axs = axs[0]
		g_axs = axs[1]
		pred_g_axs = axs[2]
		comparison_axs = axs[3]
		ptr_axs = axs[4]


		# ------- PTR plots ----------------------

		# Plot the PTR values
		ptr_ax = ptr_axs[0]
		f_ptrs_img = self.f_ptrs.reshape(shape)
		ptr_ax.imshow(f_ptrs_img, aspect='auto', extent=self.bin_extents, 
			origin='lower', cmap='viridis', vmax=10)
		ptr_ax.axvline(self.computed_plus_one, c='gray', alpha=0.25)
		ptr_ax.set_title("PTRs")

		# Plot 7 highest ptr values (7 is tentative optimal k for now)

		from src.ptr_analysis_plotter import threshold_img

		thresholded = threshold_img(f_ptrs_img)
		ptr_k_ax = ptr_axs[1]
		ptr_k_ax.imshow(thresholded, extent=self.bin_extents, origin='lower', 
				  cmap='viridis', aspect='auto')
		ptr_k_ax.axvline(self.computed_plus_one, c='gray', alpha=0.25)
		ptr_k_ax.set_xticks([])
		ptr_k_ax.set_yticks([])

		# Formatting
		for i in range(len(ptr_axs)):
			ax = ptr_axs[i]
			ax.set_xticks([])
			ax.set_yticks([])

			if self.gene.strand == '-':
				# flip the xlims
				xlims = ax.get_xlim()
				ax.set_xlim(xlims[1], xlims[0])

			if i > 1:
				ax.spines['top'].set_visible(False)
				ax.spines['bottom'].set_visible(False)
				ax.spines['left'].set_visible(False)
				ax.spines['right'].set_visible(False)

		# ----------------------------------------------------------

		for i in range(n):
			time = times[i]

			raw_ax = raw_axs[i]

			xlims = self.mnase_span
			gene = self.gene

			im = raw_ax.imshow(self.exact_bins[i], origin='lower', cmap='magma_r', 
						   aspect='auto', vmax=0.25,
						   extent=self.exact_extent)
			raw_ax.set_xlim(self.bin_extents[0], self.bin_extents[1])
			raw_ax.set_xticks([])
			raw_ax.set_yticks([])

			g_ax = g_axs[i]
			raw_image = self.G[i].reshape(self.image_shape)
			im = g_ax.imshow(raw_image, origin='lower', cmap='magma_r', 
						   aspect='auto', vmax=vmax,
						   extent=self.bin_extents)
			
			pred_ax = pred_g_axs[i]
			im = pred_ax.imshow(predicted_g_reshaped[i], origin='lower', cmap='magma_r', 
						   aspect='auto', vmax=vmax,
						   extent=self.bin_extents)

			comp_ax = comparison_axs[i]
			im = comp_ax.imshow(predicted_g_reshaped[i]-self.deconv_hist_unflattened[i], 
							origin='lower', cmap='RdBu', aspect='auto', vmin=-vmax/2, vmax=vmax/2,
							extent=self.bin_extents)
			
			raw_ax.set_ylabel(f"{time}'", fontsize=8)

			for ax in [raw_ax, g_ax, pred_ax, comp_ax]:
				ax.set_xticks([])
				ax.set_yticks([])

				# Identify the plus 1 location
				ax.axvline(self.computed_plus_one, c='gray', linewidth=1.25, linestyle='solid', alpha=0.5)

				if self.gene.strand == '-':
					# flip the xlims
					xlims = ax.get_xlim()
					ax.set_xlim(xlims[1], xlims[0])

		raw_axs[0].set_title("Raw")
		g_axs[0].set_title("Binned")
		pred_g_axs[0].set_title("Predicted")
		comparison_axs[0].set_title("Difference")

		if title is None:
			title = self.define_title()

		plt.suptitle(title, fontsize=24)

		return fig


	def find_max_plus_one_location(self):
		"""
		Find the position of the +1 by finding the max number of nucleosome reads in
		a 200bp window around the TSS.

		For the currently selected gene
		"""
		from src.chromatin_metrics import yl_rep2_len_spans
		small_lens, med_lens, nuc_lens = yl_rep2_len_spans()

		# Next, we will align at the +1
		# from the TSS, stack up all timepoints, then look up and dowstream (200 bp window) for the
		# position with the highest number of reads, and we will use that position as our +1 position
		# We will put that position into our gene data set and use that as our reference data set

		# We can get all of the  nucleosome length fragments for the gene, and stack them up by time

		cur_reads = self.gene_reads.copy()

		# Search around the TSS with a 200bp window
		window = 200
		search_peak_span = self.gene.TSS-window//2, \
			self.gene.TSS+window//2 

		cur_nuc_reads = cur_reads[(cur_reads['length'] >= nuc_lens[0]) & 
							  (cur_reads['length'] < nuc_lens[1]) & 
								 (cur_reads['mid'] >= search_peak_span[0]) &
								 (cur_reads['mid'] < search_peak_span[1])]

		counts_per_pos_search = cur_nuc_reads.groupby('mid').count()
		counts_per_pos_search = counts_per_pos_search[['start']].rename({'start': 'count'})
		pos_max = counts_per_pos_search.idxmax().start


		self.computed_plus_one = pos_max

		return pos_max

	def	setup_deconv_model(self):
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
		from src.helpers import calcH

		self.deconv_model = Model(self.config, self.orf_name, self.gamma)

		# The config for MNase and RNA-seq have a different number of timepoints, so 
		# we need to recalculate H with the chromatin number of timepoints
		self.deconv_model.H, self.deconv_model.Hpos = calcH(self.config.intervals_wt1, self.timepoints)

	def setup_solver(self, wavelet="Symmlet"):
		from src.chromatin_deconvolution_solver import ChromatinDeconvolveSolver
		from src.replication_deconvolution_solver import ReplicationChromatinDeconvolveSolver

		self.solver = ChromatinDeconvolveSolver(self.deconv_model.config, self.deconv_model.H, 
			wavelet=wavelet)
		self.solver.define_deconvolution_problem(self.G[:, 0:1])
		self.found_optimal_success = None
		self.deconvolved_f_value = None


	def deconvolve(self, solver=cvxpy.MOSEK, verbose=False, wavelet="Symmlet"):
		"""
		Deconvolve the chromatin for a single gamma value
		"""

		timer = Timer()
		self.setup_deconv_model()
		self.setup_solver(wavelet)

		print_fl(f"Deconvolving with gamma={self.gamma}")

		deconvolved_f_value = np.zeros((self.deconv_model.H.shape[1], self.G.shape[1]))

		# Setup of the solver with single dimension G
		m = self.G.shape[1]
		self.sn = 0
		self.rn = 0
		for i in range(m):

			# Set the solver's G value
			current_G = self.G[:, i:i+1]
			self.solver.define_deconvolution_problem(current_G)
			self.solver.solve(gamma_value=self.gamma, verbose=verbose)

			current_f = self.solver.f.value.flatten()
			deconvolved_f_value[:, i] = current_f

			self.rn += self.solver.rn / m
			self.sn += self.solver.sn / m

			if i % 20 == 0:
				timer.print_time(f"{i}/{m}")

		self.deconvolved_f_value = deconvolved_f_value

		print_fl(f"Deconvolved in : {timer.get_time()}")
		print_fl(f"The fitting norm is {self.rn:.2f}, "
			  f"the smoothing norm is: {self.sn:.2f}")

	def compute_ptr(self, quantiles=[0.2, 0.8]):

		# -------- Compute the PTR ---------

		f = self.deconvolved_f()
		from src.peak_to_trough import compute_ptr, compute_ptr_f

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
		self.gamma = self.find_gamma_chromatin.gamma

		print_fl(f"Found optimal gamma in: {timer.get_time()}")
		print_fl(f"Find optimal success: {self.found_optimal_success}")
		print_fl(f"The fitting norm is {self.rn:.2f}, "
			  f"the smoothing norm is: {self.sn:.2f}")


	# --------------- Beginning of histogram refactor --------------------
	#
	# Many of the histogram creation functions above will need to be removed
	#

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


	def plot_halted_f_img(self):
		plt.figure(figsize=(1.75, 0.5))
		orig_shape = self.deconv_hist_unflattened.shape
		f_imgs = self.f.reshape(-1, orig_shape[1], orig_shape[2])
		plt.imshow(f_imgs[-1], origin='lower', cmap='magma_r', vmax=25, 
				   extent=self.bin_extents, aspect='auto')
		plt.axvline(self.computed_plus_one, c='gray', lw=1, alpha=0.5)
		plt.xticks([])
		plt.yticks([])
		plt.title("Halted cells")
		

	def normalize_bins(self, exact_bins, log=True):
		"""Normalize the histogram of exact length, position counts"""

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

		# Normalization that keeps the copy number for all timepoints equal
		# The chromatin window sum should be the same for all genes
		# if log: print_fl("Applying a normalization for copy number, all timepoints will have equal sum")
		# sums_per_time = normalized_bins.sum(axis=1).sum(axis=1)
		# for i in range(normalized_bins.shape[0]):
		# 	normalized_bins[i] *= 1./sums_per_time[i] * 5000.

		return normalized_bins


	def downsample_bins(self, bin_data):
		# Now downsample to the appropriate window and resolution

		bin_width = self.bin_width
		bin_height = self.bin_height
		prom_len = self.prom_len
		gb_len = self.gb_len

		# Add half a bin width to allow for the inclusion of +1 bin 
		if self.gene.strand == '+':
			new_span = self.computed_plus_one-prom_len-bin_width//2, self.computed_plus_one+gb_len+bin_width//2
		else:
			new_span = self.computed_plus_one-gb_len-bin_width//2, self.computed_plus_one+prom_len+bin_width//2

		self.new_span = new_span

		# Next we will define our new bin locations
		x_bins = np.arange(new_span[0], new_span[1], bin_width)

		# And for y lengths
		y_bins = np.arange(0, self.max_y_len, bin_height)

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

		return downscaled_bins


	def create_deconvolution_bins(self, log=False):
		
		exact_bins = self.create_exact_bins()
		normalized_bins = self.normalize_bins(exact_bins, log=log)
		downsampled_bins = self.downsample_bins(normalized_bins)
		
		exact_extent = [self.mnase_span[0], self.mnase_span[1],
					0, 250]
		gene_extent = [self.new_span[0], self.new_span[1],
						0, 250]
		
		self.exact_extent = exact_extent
		self.bin_extents = gene_extent
		self.deconv_hist_unflattened = downsampled_bins
		self.image_shape = self.deconv_hist_unflattened.shape[1:]
		self.normalized_bins = normalized_bins

		self.exact_bins = exact_bins
		self.G = downsampled_bins.reshape(downsampled_bins.shape[0], -1)

		if log:
			print_fl(f"Unflattened the input data is of shape: {self.deconv_hist_unflattened.shape}")
			print_fl(f"The size of our input data, G is: {self.G.shape}")
		

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


	def save_deconvolved_outputs(self, out_dir, index, using_default_flag):

		orf_name = self.deconv_model.orf_name
		gene_name = self.deconv_model.gene_name
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
			index=[self.deconv_model.orf_name])
		df.to_csv(meta_save_path, float_format="%.4f")

		print_fl(f"Saved to {g_save_path}...")
		print_fl(f"Saved to {f_save_path}...")
		print_fl(f"Saved to {ptr_save_path}...")
		print_fl(f"Saved to {meta_save_path}...")


def load_chromatin_model_from_disk(gene_name, chromatin_dir, f_only=False):

	from src.config import load_yl_rg1_vst_config

	# gene name

	import os
	import glob

	f_pattern = os.path.join(chromatin_dir, f'*_f_*{gene_name}*')
	ptr_pattern = os.path.join(chromatin_dir, f'*_ptr_*{gene_name}*')
	g_pattern = os.path.join(chromatin_dir, f'*_g_*{gene_name}*')
	meta_pattern = os.path.join(chromatin_dir, f'*_meta_*{gene_name}*')

	g_filepath = glob.glob(g_pattern)[0]
	f_filepath = glob.glob(f_pattern)[0]
	ptr_filepath = glob.glob(ptr_pattern)[0]
	meta_filepath = glob.glob(meta_pattern)[0]

	config = load_yl_rg1_vst_config(1)

	f = np.load(f_filepath)

	if f_only:
		return f

	ptr = np.load(ptr_filepath)
	g = np.load(g_filepath)
	meta_data = pd.read_csv(meta_filepath)
	meta_data = meta_data.iloc[0]

	config = load_yl_rg1_vst_config(1)
	chromatin_model = ChromatinModel(config)

	chromatin_model.load_mnase_gene(gene_name)
	chromatin_model.create_deconvolution_bins()
	chromatin_model.setup_deconv_model()

	chromatin_model.deconvolved_f_value = f
	chromatin_model.f_ptrs = ptr.flatten()

	chromatin_model.rn = meta_data.rn
	chromatin_model.sn = meta_data.sn
	chromatin_model.gm = meta_data.gm

	return chromatin_model


def read_chromosome_mnase_reads(replicate, chr):
	chr_reads = pd.read_hdf(f'output/mnase/yl_rep{replicate}_mnase_reads/yl_rep{replicate}_mnase_reads_chr{chr}.h5', 
							 'mnase_data')

	return chr_reads
