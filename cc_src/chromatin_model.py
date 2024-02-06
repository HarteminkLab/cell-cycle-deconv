
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from cc_src.sgd import get_gene_name_orf_name
from cc_src.mnase_plotting import plot_mnase_density
import sys


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
		self.padding = 1000
		self.geneset = pd.read_csv('data/reference_data/geneset_nondub_w_prom_genebodies.csv').set_index('orf_name')
		self.config = config
		self.gamma = 0.01


	def load_deconvolution_results(self, gene_name):

		from cc_src.sgd import get_gene_name_orf_name, get_gene
		gene = get_gene(gene_name)


	def load_mnase_gene(self, gene_name, replicate):

		# Get some gene information
		self.computed_plus_one = None
		self.orf_name, self.gene_name = get_gene_name_orf_name(gene_name)
		self.gene = self.geneset.loc[self.orf_name]
		self.mnase_span = self.gene.TSS-self.padding, self.gene.TSS+self.padding

		print(f"Loading MNase reads for {gene_name}...", end='')
		# TODO: This may take a little while, when we've deconvolved already we may want to skip this step,
		# But that will mean needing to save the +1 location to disk.
		self.chr_reads = pd.read_hdf(f'output/mnase/yl_rep{replicate}_mnase_reads/yl_rep{replicate}_mnase_reads_chr{self.gene.chr}.h5', 
					'mnase_data')
		self.gene_reads = self.chr_reads[(self.chr_reads.mid > self.mnase_span[0]) & 
			(self.chr_reads.mid < self.mnase_span[1])]
		self.find_max_plusOne_pos()

		self.times = self.gene_reads['sample'].unique()

		# Now that we have the +1 position defined, let's realign on this position
		#
		# TODO: Note, that we may run into some weird behavior if the read counts are very low for nucleosome fragments around
		# the TSS, in that case, we will need a back up plan... maybe a minimum threshold for this procedure...
		self.mnase_span = self.computed_plus_one-self.padding, self.computed_plus_one+self.padding
		self.gene_reads = self.chr_reads[(self.chr_reads.mid > self.mnase_span[0]) & 
			(self.chr_reads.mid < self.mnase_span[1])]

		print("Done.")

		self.create_binned_structures()


	def create_binned_structures(self):
		"""Create binning structures from the loaded MNase data"""
		self.define_histogram_bins()
		self.create_bins_per_all_sample()
		self.create_deconvolution_matrices(False)


	def define_histogram_bins(self):
		"""
		Here, we will define the genomic bin positions as centered around the
		computed plus one location. Where we will want one bin. Then, 

		Define the promoter as apporximately 300 bp (3 bins backwards)
		But also take into account half a bin width, since we are centering a 
		bin on the +1 nucleosome.

		3 * 80 = 240
		+40 (the half bin from the center)
		280 bp will be the promoter region

		And the gene body as 5 bins forward:
		5 bins forward, 
		5 * 80 = 400
		+ 40

		440 bp wide will be the gene body (including sitting on the +1 nucleosome)
		"""

		from cc_src.chromatin_metrics import yl_replicate_length_bins

		x_bin_size = 80
		y_bin_size = 100

		num_promoter_bins = 3
		num_gb_bins = 5
		num_bins = num_promoter_bins+num_gb_bins+1 # Plus one, because we are centered on a bin

		y_bins = yl_replicate_length_bins()
		
		# from the center we will 
		center = self.computed_plus_one

		if self.gene.strand == "+":
			x_start = center - x_bin_size//2 - num_promoter_bins*x_bin_size
			x_end = x_start+num_bins*x_bin_size
		else:
			x_start = center - x_bin_size//2 - num_gb_bins*x_bin_size
			x_end = x_start+num_bins*x_bin_size

		x_bins = np.arange(x_start, x_end+x_bin_size, x_bin_size)

		self.bin_extents = [x_bins[0], x_bins[-1], 0, 225]
		self.x_bins = x_bins
		self.y_bins = y_bins

		
	def compute_bin_counts_sample(self, sample):

		plotting_reads = self.gene_reads[self.gene_reads['sample'] == sample]

		x_bins, y_bins = self.x_bins, self.y_bins

		hist, x_edges, y_edges = np.histogram2d(plotting_reads['mid'], 
			plotting_reads['length'], bins=[x_bins, y_bins])

		return plotting_reads, hist, x_edges, y_edges

	def create_bins_per_all_sample(self):

		samples = self.times

		self.all_plotting_reads = {}
		self.all_hists = None

		i = 0
		for sample in samples:
			plotting_reads, hist, x_edges, y_edges = self.compute_bin_counts_sample(sample)

			# Tranpose so its easier to plot (matches columns and rows more intuitively)
			hist = hist.T


			if self.all_hists is None:
				self.all_hists = np.zeros((len(samples), hist.shape[0], hist.shape[1]))

			self.all_plotting_reads[sample] = plotting_reads
			self.all_hists[i] = hist
			i += 1

		print(f"The histogram shape around the TSS is:", 
			hist.shape)

	def plot_sample(self, ax1, ax2, sample, i):
		
		x_bins, y_bins = self.x_bins, self.y_bins

		xlims = self.mnase_span
		gene = self.gene

		plotting_reads = self.all_plotting_reads[sample]
		hist = self.all_hists[i]

		plot_mnase_density(ax1, plotting_reads)
		ax1.set_xticks([])

		# This is the plot of the grid, so the extents are inset
		ax2.imshow(hist, origin='lower', aspect='auto', cmap='magma_r',
			extent=self.bin_extents)

		center = self.computed_plus_one

		for x in x_bins:
			ax1.axvline(x, c='red', lw=1, alpha=0.5)

		for y in y_bins:
			ax1.axhline(y, c='red', lw=1, alpha=0.5)

		for ax in [ax1, ax2]:
			xticks = np.arange(center-1000, center+1500, 500)
			xtick_labels = ['-1000', '-500', '+1 pos.', '500', '1000']

			ax.set_xticks(xticks)
			ax.set_xticklabels(xtick_labels)
			ax.axvline(self.computed_plus_one, c='black')
			ax.set_ylim(0, 225)

		ax1.set_xlim(*xlims)
		ax2.set_xlim(*xlims)


	def plot_raw_and_grid(self):

		times = self.times
		fig, axs = plt.subplots(len(times), 2, figsize=(19, 19))

		# Get a list of the raw and grid axes by transposing the axes
		axs = np.array(axs).T
		raw_axes = axs[0]
		grid_axes = axs[1]

		# Plot the raw data and the grid histograms for each of the time points
		for i in range(len(times)):
			time = times[i]
			raw_ax = raw_axes[i]
			raw_ax.set_ylabel(f"{time} min")
			raw_ax.set_yticks([])
			grid_ax = grid_axes[i]
			grid_ax.set_yticks([])
			self.plot_sample(raw_ax, grid_ax, time, i)

			if i < len(times)-1:
				raw_ax.set_xticks([])
				grid_ax.set_xticks([])


	def create_deconvolution_matrices(self, plot=False):

		times = self.times

		orig_shape = self.all_hists[0].shape

		# Checking if we can collapse the rows and columns, then restore them
		reshaped_hist = self.all_hists.reshape(len(times), -1)
		first_hist = reshaped_hist[0]

		if plot:
			print("The first histogram is shape:", self.all_hists[0].shape)
			print("Reshaping this histogram to a vector of shape:", first_hist.shape)
			restored_hist = first_hist.reshape(orig_shape)
			print("Then, if we were to take that first vector and restore it to its original shape:", 
				  restored_hist.shape)

			plt.subplot(1, 2, 1)
			plt.imshow(self.all_hists[0], origin='lower', cmap='magma_r')
			plt.title("Original first histogram")
			plt.xticks([])
			plt.yticks([])


			plt.subplot(1, 2, 2)
			plt.imshow(restored_hist, origin='lower', cmap='magma_r')
			plt.title("Restored histogram after flattening")
			plt.xticks([])
			plt.yticks([])

		# We will need to drop the 110 time point for this deconvolution
		# TODO: At least for now, as we have assumed we should drop this point as per 
		# Yulong's analysis
		# We probably don't need to do this anymore.
		print("Now we have a data structure that we can try to deconvolve of shape:", 
			  reshaped_hist.shape)

		# Reshape for deconvolution
		self.deconv_hist = reshaped_hist


	def create_deconvolution_plots_abbreviated(self, ax_rows=None):

		f = self.f

		if ax_rows is None:
			fig, ax_rows = plt.subplots(4, 6, figsize=(16, 6))
			plt.subplots_adjust(hspace=0.5)

		from src.model import color_for_key

		shape = self.all_hists[0].shape
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

		x_bins, y_bins = self.x_bins, self.y_bins

		plotting_index = 0
		last_phase = None

		row_titles = ["Recovery", "Mother G1", "Daughter G1", "Post G1"]
		phase_keys = ['RG1', 'CG1', 'DG1', 'postG1']

		# Plot for each deconvolved cell phase
		for row in range(len(ax_rows)):

			phase_axs = ax_rows[row]
			num_columns = len(phase_axs)
			phase = phase_keys[row]

			for column in range(num_columns):

				ax = phase_axs[column]

				if column == 0:
					ax.set_ylabel(row_titles[row], rotation=0, ha='right')

				self.plot_f_img(ax, f, phase, column, num_columns)


	def plot_f_img(self, ax, f, phase, column, num_columns, show_title=True, x_padding=0, y_padding=0):

		is_crick = self.gene.strand == '-'

		shape = self.all_hists[0].shape
		reshaped_f = f.reshape(-1, shape[0], shape[1])

		# Get the index within the f matrix of the appropriate image
		# by phase and column, num_columns signifies how many subsets of the phase
		# we are going to plot, the other returned items are for logging and for the title
		f_index, index, len_sub_f = self.f_index_for_column(phase, column, num_columns)
		if show_title:
			ax.set_title(f"{index+1}/{len_sub_f} ({(index/(len_sub_f-1))*100:.0f}%)", fontsize=9)

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

		# Plot the deconvolved chromatin for the appropriate column
		img = reshaped_f[f_index]
		im = ax.imshow(img, origin='lower', cmap='magma_r', aspect='auto', vmax=500,
			extent=self.bin_extents, zorder=1)
		ax.plot([self.computed_plus_one, self.computed_plus_one], 
				[bin_extents[2], bin_extents[3]], c='black', linestyle='solid', linewidth=5, alpha=1, zorder=0)

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
		phase_indices = self.config.phase_columns[phase]
		phase_indices_index = len(phase_indices) / (columns-1) * column
		phase_indices_index = round(phase_indices_index)
		phase_indices_index = min(phase_indices_index, len(phase_indices)-1)
		return  phase_indices[phase_indices_index], phase_indices_index, len(phase_indices)


	def create_deconvolution_plots_full(self, model):

		f = model.f
		from src.model import color_for_key

		shape = self.all_hists[0].shape
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
			ax.axvline(self.computed_plus_one, c='black', lw=1, linestyle='dotted')

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
		ax.axvline(self.computed_plus_one, c='black', lw=1, linestyle='dotted')
		return im

	def plot_prediction_comparison(self):

		f = self.f
		times = self.times
		predicted_g = np.matmul(self.deconv_model.H, f)
		shape = self.all_hists[0].shape

		n = predicted_g.shape[0]

		predicted_g_reshaped = predicted_g.reshape(n, shape[0], shape[1])

		fig, axs = plt.subplots(n, 4, figsize=(7, 10))
		axs = np.array(axs).T
		raw_axs = axs[0]
		g_axs = axs[1]
		pred_g_axs = axs[2]
		comparison_axs = axs[3]

		x_bins, y_bins = self.x_bins, self.y_bins

		for i in range(n):
			time = times[i]

			raw_ax = raw_axs[i]

			xlims = self.mnase_span
			gene = self.gene

			plotting_reads = self.all_plotting_reads[time]
			plot_mnase_density(raw_ax, plotting_reads)
			raw_ax.set_xticks([])
			raw_ax.set_yticks([])
			raw_ax.set_xlim(x_bins[0], x_bins[-1])

			g_ax = g_axs[i]
			im = g_ax.imshow(self.all_hists[i], origin='lower', cmap='magma_r', 
						   aspect='auto', vmax=300,
						   extent=self.bin_extents)
			
			pred_ax = pred_g_axs[i]
			im = pred_ax.imshow(predicted_g_reshaped[i], origin='lower', cmap='magma_r', 
						   aspect='auto', vmax=300,
						   extent=self.bin_extents)

			comp_ax = comparison_axs[i]
			im = comp_ax.imshow(predicted_g_reshaped[i]-self.all_hists[i], 
							origin='lower', cmap='RdBu', aspect='auto', vmin=-300, vmax=300,
							extent=self.bin_extents)
			
			raw_ax.set_ylabel(f"{time}'", fontsize=8)

			for ax in [raw_ax, g_ax, pred_ax, comp_ax]:
				ax.set_xticks([])
				ax.set_yticks([])
				ax.axvline(self.computed_plus_one, c='black', lw=1, linestyle='dashed')


				#if model.gene.strand == '-':
					# Flip the x-axis
				#xlim = ax.get_xlim()
				#ax.set_xlim(xlim[1], xlim[0])

		raw_axs[0].set_title("Raw")
		g_axs[0].set_title("Binned")
		pred_g_axs[0].set_title("Predicted")
		comparison_axs[0].set_title("Difference")

	def find_max_plusOne_pos(self):
		"""
		Find the position of the +1 by finding the max number of nucleosome reads in
		a 200bp window around the TSS.

		For the currently selected gene
		"""
		from cc_src.chromatin_metrics import yl_rep2_len_spans
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


	def save_deconvolved_outputs(self, out_dir, index, model, f, using_default_flag):

		orf_name = model.orf_name

		g_save_path = f'{out_dir}/{index}_g_{orf_name}.npy'
		f_save_path = f'{out_dir}/{index}_f_{orf_name}.npy'
		ptr_save_path = f'{out_dir}/{index}_ptr_{orf_name}.npy'
		meta_save_path = f'{out_dir}/{index}_meta_{orf_name}.csv'

		# ------- Reshape f ---------

		shape = self.all_hists[0].shape
		reshaped_f = f.reshape(-1, shape[0], shape[1])

		# -------- Compute the PTR ---------

		from cc_src.peak_to_trough import compute_ptr

		shape = self.all_hists[0].shape
		reshaped_f = f.reshape(-1, shape[0], shape[1])

		f_ptrs = np.zeros(f.shape[1])
		for i in range(f.shape[1]):
			cptr, dpt, ptr = compute_ptr(model, f[:, i])
			f_ptrs[i] = ptr

		reshaped_ptrs = f_ptrs.reshape(*shape)

		#---------- Save to disk -------------

		# Save the g to disk
		np.save(g_save_path, self.all_hists)

		# Save the f to disk
		np.save(f_save_path, reshaped_f)

		# Save the ptr to disk
		np.save(ptr_save_path, reshaped_ptrs)

		# Save meta information
		df = pd.DataFrame({'rn': model.rn, 'sn': model.sn, 'gm': model.gamma, 'default_gamma': using_default_flag}, 
			index=[model.orf_name])
		df.to_csv(meta_save_path, float_format="%.4f")

		print(f"Saved to {f_save_path}...")
		print(f"Saved to {ptr_save_path}...")
		print(f"Saved to {meta_save_path}...")
		sys.stdout.flush()


	def deconvolve(self):
		"""
		Deconvolve the chromatin using the gene expression's deconvolution model class

		Currently a lot of deconvolution parameters and configuration
		are built into the Model class used for gene expression.

		TODO: We will want to generalize those so that we can subclass or have a parent
		class that the chromatin and gene expression can use separately
		for now we will just instantiate this model class for use for the chromatin 
		deconvolution.

		We will try to not use the deconv_model object externally, such that it will be easier to 
		refactor in the future
		"""
		from src.deconvolve_chromatin import deconvolve_chromatin
		from src.model import Model

		self.deconv_model = Model(self.config, self.gene_name, self.gamma)
		self.f, self.rn, self.sn = deconvolve_chromatin(self.deconv_model, self.deconv_hist)
