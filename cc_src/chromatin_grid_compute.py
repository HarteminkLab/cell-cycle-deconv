
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from cc_src.sgd import get_gene_name_orf_name
from cc_src.mnase_plotting import plot_mnase_density


class ChromatinGrid:
	"""
	In this class, we will be taking mnase-seq reads for a gene, and computing a grid of occupancy 
	values for the gene's locus.

	That we can then deconvolve.

	We will be able to plot the gene's locus (raw data) as well as the histogram of the grid for 
	verification that the grid values
	are created properly.

	Notes: We will eventually need to normalize or scale (by copy number and/or by sample depth)
	"""


	def __init__(self):

		self.padding = 1000
		self.geneset = pd.read_csv('data/reference_data/geneset_nondub_w_prom_genebodies.csv').set_index('orf_name')

	def set_gene(self, gene_name):

		# Get some gene information
		self.computed_plus_one = None
		self.orf_name, self.gene_name = get_gene_name_orf_name(gene_name)
		self.gene = self.geneset.loc[self.orf_name]

		self.chr_reads = pd.read_hdf(f'output/mnase/yl_rep2_mnase_reads/yl_rep2_mnase_reads_chr{self.gene.chr}.h5', 
					'mnase_data')

		self.mnase_span = self.gene.TSS-self.padding, self.gene.TSS+self.padding

		self.gene_reads = self.chr_reads[(self.chr_reads.mid > self.mnase_span[0]) & 
			(self.chr_reads.mid < self.mnase_span[1])]

		self.times = self.gene_reads['sample'].unique()
		self.find_max_plusOne_pos()


		# Now that we have the +1 position defined, let's realign on this position
		#
		# TODO: Note, that we may run into some weird behavior if the read counts are very low for nucleosome fragments around
		# the TSS, in that case, we will need a back up plan... maybe a minimum threshold for this procedure...
		self.mnase_span = self.computed_plus_one-self.padding, self.computed_plus_one+self.padding
		self.gene_reads = self.chr_reads[(self.chr_reads.mid > self.mnase_span[0]) & 
			(self.chr_reads.mid < self.mnase_span[1])]


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

		from cc_src.chromatin_metrics import yl_rep2_len_spans

		x_bin_size = 80
		y_bin_size = 50

		num_promoter_bins = 3
		num_gb_bins = 5
		num_bins = num_promoter_bins+num_gb_bins+1 # Plus one, because we are centered on a bin

		
		# Defined by the fragment lengths we previously defined
		small_lens, med_lens, nuc_lens = yl_rep2_len_spans()
		y_bins = [small_lens[0], small_lens[1], nuc_lens[0], nuc_lens[1]]
		
		
		# from the center we will 
		center = self.computed_plus_one

		if self.gene.strand == "+":
			x_start = center - x_bin_size//2 - num_promoter_bins*x_bin_size
			x_end = x_start+num_bins*x_bin_size
		else:
			x_start = center - x_bin_size//2 - num_gb_bins*x_bin_size
			x_end = x_start+num_bins*x_bin_size

		x_bins = np.arange(x_start, x_end+x_bin_size, x_bin_size)

		return x_bins, y_bins

		
	def compute_bin_counts_sample(self, sample):

		plotting_reads = self.gene_reads[self.gene_reads['sample'] == sample]

		x_bins, y_bins = self.define_histogram_bins()

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
		
		x_bins, y_bins = self.define_histogram_bins()

		xlims = self.mnase_span
		gene = self.gene

		plotting_reads = self.all_plotting_reads[sample]
		hist = self.all_hists[i]

		plot_mnase_density(ax1, plotting_reads)
		ax1.set_xticks([])

		# This is the plot of the grid, so the extents are inset
		ax2.imshow(hist, origin='lower', aspect='auto', cmap='magma_r',
			extent=[x_bins[0], x_bins[-1], 0, 225])

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
			grid_ax = grid_axes[i]
			self.plot_sample(raw_ax, grid_ax, time, i)


	def create_deconvolution_matrices(self, plot=False):

		hist = self.all_hists[0]
		times = self.times

		# So let's create the data structure for inner 1000 bp histogram for all time points:
		# Dimension: (num_time_points, rows, columns)

		inner_hist_shape = hist[:, 5:-5].shape
		threed_hist_matrix = np.zeros((len(times), inner_hist_shape[0], inner_hist_shape[1]))

		for i in range(len(times)):
			time = times[i]
			inner_hist = self.all_hists[time][:, 5:-5]
			threed_hist_matrix[i] = inner_hist

		self.threed_hist_matrix = threed_hist_matrix

		# Checking if we can collapse the rows and columns, then restore them
		reshaped_hist = threed_hist_matrix.reshape(15, -1)
		first_hist = reshaped_hist[0]

		if plot:
			print("The first histogram is shape:", threed_hist_matrix[0].shape)
			print("Reshaping this histogram to a vector of shape:", first_hist.shape)
			restored_hist = first_hist.reshape((3, 10))
			print("Then, if we were to take that first vector and restore it to its original shape:", 
				  restored_hist.shape)

			plt.subplot(1, 2, 1)
			plt.imshow(threed_hist_matrix[0], origin='lower', cmap='magma_r')
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
		deconv_hist = np.concatenate([reshaped_hist[:-4, :], reshaped_hist[-3:, :]])
		print("Now we have a data structure that we can try to deconvolve of shape:", 
			  deconv_hist.shape)

		# Reshape for deconvolution
		self.deconv_hist = deconv_hist

	def create_deconvolution_plots(self, f, model):
		from src.model import color_for_key

		reshaped_f = f.reshape(-1, 3, 10)
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

			im = ax.imshow(reshaped_f[i], origin='lower', cmap='magma_r', aspect='auto', vmax=500)
			ax.set_xticks([])
			ax.set_yticks([])
			ax.axvline(4.5, c=color, lw=2)

			if last_phase is not None and phase != last_phase:
				plotting_index += 2
			else:
				plotting_index += 1

			last_phase = phase

	def plot_prediction_comparison(self, model, f):
		times = self.times
		predicted_g = np.matmul(model.H, f)

		n = predicted_g.shape[0]

		predicted_g_reshaped = predicted_g.reshape(n, 3, 10)

		fig, axs = plt.subplots(n, 3, figsize=(7, 6))
		axs = np.array(axs).T
		g_axs = axs[0]
		pred_g_axs = axs[1]
		comparison_axs = axs[2]

		for i in range(n):
			time = times[i]

			g_ax = g_axs[i]
			im = g_ax.imshow(self.threed_hist_matrix[i], origin='lower', cmap='magma_r', 
						   aspect='auto', vmax=300)
			
			pred_ax = pred_g_axs[i]
			im = pred_ax.imshow(predicted_g_reshaped[i], origin='lower', cmap='magma_r', 
						   aspect='auto', vmax=300)

			comp_ax = comparison_axs[i]
			im = comp_ax.imshow(predicted_g_reshaped[i]-self.threed_hist_matrix[i], 
							origin='lower', cmap='RdBu', aspect='auto', vmin=-300, vmax=300)
			
			for ax in [g_ax, pred_ax, comp_ax]:
				ax.set_xticks([])
				ax.set_yticks([])
				ax.axvline(4.5, c='black', lw=2)




		g_axs[0].set_title("Original")
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
