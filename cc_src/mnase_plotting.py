
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from cc_src.mnase_reads import load_mnase_reads


class MNasePlotter:

	def __init__(self, gene_window_padding_2):

		# TODO: hard-coded...
		self.times = [0, 10,  20,  30,  40,  50,  60,  70,  80,  90, 100, 110, 120, 130, 140]
		self.gene_window_padding_2 = gene_window_padding_2

	def set_chrom(self, chr):

		# Load all the MNase reads for a chromosome
		self.chr = chr
		self.chr_mnase_reads = load_mnase_reads(chr)

	def set_gene(self, gene, sample=True):
		"""Set the gene, window and MNase-seq reads in the window"""

		chr_mnase_reads = self.chr_mnase_reads
		self.gene = gene
		self.gene_window = self.gene['TSS']-self.gene_window_padding_2, \
			self.gene['TSS']+self.gene_window_padding_2
		self.gene_reads = chr_mnase_reads[(chr_mnase_reads['mid'] > self.gene_window[0]) & 
			(chr_mnase_reads['mid'] < self.gene_window[1])]
		self.sample = sample

		# Normalize the read plots
		if sample:
			self.sample_yl_rep2_gene_reads()
			self.plot_reads = self.sampled_reads
		else:
			self.plot_reads = self.gene_reads


	def plot(self, axs=None):
		"""Plot the MNase-seq for all time points for the gene"""

		# TODO: normalize precomputed length scalar from previous notebook
		# TODO: Set Vertical spacing to 0?
		# TODO: Maybe coloring based on length as well, to make it easier to
		#  dilineate nucleosome and small fragments
		# TODO: Density color for mnase mid points

		times = self.times

		if axs is None:
			fig, axs = plt.subplots(len(times), 1, figsize=(6, 13))
			axs = list(np.array(axs).flatten())

		for i in range(len(times)):
			
			ax = axs[i]
			time = times[i]

			time_plot_reads = self.plot_reads[self.plot_reads['sample'] == time]

			if len(time_plot_reads) > 1000:
				# Plot the reads as a colored density scatter plot
				plot_mnase_density(ax, time_plot_reads)

			ax.set_xlim(*self.gene_window)
			ax.set_yticks([])
			ax.set_xticks([])
			ax.set_ylabel(f"{time}'", rotation=0, ha='right')

	def sample_yl_rep2_gene_reads(self):
		"""Sample the gene reads by the new counts, computed from compute_scaled_counts_per_len
		length counts will be equalish per sample. (Normalized to have equivalent-ish length
		distributions)"""
		
		# --------------- Compute the new counts based on the len,time scaling matrix --------------

		gene_reads = self.gene_reads

		# Load the per time and  length time MNase-seq scaling terms
		mnase_len_time_scaling_mat = pd.read_csv('output/yl_rep2_mnase_min_len_scaling_mat.csv')\
			.set_index('time')
		mnase_len_time_scaling_mat.columns = mnase_len_time_scaling_mat.columns.astype(int)

		# Get the chromosome MNase-seq reads
		len_time_counts = gene_reads.groupby(['length', 'sample']).count()\
		.rename(columns={'start': 'count'})[['count']]

		# The length and time combinations
		lens = sorted(list(gene_reads.length.unique()))
		times = self.times

		# Create a new counts array
		new_counts = len_time_counts.copy()

		# Compute the new scaling term
		for time in times:
			for length in lens:
				scaling_term = mnase_len_time_scaling_mat.loc[time][length]

				try:
					cur_time_len_count = new_counts.loc[length].loc[time]
				except KeyError:
					# Length sample combination does not exist, skip it
					continue

				new_scaled_count = (int(cur_time_len_count.iloc[0]) * scaling_term)
				new_counts.loc[length].loc[time] = new_scaled_count

		self.sampled_counts = new_counts

		if len(gene_reads) == 0:
			self.sampled_reads = gene_reads
			return
				
		# ----------------- Do the sampling -----------------------
		
		sampled_reads = pd.DataFrame()

		for length in lens:
			for time in times:

				cur_len_time_reads = gene_reads[(gene_reads['length'] == length) & 
											 (gene_reads['sample'] == time)]

				try:
					sample_count = new_counts.loc[length].loc[time].values[0]
				except KeyError:
					# Skip combination, DNE
					continue

				cur_sampled_reads = cur_len_time_reads.sample(sample_count, replace=False)  
				sampled_reads = pd.concat([sampled_reads, cur_sampled_reads])

		self.sampled_reads = sampled_reads


def plot_mnase_density(ax, mnase_reads_df, vmax=4e-5, bw=[5, 10]):
	"""
	Function to plot mnase reads by midpoint and length colored by density of reads
	"""

	# Get the x and y values from the pandas dataframe
	x = mnase_reads_df.mid.values
	y = mnase_reads_df['length'].values

	# Compute the color using a 2-dimensional kernel density estimator
	kde = sm.nonparametric.KDEMultivariate(data=[x, y], var_type='cc', bw=bw)
	z = kde.pdf([x, y])

	# Plot reads colored by the density calculation
	ax.scatter(x, y, c=z, s=1, cmap='magma_r', vmax=vmax)

