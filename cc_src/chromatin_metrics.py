
import numpy as np
import pandas as pd

from src.timer import Timer
from matplotlib import pyplot as plt
from cc_src.read_bam import read_mnase_bam


class ChromatinMetrics:
	"""
	Class to read in an MNase-seq bam file and collect the read counts for each gene's 
	promoter and gene body small fragment and nucleosomal size reads
	"""

	def __init__(self, nucleosome_len_span, mid_frag_span, small_frag_span):
		self.timer = Timer()
		self.nucleosome_len_span = nucleosome_len_span
		self.mid_frag_span = mid_frag_span
		self.small_frag_span = small_frag_span
		self.geneset = pd.read_csv('data/geneset_nondub_w_prom_genebodies.csv')\
			.set_index('orf_name')
		self.chroms = np.arange(1, 17)

	def read_bam_file(self, filename, sample_name):
		self.sample_name = sample_name
		self.mnase_reads = read_mnase_bam(filename, sample=sample_name, 
			timer=self.timer, chroms=self.chroms)

	def set_chrom(self, chrom):
		chrom_reads = self.mnase_reads[self.mnase_reads['chr'] == chrom]
		chrom_genes = self.geneset[self.geneset['chr'] == chrom]
		self.chrom_reads = chrom_reads
		self.chrom_genes = chrom_genes

	def plot_len_dist(self):

		# Define nucleosome reads
		# Define small fragment reads
		plt.figure(figsize=(7, 3))
		plt.hist(self.mnase_reads['length'], bins=40)

		plt.axvline(self.nucleosome_len_span[0], c='red')
		plt.axvline(self.nucleosome_len_span[1], c='red')

		plt.axvline(self.mid_frag_span[0], c='orange')
		plt.axvline(self.mid_frag_span[1], c='orange')

		plt.axvline(self.small_frag_span[0], c='blue')
		plt.axvline(self.small_frag_span[1], c='blue')


	def mnase_sel_region_lens(self, mnase_dat, span, len_span=[0, 250]):

		mids = mnase_dat['mid']
		lens = mnase_dat['length']

		cur_mnase = mnase_dat[(mids >= span[0]) & 
				  (mids < span[1]) & 
				  (lens >= len_span[0]) & 
				  (lens < len_span[1])]

		return cur_mnase


	def compute_counts(self, gene, log=False):
		"""Get the reads for every partition of the gene
		(fragment length and position). Then, return the 
		number of reads in each partition.
		"""

		ret = self.get_partitions_reads(gene)
		(small_prom_reads, mid_prom_reads, nuc_prom_reads,
		 small_gb_reads, mid_gb_reads,nuc_gb_reads) = ret

		ret = (len(small_prom_reads), 
			  len(mid_prom_reads),
			  len(nuc_prom_reads),
			  len(small_gb_reads), 
			  len(mid_gb_reads),
			  len(nuc_gb_reads))

		if log:
			print("Occupancy counts for: ", gene['gene'])
			print("  Promoter small: {}\n"\
				  "  Promoter mid: {}\n"\
				  "  Promoter nuc: {}\n"\
				  "  Gene body small: {}\n"\
				  "  Gene body mid: {}\n"\
				  "  Gene body nuc: {}".format(*ret))

		return ret


	def compute_gene_entropies(self, gene):
		"""Compute the entropy scores for the different segmented regions"""

		gene_promoter_span = gene.promoter_start, gene.promoter_end
		gene_body_span = gene.gene_body_start, gene.gene_body_end

		small_prom_entropy = self.compute_partition_entropy(gene_promoter_span, 
			self.small_frag_span)
		nuc_gb_entropy = self.compute_partition_entropy(gene_body_span, self.nucleosome_len_span)

		return small_prom_entropy, nuc_gb_entropy


	def get_partitions_reads(self, gene):

		chrom_reads = self.chrom_reads

		gene_promoter_span = gene.promoter_start, gene.promoter_end
		gene_body_span = gene.gene_body_start, gene.gene_body_end

		small_prom_reads = self.mnase_sel_region_lens(chrom_reads, 
																   gene_promoter_span,
																   self.small_frag_span)

		mid_prom_reads = self.mnase_sel_region_lens(chrom_reads, 
																   gene_promoter_span,
																   self.mid_frag_span)

		nuc_prom_reads = self.mnase_sel_region_lens(chrom_reads, 
																   gene_promoter_span,
																   self.nucleosome_len_span)

		small_gb_reads = self.mnase_sel_region_lens(chrom_reads, 
																   gene_body_span,
																   self.small_frag_span)

		mid_gb_reads = self.mnase_sel_region_lens(chrom_reads, 
																   gene_body_span,
																   self.mid_frag_span)

		nuc_gb_reads = self.mnase_sel_region_lens(chrom_reads, 
																   gene_body_span,
																   self.nucleosome_len_span)

		ret = (small_prom_reads, mid_prom_reads, nuc_prom_reads,
			   small_gb_reads, mid_gb_reads,nuc_gb_reads)

		return ret


	def plot_mnase_gene(self, gene, annotate_occupancies=False, annotate_entropies=False):
		"""
		Plot the MNase-seq for a gene
		"""

		gene_window = gene['TSS']-1000, gene['TSS']+1000
		
		fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 4))
		
		from src.orf_plotter import ORFAnnotationPlotter, plot_rect

		orf_plotter = ORFAnnotationPlotter(self.geneset)
		orf_plotter.set_span_chrom(gene_window, 1)
		orf_plotter.plot_orf_annotations(ax1)
		
		gene_window_reads = self.mnase_sel_region_lens(self.chrom_reads, 
																   gene_window)

		ax2.scatter(gene_window_reads['mid'], gene_window_reads['length'], s=1, alpha=0.25, c='orange')

		
		prom_span = gene.promoter_start, gene.promoter_end
		gb_span = gene.gene_body_start, gene.gene_body_end

		for ax in [ax1, ax2]:
			ax.axvline(gene.TSS, c='black', lw=1, alpha=0.5)    
			plot_rect(ax, prom_span[0], -250, prom_span[1]-prom_span[0], 500, zorder=0, 
				color='#eef', fill_alpha=0.5)
			plot_rect(ax, gb_span[0], -250, gb_span[1]-gb_span[0], 500, zorder=0, 
				color='#eee', fill_alpha=0.5)

		# annotate the counts for each region
		if annotate_occupancies:

			# If plotting text for annotations, here are the x positions
			# for each region
			xs = [
				(prom_span[0]+prom_span[1])/2,
				(prom_span[0]+prom_span[1])/2,
				(prom_span[0]+prom_span[1])/2,
				(gb_span[0]+gb_span[1])/2,
				(gb_span[0]+gb_span[1])/2,
				(gb_span[0]+gb_span[1])/2,
			]

			gene_counts = self.compute_counts(gene, log=True)

			# Define the y position for where the counts will be placed
			# TODO: hard-coded based on yl2 data
			ys = [
				50, 120, 167,
				50, 120, 167,
			]

			for i in range(len(gene_counts)):
				gene_count = gene_counts[i]
				plt.text(xs[i], ys[i], str(gene_count), fontsize=12, weight='normal', 
				color='black', va='center', ha='center', zorder=10)

		if annotate_entropies:

			# If plotting text for annotations, here are the x positions
			# for each region
			xs = [
				(prom_span[0]+prom_span[1])/2,
				(gb_span[0]+gb_span[1])/2,
			]

			prom_sm_entropy, gb_nuc_entropy = self.compute_gene_entropies(gene)

			# Define the y position for where the counts will be placed
			# TODO: hard-coded based on yl2 data
			ys = [
				50, 167,
			]

			entropies = [prom_sm_entropy, gb_nuc_entropy]

			for i in range(len(entropies)):
				cur_entropy = entropies[i]
				plt.text(xs[i], ys[i], f"{cur_entropy:.02f}", fontsize=12, weight='normal', 
				color='black', va='center', ha='center', zorder=10)

		ax2.set_ylim(0, 250)
		ax2.set_xlim(gene_window[0], gene_window[1])
		ax1.set_xlim(gene_window[0], gene_window[1])
		ax1.set_xticks([])
		
		ax2.axhline(self.nucleosome_len_span[0], c='red', lw=1, ls='dotted', alpha=0.5)
		ax2.axhline(self.nucleosome_len_span[1], c='red', lw=1, ls='dotted', alpha=0.5)
		ax2.axhline(self.small_frag_span[1], c='green', lw=1, ls='dotted', alpha=0.5)
		plt.suptitle(f"{gene.gene} - chr {gene.chr}: {gene_window[0]}-{gene_window[1]}")

	def compute_all_gene_entropies(self):
		"""Compute the gene entropies for all chromosomes for the current gene set (chromosome)"""

		# Create a gene counts data frame that will collect the entropies for each gene
		# at the time point
		gene_entropies = self.geneset[['chr']].copy()

		gene_entropies['prom_sm_occ'] = 0
		gene_entropies['gb_nuc_occ'] = 0

		self.timer.start()

		# For each chromosome
		print(f"Computing gene entropies for all chromosomes...")
		for chrom in self.chroms:    

			# Set the chromosome number to filter the mnase reads and the genes
			self.set_chrom(chrom)    
			for orf_name, gene_row in self.chrom_genes.iterrows():
				
				# Compute the counts and set them in the data frame
				(entropy_p_sm, entropy_gb_nuc) = self.compute_gene_entropies(gene_row)
				
				gene_entropies.loc[orf_name, 'prom_sm_occ'] = entropy_p_sm
				gene_entropies.loc[orf_name, 'gb_nuc_occ'] = entropy_gb_nuc

			# Progress logging
			print(f"Done with chromosome {chrom} - {self.timer.get_time()}")

		self.gene_entropies = gene_entropies
		return gene_entropies


	def compute_all_gene_counts(self):

		# Create a gene counts data frame that will collect the occupancy for each gene
		# at the time point
		gene_counts = self.geneset[['chr']].copy()

		gene_counts['prom_sm_occ'] = 0
		gene_counts['prom_mid_occ'] = 0
		gene_counts['prom_nuc_occ'] = 0

		gene_counts['gb_sm_occ'] = 0
		gene_counts['gb_mid_occ'] = 0
		gene_counts['gb_nuc_occ'] = 0

		self.timer.start()

		# For each chromosome
		print(f"Computing metrics...")
		for chrom in self.chroms:    

			# Set the chromosome number to filter the mnase reads and the genes
			self.set_chrom(chrom)    
			for orf_name, gene_row in self.chrom_genes.iterrows():
				
				# Compute the counts and set them in the data frame
				(num_p_sm, num_p_mid, num_p_nuc, 
				 num_gb_sm, num_gb_mid, num_gb_nuc) = self.compute_counts(gene_row, log=False)
				
				gene_counts.loc[orf_name, 'prom_sm_occ'] = num_p_sm
				gene_counts.loc[orf_name, 'prom_mid_occ'] = num_p_mid
				gene_counts.loc[orf_name, 'prom_nuc_occ'] = num_p_nuc
				
				gene_counts.loc[orf_name, 'gb_sm_occ'] = num_gb_sm
				gene_counts.loc[orf_name, 'gb_mid_occ'] = num_gb_mid
				gene_counts.loc[orf_name, 'gb_nuc_occ'] = num_gb_nuc

			# Progress logging
			print(f"Done with chromosome {chrom} - {self.timer.get_time()}")

		self.gene_counts = gene_counts
		return gene_counts

	def compute_length_hist(self):
		mnase_reads = self.mnase_reads
		mn_min, mn_max = mnase_reads['length'].min(), mnase_reads['length'].max()
		print(f"The minimum length of reads is {mn_min}, the maximal length is {mn_max}.")

		# Add 2 because we want to include the last length count, and 1 more to create bins that:
		# encompass the length read:
		#
		#   bins:        |       ...           |             |
		#                mn_min                mn_max    mn_max+1
		#

		# Range add one to include last value, add one more to create bin edge greater than last value:
		bins = np.arange(mn_min, mn_max+1+1)
		counts, bins = np.histogram(mnase_reads['length'], bins=bins)
		self.counts, self.bins = counts, bins


	def compute_partition_entropy(self, genome_span, len_span):
		"""Compute the entropy for the gene's len and position span"""
		
		# Get the relevant reads
		reads = self.mnase_sel_region_lens(self.chrom_reads, genome_span,
			len_span)

		# Define the bins from the given spans
		genome_bins = np.arange(genome_span[0], genome_span[1]+2)
		len_bins = np.arange(len_span[0], len_span[1]+2)

		# Create a 2D histogram of counts for the bins, flatten to
		# 1D
		x = reads['mid']
		y = reads['length']
		hist, x_edges, y_edges = np.histogram2d(x, y, 
			bins=(genome_bins, len_bins))
		
		# then compute and return the entropy
		entropy = compute_entropy(hist.flatten())
		return entropy


def compute_entropy(flat_data):
	"""Compute the entropy of flattened 1D array"""
	
	# Compute the probability distribution
	unique_elements, counts = np.unique(flat_data, return_counts=True)
	probabilities = counts / len(flat_data)

	# Calculate entropy
	entropy = -np.sum(probabilities * np.log2(probabilities))

	return entropy



def yl_replicate_length_bins_doubled():
	"""Predifined bins for length fragments"""

	return [
		50, 100, 122, 145, 170, 195, 225
	]

def yl_replicate_length_bins():
	"""Predifined bins for length fragments"""

	return [
		50, 100, 145, 195
	]


def yl_rep2_len_spans():
	nucleosome_len_span=(145, 195)
	mid_frag_span=(100, 145)
	small_frag_span=(0, 100)
	return small_frag_span, mid_frag_span, nucleosome_len_span

	
def make_matrix_pvt_df(all_gene_chrom_occs, chrom_col_name):
    # Make a pivot table (matrix) where the rows are each gene,
    # the columns are the time points, and the values of each cell are one of the measures
    pvt = all_gene_chrom_occs[[chrom_col_name]].reset_index().pivot_table(index='orf_name', 
        columns='time', values=chrom_col_name)
    return pvt


# Normalize the occupancy counts
def normalize_length_dist_pvt(prom_sm_occ_pvt, scaling_counts_df):
    """
    Normalize the fragment counts using the precomputed per lengths scaling factors
    """
    prom_sm_occ_pvt_normed = prom_sm_occ_pvt.copy()

    # Select the specific span of lengths we want to scale
    # For example if we want small fragments (0-100), we are going to retrieve these columns
    # from the scaling matrix and take the average scaling value to apply to fragment 
    # occupancy counts for small fragments for each time point
    small_span = list(chromatin_metrics.small_frag_span)
    small_span[0] = max(small_span[0], min(scaling_counts_df.columns))

    # The scaling terms for each time point for given length spans
    scaling_term_per_time = scaling_counts_df[range(small_span[0], 
                                                    small_span[1])].mean(axis=1)

    for gene in prom_sm_occ_pvt.index:
        prom_sm_occ_pvt_normed.loc[gene] = prom_sm_occ_pvt.loc[gene] * \
        scaling_term_per_time

    return prom_sm_occ_pvt_normed


def create_save_deconvolution_datasets(counts_df, key, save_filename):

    # Now we need to make this formatted for how Matlab expects the input data to look
    counts_df.to_csv(save_filename, sep='\t', index=False, 
                                              header=False)
    # A table with just the orf names, one per line
    counts_df.reset_index()[['orf_name']].to_csv('output/orf_names.csv', 
        index=False, header=False)

    # And a list of the time points that we will put into the config file
    counts_df.iloc[[]].to_csv('output/times.csv', index=False)

    print(f"Saved {save_filename}")

