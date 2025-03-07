
import numpy as np
import pandas as pd

from src.timer import Timer
from matplotlib import pyplot as plt
from src.read_bam import read_mnase_bam
from src.figure_configs import FiguresConfig


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
		self.geneset = pd.read_csv('data/reference_data/geneset_nondub_w_prom_genebodies.csv')\
			.set_index('orf_name')
		self.chroms = np.arange(1, 17)

	def set_chrom(self, chrom, replicate):
		chrom_reads = pd.read_hdf(f'output/mnase/yl_rep{replicate}_mnase_reads/yl_rep{replicate}_mnase_reads_chr{chrom}.h5', 
					'mnase_data')
		chrom_genes = self.geneset[self.geneset['chr'] == chrom]

		if chrom == 12:
			# Mask chromosome 12, rDNA region. Rough estimate of region
			# but for the purposes of normalization, this appears appropriate.
			mask_span = 450000, 470000
			masked_chrom_12_reads = chrom_reads
			masked_chrom_12_reads = masked_chrom_12_reads[(masked_chrom_12_reads.mid < mask_span[0]) | 
											(masked_chrom_12_reads.mid > mask_span[1])]
			chrom_reads = masked_chrom_12_reads

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

	def compute_length_hist(self, sample, len_span=(0, 250)):
		mnase_reads = self.chrom_reads
		sample_reads = mnase_reads[mnase_reads['sample'] == sample]

		# Add 2 because we want to include the last length count, and 1 more to create bins that:
		# encompass the length read:
		#
		#   bins:        |       ...           |             |
		#                mn_min                mn_max    mn_max+1
		#
		return self.compute_length_hist_reads(sample_reads, len_span)

	def compute_length_hist_reads(self, sample_reads, len_span=(0, 250)):

		mn_min, mn_max = len_span

		# Range add one to include last value, add one more to create bin edge greater than last value:
		bins = np.arange(mn_min, mn_max+1+1)
		counts, bins = np.histogram(sample_reads['length'], bins=bins)
		self.counts, self.bins = counts, bins

		return bins[:-1], counts


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

# def yl_replicate_length_bins_doubled():
# 	"""Predifined bins for length fragments"""

# 	return [
# 		50, 100, 122, 145, 170, 195, 225
# 	]

# def yl_replicate_length_bins():
# 	"""Predifined bins for length fragments"""
# 	return [
# 		50, 100, 145, 195
# 	]

def fragment_lengths_definitions():
	"""Length spans as defined from the replicate 2 dataset,
	these should also match replicate 1
	"""
	nucleosome_len_span=(144, 192)
	mid_frag_span=(96, 144)
	small_frag_span=(0, 96)
	return small_frag_span, mid_frag_span, nucleosome_len_span


def len_bins():
	from src.global_config import GlobalConstants
	small_lens, med_lens, nuc_lens = fragment_lengths_definitions()
		
	small_bins = small_lens[0]//GlobalConstants.BIN_HEIGHT,  \
		small_lens[1]//GlobalConstants.BIN_HEIGHT
	med_bins = med_lens[0]//GlobalConstants.BIN_HEIGHT, \
		med_lens[1]//GlobalConstants.BIN_HEIGHT
	nuc_bins = nuc_lens[0]//GlobalConstants.BIN_HEIGHT, \
		nuc_lens[1]//GlobalConstants.BIN_HEIGHT

	return small_bins, med_bins, nuc_bins

def plot_len_counts(len_counts_df, show_len_cutoffs=True, title=None):
	
	small_frag_span, mid_frag_span, nucleosome_len_span = fragment_lengths_definitions()
	len_counts_df = len_counts_df.copy().T
	cmap = plt.get_cmap('viridis')
	colors = [cmap(i/(len(len_counts_df))) for i in range(len(len_counts_df))]

	for i in range(len(len_counts_df)):
		row = len_counts_df.iloc[i]
		plt.plot(row, color=colors[i], label=f"{row.name} min")
		
	plt.legend(ncol=2)
	plt.xlim(30, 250)
	
	og_max_ylim = plt.ylim()[1]
	og_ylim = -og_max_ylim*0.1, og_max_ylim

	y = -og_max_ylim*0.025
	plt.xlabel("Fragment length, nt")
	plt.ylabel("Frequency")

	plt.ylim(*og_ylim)

	if show_len_cutoffs:
		for len_spans in [small_frag_span, mid_frag_span, nucleosome_len_span]:
			plt.axvline(len_spans[1], linestyle='dotted', c='red')

		plt.text(small_frag_span[1]//2+25, y, "Small\nfragments", ha='center', va='bottom',
				 color='red', fontsize=12)
		plt.text((mid_frag_span[1]+mid_frag_span[0])//2, y, "Mid\nlength\nfragments", 
				 ha='center', va='bottom', 
				 color='red', fontsize=12)
		plt.text((nucleosome_len_span[1]+nucleosome_len_span[0])//2, y,
				 "Nucleosome\nlength\nfragments", ha='center', va='bottom',
				 color='red', fontsize=12)

	plt.title(title, fontsize=FiguresConfig.FIG_SUPTITLE_FONTSIZE, pad=15)


def compute_scaling_matrix(tp_counts):
	# Compute the minimum fragment occupancy for each length 
	# (what time had the fewest number of reads for 
	# a given fragment length)

	# Ensure that the total distribution is mean centered
	# around 1
	len_counts_df = tp_counts.copy().T
	len_counts_df = len_counts_df/len_counts_df.mean().mean()

	# After min length computation, again center around 1
	min_counts_per_len = len_counts_df.min(axis=0)
	min_counts_per_len = min_counts_per_len/min_counts_per_len.mean()

	# Compute the scaling matrix
	min_len_scaling_matrix = min_counts_per_len / len_counts_df
	min_len_scaling_matrix = min_len_scaling_matrix.fillna(0)

	return min_len_scaling_matrix.T


def convert_scaling_matrix_to_len_bins(min_len_scaling_matrix):
	
	min_len_scaling_matrix = min_len_scaling_matrix.T
	small_len, mid_len, nuc_len = fragment_lengths_definitions()
	subset_scaling_mat = min_len_scaling_matrix[[]].copy()

	def _add_subset_by_len(subset_scaling_mat, len_span):
		subset_lens = min_len_scaling_matrix[np.arange(*len_span)]
		subset_mean_scales = subset_lens.mean(axis=1)
		subset_scaling_mat[str(len_span)] = subset_mean_scales
		return subset_scaling_mat

	_add_subset_by_len(subset_scaling_mat, small_len)
	_add_subset_by_len(subset_scaling_mat, mid_len)
	_add_subset_by_len(subset_scaling_mat, nuc_len)

	return subset_scaling_mat
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


