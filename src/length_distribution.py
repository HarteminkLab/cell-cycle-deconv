
import numpy as np
import pandas as pd

from src.timer import Timer
from matplotlib import pyplot as plt
from src.read_bam import read_mnase_bam
from src.figure_configs import FiguresConfig
from src.global_config import GlobalConstants
from src.mnase_10kb_loader import MNase10kbLoader
from src.utils import print_fl


class LengthDistributionCalculator:
	"""
	Load reads for each sample and chromosome and create a target length distribution that
	is the mean of the two replicates.
	"""

	def __init__(self, replicate):

		self.mnase_loader = MNase10kbLoader()
		self.replicate = replicate


	def load_distributions_for_chrom(self, chrom):

		chrom_reads = self.mnase_loader.load_mnase_data(self.replicate, chrom, (0, 251))
		if chrom == 12:
			chrom_reads = mask_chromosome_12_repetitive_region(chrom_reads)
		self.chrom_reads = chrom_reads

		# Compute the per sample length counts and distribution (normalized by
		# sample means)
		self.length_counts, self.lengths_dists = self.compute_length_dists()

		return self.length_counts, self.lengths_dists

	def compute_length_distributions_for_all_chroms(self):

		from src.timer import Timer
		timer = Timer()

		chroms = range(1, 17)

		all_length_dists = pd.DataFrame()
		all_length_counts = pd.DataFrame()

		for chrom in chroms:
			print_fl(f"{chrom}", end=",")
			length_counts, lengths_dists = self.load_distributions_for_chrom(chrom)
			length_counts['chrom'] = chrom
			lengths_dists['chrom'] = chrom

			all_length_counts = pd.concat([all_length_counts, length_counts])
			all_length_dists = pd.concat([all_length_dists, lengths_dists])
		timer.print_time()

		all_length_counts = all_length_counts.set_index(['chrom'], append=True).sort_index()
		all_length_dists = all_length_dists.set_index(['chrom'], append=True).sort_index()

		self.all_length_dists = all_length_dists
		self.all_length_counts = all_length_counts


	def create_rep_distribution(self):    
		rep_dist = self.all_length_counts.reset_index().mean()
		rep_dist = rep_dist[rep_dist.index[2:]]
		rep_dist = rep_dist/(rep_dist.mean()+1e-5)    
		return rep_dist


	def compute_length_dists(self):
		length_counts = self.chrom_reads.groupby(['sample', 
			'length']).count()[['start']]
		length_counts = length_counts.rename(columns={'start': 'count'})

		all_lengths = np.arange(0, 251)
		length_counts = length_counts.reset_index().pivot(index='sample', 
			columns='length', values='count')

		length_counts = length_counts.reindex(columns=all_lengths, fill_value=0)
		lengths_dists = pd.DataFrame(length_counts.values / length_counts.mean(axis=1).values[:, None],
									index=length_counts.index,
									columns=length_counts.columns)
		return length_counts, lengths_dists


def mask_chromosome_12_repetitive_region(chrom_reads):

	# Mask chromosome 12, rDNA region. Rough estimate of region
	# but for the purposes of normalization, this appears appropriate.
	mask_span = 450000, 470000
	masked_chrom_12_reads = chrom_reads
	masked_chrom_12_reads = masked_chrom_12_reads[(masked_chrom_12_reads.mid < mask_span[0]) | 
									(masked_chrom_12_reads.mid > mask_span[1])]
	chrom_reads = masked_chrom_12_reads

	return chrom_reads


def plot_fragment_length_distributions(all_length_dists1, all_length_dists2):
	
	def plot_distribution_per_sample(all_length_dists):
		len_counts_df = all_length_dists.groupby('sample').mean()
		cmap = plt.get_cmap('inferno_r')
		colors = [cmap(i/(len(len_counts_df))) for i in range(len(len_counts_df))]

		for i in range(len(len_counts_df)):
			row = len_counts_df.iloc[i]
			plt.plot(row, color=colors[i], label=f"{row.name} min")
		plt.legend(ncol=2)
		plt.xlabel("Fragment length")

	plt.figure(figsize=(13, 4))
	plt.subplot(1, 2, 1)
	plot_distribution_per_sample(all_length_dists1)
	plt.title("Replicate 1", fontsize=16)

	plt.subplot(1, 2, 2)
	plot_distribution_per_sample(all_length_dists2)
	plt.title("Replicate 2", fontsize=16)

	plt.suptitle("Unnormalized fragment length distributions", fontsize=24)
	plt.subplots_adjust(top=0.8)

