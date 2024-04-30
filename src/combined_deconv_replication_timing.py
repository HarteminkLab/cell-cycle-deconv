


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.combined_chromatin_model import CombinedChromatinModel
from src.dynamic_config_alpha import create_dynamic_alpha_config
from src.geneset import get_deconvolved_geneset


class CombinedDeconvReplicationProfileAnalysis:
	"""Combined deconvolution replication timing analysis
	"""

	def __init__(self, deconv_rep_analysis1, deconv_rep_analysis2):
		self.deconv_rep_analysis1 = deconv_rep_analysis1
		self.deconv_rep_analysis2 = deconv_rep_analysis2

		config1 = create_dynamic_alpha_config(deconv_rep_analysis1.alpha, 1)
		config2 = create_dynamic_alpha_config(deconv_rep_analysis2.alpha, 2)

		combined_model = CombinedChromatinModel(config1, config2)
		combined_model.load_combined_mnase_gene('CLB2')

		self.combined_model = combined_model
		self.geneset = get_deconvolved_geneset()

	def set_chromosome(self, chrom):
		self.chrom = chrom
		self.deconv_rep_analysis1.create_chr_bin_curves(1, chrom)
		self.deconv_rep_analysis2.create_chr_bin_curves(2, chrom)

	def generate_g1_g2(self):

		# Create the data set of G1, G2 using the chromosome bin curves
		# Make sure both replicates have the same set of genes.

		# Combine the index sets and sort
		intersect_index_set = self.get_combined_index_set()

		# Add 1. for stability of deconvolution
		G1 = self.deconv_rep_analysis1.chr_bin_curves.loc[intersect_index_set].values.T
		G2 = self.deconv_rep_analysis2.chr_bin_curves.loc[intersect_index_set].values.T
		self.G1 = G1 + 1.
		self.G2 = G2 + 1.

	def get_combined_index_set(self):
		index_set1 = set(self.deconv_rep_analysis1.chr_bin_curves.index)
		index_set2 = set(self.deconv_rep_analysis2.chr_bin_curves.index)
		intersect_index_set = np.array(index_set1.intersection(index_set2))
		intersect_index_set = self.geneset.loc[intersect_index_set].sort_values('start').index
		return intersect_index_set

	def deconvolve_bin_curves(self, verbose=False):

		self.generate_g1_g2()
		self.gamma = 0.001
		self.combined_model.deconvolve(gamma=self.gamma, G1=self.G1, G2=self.G2, verbose=verbose)

	def compute_combined_replication_timing(self):

		from src.deconv_replication_timing import compute_replication_timing

		f = self.combined_model.f
		intersect_index_set = self.get_combined_index_set()

		combined_mother_repl_timing, \
		combined_daughter_repl_timing = compute_replication_timing(f, self.combined_model.chrom1_model.config, 
			intersect_index_set, threshold=0.9)

		self.combined_mother_repl_timing = combined_mother_repl_timing
		self.combined_daughter_repl_timing = combined_daughter_repl_timing

		combined_repl_timing_df = self.combined_mother_repl_timing.join(
			self.combined_daughter_repl_timing,
			lsuffix='_mother', rsuffix='_daughter'
		)
		combined_repl_timing_df['chr'] = self.chrom
		self.combined_repl_timing_df = combined_repl_timing_df


	def plot_replication_heatmap(self):

		config = self.combined_model.chrom1_model.config
		f = self.combined_model.f
				
		t_indices = config.get_Hpositions_for_branch('t')
		b_indices = config.get_Hpositions_for_branch('b')

		t_tps = config.get_timepoints_for_branch('t')
		b_tps = config.get_timepoints_for_branch('b')

		plt.figure(figsize=(13, 4))

		plt.subplot(2, 1, 1)
		plt.imshow(f[t_indices], origin='lower', 
				   vmin=1, vmax=3., aspect='auto',
				  extent=[0, f.shape[1], t_tps[0], t_tps[-1]])
		plt.plot(self.combined_mother_repl_timing.tp, c='red')
		plt.title("Mother")
		plt.xticks([])

		plt.subplot(2, 1, 2)
		plt.imshow(f[b_indices], origin='lower', vmin=1, vmax=3., aspect='auto',
				  extent=[0, f.shape[1], b_tps[0], b_tps[-1]])
		plt.plot(self.combined_daughter_repl_timing.tp, c='red')

		plt.title("Daughter")
		plt.xticks([])

		plt.suptitle(f"Deconvolved gene replication profile, chr{self.chrom}, combined model, "
			f"$\\alpha$={self.deconv_rep_analysis1.alpha,self.deconv_rep_analysis2.alpha}")

	def plot_replication_timing(self):
		
		from src.sgd import get_chromosome_length

		chrom = self.chrom
		
		config = self.combined_model.chrom1_model.config
		alpha = config.intervals_wt1[0][5]
		
		chrom_len = get_chromosome_length(chrom)
		repl_timing = self.combined_mother_repl_timing.join(self.geneset[['start']])
		
		n = len(repl_timing)
		xs = repl_timing.start
		
		plt.figure(figsize=(13, 2))
		plt.plot(xs, repl_timing.tp + alpha)
		plt.ylim(45, 10)
		plt.xlim(0, chrom_len)
		plt.title(f"Combined replicate timing, chr{chrom}")
		plt.xlabel("Chrom position, bp")
		plt.ylabel("Replication timing, min")
		
		
	def plot_raw_bin_curves(self):
		plt.figure(figsize=(13, 6))
		plt.subplot(2, 1, 1)
		plt.imshow(self.deconv_rep_analysis1.chr_bin_curves.T, 
				   aspect='auto', origin='lower', vmin=0, vmax=1.)
		plt.xticks([])
		plt.yticks([])

		plt.title(f"Chr{self.chrom}, Replicate 1, raw data")
		plt.subplot(2, 1, 2)
		plt.imshow(self.deconv_rep_analysis2.chr_bin_curves.T, 
				   aspect='auto', origin='lower', vmin=0, vmax=1.)
		plt.title(f"Chr{self.chrom}, Replicate 2, raw data")
		plt.xticks([])
		plt.yticks([])

	def save_results(self, directory):

		f_file = f"{directory}/chr{self.chrom}_f.npy"
		repl_file = f"{directory}/chr{self.chrom}_repl.csv"

		np.save(f_file, self.combined_model.f)
		self.combined_repl_timing_df.to_csv(repl_file)

		print(f"Saved {f_file}")
		print(f"Saved {repl_file}")
