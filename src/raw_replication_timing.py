
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class RawReplicationProfileAnalysis(object):

	def __init__(self, mnase_analysis_rep1, mnase_analysis_rep2):
		"""Compute the replication timing profiles from mnase seq occupancies.

		Deconvolve and compute the smoothed replication timing profiles after an alpha parameter search
		"""

		self.mnase_analysis_rep1 = mnase_analysis_rep1
		self.mnase_analysis_rep2 = mnase_analysis_rep2
		
		merged_replication_timing = pd.DataFrame(mnase_analysis_rep1.normalized_timing, columns=['repl_timing1']).join(
	    pd.DataFrame(mnase_analysis_rep2.normalized_timing, columns=['repl_timing2']))
		merged_replication_timing['mean'] = merged_replication_timing.mean(axis=1)
		self.merged_replication_timing = merged_replication_timing


	def plot_raw_replication_timing(self):
		"""Plot the raw replicating timing histogram"""

		mnase_analysis_rep1 = self.mnase_analysis_rep1
		mnase_analysis_rep2 = self.mnase_analysis_rep2

		bins = np.arange(0, 70, 5)-5

		plt.figure(figsize=(9, 3))
		plt.subplots_adjust(top=0.77)

		plt.subplot(1, 2, 1)
		plt.hist(mnase_analysis_rep1.repl_timing_df, bins=bins)

		def plot_lines(lines):
		    for line in lines:
		        plt.axvline(line, c='black', lw=1, ls='dotted')

		plot_lines([
		    mnase_analysis_rep1.g1_recovery_would_start_here,
		    mnase_analysis_rep1.first_s_start,
		    mnase_analysis_rep1.first_s_end,
		    mnase_analysis_rep1.end_of_first_lambd,
		])

		plt.xlim(-10, 70)
		plt.title("Replicate 1")
		plt.xlabel("Timing of replication, min")

		plt.subplot(1, 2, 2)
		plt.hist(mnase_analysis_rep2.repl_timing_df, bins=bins)

		plot_lines([
		    mnase_analysis_rep2.g1_recovery_would_start_here,
		    mnase_analysis_rep2.first_s_start,
		    mnase_analysis_rep2.first_s_end,
		    mnase_analysis_rep2.end_of_first_lambd,
		])

		plt.xlim(-10, 70)
		plt.title("Replicate 2")
		plt.xlabel("Timing of replication, min")

		plt.suptitle("Distribution of replication timing", fontsize=16)


	def plot_merged_raw_replication_timing(self):

		bins = np.arange(0, 1, 0.07)

		merged_replication_timing = self.merged_replication_timing

		plt.figure(figsize=(23, 4))
		plt.subplots_adjust(top=0.77, wspace=0.3)

		plt.subplot(1, 5, 1)
		plt.hist(merged_replication_timing.repl_timing1, bins=bins)
		plt.title("Replicate 1")
		plt.xlabel("Normalized timing")

		plt.subplot(1, 5, 2)
		plt.hist(merged_replication_timing.repl_timing2, bins=bins)
		plt.title("Replicate 2")
		plt.xlabel("Normalized timing")

		plt.subplot(1, 5, 3)
		plt.hist(merged_replication_timing['mean'], bins=bins)
		plt.title("Mean replicate 1 and 2")
		plt.xlabel("Normalized timing")

		plt.suptitle("Distribution of replication timing", fontsize=16)

		plt.subplot(1, 5, 4)
		from scipy.stats.distributions import norm

		n = len(merged_replication_timing)
		x_jitter = norm.rvs(loc=0, scale=0.005, size=n)
		y_jitter = norm.rvs(loc=0, scale=0.005, size=n)

		plt.scatter(merged_replication_timing.repl_timing1 + x_jitter, 
		            merged_replication_timing.repl_timing2 + y_jitter,
		           s=1, alpha=0.1)
		plt.xlabel("Replicate 1 timing")
		plt.ylabel("Replicate 2 timing")
		plt.title("Distribution comparison (with jitter)")

		plt.subplot(1, 5, 5)
		x = np.arange(n)
		plt.plot(merged_replication_timing.sort_values('mean')['mean'], x)
		plt.title("Genes sorted by replication timing")
		plt.xlabel("Mean normalized replication timing")
		plt.ylabel("Gene rank")
		plt.yticks([])

	def load_chromatin_images(self, chromatin_dir):

		# Load the F images, currently we have them ready to plot in the promoter analysis
		# but we will move them to its own class

		from src.promoter_ptr_analysis import PromoterPTRAnalysis

		promoter_analysis = PromoterPTRAnalysis(chromatin_dir)
		promoter_analysis.load_f_files()
		promoter_analysis.correct_f_images_by_strand()
		self.promoter_analysis = promoter_analysis


	def index_f_images_by_repl_timing(self):
		"""The f images are in a numpy array, so we will need a dataframe that contains
		the replication timing and gene index associated with the f images indexing"""

		promoter_analysis = self.promoter_analysis
		strand_corrected_f_images = promoter_analysis.strand_corrected_f_images

		# The f images are in a numpy array so keep track of the indices
		gene_indices = promoter_analysis.geneset[['gene']].copy()
		gene_indices['gene_index'] = np.arange(len(gene_indices))

		# Replication timing is a subset, so let's subset the f images
		merged_replication_timing = self.merged_replication_timing
		gene_indices = gene_indices.loc[merged_replication_timing.index]
		repl_timing = gene_indices.join(merged_replication_timing[['mean']])

		self.sc_f_imgs_with_rep = strand_corrected_f_images[gene_indices.gene_index.values]

		# Reset the gene index because we have now subsetted
		repl_timing['gene_index'] = np.arange(len(repl_timing))

		repl_timing = repl_timing.sort_values('mean')
		self.repl_timing = repl_timing


	def subset_raw_f_images(self, k=500):
		""

		# Now we have the indices and the replication timing, let's plot the 
		# top 500 genes in aggregate and the lower 500 genes

		self.k = k

		top_500_early = self.repl_timing.iloc[0:k]
		bottom_500_late = self.repl_timing.iloc[-k:]

		top_500_early_imgs_aggregated = self.sc_f_imgs_with_rep[top_500_early.gene_index].mean(axis=0)
		bottom_500_late_imgs_aggregated = self.sc_f_imgs_with_rep[bottom_500_late.gene_index].mean(axis=0)

		self.top_500_early_imgs_aggregated = top_500_early_imgs_aggregated
		self.bottom_500_late_imgs_aggregated = bottom_500_late_imgs_aggregated


	def plot_raw_subsets_replication_timing(self):
		# Plot CG1 and DG1 and PostG1
		# Compare top 500 and bottom 500 replication timing genes

		# Let's plot the aggreagete plots through various segments of the cell cycle
		# CG1:
		from src.config import load_yl_rg1_vst_config

		config = load_yl_rg1_vst_config(1)

		cg1_indices = config.get_Hpositions_for_phase('CG1')
		dg1_indices = config.get_Hpositions_for_phase('DG1')
		postg1_indices = config.get_Hpositions_for_phase('postG1')

		def subset_the_indices(indices, size):
		    subset_indices = np.linspace(0, len(indices)-1, size).astype(int)
		    return indices[subset_indices]

		n = 10
		k = self.k
		cg1_indices_subset = subset_the_indices(cg1_indices, n)
		dg1_indices_subset = subset_the_indices(dg1_indices, n)
		postg1_indices_subset = subset_the_indices(postg1_indices, n)

		def plot_phase_subset(aggregate_imgs1, aggregate_imgs2,
		                      subset_indices, phase):
		    phase_imgs1 = aggregate_imgs1[subset_indices]
		    phase_imgs2 = aggregate_imgs2[subset_indices]
		    n = len(phase_imgs1)

		    fig, axs = plt.subplots(n, 3, figsize=(5, 5))
		    plt.subplots_adjust(wspace=0.05)
		    
		    for i in range(n):
		        img1 = phase_imgs1[i]
		        img2 = phase_imgs2[i]
		        
		        ax_row = axs[i]
		        
		        ax = ax_row[0]
		        ax.imshow(img1, origin='lower', cmap='magma_r', vmax=20, aspect='auto')
		        ax.set_xticks([])
		        ax.set_yticks([])
		        if i == 0: ax.set_title(f"Early {k}")
		        
		        ax = ax_row[1]
		        ax.imshow(img2, origin='lower', cmap='magma_r', vmax=20, aspect='auto')
		        ax.set_xticks([])
		        ax.set_yticks([])
		        if i == 0: ax.set_title(f"Late {k}")
		            
		        ax = ax_row[2]
		        im = ax.imshow(img1-img2, origin='lower', cmap='RdBu_r', vmax=2, vmin=-2,
		                  aspect='auto')
		        ax.set_xticks([])
		        ax.set_yticks([])

		        if i == 0: ax.set_title(f"Difference\nEarly-Late")

		    plt.suptitle(phase)


		plot_phase_subset(self.top_500_early_imgs_aggregated, self.bottom_500_late_imgs_aggregated,
		                  cg1_indices_subset, phase="CG1")
		plot_phase_subset(self.top_500_early_imgs_aggregated, self.bottom_500_late_imgs_aggregated,
		                  dg1_indices_subset, phase="DG1")
		plot_phase_subset(self.top_500_early_imgs_aggregated, self.bottom_500_late_imgs_aggregated,
		                  postg1_indices_subset, phase="PostG1")



		fig, axs = plt.subplots(1, 3, figsize=(9, 2))

		indices = np.concatenate([cg1_indices, dg1_indices, postg1_indices])
		all_top_values = self.top_500_early_imgs_aggregated[indices].sum(axis=1).sum(axis=1)
		all_bottom_values = self.bottom_500_late_imgs_aggregated[indices].sum(axis=1).sum(axis=1)

		all_values = np.concatenate([all_top_values, all_bottom_values])

		ymin = all_values.min()
		ymax = all_values.max()
		ydelta = ymax-ymin
		ylim = ymin-ydelta*0.15, ymax+ydelta*0.15

		def plot_phase_sum(ax, indices, phase):
		    ax.plot(self.top_500_early_imgs_aggregated[indices].sum(axis=1).sum(axis=1), 
		            label=f"Early {k}", c='red')
		    ax.plot(self.bottom_500_late_imgs_aggregated[indices].sum(axis=1).sum(axis=1),
		            label=f"Late {k}",
		           c='blue')
		    ax.set_title(phase)
		    ax.legend()
		    ax.set_ylim((*ylim))
		    ax.set_xticks([])
		    ax.set_yticks([])
		    
		plot_phase_sum(axs[0], cg1_indices, 'CG1')
		plot_phase_sum(axs[1], dg1_indices, 'DG1')
		plot_phase_sum(axs[2], postg1_indices, 'PostG1')
		plt.suptitle("Early vs Late replication, total read counts in gene windows")
		plt.subplots_adjust(top=0.7, wspace=0.15)