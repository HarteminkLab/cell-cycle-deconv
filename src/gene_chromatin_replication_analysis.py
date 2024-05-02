
import cvxpy
import pywt

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from src.utils import print_fl
from src.geneset import get_deconvolved_geneset
from src.config import load_yl_rg1_vst_config


class GeneChromatinReplicationAnalysis:
	"""
	Analysis to examine chromatin pre/post replication fork passing through.

	Goal: Make it easy to examine different subsets of genes at different time deltas
	before and after the replication.

	To identify potential effects of the replication fork in the chromatin at genes (expressed/unexpressed), transcribing
	with and towards the replication fork, etc.
	"""

	def __init__(self, promoter_analysis, ge_analysis, repl_timing):

		repl_timing = repl_timing.dropna()

		# Using the timepoint values from config 1
		self.config = load_yl_rg1_vst_config(1)

		geneset = get_deconvolved_geneset()

		# Now we will need to index the geneset so we can subset the f images properly
		# Replication timing profiles were computed for a subset ~5200 genes
		# So we will need to keep track of the indices of the genes and F properly

		geneset_repl = geneset[['gene', 'TSS', 'chr', 'strand']].copy()
		geneset_repl['f_gene_index'] = np.arange(len(geneset_repl))

		# Subset by the genes for which we have replication timing
		geneset_repl = geneset_repl.loc[repl_timing.index]
		geneset_repl['replication_timing'] = repl_timing.timing
		geneset_repl['replication_H_index'] = repl_timing.H_index

		# There may be some genes for which we don't have gene expression data as well
		gene_index = geneset_repl.join(ge_analysis.gene_expression_f[[]], how='inner').f_gene_index
		gene_orf_name_index = geneset_repl.join(ge_analysis.gene_expression_f[[]], how='inner').index.values
		geneset_repl = geneset_repl.loc[gene_orf_name_index]

		# Subset the set f images (np array) so use the integer index
		sc_f_w_repl_images = promoter_analysis.strand_corrected_f_images[gene_index]

		print(len(geneset_repl), "genes with replication timing")
		print("Shape of the f images: ", sc_f_w_repl_images.shape)

		# Now that we have subset the f images, reset the f gene indexing
		geneset_repl['f_gene_index'] = np.arange(len(geneset_repl))

		self.gene_expression_f = ge_analysis.gene_expression_f.loc[gene_orf_name_index]
		self.geneset = geneset
		self.geneset_repl = geneset_repl
		self.sc_f_w_repl_images = sc_f_w_repl_images

		# Next compute the expression value at replication time
		repl_h_index = self.geneset_repl.replication_H_index
		expression_at_replication = self.gene_expression_f.apply(lambda row: row.iloc[repl_h_index[row.name]], axis=1)
		self.geneset_repl['expression_at_replication'] = expression_at_replication


	def select_subset(self, subset_orfs=None, delta_index=25):

		geneset_repl = self.geneset_repl.copy()

		if subset_orfs is not None:
			self.subset_orfs = subset_orfs
			subset_genes = geneset_repl.loc[subset_orfs]
			sc_subset_f_imgs = self.sc_f_w_repl_images[subset_genes.f_gene_index]
		else:
			subset_genes = geneset_repl
			sc_subset_f_imgs = self.sc_f_w_repl_images

		n = len(sc_subset_f_imgs)

		# Set nans to 0
		sc_subset_f_imgs[np.isnan(sc_subset_f_imgs)] = 0

		# Get the prior and post replication indices
		# trying delta index of 10 columns (not sure how many minutes this is)
		self.delta_index = delta_index

		prior_index = subset_genes.replication_H_index.values-self.delta_index
		at_index = subset_genes.replication_H_index.values
		post_index = subset_genes.replication_H_index.values+self.delta_index

		# If post index is greater than the available indices, wrap around
		tp_indices = self.config.get_Hpositions_for_branch('t')
		last_tp_index = tp_indices[-1]
		sel_wrap_around = post_index > last_tp_index

		# wrap around the index
		# todo: note we need to wrap around the timepoints as well
		post_index[sel_wrap_around] = post_index[sel_wrap_around]-(last_tp_index+1)

		rg1_len, cg1_len, dg1_len = self.config.get_g1_lens()

		subset_genes['prior_i'] = prior_index
		subset_genes['post_i'] = post_index

		subset_genes['prior_tp'] = [self.config.get_timepoint_for_index(i)+cg1_len for i in prior_index]
		subset_genes['post_tp'] = [self.config.get_timepoint_for_index(i)+cg1_len for i in post_index]

		def compute_delta_timing(subset_genes):
			subset_genes = subset_genes.copy().dropna()
			row = subset_genes.iloc[-1]
			return row.replication_timing-row.prior_tp

		self.delta_min = compute_delta_timing(subset_genes)

		prior_f_images = sc_subset_f_imgs[np.arange(n), prior_index]
		at_f_images = sc_subset_f_imgs[np.arange(n), at_index]
		post_f_images = sc_subset_f_imgs[np.arange(n), post_index]

		self.prior_f_images = prior_f_images
		self.at_f_images = at_f_images
		self.post_f_images = post_f_images
		self.subset_genes = subset_genes

	def plot_pre_post_images(self, title=None):
		plt.figure(figsize=(8, 5))
		plt.subplots_adjust(hspace=0.5, wspace=0.2)

		def plot_f_img(img):
			plt.imshow(img, cmap='magma_r', 
					   origin='lower', aspect='auto', vmin=0, vmax=25)
			plt.xticks([])
			plt.yticks([])

		prior = self.prior_f_images.mean(axis=0)
		post = self.post_f_images.mean(axis=0)
		at = self.at_f_images.mean(axis=0)
		
		plt.subplot(4, 2, 1)
		plot_f_img(prior)
		plt.ylabel(f"Prior to replication\n-{self.delta_min:.1f} min", rotation=0, ha='right')

		plt.subplot(4, 2, 3)
		plot_f_img(at)
		plt.ylabel("At replication", rotation=0, ha='right')

		plt.subplot(4, 2, 5)
		plot_f_img(post)
		plt.ylabel(f"Post-replication\n{self.delta_min:.1f} min", rotation=0, ha='right')

		# -------- Difference plots ---------------

		def plot_f_img_diff(img):
			plt.imshow(img, origin='lower', aspect='auto', 
					   cmap='RdBu_r', vmin=-1, vmax=1)
			plt.xticks([])
			plt.yticks([])
			plt.gca().yaxis.set_label_position("right")

		plt.subplot(4, 2, 2)
		plot_f_img_diff(at-prior)
		plt.ylabel("at-Prior", rotation=0, ha='left')

		plt.subplot(4, 2, 4)
		plot_f_img_diff(post-prior)

		plt.ylabel("Post-Prior", rotation=0, ha='left')

		plt.subplot(4, 2, 6)
		plot_f_img_diff(post-at)
		plt.ylabel("Post-at", rotation=0, ha='left')

		if title is not None:
			title = f"{title}\nPrior/Post Replication, n={len(self.subset_genes)}"
		else:
			title = f"Prior/Post Replication, n={len(self.subset_genes)}"

		plt.suptitle(title)
