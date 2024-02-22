
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cc_src.reference_data import load_spellman_orfs


class PeakToTroughAnalysis:
	"""This class allows us to analyze the peak-to-trough deconvolution results of the chromatin.

	The objective for this analysis is to identify cell cycling genes, what threshold a gene can be considered
	cell cycling, how we can categorize, dilineate between cell cycling genes, and identify novel cell cycling regulatory
	chromatin that is improved by the deconvolution algorithm.
	"""

	def __init__(self, chromatin_dir):

		self.geneset = pd.read_csv('data/reference_data/geneset_nondub_w_prom_genebodies.csv').set_index('orf_name')
		self.chromatin_dir = chromatin_dir
		self.file_paths = glob.glob(f'{self.chromatin_dir}/*_ptr_*.npy')

	def load_ptr_files(self):

		# Load ptr values

		# Load the size of a flattened image
		loaded_ptrs = np.load(self.file_paths[0])
		m = loaded_ptrs.flatten().shape[0]

		ptrs_df = pd.DataFrame(index=self.geneset.index, columns=np.arange(m))

		# For each deconvolved gene, load the ptr values and place them into the PTRs dataframe
		for path in self.file_paths:
			filename = path.split('/')[-1]
			orf_name = filename.split('_')[2]
			loaded_ptrs = np.load(path)
			ptrs_df.loc[orf_name] = loaded_ptrs.flatten()

		self.ptrs_df = ptrs_df.dropna()

	def sort_gene_ptrs(self):

		# Sort each gene by the highest ptr values first
		# We should now be able to select a column and this will indicate the kth highest
		# ptr value (0 indexed, so highest is 0)
		ptr_vals = self.ptrs_df.values
		sorted_ptr_array = np.array([row[np.argsort(row)[::-1]] for row in ptr_vals])

		sorted_ptrs_df = self.ptrs_df.copy()
		sorted_ptrs_df.loc[:] = sorted_ptr_array

		self.sorted_ptrs_df = sorted_ptrs_df


	def compute_gene_ptr_rank_per_k(self):
		"""Per k, each gene will have a rank for its peak to trough ratio value, compute this rank.
	
		Then compute the standard deviation of these ranks, such that each gene will will have a 
		standard deviation for how much variation from 1:k the rank changes... TODO: there
		may be a better measure here.
		"""
		sorted_ptrs_ranks_df = self.sorted_ptrs_df.dropna().copy()

		ptr_ranks = ranks = np.argsort(np.argsort(sorted_ptrs_ranks_df.values, axis=0), axis=0) + 1

		sorted_ptrs_ranks_df.loc[:] = ptr_ranks
		sorted_ptrs_ranks_df = len(sorted_ptrs_ranks_df) - sorted_ptrs_ranks_df.astype(int)
		self.sorted_ptrs_ranks_df = sorted_ptrs_ranks_df

		# Compute the rank of the standard deviations of selected k values
		rank_stds_df = sorted_ptrs_ranks_df.copy()
		rank_stds_df.loc[:] = np.nan
		for k in range(1, len(sorted_ptrs_ranks_df.columns), 1):
			rank_stds_df.loc[:, k] = sorted_ptrs_ranks_df.loc[:, 0:k].std(axis=1)
		self.ptr_rank_stds_df = rank_stds_df


	def compute_optimal_k(self):
		rank_mean_dat = self.ptr_rank_stds_df.mean(axis=0)[1:100]

		plt.plot(rank_mean_dat.index, rank_mean_dat.values)

		self.optimal_k = rank_mean_dat.argmax()+1

		k_sorted_genes = k_sorted_genes.join(self.sorted_ptrs_df[[self.optimal_k]], lsuffix='rank', rsuffix='ptr')
		self.k_sorted_genes = k_sorted_genes.rename(
			columns={
				f'{self.optimal_k}rank': 'rank',
				f'{self.optimal_k}ptr': 'ptr'
			})

		plt.axvline(self.optimal_k, c='red')
		plt.xlabel("k")
		plt.ylabel("Average standard deviation in rank")
		plt.title(f"Average std of gene PTR rank with increased k, optimal k={self.optimal_k}")


	def plot_ptr_k_gene(self):

		spellman_orfs = load_spellman_orfs()

		sorted_ptrs_df = self.sorted_ptrs_df

		plt.figure(figsize=(9, 3))

		select_ks = np.arange(1, 300, 1)

		for orf_name, row in sorted_ptrs_df.dropna().iterrows():

			color = 'red' if orf_name in spellman_orfs else '#555'
			ptrs = row[select_ks]
			plt.plot(ptrs, select_ks, c=color, alpha=0.25)

		plt.xlabel("PTR")
		plt.ylabel("k")
		plt.title("Peak to trough ratio per k for each gene")


	def spellman_analysis(self):
		"""Plot a graph of counting up the spellman genes from accumulating the 
		genes with the highest PTR values for the chosen k value"""

		spellman_orfs = load_spellman_orfs()
		self.k_sorted_genes['spellman'] = False
		k_spellman = list(set(spellman_orfs).intersection(set(self.k_sorted_genes.index.values)))
		self.k_sorted_genes.loc[k_spellman, 'spellman'] = True

		spellman_cumsum = self.k_sorted_genes.spellman.cumsum()

		num_scs = len(k_spellman)
		n = len(spellman_cumsum)

		plt.plot(np.arange(n)/n, spellman_cumsum/num_scs)
		plt.plot([0, 1], [0, 1], c='gray', ls='dotted', lw=1)
		plt.title("Proportion of Spellman genes in ordered chromatin PTR list")
		plt.xlabel("Proportion of all deconvolved genes")
		plt.ylabel("Proportion of spellman genes")
