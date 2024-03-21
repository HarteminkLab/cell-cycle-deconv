
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cc_src.reference_data import load_spellman_orfs, load_analysis_genes


class PeakToTroughAnalysis:
	"""This class allows us to analyze the peak-to-trough deconvolution results of the chromatin.

	The objective for this analysis is to identify cell cycling genes, what threshold a gene can be considered
	cell cycling, how we can categorize, dilineate between cell cycling genes, and identify novel cell cycling regulatory
	chromatin that is improved by the deconvolution algorithm.
	"""

	def __init__(self, chromatin_dir):
		self.geneset = load_analysis_genes()
		self.chromatin_dir = chromatin_dir
		self.file_paths = glob.glob(f'{self.chromatin_dir}/*_ptr_*.npy')


		from src.config import load_yl_rg1_vst_config
		from cc_src.chromatin_model import ChromatinModel

		# config and chromatin model to hold the image shape and config information
		# we may need later, should be consistent across rep1, rep2, and combined models
		config = load_yl_rg1_vst_config(1)
		self.chromatin_model = ChromatinModel(config)
		self.image_shape = self.chromatin_model.num_bins_y, self.chromatin_model.num_bins_x

		self.load_ptr_files()
		self.sort_gene_ptrs()


	def compute_summary_metrics(self):
		mean = self.summarize_ptrs(np.mean)
		median = self.summarize_ptrs(np.median)
		q95 = self.compute_q_vals(0.95)
		q90 = self.compute_q_vals(0.9)

		metrics_df = self.geneset[['gene']].loc[mean.index.values]
		metrics_df['mean_ptr'] = mean
		metrics_df['q95'] = q95
		metrics_df['median'] = median

		self.metrics_df = metrics_df


	def load_ptr_files(self):

		# Load ptr values

		# Load the size of a flattened image
		loaded_ptrs = np.load(self.file_paths[0])
		m = self.image_shape[0]*self.image_shape[1]

		ptrs_df = pd.DataFrame(index=self.geneset.index, columns=np.arange(m))

		# For each deconvolved gene, load the ptr values and place them into the PTRs dataframe
		for path in self.file_paths:
			filename = path.split('/')[-1]
			orf_name = filename.split('_')[2]

			# Skip genes not in our analysis set
			# for runs in which we haven't filtered for low coverage genes yet
			if not orf_name in self.geneset.index.values: continue

			loaded_ptrs = np.load(path)
			ptrs_df.loc[orf_name] = loaded_ptrs.flatten()

		self.ptr_imgs = ptrs_df.values.reshape((-1, 
			*self.image_shape))
		self.undropped_ptrs_df = ptrs_df.copy()
		self.ptrs_df = ptrs_df.dropna()
		self.n = len(self.ptrs_df)

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

		# The first 100 bins should be enough to find compute the optimal k
		# any large and it may be introducing too much noise
		rank_mean_dat = self.ptr_rank_stds_df.mean(axis=0)[1:700]
		self.set_optimal_k(73)

		plt.figure(figsize=(5, 4))
		plt.plot(rank_mean_dat.index, rank_mean_dat.values)
		plt.axvline(self.optimal_k, c='red')
		plt.xlabel("k")
		plt.ylabel("Average standard deviation in rank")
		plt.title(f"Average $\\sigma$ of gene PTR rank with increasing k\nn={self.n}, optimal k={self.optimal_k}")

	def set_optimal_k(self, k):

		self.optimal_k = k

		k_sorted_genes = self.sorted_ptrs_ranks_df[[self.optimal_k]].sort_values(self.optimal_k).join(self.geneset[['gene']], 
			how='inner')
		k_sorted_genes = k_sorted_genes.join(self.sorted_ptrs_df[[self.optimal_k]], lsuffix='rank', rsuffix='ptr')
		self.k_sorted_genes = k_sorted_genes.rename(
			columns={
				f'{self.optimal_k}rank': 'rank',
				f'{self.optimal_k}ptr': 'ptr'
			})

	def plot_ptrs_per_gene(self):

		plt_data = self.k_sorted_genes
		n = len(plt_data)

		plt.figure(figsize=(3, 4))
		plt.plot(plt_data['ptr'], plt_data['rank']+1)
		plt.xlabel("PTR")
		plt.ylabel("Gene rank")
		plt.yticks([1] + list(np.arange(500, n, 500)))
		plt.ylim(n+100, 1-100)

		ptr_values = plt_data['ptr']
		q05, q95 = np.quantile(ptr_values, q=[0.05, 0.95])

		plt.axvline(q05, c='red', lw=1)
		plt.axvline(q95, c='red', lw=1)
		plt.title(f"Gene PTR values for k={self.optimal_k}\n" +
				 f"n={n}, q05={q05:0.1f}, q95={q95:.1f}")

	def plot_mean_chromatin(self):
		plt.figure(figsize=(13, 4))
		plt.subplot(1, 3, 1)
		plt.scatter(tf_joined_df['Fourier_score'], 
		            tf_joined_df['mean_ptr'], s=1, alpha=0.5)
		plt.title("Promoter fourier score vs\nMean Chromatin PTR")

		plt.subplot(1, 3, 2)
		plt.scatter(nuc_joined_df['Fourier_score'], 
		            nuc_joined_df['mean_ptr'], s=1, alpha=0.5)
		plt.title("Nucleosome fourier score vs\nMean Chromatin PTR")

		plt.subplot(1, 3, 3)
		plt_data = tf_joined_df.join(nuc_joined_df, lsuffix='_tf', rsuffix='_nuc')
		plt.scatter(plt_data.Fourier_score_nuc, plt_data.Fourier_score_tf, s=1)
		plt.title("Nucleosome fourier score vs TF fourier score")

	def load_Fourier_scores(self):
		# Load the Yulong Fourier Score Calculations

		metrics_df = self.metrics_df

		def load_join_yulong_fourier_score(csv_path, metrics_df):
			# Load the Yulong Fourier scores for each gene
			promoter_tf_scores = pd.read_csv(csv_path)
			promoter_tf_scores = promoter_tf_scores.set_index('Gene_ID')
			prom_tf_fourier_vs_ptr_comparison = promoter_tf_scores.join(metrics_df)
			return prom_tf_fourier_vs_ptr_comparison

		prom_tf_path = 'data/reference_data/yulongs_2023_Table_S1_promoter_tf_score.csv'
		self.tf_joined_df = load_join_yulong_fourier_score(prom_tf_path, 
													  metrics_df)
		self.tf_joined_df = self.tf_joined_df.dropna()

		gb_nuc_path = 'data/reference_data/yulongs_2023_Table_S1_gene_body_nuc_score.csv'
		self.nuc_joined_df = load_join_yulong_fourier_score(gb_nuc_path,
													   metrics_df)
		self.nuc_joined_df = self.nuc_joined_df.dropna()

	def plot_ptr_comparison_to_fourier(self):

		plt.figure(figsize=(13, 4))
		plt.subplot(1, 3, 1)
		plot_comparison_scatter(self.tf_joined_df['Fourier_score'], 
			self.tf_joined_df['mean_ptr'], "Promoter fourier score vs\nMean Chromatin PTR", 
			"TF Fourier score", "Mean chromatin PTR")

		plt.subplot(1, 3, 2)
		plot_comparison_scatter(self.nuc_joined_df['Fourier_score'], 
			self.nuc_joined_df['mean_ptr'], "Nucleosome fourier score vs\nMean Chromatin PTR",
			"GB nucleosome Fourier score", "Mean chromatin PTR")

		plt.subplot(1, 3, 3)
		plt_data = self.tf_joined_df.join(self.nuc_joined_df, lsuffix='_tf', rsuffix='_nuc')
		plot_comparison_scatter(plt_data.Fourier_score_nuc, plt_data.Fourier_score_tf, 
			"Nucleosome fourier score vs TF fourier score",
			"GB nucleosome Fourier score", "TF Fourier Score")

	def plot_ptr_k_gene(self):

		spellman_orfs = load_spellman_orfs()

		sorted_ptrs_df = self.sorted_ptrs_df

		plt.figure(figsize=(5, 5))

		select_ks = np.arange(1, 700, 1)

		for orf_name, row in sorted_ptrs_df.dropna().iterrows():

			color = 'red' if orf_name in spellman_orfs else '#888'
			ptrs = row[select_ks]
			plt.plot(select_ks, ptrs, c=color, alpha=0.25)

		plt.xlabel("$k$")
		plt.ylabel("PTR")
		plt.title(f"Peak-to-trough ratio per $k$ for each gene\nn={self.n}")


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

		plt.figure(figsize=(4, 4))
		plt.plot(np.arange(n)/n, spellman_cumsum/num_scs)
		plt.plot([0, 1], [0, 1], c='gray', ls='dotted', lw=1)
		plt.title("Proportion of Spellman genes in ordered chromatin PTR list\n" + 
			f"n={n}, k={self.optimal_k}")
		plt.xlabel("Proportion of all deconvolved genes")


	def summarize_ptrs(self, metric_func):
		summary_ptrs = self.ptrs_df.copy()   
		metric_val = summary_ptrs.apply(metric_func, axis=1)
		return metric_val


def compute_q_vals(self, q_val):
	q_func = lambda val : np.quantile(val, q=q_val)
	q_vals = self.summarize_ptrs(q_func)
	return q_vals

def plot_comparison_scatter(x, y, title, xlabel, ylabel, c='#aaa', 
	highlighted_orfs=[]):

	from scipy.stats import pearsonr

	pearsonr, pval = pearsonr(x, y)
	title = f"{title}\nPearson R={pearsonr:.2f}, P-value={pval:.2f}, N={len(x)}"

	plt.scatter(x, y, s=1, alpha=0.5, c=c, label='_none')

	for (sel_orfs, color, label) in highlighted_orfs:

		x_sel = x.loc[list(sel_orfs)]
		y_sel = y.loc[list(sel_orfs)]

		label = f"{label}, n={len(sel_orfs)}"

		plt.scatter(x_sel, y_sel, s=8, alpha=0.5, facecolors='none', 
			edgecolors=color, label=label, lw=1, marker='D')

		#plt.scatter(x, y, marker='d', facecolors='none', edgecolors='b', s=100, linewidth=1)





	plt.title(title)

	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.legend()


def filter_index(select_index, primary_index):
    """Filters out an index values that do not appear in the primary index.
    Useful for selecting subsets of a dataframe, but discards missing values
    if that selected index does not appear in the primary index"""
    keep_index = set(primary_index).intersection(select_index)
    return list(keep_index)
