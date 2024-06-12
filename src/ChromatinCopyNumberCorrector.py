
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.mnase_replication_timing_analysis import get_bin_for_position
from src.geneset import get_deconvolved_geneset
from src.helpers import calcH_config
from src.delta_config import load_yl_delta_config
from src.config import read_yl_vst_data_rep
from src.CopyNumberCorrection import normalize_total_reads, get_gene_replication_profile, copy_number_correct_H
from src.stepwise_replication_solver import load_replication_profile


NORMALIZE_TARGET = 7e6

class ChromatinCopyNumberCorrector:

	def __init__(self):
		
		from src.helpers import calcH_config

		# Load the gene expression data and geneset
		self.genes = get_deconvolved_geneset()
		self.repl_profile = load_replication_profile()

		self.config = load_yl_delta_config(1)
		self.H, Hpos = calcH_config(self.config)

	def correct_for_copy_number(self):
		from src.timer import Timer
		
		genes = self.genes_w_repl_timing
		H = self.H
		repl_profile = self.repl_profile
			
		geneset_w_repl = self.genes_w_repl_timing
		corrected_chromatin_g_data_sum_window_df = self.normalized_chrom_sum_data.copy().loc[geneset_w_repl.index]

		for orf_name, gene in geneset_w_repl.iterrows():
		    data = self.normalized_chrom_sum_data.loc[orf_name]
		    corrected_data = copy_number_correct_H(self.H, gene,
		                                           data,  
		        self.repl_profile, geneset_w_repl)
		    corrected_chromatin_g_data_sum_window_df.loc[orf_name] = corrected_data
		    
		normalized_corrected_g_data = normalize_total_reads(corrected_chromatin_g_data_sum_window_df, 
		    NORMALIZE_TARGET)

		self.corrected_chromatin_g_data_sum_window_df = corrected_chromatin_g_data_sum_window_df
		self.normalized_corrected_g_data = normalized_corrected_g_data

	def compute_replication_timing(self):
		from src.timer import Timer

		genes = self.genes
		repl_profile = self.repl_profile
		H = self.H
		config = self.config
			
		repl_timing = self.genes[[]].copy()
		repl_timing['replication_time'] = np.nan
		repl_timing['replication_H_index'] = np.nan

		for orf_name, gene in genes.iterrows():

			replication_idx = get_gene_replication_profile(gene.name, repl_profile, self.genes)

			repl_timing.loc[orf_name, 'replication_H_index'] = replication_idx
			repl_timing.loc[orf_name, 'replication_time'] = config.get_timepoint_for_index(replication_idx)

		self.genes_w_repl_timing = self.genes.join(repl_timing)


	def load_chromatin_data(self, replicate):

		from src.config import load_yl_rg1_vst_config
		from src.chromatin_model import ChromatinModel

		config = load_yl_rg1_vst_config(replicate)
		chromatin_model = ChromatinModel(config)

		geneset = self.genes_w_repl_timing

		from src.global_config import GlobalConstants
		from src.timer import Timer

		"""Load the normalized deconvolution bins for all chromatin data for the replicate"""

		i = 0

		timer = Timer()
		n = len(geneset)
		rows, columns = GlobalConstants.IMAGE_SHAPE
		num_timepoints = len(config.WT1_TIMEPOINTS)
		chromatin_g_data = np.zeros((n, num_timepoints, rows, columns))

		for orf_name, gene in geneset.iterrows():
			chromatin_model.load_mnase_gene(orf_name, log=False)
			chromatin_model.create_deconvolution_bins()

			chromatin_g_data[i] = chromatin_model.deconv_hist_unflattened

			if i % 200 == 0:
				timer.print_time(f"{i}/{n}")

			i += 1

		chromatin_g_data_sum_window = chromatin_g_data.sum(axis=2).sum(axis=2)
		chromatin_g_data_sum_window_df = pd.DataFrame(chromatin_g_data_sum_window, 
		    index=geneset.index, columns=config.WT1_TIMEPOINTS)

		self.chromatin_g_data = chromatin_g_data
		self.chromatin_g_data_sum_window_df = chromatin_g_data_sum_window_df
		self.normalized_chrom_sum_data = normalize_total_reads(chromatin_g_data_sum_window_df, 
		    NORMALIZE_TARGET)


	def compute_ptrs(self):
		from src.peak_to_trough import compute_quantile_ptr

		geneset_w_repl = self.genes_w_repl_timing

		unnormalized_ptr = np.apply_along_axis(lambda mat: compute_quantile_ptr(mat,
		   0.2, 0.8), 1, self.chromatin_g_data_sum_window_df.loc[geneset_w_repl.index].values)
		chrom_ptrs = np.apply_along_axis(lambda mat: compute_quantile_ptr(mat,
		   0.2, 0.8), 1, self.normalized_chrom_sum_data.loc[geneset_w_repl.index].values)
		corrected_chrom_ptrs = np.apply_along_axis(lambda mat: compute_quantile_ptr(mat,
		   0.2, 0.8), 1, self.corrected_chromatin_g_data_sum_window_df.values)
		normalized_corrected_chrom_ptrs = np.apply_along_axis(lambda mat: compute_quantile_ptr(mat,
		   0.2, 0.8), 1, self.normalized_corrected_g_data.values)

		comparison_ptr_df = pd.DataFrame({
		    "unnormalized_raw_ptr": unnormalized_ptr,
		    "normalized_raw_ptr": chrom_ptrs,
		    "corrected_ptr": corrected_chrom_ptrs,
		    "normalized_corrected_ptr": normalized_corrected_chrom_ptrs,
		    "replication_time": self.genes_w_repl_timing.replication_time
		}, index=geneset_w_repl.index)
		self.comparison_ptr_df = comparison_ptr_df


	def plot_ptrs(self, orfs=None):
		plt.figure(figsize=(6, 5))
		plt_data = self.comparison_ptr_df

		plt.scatter(plt_data.normalized_raw_ptr,
		    plt_data.normalized_corrected_ptr, s=1, vmin=5, vmax=15,
		            c=plt_data.replication_time, cmap='Spectral')
		plt.colorbar()
		plt.title("Copy number correction of chromatin data")
		plt.plot([1])
		plt.xlim(0.99, 1.5)
		plt.ylim(0.99, 1.5)
		plt.xlabel("Uncorrected PTR")
		plt.ylabel("Corrected PTR")


	def plot_corrected_gene(self, orf_name, fig=None):
		gene = self.genes.loc[orf_name]

		if fig is None:
			fig = plt.figure(figsize=(4, 3))
		plt.plot(self.normalized_ge.loc[gene.name])
		plt.plot(self.normalized_corrected_ge.loc[gene.name], ls='dotted')

		from src.sgd import get_gene_title_name
		title = get_gene_title_name(orf_name)

		plt.title(title, pad=10)
