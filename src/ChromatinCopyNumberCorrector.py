
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.mnase_replication_timing_analysis import get_bin_for_position
from src.geneset import get_deconvolved_geneset
from src.helpers import calcH_config
from src.config import load_configs_by_config_type
from src.config import read_yl_vst_data_rep
from src.CopyNumberCorrection import normalize_total_reads
from src.stepwise_replication_solver import load_gene_replication_profile


NORMALIZE_TARGET = 7e6

class ChromatinCopyNumberCorrector:

	def __init__(self):
		
		from src.helpers import calcH_config

		# Load the gene expression data and geneset
		self.genes = get_deconvolved_geneset()


	def load_config_type(self, config_type):
		self.repl_profile = load_gene_replication_profile(config_type)
		self.config1, self.config2 = load_configs_by_config_type(config_type, with_copy_correction=False)
		self.config = self.config1 if self.replicate == 1 else self.config2
		self.H, Hpos = self.config.calcH_function(self.config.intervals_wt1, self.config.WT1_TIMEPOINTS)
		self.config_type = config_type


	def correct_for_copy_number(self):
		from src.timer import Timer
		from src.CopyNumberCorrection import copy_number_correct_H_index
		
		H = self.H
		repl_profile = self.repl_profile
		corrected_chromatin_g_data_sum_window_df = self.normalized_chrom_sum_data.copy().loc[repl_profile.index]

		for orf_name, gene in repl_profile.iterrows():
			data = self.normalized_chrom_sum_data.loc[orf_name]
			replication_index = int(repl_profile.loc[gene.name].replication_H_index)
			corrected_data = copy_number_correct_H_index(self.H, data, replication_index)
			corrected_chromatin_g_data_sum_window_df.loc[orf_name] = corrected_data
			
		normalized_corrected_g_data = normalize_total_reads(corrected_chromatin_g_data_sum_window_df, 
			NORMALIZE_TARGET)

		self.corrected_chromatin_g_data_sum_window_df = corrected_chromatin_g_data_sum_window_df
		self.normalized_corrected_g_data = normalized_corrected_g_data

	def load_chromatin_data(self, replicate):

		from src.config import load_yl_rg1_vst_config
		from src.chromatin_model import ChromatinModel
		self.replicate = replicate

		# Config appears only used for loading the raw chromatin data
		config = load_yl_rg1_vst_config(replicate)
		chromatin_model = ChromatinModel(config)
		geneset = self.genes

		from src.global_config import GlobalConstants
		from src.timer import Timer

		timepoints = GlobalConstants.CHROM_WT1_TIMEPOINTS if replicate == 1 \
			else GlobalConstants.CHROM_WT2_TIMEPOINTS

		i = 0

		timer = Timer()
		n = len(geneset)
		rows, columns = GlobalConstants.IMAGE_SHAPE
		num_timepoints = len(timepoints)
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
			index=geneset.index, columns=timepoints)

		self.chromatin_g_data = chromatin_g_data
		self.chromatin_g_data_sum_window_df = chromatin_g_data_sum_window_df
		self.normalized_chrom_sum_data = normalize_total_reads(chromatin_g_data_sum_window_df, 
			NORMALIZE_TARGET)


	def save_files(self, save_dir):
		from src.utils import save_print_df
		scaling = self.normalized_corrected_g_data / \
			self.chromatin_g_data_sum_window_df
		save_print_df(self.comparison_ptr_df, 
			f"{save_dir}/ptrs_rep{self.replicate}_{self.config_type}.csv")
		save_print_df(scaling, 
			f"{save_dir}/norm_corr_scaling_rep{self.replicate}_{self.config_type}.csv")
		save_print_df(self.chromatin_g_data_sum_window_df, 
			f"{save_dir}/raw_sums_rep{self.replicate}_{self.config_type}.csv")
		save_print_df(self.normalized_corrected_g_data, 
			f"{save_dir}/normalized_corrected_rep{self.replicate}_{self.config_type}.csv")


	def compute_ptrs(self):
		from src.peak_to_trough import compute_quantile_ptr

		genes = self.genes

		unnormalized_ptr = np.apply_along_axis(lambda mat: compute_quantile_ptr(mat,
		   0.2, 0.8), 1, self.chromatin_g_data_sum_window_df.loc[genes.index].values)
		chrom_ptrs = np.apply_along_axis(lambda mat: compute_quantile_ptr(mat,
		   0.2, 0.8), 1, self.normalized_chrom_sum_data.loc[genes.index].values)
		corrected_chrom_ptrs = np.apply_along_axis(lambda mat: compute_quantile_ptr(mat,
		   0.2, 0.8), 1, self.corrected_chromatin_g_data_sum_window_df.values)
		normalized_corrected_chrom_ptrs = np.apply_along_axis(lambda mat: compute_quantile_ptr(mat,
		   0.2, 0.8), 1, self.normalized_corrected_g_data.values)

		comparison_ptr_df = pd.DataFrame({
			"unnormalized_raw_ptr": unnormalized_ptr,
			"normalized_raw_ptr": chrom_ptrs,
			"corrected_ptr": corrected_chrom_ptrs,
			"normalized_corrected_ptr": normalized_corrected_chrom_ptrs,
			"replication_time": self.repl_profile.replication_time
		}, index=genes.index)
		self.comparison_ptr_df = comparison_ptr_df


	def plot_ptrs(self, orfs=None):
		plt.figure(figsize=(6, 5))
		plt_data = self.comparison_ptr_df

		plt.scatter(plt_data.normalized_raw_ptr,
			plt_data.normalized_corrected_ptr, s=1, vmin=5, vmax=15,
					c=plt_data.replication_time, cmap='Spectral')
		plt.colorbar()
		plt.title(f"Copy # correction PTR change\n{self.config_type.title()} model, replicate {self.replicate}")
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


def load_chromatin_copy_correction(config_type, replicate):
    path = f'data/copy_correction/chromatin/norm_corr_scaling_rep{replicate}_{config_type}.csv'
    correction = pd.read_csv(path).set_index('orf_name')
    return correction

