

import glob
import math

import numpy as np
import pandas as pd

from src.model import color_for_key
from matplotlib import pyplot as plt
from src.chromatin_model import read_chromosome_mnase_reads
from src.reference_data import load_spellman_orfs, load_analysis_genes
from src.config import load_yl_rg1_vst_config


class GeneExpressionAnalysis:
	"""Analysis to classify and characterize the deconvolved gene expression"""

	def __init__(self, gene_expression_dir):
		self.gene_expression_dir = gene_expression_dir
		self.file_paths = glob.glob(f'{gene_expression_dir}/*_f_*.npy')
		self.geneset = load_analysis_genes()
		self.config = load_yl_rg1_vst_config(1)

	def load_gene_expression_fs(self):

		gene_expression_f = None

		# For each deconvolved gene, load the ptr values and place them into the PTRs dataframe
		for path in self.file_paths:
			filename = path.split('/')[-1]
			orf_name = filename.split('_')[2].replace('.npy', '')

			# Skip genes not in our analysis set
			# for runs in which we haven't filtered for low coverage genes yet
			if not orf_name in self.geneset.index.values: continue

			loaded_f = np.load(path)

			if gene_expression_f is None:
				m = len(loaded_f)
				gene_expression_f = pd.DataFrame(index=self.geneset.index, columns=np.arange(m))

			gene_expression_f.loc[orf_name] = loaded_f

		self.gene_expression_f = gene_expression_f.dropna()
		self.n = len(self.gene_expression_f)


	def plot_cg1_dg1_ratio(self):
		# And select by DG1 and CG1 curves
		cg1_indices = self.config.get_Hpositions_for_phase('CG1')
		dg1_indices = self.config.get_Hpositions_for_phase('DG1')

		cg1_genes_f = self.gene_expression_f.astype(float)[cg1_indices]
		dg1_genes_f = self.gene_expression_f.astype(float)[dg1_indices]

		cg1_dg1_ratio = self.gene_expression_f[[]].copy()
		cg1_dg1_ratio['c_d_ratio'] = cg1_genes_f.sum(axis=1) / dg1_genes_f.sum(axis=1)
		cg1_dg1_ratio = cg1_dg1_ratio.sort_values('c_d_ratio', ascending=False)

		plt.figure(figsize=(6, 3))
		plt.hist(np.log(cg1_dg1_ratio['c_d_ratio']), bins=100)
		plt.yscale('log')
		plt.ylabel("Count")
		plt.xlabel("log CG1/DG1, gene expression ratio")
		plt.title("CG1/DG1 ratio distribution")


	def compute_min_max_df(self):

		from src.peak_to_trough import compute_max_min_locations

		gene_fs = self.gene_expression_f

		min_max_arr = np.apply_along_axis(lambda row: compute_max_min_locations(self.config, row), 1, gene_fs)
		min_rets_df = pd.DataFrame(min_max_arr[:, 0, :], index=gene_fs.index,
			 columns=['min_value', 'min_f_idx', 'min_tp', 'min_phase'])
		max_rets_df = pd.DataFrame(min_max_arr[:, 1, :], index=gene_fs.index,
			 columns=['max_value', 'max_f_idx', 'max_tp', 'max_phase'])
		min_max_df = min_rets_df.join(max_rets_df)

		return min_max_df


	def compute_ptr_maxmins_tps(self):

		from src.peak_to_trough import compute_ptr

		min_max_df = self.compute_min_max_df()

		ptr_rets_arr_80_20 = np.apply_along_axis(lambda row: compute_ptr(self.config, row,
		    return_indices=False), 1, self.gene_expression_f)
		ptrs_ret_df = pd.DataFrame(ptr_rets_arr_80_20, index=self.gene_expression_f.index,
					 columns=['mother_ptr', 'daughter_ptr', 'ptr'])

		self.ptrs_min_maxs = ptrs_ret_df.join(min_max_df)

	def threshold_ptrs(self, ptr_threshold):

		self.ptr_threshold = ptr_threshold

		ptrs_ret_df = self.ptrs_min_maxs
		# Select a PTR threshold and count the number of genes in each phase:
		thresholded_ptrs = ptrs_ret_df[ptrs_ret_df.ptr > ptr_threshold]

		phase_counts = {}
		format_str = ""
		for phase in thresholded_ptrs.max_phase.unique():
			selected_rows = thresholded_ptrs[thresholded_ptrs.max_phase == phase]
			if phase.endswith("S"): phase = "S"
			if phase.endswith('G2M'): phase = "G2M"    
			count = len(selected_rows)

			if phase in phase_counts.keys():
				phase_counts[phase] += count
			else:
				phase_counts[phase] = count

		for phase, count in phase_counts.items():
			format_str += f"{phase}:\t\t{count}\n"

		self.thresholded_ptrs = thresholded_ptrs
		return thresholded_ptrs, format_str, phase_counts