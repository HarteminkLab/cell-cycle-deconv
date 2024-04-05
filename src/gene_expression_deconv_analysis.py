

import glob
import math

import numpy as np
import pandas as pd

from src.model import color_for_key
from matplotlib import pyplot as plt
from src.chromatin_model import read_chromosome_mnase_reads
from src.reference_data import load_spellman_orfs, load_analysis_genes


class GeneExpressionAnalysis:
	"""Analysis to classify and characterize the deconvolved gene expression"""

	def __init__(self, gene_expression_dir):
		self.gene_expression_dir = gene_expression_dir
		self.file_paths = glob.glob(f'{gene_expression_dir}/*_f_*.npy')
		self.geneset = load_analysis_genes()

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

	def compute_ptr_inds_tps(self):

		from src.config import load_yl_rg1_vst_config

		# Use config of replicate 1, this won't matter between the two replicates
		# because both will map to the same columns in the final F vector
		self.config = load_yl_rg1_vst_config(1)

		from src.peak_to_trough import compute_ptr

		gene_expression_f = self.gene_expression_f

		ptr_rets_arr_80_20 = np.apply_along_axis(lambda row: compute_ptr(self.config, row,
																   return_indices=Tru), 1,
						   gene_expression_f)

		self.ptrs_ret_df = pd.DataFrame(ptr_rets_arr_80_20, index=gene_expression_f.index,
					 columns=['mother_ptr', 'daughter_ptr', 'ptr',
							  
							  # The index of the hi and low values for 
							  # comparing to F in the chromatin.
							 'm_lo_idx', 'm_hi_idx', 'd_lo_idx', 'd_hi_idx',
							  
							  # The hi and lo values
							 'm_lo', 'm_hi', 'd_lo', 'd_hi',
							  
							  # The timepoints of the hi and lo values
							 'm_lo_tp','m_hi_tp', 'd_lo_tp','d_hi_tp'])

	def compute_c_d_timepoints_radians(self):

		config = self.config
		ptrs_ret_df = self.ptrs_ret_df
		
		mother_timepoints = config.get_timepoints_for_branch('t')
		daughter_timepoints = config.get_timepoints_for_branch('b')

		cg1_timepoints = config.get_timepoints_phases_Hpositions_for_branch('t')[0][1].values
		postcg1_timepoints = config.get_timepoints_phases_Hpositions_for_branch('t')[1][1].values
		dg1_timepoints = config.get_timepoints_phases_Hpositions_for_branch('b')[0][1].values
		postdg1_timepoints = config.get_timepoints_phases_Hpositions_for_branch('b')[1][1].values

		# Append end of G1 for contiguous timepoints for plotting
		cg1_timepoints = np.concatenate([cg1_timepoints, postcg1_timepoints[0:1]])
		dg1_timepoints = np.concatenate([dg1_timepoints, postdg1_timepoints[0:1]])

		# Calculate S-phase
		gamma1, gamma2 = config.intervals_wt1[0][7], config.intervals_wt1[0][8]
		lambda_val = config.intervals_wt1[0][1]

		s_start, s_end = lambda_val*gamma1, lambda_val*gamma2

		c_s_timepoints = postcg1_timepoints[postcg1_timepoints < s_end]
		c_g2m_timepoints = postcg1_timepoints[postcg1_timepoints >= s_end]

		# contiguous plotting
		c_s_timepoints = np.concatenate([c_s_timepoints, c_g2m_timepoints[:1]])

		d_s_timepoints = postdg1_timepoints[postdg1_timepoints < s_end]
		d_g2m_timepoints = postdg1_timepoints[postdg1_timepoints >= s_end]

		# contiguous plotting
		d_s_timepoints = np.concatenate([d_s_timepoints, d_g2m_timepoints[:1]])
		
		
		# -------- In radians --------------
		cg1_tp_radians = convert_tps_to_radians(mother_timepoints, cg1_timepoints)
		c_s_tp_radians = convert_tps_to_radians(mother_timepoints, c_s_timepoints)
		c_g2m_tp_radians = convert_tps_to_radians(mother_timepoints, c_g2m_timepoints)
		
		dg1_tp_radians = convert_tps_to_radians(daughter_timepoints, dg1_timepoints)
		d_s_tp_radians = convert_tps_to_radians(daughter_timepoints, d_s_timepoints)
		d_g2m_tp_radians = convert_tps_to_radians(daughter_timepoints, d_g2m_timepoints)

		return mother_timepoints, daughter_timepoints, (cg1_tp_radians, c_s_tp_radians, c_g2m_tp_radians), \
			(dg1_tp_radians, d_s_tp_radians, d_g2m_tp_radians)


	def plot_scatter_polar(self, selected_genes=[]):

		mother_timepoints, daughter_timepoints, c_tps, d_tps = self.compute_c_d_timepoints_radians()
		ptrs_ret_df = self.ptrs_ret_df
		geneset = self.geneset

		plt.figure(figsize=(8, 4))

		mother_tp_radians = convert_tps_to_radians(mother_timepoints, ptrs_ret_df.m_hi_tp)
		daughter_tp_radians = convert_tps_to_radians(daughter_timepoints, ptrs_ret_df.d_hi_tp)

		plt.subplot(121, polar=True)
		plt.scatter(-mother_tp_radians, ptrs_ret_df.mother_ptr,  c='#5f728c', s=1, zorder=10)

		# Plot the selected genes
		selected_orfs = geneset[geneset.gene.isin(selected_genes)].index.values
		selected_rows = self.ptrs_ret_df.loc[selected_orfs]
		selected_rows_tp_radians = convert_tps_to_radians(mother_timepoints, selected_rows.m_hi_tp)
		plt.scatter(-selected_rows_tp_radians, selected_rows.mother_ptr, facecolors='none',
			edgecolor='red', s=15, marker='D', 
			zorder=11)

		def plot_annotations(tp_sets, phases):
			xtick_locs = []
			for i in range(len(tp_sets)):
				tp_set = tp_sets[i]
				phase = phases[i]
				plt.fill_between(-tp_set, 1, 3, color=color_for_key(phase))
				xtick_locs.append(-tp_set[len(tp_set)//2])
			plt.xticks(xtick_locs, phases, minor=True)

		plot_annotations(c_tps, ['CG1', 'S', 'G2M'])

		plt.title("Mother PTR values", pad=15)
		plt.ylim(0, 3)

		plt.xticks([], minor=False)
		plt.yticks([])
		plt.xlim(0, -math.pi*2)

		# --------- Daughter PTR plot ---------------

		plt.subplot(122, polar=True)

		plot_annotations(d_tps, ['DG1', 'S', 'G2M'])

		plt.scatter(-daughter_tp_radians, ptrs_ret_df.daughter_ptr, c='#5f728c', s=1, zorder=10)

		selected_rows_tp_radians = convert_tps_to_radians(daughter_timepoints, selected_rows.d_hi_tp)
		plt.scatter(-selected_rows_tp_radians, selected_rows.daughter_ptr, facecolors='none',
			edgecolor='red', s=15, marker='D', 
			zorder=11)

		plt.ylim(0, 3)
		plt.title("Daughter PTR values", pad=15)

		plt.xticks([], minor=False)
		plt.yticks([])
		plt.xlim(0, -math.pi*2)

		plt.suptitle("Gene Expression PTR values, combined model", fontsize=16)
		plt.subplots_adjust(top=0.75)

def convert_tps_to_radians(tps, input_tps):
	"""Convert a set of timepoint values to radians given a set of all timepoints

	Parameters:
	tps - Exhaustive list of timepoints from start to finish
	input_tps - timepoints to convert
	"""

	tp_min, tp_max = tps.min(), tps.max()
	len_tps = tp_max - tp_min
	proportion_through_m_cc = (input_tps - tp_min) / len_tps
	radians = proportion_through_m_cc * 2*math.pi

	return radians