

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


	def compute_c_d_timepoints_radians(self):

		config = self.config
		
		mother_timepoints = config.get_timepoints_for_branch('t')
		daughter_timepoints = config.get_timepoints_for_branch('b')

		# ------------- timepoints per phase --------------------

		(cg1_timepoints, c_s_timepoints, c_g2m_timepoints), \
		(dg1_timepoints, d_s_timepoints, d_g2m_timepoints) = self.config.get_phase_timepoints_for_plotting()
		
		# -------- In radians --------------
		cg1_tp_radians = convert_tps_to_radians(mother_timepoints, cg1_timepoints)
		c_s_tp_radians = convert_tps_to_radians(mother_timepoints, c_s_timepoints)
		c_g2m_tp_radians = convert_tps_to_radians(mother_timepoints, c_g2m_timepoints)
		
		dg1_tp_radians = convert_tps_to_radians(daughter_timepoints, dg1_timepoints)
		d_s_tp_radians = convert_tps_to_radians(daughter_timepoints, d_s_timepoints)
		d_g2m_tp_radians = convert_tps_to_radians(daughter_timepoints, d_g2m_timepoints)

		return mother_timepoints, daughter_timepoints, (cg1_tp_radians, c_s_tp_radians, c_g2m_tp_radians), \
			(dg1_tp_radians, d_s_tp_radians, d_g2m_tp_radians)


	def plot_scatter_polar_full(self, selected_genes=[]):

		mother_timepoints, daughter_timepoints, c_tps, d_tps = self.compute_c_d_timepoints_radians()
		ptrs_ret_df = self.ptrs_min_maxs
		geneset = self.geneset

		plt.figure(figsize=(8, 4))

		# Offset the daughter timepoints such that they follow the mother's timepoints
		offset = -daughter_timepoints[0] + mother_timepoints[-1]
		daughter_timepoints += offset
		mother_and_daughter_tps = np.concatenate([mother_timepoints, daughter_timepoints])

		plt.subplot(121, polar=True)

		(cg1_timepoints, c_s_timepoints, c_g2m_timepoints), \
		(dg1_timepoints, d_s_timepoints, d_g2m_timepoints) = self.config.get_phase_timepoints_for_plotting()
		
		# -------- label the subsectors --------------
		cg1_tp_radians = convert_tps_to_radians(mother_and_daughter_tps, cg1_timepoints)
		c_s_tp_radians = convert_tps_to_radians(mother_and_daughter_tps, c_s_timepoints)
		c_g2m_tp_radians = convert_tps_to_radians(mother_and_daughter_tps, c_g2m_timepoints)
		
		dg1_tp_radians = convert_tps_to_radians(mother_and_daughter_tps, dg1_timepoints+offset)
		d_s_tp_radians = convert_tps_to_radians(mother_and_daughter_tps, d_s_timepoints+offset)
		d_g2m_tp_radians = convert_tps_to_radians(mother_and_daughter_tps, d_g2m_timepoints+offset)

		c_xticks_major, c_xticks_minor = plot_annotations((cg1_tp_radians, c_s_tp_radians, 
			c_g2m_tp_radians), ['CG1', 'S', 'G2M'], set_xticks=False)
		d_xticks_major, d_xticks_minor = plot_annotations((dg1_tp_radians, d_s_tp_radians, 
			d_g2m_tp_radians), ['DG1', 'S', 'G2M'], set_xticks=False, colors=[
			color_for_key('DG1'), '#ddd', '#dfdfdf'])

		# -----------------------

		# copy to make manipulations just for plotting
		ptrs_ret_df = ptrs_ret_df.copy()

		sel_mothers = ptrs_ret_df.max_phase.str.startswith('C')
		sel_daughters = ptrs_ret_df.max_phase.str.startswith('D')

		ptrs_mother_hi = ptrs_ret_df[sel_mothers]
		ptrs_daughter_hi = ptrs_ret_df[sel_daughters]

		mother_tp_radians = convert_tps_to_radians(mother_and_daughter_tps, 
			ptrs_mother_hi.max_tp.values.astype(float))
		daughter_tp_radians = convert_tps_to_radians(mother_and_daughter_tps, 
			ptrs_daughter_hi.max_tp.values.astype(float) + offset) # add offset such that daughter timepoints
		# follow mother time points
	
		# Negative to move the plot clockwise
		ptrs_ret_df.loc[sel_mothers, 'tp_rad'] = -mother_tp_radians
		ptrs_ret_df.loc[sel_daughters, 'tp_rad'] = -daughter_tp_radians

		# #  ---------- Move the Daughter S and G2M ptrs back to the mother branch sectors ------------------

		# Move all S and PostG1 genes into the same coordinate space (into mother's time space)
		# which means any timepoints in which the daughter's hi
		# is greater than the start of S phase, should be offset to the start of common G1's S and G2M phase

		# For the daughter G1 genes, their S-phase, be offset backwards by the length of DG1, C-G2M, C-S
		dg1_len = dg1_tp_radians[-1]-dg1_tp_radians[0]
		c_g2m_len = c_g2m_tp_radians[-1]-c_g2m_tp_radians[0]
		c_s_len = c_s_tp_radians[-1]-c_s_tp_radians[0]
		offset_to_move_d_s = dg1_len+c_g2m_len+c_s_len

		sel_d_s_and_d_post_g1 = (ptrs_ret_df.max_phase == 'D_S') | (ptrs_ret_df.max_phase == 'D_G2M')
		ptrs_ret_df.loc[sel_d_s_and_d_post_g1, 'tp_rad'] = ptrs_ret_df[sel_d_s_and_d_post_g1].tp_rad + \
			offset_to_move_d_s 

		# # ----------------

		# # Plot the data 
		plt.scatter(ptrs_ret_df.tp_rad, ptrs_ret_df.ptr,  c='#5f728c', s=1, zorder=10)

		# -------------- Plot highlighted genes

		# plot the selected data 
		selected_orfs = geneset[geneset.gene.isin(selected_genes)].index.values

		# plot the selected data where the mother is highest value
		selected_rows = ptrs_ret_df.loc[selected_orfs]
		plt.scatter(selected_rows.tp_rad, selected_rows.ptr, facecolors='none',
			edgecolor='red', s=15, marker='D', 
			zorder=11)

		# ------- Format the plot -----------

		plt.xticks(np.concatenate([c_xticks_minor, d_xticks_minor]), 
			['CG1', 'S', 'G2M', 'DG1', 'S', 'G2M'], minor=True)
		plt.xticks([], minor=False)

		plt.yticks([self.ptr_threshold], [])

		plt.xlim(0, -math.pi*2)
		plt.ylim(0, 2.25)

		plt.grid(axis='y', linestyle='dotted', linewidth=1, color='gray')

		return ptrs_ret_df

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

def plot_annotations(tp_sets, phases, set_xticks=True, colors=None):

	xtick_major_locs = []
	xtick_minor_locs = []

	for i in range(len(tp_sets)):
		tp_set = tp_sets[i]
		phase = phases[i]

		if colors is not None:
			color = colors[i]
		else:
			color = color_for_key(phase)

		plt.fill_between(-tp_set, 1, 3, color=color, zorder=0)

		xtick_minor_locs.append(-tp_set[len(tp_set)//2])
		xtick_major_locs.append(-tp_set[0])

	if set_xticks:
		plt.xticks(xtick_minor_locs, phases, minor=True)
		plt.xticks(xtick_major_locs, phases, minor=False)

	return xtick_major_locs, xtick_minor_locs