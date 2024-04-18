# Preamble, notebook setup and imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class TimeDeltaAnalysis:
	"""This class will allow us to compute the time delta between two 
	measures of the cell cycle space. For example, we may want to know
	how much time elapses between a promoter TF binding event and gene
	expression.

	Note: We will keep in mind that the daughter and mother branches
	may need to be analyzed separately for simplicity sakes to start.
	"""
	def __init__(self, ge_analysis, promoter_analysis):
		self.ge_analysis = ge_analysis
		self.promoter_analysis = promoter_analysis
		
	def join_analyses_compute_deltas(self):

		ge_analysis = self.ge_analysis
		promoter_analysis = self.promoter_analysis

		ge_prom_tp_diff = ge_analysis.ptrs_min_maxs[['ptr', 'max_tp', 'max_phase']].join(
			promoter_analysis.promoter_min_maxs[['ptr', 'max_tp', 'max_phase']], 
			lsuffix='_ge', rsuffix='_prom')

		# Note: daughter-specific delta has not been handled yet.
		lambda_val = ge_analysis.config.intervals_wt1[0][1]
		delta_val = ge_analysis.config.intervals_wt1[0][2]

		_, min_case, signed_abs_min = get_minimum_delta_term(ge_prom_tp_diff, 
			lambda_val, 'max_tp_prom', 'max_tp_ge')

		ge_prom_tp_diff['min_case'] = min_case
		ge_prom_tp_diff['signed_abs_min'] = signed_abs_min

		self.ge_prom_tp_diff = ge_prom_tp_diff

	def set_thresholds(self, ge_t=1.2, prom_t=1.2):
		self.ge_t = ge_t
		self.prom_t = prom_t

	def plot_delta_distribution(self):

		ge_t, prom_t = self.ge_t, self.prom_t

		plt.figure(figsize=(8, 3))
		plt.subplot(1, 2, 1)

		ge_prom_tp_diff = self.ge_prom_tp_diff

		selected_ge_prom_diff_df = ge_prom_tp_diff[(ge_prom_tp_diff.ptr_ge > ge_t) & 
			(ge_prom_tp_diff.ptr_prom > prom_t)]
		self.selected_ge_prom_diff_df = selected_ge_prom_diff_df

		plt.hist(selected_ge_prom_diff_df.signed_abs_min, bins=40)
		plt.title("Distribution of thresholded time deltas\n"
				  f"for gene expression and promoter occupancy\nthresholds: {ge_t, prom_t}, "
				  f"n={len(selected_ge_prom_diff_df)}")

		plt.subplot(1, 2, 2)
		plt.scatter(ge_prom_tp_diff.ptr_ge, ge_prom_tp_diff.ptr_prom, s=8, 
			color='none', edgecolors='black', lw=1)

		plt.scatter(ge_prom_tp_diff.ptr_ge, ge_prom_tp_diff.ptr_prom, s=5, 
			c=ge_prom_tp_diff['signed_abs_min'], cmap='RdBu_r')

		plt.axvline(ge_t, c='black', lw=1, ls='dotted')
		plt.axhline(prom_t, c='black', lw=1, ls='dotted')

		plt.title("Gene expression PTR vs\nPromoter Occupancy PTR, time deltas")
		plt.xlabel("Gene expression, PTR")
		plt.ylabel("Promoter occupancy, PTR")
		plt.colorbar()


def get_minimum_delta_term(data_df, lambda_val, A_key, B_key):
	"""
	Compute the absolute minimum difference of the three cases:
	
	1. Difference of: B - A
	2. Difference if A was moved up a cell cycle (lambda)
	3. Difference if B was moved up a cell cycle (lambda)
	
	We want the absolute minimum, as we are interested in minimizing
	the delta.
	
	Returns:
	1. the absolute minimum vector
	2. the index of the column of the minimum for each row
	3. the minimum vector with the original sign returned
	"""
		
	B = data_df[B_key].astype(float)
	A = data_df[A_key].astype(float)

	diff_B_A = B - A
	diff_B_2A = B - (A+lambda_val)
	diff_2B_A = (lambda_val+B) - A

	# Step 1: Calculate absolute values
	abs_B_A = np.abs(diff_B_A)
	abs_B_2A = np.abs(diff_B_2A)
	abs_2B_A = np.abs(diff_2B_A)
	
	# Step 2 and 3: Find minimum values and their indices
	combined_abs = np.stack((abs_B_A, abs_B_2A, abs_2B_A), axis=-1)
	min_values = np.min(combined_abs, axis=-1)
	min_indices = np.argmin(combined_abs, axis=-1)
	
	# Step 4: Determine signs
	signs = np.choose(min_indices, [np.sign(diff_B_A), 
		np.sign(diff_B_2A), np.sign(diff_2B_A)])
	
	return min_values, min_indices, signs*min_values



def plot_proportion_sankeyesque(ge_prom_tp_diff):
	phases = ['C_G1', 'D_G1', 'D_S', 'D_G2M']

	# Next a bar plot of the phase differences....
	prom_ge_phase_counts = ge_prom_tp_diff.groupby(['max_phase_prom', 'max_phase_ge']).count()[['ptr_ge']].rename(
	columns={'ptr_ge': 'count'})
	
	from src.model import color_for_key

	# Accumulate and stack the phase counts for the
	# first and second columns
	accum_y_first = 0
	accum_y_second = 0

	x_padding = 0.1
	first_y_padding = 10
	second_y_padding = 10

	for phase in phases:

		if not phase in prom_ge_phase_counts.index: continue

		first_phase_counts = prom_ge_phase_counts.loc[phase]
		sum_count = first_phase_counts.sum()['count']
		
		y1 = accum_y_first
		y2 = sum_count + accum_y_first

		phase = phase.replace("_G1", "G1").replace("D_", "")
		plt.fill_between([0, 1-x_padding], [y1, y1], [y2, y2],
						 color=color_for_key(phase))
		plt.yticks([])
		plt.xticks([0.5, 1.5], ['Peak promoter occupancy', 'Peak gene expression'])

		accum_y_first = y2 + first_y_padding
		plt.text(0.1, (y1+y2)/2, 
			f"{phase}, {sum_count}", ha='left', va='center', c='black')
		
		# second phase counts
		for phase in phases:

			if not phase in first_phase_counts.index: continue

			second_count = first_phase_counts.loc[phase]['count']
			
			phase = phase.replace("_G1", "G1").replace("D_", "")
			
			y = second_count + accum_y_second
			plt.fill_between([1+x_padding, 2], [accum_y_second, accum_y_second], [y, y], 
							 color=color_for_key(phase))
			accum_y_second = y

		accum_y_second += second_y_padding
		
	return prom_ge_phase_counts



class TracePlotter:


	def __init__(self, ge_analysis, promoter_analysis):
		self.ge_analysis, self.promoter_analysis = ge_analysis, promoter_analysis


	def set_branch_indices_tps(self):

		ge_analysis = self.ge_analysis
		promoter_analysis = self.promoter_analysis

		# Note: daughter-specific delta has not been handled yet.
		lambda_val = ge_analysis.config.intervals_wt1[0][1]
		delta_val = ge_analysis.config.intervals_wt1[0][2]

		# And select by DG1 and CG1 curves
		top_branch_indices = ge_analysis.config.get_Hpositions_for_branch('t')
		bottom_branch_indices = ge_analysis.config.get_Hpositions_for_branch('b')
		top_bottom_indices_concat = np.concatenate([top_branch_indices, bottom_branch_indices])

		top_branch_tps = ge_analysis.config.get_timepoints_for_branch('t')
		bottom_branch_tps = ge_analysis.config.get_timepoints_for_branch('b')
		bottom_branch_tps += -bottom_branch_tps[0] + top_branch_tps[-1]
		tps = np.concatenate([top_branch_tps, bottom_branch_tps])

		return top_bottom_indices_concat, tps


	def plot_time_delta_curves(self, orf_name):
		"""Plot the selected ORFs trace plots"""

		config = self.ge_analysis.config

		ge_time_delta_df = self.ge_analysis.gene_expression_f.astype(float)
		gene_expression_f = ge_time_delta_df.loc[orf_name]

		prom_f = self.promoter_analysis.sm_prom_occ_df.astype(float)
		gene_promoter_f = prom_f.loc[orf_name]

		# Normalize by only the top and bottom branches
		top_branch_indices = config.get_Hpositions_for_branch('t')
		bottom_branch_indices = config.get_Hpositions_for_branch('b')
		top_bottom_indices_concat = np.concatenate([top_branch_indices, bottom_branch_indices])
		gene_expression_f = normalize_max_min(gene_expression_f, indices=top_bottom_indices_concat)
		gene_promoter_f = normalize_max_min(gene_promoter_f, indices=top_bottom_indices_concat)

		from src.sgd import get_gene_title_name
		self.plot_f_curves([
			("Gene expression", 'red', gene_expression_f),
			("Promoter occupancy", 'blue', gene_promoter_f),
		], title=get_gene_title_name(orf_name))



	def plot_f_curves(self, data_sets={}, title=""):
		"""Plot the selected ORFs trace plots"""

		fig, (t_ax, b_ax) = plt.subplots(1, 2, figsize=(9, 2))
		plt.subplots_adjust(wspace=0., top=0.8)
		plt.suptitle(title)

		b_ax.set_yticks([])

		config = self.ge_analysis.config
		promoter_analysis = self.promoter_analysis

		# Note: daughter-specific delta has not been handled yet.
		lambda_val = config.intervals_wt1[0][1]
		delta_val = config.intervals_wt1[0][2]

		# And select by DG1 and CG1 curves
		top_branch_indices = config.get_Hpositions_for_branch('t')
		bottom_branch_indices = config.get_Hpositions_for_branch('b')
		top_bottom_indices_concat = np.concatenate([top_branch_indices, bottom_branch_indices])

		top_branch_tps = config.get_timepoints_for_branch('t')
		bottom_branch_tps = config.get_timepoints_for_branch('b')

		for name, color, data_f in data_sets:
			min_f, max_f = data_f[top_bottom_indices_concat].min(), \
				data_f[top_bottom_indices_concat].max()
			delta = max_f - min_f
			ylims = (min_f - delta*.15), (max_f + delta*.15)

			top_f = data_f[top_branch_indices]
			t_ax.plot(top_branch_tps, top_f, color=color, label=name)
			t_ax.set_xlim(top_branch_tps[0], top_branch_tps[-1])
			t_ax.set_ylim(*ylims)

			bottom_f = data_f[bottom_branch_indices]
			b_ax.plot(bottom_branch_tps, bottom_f, color=color)
			b_ax.set_xlim(bottom_branch_tps[0], bottom_branch_tps[-1])
			b_ax.set_ylim(*ylims)

			# ---------- max and mins -----------------

			# Plot max indices
			top_max_idx = top_f.argmax()		
			t_ax.scatter(top_branch_tps[top_max_idx], top_f.values[top_max_idx], marker='^',
				color=color, zorder=10)

			bottom_max_idx = bottom_f.argmax()
			b_ax.scatter(bottom_branch_tps[bottom_max_idx], bottom_f.values[bottom_max_idx], marker='^',
				color=color, zorder=10)

			# Plot min indices
			top_min_idx = top_f.argmin()
			t_ax.scatter(top_branch_tps[top_min_idx], top_f.values[top_min_idx], marker='v',
				color=color, zorder=10)

			bottom_min_idx = bottom_f.argmin()
			b_ax.scatter(bottom_branch_tps[bottom_min_idx], bottom_f.values[bottom_min_idx], marker='v',
				color=color, zorder=10)
			t_ax.set_yticks([])

		format_top_branch_annotations(config, t_ax)
		format_bottom_branch_annotations(config, b_ax)
		t_ax.legend()


def normalize_max_min(dat, indices):
	"""Normalize the input data to the min and max for comparing"""
	min_v, max_v = dat[indices].min(), dat[indices].max()
	delta = max_v - min_v
	dat = dat.copy()
	dat = (dat - min_v) / delta
	return dat


def format_top_branch_annotations(config, ax):

	# Retrieve the timepoints and indices for the branch and phases
	top_tps = config.get_timepoints_for_branch('t')
	top_indices = config.get_Hpositions_for_branch('t')

	t_timepoints_struct = config.get_timepoints_phases_Hpositions_for_branch('t')

	cg1_timepoints, cg1_indices = t_timepoints_struct[0][1].values, t_timepoints_struct[0][2]
	postcg1_timepoints, postcg1_indices = t_timepoints_struct[1][1].values, t_timepoints_struct[1][2]

	gamma1, gamma2 = config.intervals_wt1[0][7], config.intervals_wt1[0][8]
	lambda_val = config.intervals_wt1[0][1]

	s_start, s_end = lambda_val*gamma1, lambda_val*gamma2
	c_s_timepoints = postcg1_timepoints[postcg1_timepoints < s_end]
	c_g2m_timepoints = postcg1_timepoints[postcg1_timepoints >= s_end]

	# Compute the label and boundary positions for xticks
	cg1_label_position = cg1_timepoints[len(cg1_timepoints)//2]
	cs_label_position = c_s_timepoints[len(c_s_timepoints)//2]
	cg2m_label_position = c_g2m_timepoints[len(c_g2m_timepoints)//2]

	boundaries = [
		cg1_timepoints[0],
		c_s_timepoints[0],
		c_g2m_timepoints[0],
		c_g2m_timepoints[-1]
	]

	# Format the axis
	ax.set_xlim(top_tps[0], top_tps[-1])

	ax.set_xticks([cg1_label_position, cs_label_position, cg2m_label_position])
	ax.set_xticklabels(['CG1', 'S', 'G2M'], minor=False, fontsize=14)
	ax.set_xticks(boundaries, minor=True)
	ax.tick_params(axis='x', which='major', length=0)
	ax.tick_params(axis='x', which='minor', length=15)


def format_bottom_branch_annotations(config, ax):

	# Retrieve the timepoints and indices for the branch and phases
	bottom_tps = config.get_timepoints_for_branch('b')
	bottom_indices = config.get_Hpositions_for_branch('b')

	b_timepoints_struct = config.get_timepoints_phases_Hpositions_for_branch('b')

	dg1_timepoints, dg1_indices = b_timepoints_struct[0][1].values, b_timepoints_struct[0][2]
	postdg1_timepoints, postdg1_indices = b_timepoints_struct[1][1].values, b_timepoints_struct[1][2]

	gamma1, gamma2 = config.intervals_wt1[0][7], config.intervals_wt1[0][8]
	lambda_val = config.intervals_wt1[0][1]

	s_start, s_end = lambda_val*gamma1, lambda_val*gamma2
	d_s_timepoints = postdg1_timepoints[postdg1_timepoints < s_end]
	d_g2m_timepoints = postdg1_timepoints[postdg1_timepoints >= s_end]

	# Compute the label and boundary positions for xticks
	dg1_label_position = dg1_timepoints[len(dg1_timepoints)//2]
	ds_label_position = d_s_timepoints[len(d_s_timepoints)//2]
	dg2m_label_position = d_g2m_timepoints[len(d_g2m_timepoints)//2]

	boundaries = [
		dg1_timepoints[0],
		d_s_timepoints[0],
		d_g2m_timepoints[0],
		d_g2m_timepoints[-1]
	]

	# Format the axis
	ax.set_xlim(bottom_tps[0], bottom_tps[-1])

	ax.set_xticks([dg1_label_position, ds_label_position, dg2m_label_position])
	ax.set_xticklabels(['DG1', 'S', 'G2M'], minor=False, fontsize=14)
	ax.set_xticks(boundaries, minor=True)
	ax.tick_params(axis='x', which='major', length=0)
	ax.tick_params(axis='x', which='minor', length=15)

