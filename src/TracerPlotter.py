
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


	def plot_time_delta_curves(self, orf_name, title=None):
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

		self.plot_f_curves([
			("Gene expression", '#886cad', gene_expression_f),
			("Promoter occupancy", '#ffa02b', gene_promoter_f),
		], title=title)


	def plot_f_curves(self, data_sets={}, title=""):
		"""Plot the selected ORFs trace plots"""

		fig, (t_ax, b_ax) = plt.subplots(1, 2, figsize=(9, 2.5))
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
			t_ax.plot(top_branch_tps, top_f, color=color, lw=2, label=name)
			t_ax.set_xlim(top_branch_tps[0], top_branch_tps[-1])
			t_ax.set_ylim(*ylims)

			bottom_f = data_f[bottom_branch_indices]
			b_ax.plot(bottom_branch_tps, bottom_f, lw=2, color=color)
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


def normalize_max_min(dat, indices=None):
	"""Normalize the input data to the min and max for comparing"""

	if indices is None: indices = np.arange(len(dat))

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
	ax.tick_params(axis='x', which='minor', length=15, width=1)


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
