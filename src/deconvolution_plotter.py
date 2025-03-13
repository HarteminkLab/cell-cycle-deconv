
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.gridspec as gridspec


class DeconvolutionPlotter():


	def __init__(self, ge_model, chromatin_gridder, chrom_model):
		self.ge_model = ge_model
		self.chrom_model = chrom_model
		self.chromatin_gridder = chromatin_gridder

	def layout_axes(self):
		# Create a 10x10 figure
		fig = plt.figure(figsize=(36, 13))
		plt.subplots_adjust(hspace=0.0, wspace=0.0, right=0.8)

		num_phase_cols = 4
		gene_expression_plt_height = 2

		# Define the grid
		# Rows are defined as: size of the gene expression plot + 1 (chromatin plot height) + 1 (spacing between two branches)
		gs = gridspec.GridSpec((gene_expression_plt_height+1)*2+1, num_phase_cols*4, figure=fig)

		def _add_empty_ax(start_row, end_row, start_col, end_col):
			ax1 = fig.add_subplot(gs[start_row:end_row+1, start_col:end_col+1])
			ax1.set_xticks([])
			ax1.set_yticks([])
			return ax1

		def _add_branch_plots(start_row, start_column):

			# G1 Plots
			g1_ge_ax = _add_empty_ax(start_row, start_row+(gene_expression_plt_height-1), 
						  start_column, start_column+(num_phase_cols-1))
			g1_chrom_axes = []
			for i in range(num_phase_cols):    
				ax = _add_empty_ax(start_row+gene_expression_plt_height, 
					start_row+gene_expression_plt_height, start_column+i, start_column+i)
				g1_chrom_axes.append(ax)

			# Post G1 plots
			post_g1_ge_ax = _add_empty_ax(start_row, start_row+(gene_expression_plt_height-1),
						  start_column+num_phase_cols, start_column+(num_phase_cols*2-1))
			post_g1_chrom_axes = []
			for i in range(num_phase_cols, num_phase_cols*2):    
				ax = _add_empty_ax(start_row+gene_expression_plt_height,
							  start_row+gene_expression_plt_height, 
							  start_column+i, start_column+i) 
				post_g1_chrom_axes.append(ax)

			return g1_ge_ax, g1_chrom_axes, post_g1_ge_ax, post_g1_chrom_axes
			
		self.initial_axes = _add_branch_plots(0, 0)
		self.top_axes = _add_branch_plots(0, num_phase_cols*2)
		self.bottom_axes = _add_branch_plots(4, num_phase_cols*2)
		self.between_t_b_axis = _add_empty_ax(3, 3, (num_phase_cols*2), (num_phase_cols*4))

		# For displaying statistics
		self.stats_ax = _add_empty_ax(4, 4+gene_expression_plt_height, 0, num_phase_cols*2-1)

		self.fig = fig


	def plot_gene_expression(self, initial_axes, top_axes, bottom_axes, ymax=None):

		from src.plot_helpers import color_for_key
		from src.orf_plotter import plot_rect
		
		branches = ['i', 't', 'b']
		axes = [initial_axes, top_axes, bottom_axes]

		f = self.ge_model.f

		if ymax is None:
			ymax = f.max()*1.25

		# The phase label should be fixed as a proportion of the ymax value
		phase_label_height = 0.25 * ymax

		map_phase_name = {
			'R': "Recovery G1",
			'RG1': "Recovery G1",
			'CG1': "Mother G1",
			'DG1': "Daughter G1",
			'postG1': "G2/M",
		}

		branch_mapping = {
			'i': "Initial",
			't': "Top",
			'b': "Bottom",
		}

		for branch_idx in range(len(branches)):
			
			branch = branches[branch_idx]
			branch_axs = axes[branch_idx]

			phase_tp_idx_list = self.ge_model.config.get_timepoints_phases_Hpositions_for_branch(branch)

			for k in range(2):
				
				phase, timepoints, indices = phase_tp_idx_list[k]
				timepoints = timepoints.values
				
				color = color_for_key(phase)

				ax = branch_axs[k]
				ax.fill_between(timepoints, 0, f[indices], lw=5, linestyle='solid', color=color)
				ax.set_xticks([])
				ax.set_yticks([])

				# Plot the text annotations that label the phase of the cell cycle
				xlims = ax.get_xlim()
				plot_rect(ax, xlims[0], -phase_label_height, xlims[1]-xlims[0], 
					phase_label_height, zorder=0, color='#ddd', fill_alpha=1.)
				ax.text((xlims[0]+xlims[1])/2, -phase_label_height/1.75, map_phase_name[phase], 
					zorder=9, fontsize=39, va='center', ha='center')

				# Label the branch name
				if k == 0:
					ax.text(xlims[0] + (xlims[1]-xlims[0])*0.139, ymax*0.93, branch_mapping[branch], 
					zorder=9, fontsize=63, va='top', ha='left')

				# Put the ticks on the right side of the last branch plots
				if branch_idx > 0 and k == 1:

					yticks = np.arange(0, 1200, 200)
					ytick_labels = [f"{y:0.0f}" for y in yticks]
					ax.set_yticks(yticks)
					ax.set_yticklabels(ytick_labels)

					ax.yaxis.tick_right()
					ax.tick_params(axis='both', which='major', labelsize=16)
					ax.yaxis.set_label_position("right")
					ax.set_ylabel('Expression, TPM', rotation=270, va='bottom', fontsize=23)

				ax.set_ylim(-phase_label_height, ymax)
				ax.set_xlim(timepoints[0], timepoints[-1])



	# Plot the chromatin
	def plot_chrom_imgs_for(self, chrom_model, f, chromatin_gridder, phase, axes):
		num_columns = len(axes)

		x_padding = 90
		y_padding = 160

		for column in range(num_columns):
			ax = axes[column]
			chromatin_gridder.plot_f_img(ax, chrom_model, f, phase, column, num_columns, 
										 show_title=False, x_padding=x_padding,
										y_padding=y_padding)

	def plot_deconvolved_models(self):

		chrom_model, chromatin_gridder, ge_model = self.chrom_model, self.chromatin_gridder, self.ge_model
		# chrom_meta_data = chrom_model.chrom_meta_data

		self.layout_axes()

		# gene_title = ge_model.gene_name + "\ /\ " + ge_model.orf_name
		# gene_title = "$\it{"+ gene_title + "}$"

		# chrom_meta = chrom_meta_data.iloc[0]
		# chrom_meta.rn, chrom_meta.sn, chrom_meta.gm
		# gene_name = chrom_model.gene_name
		# orf_name = chrom_model.orf_name

		# title_string = (f"{gene_title}")
		# stats_str = f"rn = {chrom_meta.rn:.3f}\nsn = {chrom_meta.sn:.3f}\ngm = {chrom_meta.gm:.4f}"


		# self.stats_ax.text(0, 0, stats_str, ha='center', va='center', fontsize=36)
		self.stats_ax.set_ylim(-10, 10)
		self.stats_ax.set_xlim(-10, 10)
		self.stats_ax.spines['left'].set_visible(False)
		self.stats_ax.spines['right'].set_visible(False)
		self.stats_ax.spines['top'].set_visible(False)
		self.stats_ax.spines['bottom'].set_visible(False)
		
		# plt.suptitle(title_string, fontsize=63)

		# Plot the gene expression
		initial_ge_axes = self.initial_axes[0], self.initial_axes[2]
		top_ge_axes = self.top_axes[0], self.top_axes[2]
		bottom_ge_axes = self.bottom_axes[0], self.bottom_axes[2]
		self.plot_gene_expression(initial_ge_axes, top_ge_axes, bottom_ge_axes)

		f = chrom_model.f

		self.plot_chrom_imgs_for(chrom_model, f, chromatin_gridder, 'RG1',
										   self.initial_axes[1])
		self.plot_chrom_imgs_for(chrom_model, f, chromatin_gridder, 'postG1',
										   self.initial_axes[3])

		self.plot_chrom_imgs_for(chrom_model, f, chromatin_gridder, 'CG1',
										   self.top_axes[1])
		self.plot_chrom_imgs_for(chrom_model, f, chromatin_gridder, 'postG1',
										   self.top_axes[3])

		self.plot_chrom_imgs_for(chrom_model, f, chromatin_gridder, 'DG1',
										   self.bottom_axes[1])
		self.plot_chrom_imgs_for(chrom_model, f, chromatin_gridder, 'postG1',
										   self.bottom_axes[3])

		# Bold the start of the branches axes
		for ax in [initial_ge_axes[0], top_ge_axes[0], bottom_ge_axes[0], 
				   self.initial_axes[1][0], self.top_axes[1][0], 
				   self.bottom_axes[1][0], self.between_t_b_axis]:
			ax.spines['left'].set_linewidth(6)

