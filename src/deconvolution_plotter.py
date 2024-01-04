
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class DeconvolutionPlotter():


	def __init__(self, model):
		self.model = model

	def layout_axes(self):
		# Create a 10x10 figure
		fig = plt.figure(figsize=(36, 13))
		plt.subplots_adjust(hspace=0.0, wspace=0.0)

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
		self.fig = fig


	def plot_gene_expression(self, initial_axes, top_axes, bottom_axes):
		from src.model import color_for_key
		
		branches = ['i', 't', 'b']
		axes = [initial_axes, top_axes, bottom_axes]

		f = self.model.f.value
			
		ylim = f.max()*1.25

		for branch_idx in range(len(branches)):
			
			branch = branches[branch_idx]
			branch_axs = axes[branch_idx]

			phase_tp_idx_list = self.model.config.get_timepoints_phases_Hpositions_for_branch(branch)

			for k in range(2):
				
				phase, timepoints, indices = phase_tp_idx_list[k]
				timepoints = timepoints.values
				
				color = color_for_key(phase)

				ax = branch_axs[k]
				ax.fill_between(timepoints, 0, f[indices], lw=5, linestyle='solid', color=color)
				ax.set_xticks([])
				ax.set_yticks([])
				ax.set_ylim(0, ylim)
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

