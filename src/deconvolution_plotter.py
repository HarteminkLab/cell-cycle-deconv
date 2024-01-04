
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class DeconvolutionPlotter():


	def __init__(self):
		return

	def layout_axes(self):
		# Create a 10x10 figure
		fig = plt.figure(figsize=(12, 12))
		plt.subplots_adjust(hspace=0.0, wspace=0.0)

		# Define the grid
		gs = gridspec.GridSpec(24, 24, figure=fig)

		def _add_empty_ax(start_row, end_row, start_col, end_col):
			ax1 = fig.add_subplot(gs[start_row:end_row+1, start_col:end_col+1])
			ax1.set_xticks([])
			ax1.set_yticks([])
			return ax1

		def _add_branch_plots(start_row, start_column):
			
			gene_expression_plt_height = 2

			# G1 Plots
			g1_ge_ax = _add_empty_ax(start_row, start_row+(gene_expression_plt_height-1), 
						  start_column, start_column+5) 
			g1_chrom_axes = []
			for i in range(6):    
				ax = _add_empty_ax(start_row+2, start_row+2, start_column+i, start_column+i)
				g1_chrom_axes.append(ax)

			# Post G1 plots
			post_g1_ge_ax = _add_empty_ax(start_row, start_row+(gene_expression_plt_height-1),
						  start_column+6, start_column+11)
			post_g1_chrom_axes = []
			for i in range(6, 12):    
				ax = _add_empty_ax(start_row+gene_expression_plt_height,
							  start_row+gene_expression_plt_height, 
							  start_column+i, start_column+i) 
				post_g1_chrom_axes.append(ax)

			return g1_ge_ax, g1_chrom_axes, post_g1_ge_ax, post_g1_chrom_axes
			
		self.initial_axes = _add_branch_plots(4, 0)
		self.top_axes = _add_branch_plots(0, 12)
		self.bottom_axes = _add_branch_plots(8, 12)
		self.fig = fig
