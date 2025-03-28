import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from src.plot_helpers import annotate_points
from src.plot_helpers import plot_density
from src.linear_regress import fit_linear_regression


class ScatterChromatinPlot:
	def __init__(self, figsize=(4.5, 4.2), main_height_ratios=0.9, main_width_ratios=0.9):
		"""
		Initialize the plot with marginal distributions.
		
		Parameters:
		-----------
		figsize : tuple
			Figure size (width, height)
		main_height_ratios : float
			Ratio of the main plot height to total height
		main_width_ratios : float
			Ratio of the main plot width to total width
		"""
		self.figsize = figsize
		
		# Set up the figure and grid
		self.fig = plt.figure(figsize=self.figsize)
		height_ratios = [1-main_height_ratios, main_height_ratios]
		width_ratios = [1-main_width_ratios, main_width_ratios]
		
		self.gs = GridSpec(2, 2, width_ratios=width_ratios, height_ratios=height_ratios)
		
		# Create the three axes - top marginal, left marginal, and main plot
		self.ax_top = plt.subplot(self.gs[0, 1])
		self.ax_left = plt.subplot(self.gs[1, 0])
		self.ax_main = plt.subplot(self.gs[1, 1])
		
		# Remove tick labels for marginal plots
		self.ax_top.set_xticks([])
		self.ax_top.set_yticks([])
		self.ax_main.set_yticks([])
		self.ax_left.set_xticks([])

		
	def plot(self, dat, x_key='x', y_key='y', highlight_genes=[], xlim=(-2, 2), 
			 orf_groups=[], ylim=None, xlabel="", ylabel="", plot_density=True,
			 cmap=None, plot_fit=True, bw=[0.03, 0.03]):
		"""
		Plot the scatter plot with marginal distributions.
		"""
			
		y_data = dat[y_key]
		x_data = dat[x_key]
		
		self.ax_main.axvline(0, c='black', lw=1, zorder=0, ls='dotted')

		# Plot main scatter
		self.ax_main.scatter(x_data, y_data, c='gray', s=1, alpha=0.2, label=f'All genes, n={len(dat)}')

		results = fit_linear_regression(x_data, y_data)

		# Plot the regression line with the results
		if plot_fit:
			m, b = results['slope'], results['intercept']
			plot_x = np.linspace(xlim[0], xlim[1], 2)
			plot_y = plot_x*m + b
			self.ax_main.plot(plot_x, plot_y, c='black', lw=1, ls='dotted', zorder=12,
				label=f"Fit $y={m:.2f}x+{b:.2f}$")

		if plot_density:
			from src.DensityScatterPlotter import DensityScatterPlotter
			from src.plot_helpers import create_sub_colormap

			if cmap is None:
				cmap = create_sub_colormap('gray_r', 0.1, 0.25, 'gray_r_lighter')
			else:
				cmap = cmap
			density_scatter_plotter = DensityScatterPlotter()
			density_scatter_plotter.cmap = cmap
			density_scatter_plotter.alpha = 1.
			density_scatter_plotter.s = 9
			density_scatter_plotter.bw = bw

			density_scatter_plotter.set_data(x_data, y_data)
			density_scatter_plotter.plot_ax(self.ax_main, plot_colorbar=False)

		# Set axes limits
		self.ax_main.set_xlim(*xlim)
		if ylim is not None:
			self.ax_main.set_ylim(*ylim)
		
		# Handle highlighting genes
		if highlight_genes:
			highlight_orfnames = self._get_orfnames(highlight_genes)
			high_xs, high_ys = x_data.loc[highlight_orfnames], y_data.loc[highlight_orfnames]
			
			self.ax_main.scatter(high_xs, high_ys, s=10, 
						   facecolor='none', edgecolor='red', marker='D')
			
			# Annotate points
			annotate_points(high_xs, high_ys, highlight_genes, 
						ax=self.ax_main, use_adjust_text=True, zorder=11)

		# Add marginal distribution for all data
		self._add_marginal_density(x_data, y_data, fill=True)
		
		# # Highlight another subset of orfs
		if len(orf_groups) > 0:

			for name, color, orf_group in orf_groups:

				another_selection_of_orf = list(set(dat.index.values).intersection(set(orf_group)))
				xs = x_data.loc[another_selection_of_orf]
				ys = y_data.loc[another_selection_of_orf]
					
				self.ax_main.scatter(xs, ys, edgecolor='#777777', lw=1, facecolor='none', s=17, zorder=9)
				self.ax_main.scatter(xs, ys, color=color, s=9, label=name, zorder=9)
					
				# Add these to the marginal plots
				self._add_marginal_density(xs, ys, color=color)

				results = fit_linear_regression(xs, ys)

				# Plot the regression line with the results
				m, b = results['slope'], results['intercept']

				plot_x = np.linspace(xlim[0], xlim[1], 2)
				plot_y = plot_x*m + b
				self.ax_main.plot(plot_x, plot_y, c=color, lw=1, ls='dotted', zorder=12,
					label=f"{m:.2f}*x+{b:.2f}")


		# Adjust layout
		plt.subplots_adjust(wspace=0.1, hspace=0.1, top=0.86)

		self.ax_left.set_ylabel(ylabel)
		self.ax_main.set_xlabel(xlabel)

		plt.legend()
		
		return self.fig
	
	def _add_marginal_density(self, x_data, y_data, fill=False, color='#e0e0e0'):
		"""Add density plots to marginal axes using the custom plot_density function"""

		# Convert pandas Series to numpy array if needed
		if hasattr(x_data, 'values'):
			x_data = x_data.values
		if hasattr(y_data, 'values'):
			y_data = y_data.values

		num_domain_values = 100
		bw_divisor = 40
			
		# Top marginal - x distribution
		x_range = self.ax_main.get_xlim()
		plot_density(
			x_data, 
			ax=self.ax_top, 
			color=color, 
			domain_values=np.linspace(x_range[0], x_range[1], num_domain_values),
			fill=fill, 
			alpha=1, 
			lw=2,
			bw=(x_range[1]-x_range[0])/bw_divisor  # Adjust bandwidth based on range
		)
		
		# Left marginal - y distribution (rotated)
		y_range = self.ax_main.get_ylim()
		plot_density(
			y_data, 
			ax=self.ax_left, 
			color=color, 
			domain_values=np.linspace(y_range[0], y_range[1], num_domain_values),
			fill=fill, 
			alpha=1, 
			lw=2,
			bw=(y_range[1]-y_range[0])/bw_divisor,  # Adjust bandwidth based on range
			flip_axes=True,  # Flip to align with y-axis
			flip_x_axis=True
		)
	
	def _get_orfnames(self, genes):
		"""Convert gene names to ORF names - assuming same implementation"""
		# This is a placeholder for your get_orfnames function
		# You should replace this with the actual implementation
		from src.sgd import get_orfnames as _get_orfnames
		return _get_orfnames(genes)
	
