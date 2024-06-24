
import matplotlib.pyplot as plt
import numpy as np

class Figure1Deconvolution(object):
	"""Load and plot figures for the first result figure"""

	def __init__(self):

		# Create the H for the updated model config to include the H config
		from src.config import load_configs_by_config_type
		config1, config2 = load_configs_by_config_type('shared')

		H, Hpositions = config1.calcH_function(config1.intervals_wt1, config1.WT1_TIMEPOINTS)
		self.H = H
		self.config1 = config1
		self.config2 = config2


	def plot_H(self):
		from src.config import plot_H
		plot_H(self.config1, self.H)


	def plot_H_fill_betweens(self):

		from src.helpers import combine_with_bins
		from src.model import color_for_key

		H = self.H
		fig = plt.figure(figsize=(6, 6))
		ax = plt.gca()

		def plot_phase_fills(ax, phase, x_offset=0):
			phase_indices = self.config1.get_Hpositions_for_phase(phase)
			cur_H = H[:, phase_indices]

			# Combine rows to make the plot more clear
			combined_cur_H = combine_with_bins(cur_H, np.arange(0, 16, 1), axis=0)
			n, m = combined_cur_H.shape

			# Plot fill betweens for each row bunch
			scale = 20
			x_values = np.arange(m)+x_offset
			color = color_for_key(phase)
			
			for i in range(n):
				y2_values = combined_cur_H[i]*scale-i
				y1_values = -i
				
				# Special case halted, 1 index, but make it larger to make
				# it more visible
				if phase == 'H':
					scale = 3.
					y2_values = np.repeat(combined_cur_H[i]*scale-i, 2)
					y1_values = np.array([-i, -i])
					x_values = [x_offset, x_offset+7]
					
					plt.fill_between(x_values, y2_values, y1_values, 
						color=color, lw=0.1)
				else:
					plt.fill_between(x_values, y2_values, y1_values, 
						color=color, lw=0.1)

			annotation_y = -n-0.25
			
			plt.plot([x_values[0], x_values[-1]], [annotation_y, annotation_y],
				lw=20, c=color, solid_capstyle='butt')
			x_mid = (x_values[-1]+x_values[0])/2
			
			phase_txt = phase.replace("Delta", 'D')
			ax.text(x_mid, annotation_y-.25, phase_txt, ha='center', c='white')
			
			return x_values[-1]

		last_x = plot_phase_fills(ax, 'RG1')
		last_x = plot_phase_fills(ax, 'CG1', last_x)
		# last_x = plot_phase_fills(ax, 'Delta', last_x)
		last_x = plot_phase_fills(ax, 'postG1', last_x)
		last_x = plot_phase_fills(ax, 'H', last_x)

		n, m = H.shape

		yticks = np.arange(0, -n+1, -1)
		yticks_labels = ["$t_{"+str(-yt+1)+"}$" for yt in yticks]

		yticks = np.concatenate([yticks[0:3], yticks[7:10], yticks[-1:]])
		yticks_labels = np.concatenate([yticks_labels[0:3], yticks_labels[7:10], 
			yticks_labels[-1:]])
		yticks_labels[-1] = "$t_n$"
		yticks_labels[-2] = '.'
		yticks_labels[-3] = '.'
		yticks_labels[-4] = '.'

		ax.set_yticks(yticks+0.35)
		ax.set_yticklabels(yticks_labels, fontsize=9)
		ax.yaxis.set_tick_params(pad=5, length=0)
		ax.set_xlim(-.5, m+2.5)
		ax.set_ylim(-n-0.5, 1.5)
		ax.set_xticks([])

		from src.plot_helpers import hide_spines

		hide_spines(ax, False)

		# Create custom spines for the annotations
		bottom_y = -n+1.25
		right_x = m+2
		plt.plot([0, right_x], [bottom_y, bottom_y], c='black', lw=0.75)
		plt.plot([0, 0], [1, bottom_y], c='black', lw=0.75)
		plt.plot([0, right_x], [1, 1], c='black', lw=0.75)
		plt.plot([right_x, right_x], [1, bottom_y], c='black', lw=0.75)

		# Plot the backgorund
		from src.plot_helpers import plot_rect2
		plot_rect2(ax, 1, 1, right_x, bottom_y, zorder=0, color='white')

		from src.figure_configs import FiguresConfig
		plt.title("Convolution Kernel, $\\bf{H}$", fontsize=FiguresConfig.FIG_SUPTITLE_FONTSIZE)
		plt.xlabel("Single cell deconvolution time", fontsize=FiguresConfig.FIG_LABEL_FONTSIZE)
		plt.ylabel("Experimental time", fontsize=FiguresConfig.FIG_LABEL_FONTSIZE)


	def compute_clb2_deconvolution(self):
		from src.delta_config import load_delta_combined_gene_expression_config
		combined_ge_config = load_delta_combined_gene_expression_config()

		from src.model import Model
		gene_name = 'CLB2'
		combined_ge_model = Model(combined_ge_config, gene_name)
		combined_ge_model.deconvolve_find_optimal_gamma()
		from src.combined_chromatin_model import CombinedChromatinModel
		combined_model = CombinedChromatinModel(self.config1, self.config2)
		combined_model.load_combined_mnase_gene(gene_name)
		combined_model.gamma = 0.007
		combined_model.deconvolve()

		self.combined_model = combined_model
		self.combined_ge_model = combined_ge_model


	def plot_clb2_deconvolution(self):

		fig = self.combined_model.create_deconvolution_plots_abbreviated_flipped(
			ge_model=self.combined_ge_model, vmax=50)

	def plot_raw_clb2(self):

		# Let's plot the Raw data for figure 1, as timepoints 1, 2, 3 .. n
		from src.plot_helpers import hide_spines

		self.combined_model.chrom1_model.deconv_hist_unflattened.shape

		imgs = self.combined_model.chrom1_model.deconv_hist_unflattened
		plt_imgs = imgs[[0, 1, 2, 3, 4, -1]]

		fig, axs = plt.subplots(5, 1, figsize=(3, 6))
		n = len(axs)

		for i in range(n):
			ax = axs[i]
			
			if i == n-2:
				hide_spines(ax)
				ax.set_xlim(0, 1)
				ax.set_ylim(0, 1)
				ax.scatter([0.5, 0.5, 0.5], [0.3, 0.5, 0.7], c='black', s=5)
			else:
				self.combined_model.chrom1_model.plot_f_img(ax, plt_imgs[i], vmax=50)

				if i == n-1:
					ylabel = '$t_n$'
				else:
					ylabel = f'$t_{i+1}$'
				ax.set_ylabel(ylabel, rotation=0, ha='right', labelpad=8, fontsize=13)

