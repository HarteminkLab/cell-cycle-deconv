
import matplotlib.pyplot as plt
import numpy as np
from src.figure_configs import FiguresConfig


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
					

				from src.plot_helpers import adjust_lightness_saturation
				# edgecolor = adjust_lightness_saturation(color, 0.6, 1.0)
				plt.fill_between(x_values, y2_values, y1_values, 
					facecolor=color, edgecolor=color, lw=0.5)

			annotation_y = -n-0.25
			
			phase_txt = phase.replace("RG1", 'Recovery G1')
			phase_txt = phase_txt.replace("CG1", 'Shared G1')
			phase_txt = phase_txt.replace("postG1", 'S G2/M')

			annotation_tuple = (x_values, annotation_y, phase_txt, color)
			
			return x_values[-1], annotation_tuple

		annotation_tuples = []
		last_x, annot = plot_phase_fills(ax, 'RG1')
		annotation_tuples.append(annot)

		last_x, annot = plot_phase_fills(ax, 'CG1', last_x)
		annotation_tuples.append(annot)

		last_x, annot = plot_phase_fills(ax, 'S', last_x)
		annotation_tuples.append(annot)

		last_x, annot = plot_phase_fills(ax, 'G2/M', last_x)
		annotation_tuples.append(annot)

		last_x, annot = plot_phase_fills(ax, 'H', last_x)
		annotation_tuples.append(annot)

		for (x_values, annotation_y, phase_txt, color) in annotation_tuples:
			
			plt.plot([x_values[0], x_values[-1]], [annotation_y, annotation_y],
				lw=20, c=color, solid_capstyle='butt')

			x_mid = (x_values[-1]+x_values[0])/2
			ax.text(x_mid, annotation_y-0.25, phase_txt, ha='center', c='white')

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
		right_x = m+2.25
		plt.plot([0, right_x], [bottom_y, bottom_y], c='black', lw=0.75)
		plt.plot([0, 0], [1, bottom_y], c='black', lw=0.75)
		plt.plot([0, right_x], [1, 1], c='black', lw=0.75)
		plt.plot([right_x, right_x], [1, bottom_y], c='black', lw=0.75)

		# Plot the backgorund
		from src.plot_helpers import plot_rect2
		plot_rect2(ax, 1, 1, right_x, bottom_y, zorder=0, color='white')

		plt.title("Convolution Kernel, $\\bf{H}$", fontsize=FiguresConfig.FIG_SUPTITLE_FONTSIZE)
		plt.xlabel("Single cell deconvolution time", fontsize=FiguresConfig.FIG_LABEL_FONTSIZE)
		plt.ylabel("Experimental time", fontsize=FiguresConfig.FIG_LABEL_FONTSIZE)


	def compute_deconvolution(self, gene_name, center_on_TSS=True):
		from src.config import load_configs_by_config_type, load_combined_gene_expression_by_config_type
		combined_ge_config = load_combined_gene_expression_by_config_type('shared')

		from src.model import Model
		combined_ge_model = Model(combined_ge_config, gene_name)
		combined_ge_model.deconvolve_find_optimal_gamma()

		from src.combined_chromatin_model import CombinedChromatinModel
		combined_model = CombinedChromatinModel(self.config1, self.config2)
		combined_model.chrom1_model.center_on_TSS = center_on_TSS
		combined_model.chrom2_model.center_on_TSS = center_on_TSS

		self.combined_model = combined_model

		combined_model.load_combined_mnase_gene(gene_name)
		combined_model.gamma = 0.0066
		combined_model.deconvolve()

		self.combined_ge_model = combined_ge_model


	def plot_deconvolution(self, should_smooth_data=True):
		fig = self.combined_model.create_deconvolution_plots_abbreviated_flipped(
			ge_model=self.combined_ge_model, vmax=10, should_smooth_data=should_smooth_data)

	def plot_raw_example(self):

		# Let's plot the Raw data for figure 1, as timepoints 1, 2, 3 .. n
		from src.plot_helpers import hide_spines

		self.combined_model.chrom1_model.deconv_hist_unflattened.shape

		img_indices = [0, 1, 2, 3, -2, -1]
		n = len(img_indices)

		fig, axs = plt.subplots(n, 2, figsize=(6, 6))
		axs = np.array(axs).T

		def plot_column_imgs(axs_col, chrom_model, title, show_labels=True):
			imgs = chrom_model.deconv_hist_unflattened
			plt_imgs = imgs[img_indices]

			for i in range(n):
				ax = axs_col[i]
				
				if i == n-3:
					hide_spines(ax)
					ax.set_xlim(0, 1)
					ax.set_ylim(0, 1)
					ax.scatter([0.5, 0.5, 0.5], [0.3, 0.5, 0.7], c='black', s=5)
				else:

					if i == 0:
						ax.set_title(title, fontsize=FiguresConfig.FIG_TITLE_FONTSIZE, pad=11)

					chrom_model.plot_f_img(ax, plt_imgs[i], vmax=25)

					if i == n-1:
						ylabel = '$t_n$'
					elif i == n-2:
						ylabel = '$t_{n-1}$'
					else:
						ylabel = f'$t_{i+1}$'

					if show_labels:
						ax.set_ylabel(ylabel, rotation=0, ha='right', labelpad=8, fontsize=13)

		plot_column_imgs(axs[0], self.combined_model.chrom1_model, title="Replicate 1")
		plot_column_imgs(axs[1], self.combined_model.chrom2_model, show_labels=False, title="Replicate 2")


	def plot_deconvolved_phase_annotated(self):

		from src.plot_helpers import plot_rect2
		from src.plot_helpers import hide_spines

		# Plot the deconvolved data as a stack for the diagram of the deconvolution
		chrom_model = self.combined_model.chrom1_model
		config = chrom_model.config
		rg1_i = config.get_Hpositions_for_phase('RG1')
		cg1_i = config.get_Hpositions_for_phase('CG1')
		s_i = config.get_Hpositions_for_phase('S')
		pg1_i = config.get_Hpositions_for_phase('postG1')

		imgs = chrom_model.get_f_images()


		i = 0

		fig = plt.figure(figsize=(5, 5))
		ax = plt.gca()

		w, h = 0.7, 0.2
		x, y = 0.5, -0.125
		padding = 0.2

		phases = ['RG1', 'CG1', 'S', 'G2/M']
		img_indices = [rg1_i[0], cg1_i[0], s_i[0], pg1_i[0]]
		img_indices = list(reversed(img_indices))

		plt_imgs =  imgs[img_indices]
		n = len(img_indices)

		# Flip the vertical indices such that we are plotting top to bottom
		phases = list(reversed(phases))

		from src.model import color_for_key
		flip = chrom_model.gene.strand == '-'

		for i in range(n):
			x1, x2, y1, y2 = x, x+w, y+i*(h+padding), y+h+i*(h+padding)

			if flip:
				img_data = np.flip(plt_imgs[i], axis=1)
			else:
				img_data = plt_imgs[i]

			chrom_model.plot_f_img(ax, img_data, vmax=10, extent=[x1, x2, y1, y2], 
				should_smooth_data=True)
			plt.plot([x1+w/2., x1+w/2.], [y1, y2], c='black', lw=1, alpha=0.25)
			plot_rect2(ax, x1, y1, x2, y2, edgecolor='black', fill=None, lw=0.5, zorder=100)
			phase = phases[i]

			# Stack offset underneath for the appearance of a set of stacked images
			stack_offset = 0.02
			num_stack = 4
			for j in range(num_stack, 0, -1):
				plot_rect2(ax, x1+j*stack_offset, y1+j*stack_offset, 
							   x2+j*stack_offset, y2+j*stack_offset, 
						   edgecolor='black', color='#AB9B7F',
						   lw=0.5, zorder=0)
				
				from matplotlib.patches import Rectangle, FancyBboxPatch


				rounded_rect = FancyBboxPatch((x1-0.3, y1-0.04), 1.13, 0.37,
					boxstyle='Round, pad=0, rounding_size=0.05', color=color_for_key(phase),
							 alpha = 1., zorder=-1)
				
				rounded_patch = ax.add_patch(rounded_rect)

				phase_name = phase.replace('CG1', 'SG1')
				ax.text(x1-0.15, (y1+y2)/2+0.02, phase_name, ha='center', color='white')

		plt.xlim(-0.25, 1.5)
		plt.ylim(-0.25, 1.5)
		hide_spines(ax)

	def plot_mnase_reads_histogram(self):
		from src.DensityScatterPlotter import DensityScatterPlotter

		chrom_model = self.combined_model.chrom1_model
		reads = chrom_model.locus_reads
		reads = reads[reads['sample'] == 50]
		index = 5

		plt.figure(figsize=(5, 4.5))
		plt.subplot(2, 1, 1)

		ax = plt.gca()
		dsc_plotter = DensityScatterPlotter()
		x, y = reads.mid, reads['length']

		flip = chrom_model.gene.strand == '-'

		plt.scatter(x, y, s=9, edgecolors='#afafaf', facecolor='none')

		dsc_plotter.set_data(x.values, y.values)
		dsc_plotter.bw = (5, 10)
		dsc_plotter.cmap = 'magma_r'
		dsc_plotter.plot_ax(ax)
		dsc_plotter.s = 7
		ax.set_yticks(np.arange(50, 300, 100))
		plt.ylim(0, 250)

		xlims = chrom_model.bin_extents[0], chrom_model.bin_extents[1]

		p1 = chrom_model.computed_plus_one
		xticks = np.arange(p1-1000, p1+1000, 250)

		xtick_labels = [f'+{x-p1}' if x>p1 else str(x-p1) for x in xticks]
		xtick_labels = ['+1' if x == '0' else x for x in xtick_labels]
		ax.set_xticks([])

		if flip:
			plt.xlim(xlims[1], xlims[0])
		else:
			plt.xlim(*xlims)

		plt.ylabel("Fragment length, nt")
		ax.axvline(p1, c='gray', lw=2, alpha=0.5)
		plt.title("MNase-seq reads", fontsize=FiguresConfig.FIG_TITLE_FONTSIZE, pad=9)

		plt.subplot(2, 1, 2)
		ax  = plt.gca()
		chrom_model.exact_bins.shape

		img = chrom_model.deconv_hist_unflattened[index]

		if flip:
			img = np.flip(img, axis=1)

		plt.imshow(img, origin='lower', cmap='magma_r',
				  aspect='auto', extent=chrom_model.bin_extents, vmax=25)
		ax.set_xticks(xticks)
		ax.set_xticklabels(xtick_labels)
		ax.set_yticks(np.arange(50, 300, 100))
		ax.set_xlim(*xlims)
		ax.axvline(p1, c='gray', lw=2, alpha=0.5)
		plt.xlabel("Genomic position, nt")
		plt.ylabel("Fragment length, nt")
		plt.title("2D Histogram", fontsize=FiguresConfig.FIG_TITLE_FONTSIZE, pad=9)

		plt.subplots_adjust(hspace=0.5)


	def print_posteriors(self):
		"""Print posteriors for supplemental table of CLOCCS fits"""
		from src.create_models import ModelCreation

		posteriors1 = 'data/2019_cloccs_fits/yl_2019_replicate1/posteriors.txt'
		posteriors2 = 'data/2019_cloccs_fits/yl_2019_replicate2/posteriors.txt'

		def load_posterior_mapping(posteriors1):
			with open(posteriors1, 'r') as p_file:
				lines = p_file.readlines()

			parameters = ['mu0', 'delta', 'sigma0', 'sigmav', 'lambda',
				'gamma1', 'gamma2', 'mua1', 'sigmaa1', 'mua2', 'sigmaa2', 
						  'mut', 'sigmat', 'halted']

			posterior_mapping = []
			for l in lines:

				line_tokens = l.strip().split(' ')
				line_tokens = [x for x in line_tokens if x != '']

				param_name = line_tokens[0]
				if param_name in parameters:
					_, mean, q025, q975, _ = tuple(line_tokens)
					mean, q025, q975 = float(mean), float(q025), float(q975)

					posterior_mapping.append((param_name, mean, q025, q975))
			return posterior_mapping

		mapping1 = load_posterior_mapping(posteriors1)
		mapping2 = load_posterior_mapping(posteriors2)

		parameter_latex_names = ["$\\mu0$", "$\\delta$", "$\\sigma_0$", "$\\sigma_v$", "$\\lambda$", "$\\gamma_1$", "$\\gamma_2$", "$\\mu_{\\alpha1}$", "$\\sigma_{\\alpha1}$", "$\\mu_{\\alpha2}$", "$\\sigma_{\\alpha2}$", "$\\mu_t$", "$\\sigma_t$", "halted"]

		for i in range(len(mapping1)):

			_, mean_1, q025_1, q975_1 = mapping1[i]
			_, mean_2, q025_2, q975_2 = mapping2[i]

			param_name = parameter_latex_names[i]

			print(f"{param_name}\t&\t{mean_1:.3f}\t&\t({q025_1:.3f}," +
				  f"{q975_1:.3f})\t&&\t{mean_2:.3f}\t&\t({q025_2:.3f},{q975_2:.3f}) \\\\")
