
import matplotlib.pyplot as plt
import numpy as np
from src.figure_configs import FiguresConfig
from src.chromatin_model import plot_img
from src.utils import mkdir_safe
from src.figure_configs import save_figure_for_paper

SUBPANEL_COLOR = '#f5f5f5'

class Figure1Deconvolution(object):
	"""Load and plot figures for the first result figure"""

	def __init__(self, output_dir):

		self.output_dir = output_dir

		self.save_dir = f'{output_dir}/fig_chromatin_deconvolution'
		mkdir_safe(self.save_dir)

		# Create the H for the updated model config to include the H config
		from src.config import load_default_chrom_configs
		config1, config2 = load_default_chrom_configs()
		from src.combined_chromatin_model import CombinedChromatinModel

		H1 = config1.calculate_H()
		H2 = config2.calculate_H()
		self.H = np.concatenate([H1, H2])
		self.config1 = config1
		self.config2 = config2
		self.chrom_model = CombinedChromatinModel(config1, config2)
		self.chrom_model.load_combined_mnase_gene("CLN2")

	def run_and_save_all(self, chromatin_data_path=
			'output/draft3_run/chromatin_deconvolution_partial_daughter/deconvolution_data/'):

		save_dir = self.save_dir
		fig, axs = self.plot_H_matrices()
		save_figure_for_paper(f"{save_dir}/Kernel_H_diagram.png")

		self.plot_mnase_reads_histogram()
		save_figure_for_paper(f"{save_dir}/MNase_2D_Histogram.png")

		self.plot_chromatin_profiles_G()
		save_figure_for_paper(f"{save_dir}/Chromatin_profiles_G.png")

		print(f"todo: temporary chromatin data path {chromatin_data_path}")
		self.plot_deconvolved_phase_annotated(chromatin_data_path)
		save_figure_for_paper(f"{save_dir}/Deconvolved_Profiles_F.png")

	def create_panel(self):
		layout_figure_panel(self.save_dir)


	def plot_H(self):
		from src.config import plot_H
		plot_H(self.config1, self.H)

	def plot_H_matrices(self, figsize=(7, 6), padding=0.0):
		"""
		Plot H1 and H2 matrices separately on two vertically stacked axes.
		
		Parameters:
		-----------
		figsize : tuple
			Figure size (width, height)
		padding : float
			Padding between the two plots
		"""
		from src.helpers import combine_with_bins
		from src.plot_helpers import color_for_key, hide_spines, plot_rect2, adjust_lightness_saturation
		
		# Get matrices and calculate dimensions
		H1 = self.config1.calculate_H()
		H2 = self.config2.calculate_H()
		n1, m = H1.shape
		n2, _ = H2.shape
		
		# Create figure with two subplots
		fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
									   gridspec_kw={'height_ratios': [n1, n2],
												   'hspace': padding/10})
		
		# Define helper function to plot H matrix on a given axis
		def plot_H_on_axis(ax, H, config, add_annotations=False):
			annotation_tuples = []
			last_x = 0
			
			# Plot each phase
			for phase in ['RG1', 'CG1', 'DG1', 'S', 'G2M', 'H']:
				last_x, annot = plot_phase_fills(ax, H, config, phase, last_x)
				annotation_tuples.append(annot)
			
			# Add phase annotations if requested
			if add_annotations:
				for (x_values, annotation_y, phase_txt, color) in annotation_tuples:
					ax.plot([x_values[0], x_values[-1]], [annotation_y, annotation_y],
							lw=20, c=color, solid_capstyle='butt')
					
					x_mid = (x_values[-1] + x_values[0]) / 2
					ax.text(x_mid, annotation_y-0.25, phase_txt, ha='center', c='white')
			
			# Set up axis limits and appearance
			n, m = H.shape
			ax.set_xlim(0, m-1)

			if add_annotations:
				ax.set_ylim(-n, 1.5)
			else:
				ax.set_ylim(-n+1.5, 1.5)

			ax.set_yticks([])
			ax.yaxis.set_tick_params(pad=5, length=0)
			ax.set_xticks([])

			return ax
		
		# Helper function to plot the fill_betweens for a specific phase
		def plot_phase_fills(ax, H, config, phase, x_offset=0):
			phase_indices = config.get_Hpositions_for_phase(phase)
			cur_H = H[:, phase_indices]
			n, _ = cur_H.shape

			# Combine rows to make the plot more clear
			combined_cur_H = combine_with_bins(cur_H, np.arange(0, n, 1), axis=0)
			n, m = combined_cur_H.shape

			# Plot fill betweens for each row bunch
			scale = 10
			x_values = phase_indices
			color = color_for_key(phase)

			# The indices between phases are separated by 1 index,
			# thus we need to expand the edges by a half
			x_values[0] = x_values[0]-0.5
			x_values[-1] = x_values[-1]+0.5
			
			for i in range(n):
				y2_values = combined_cur_H[i] * scale - i
				y1_values = -i
				
				# Special case for halted phase - make it larger for visibility
				if phase == 'H':
					scale = 3.0
					y2_values = np.repeat(combined_cur_H[i] * scale - i, 2)
					y1_values = np.array([-i, -i])
					x_values = [x_offset, x_offset + 1]
				
				ax.fill_between(x_values, y2_values, y1_values, 
							  facecolor=color, edgecolor=color, lw=0.5)
			
			# Prepare annotation information
			annotation_y = -n - 0.25

			# Format phase names for readability
			phase_mapping = {
				'RG1': 'Recovery G1',
				'CG1': 'Mother G1',
				'DG1': 'Daughter G1',
				'S': 'S',
				'G2M': 'G2M',
				'H': '',
				'postG1': 'S G2/M'
			}
			phase_txt = phase_mapping.get(phase, phase)
			annotation_tuple = (x_values, annotation_y, phase_txt, color)
			
			return x_values[-1], annotation_tuple
		
		# Plot H1 on top axis
		plot_H_on_axis(ax1, H1, self.config1, add_annotations=False)
		
		# Plot H2 on bottom axis with phase annotations
		plot_H_on_axis(ax2, H2, self.config2, add_annotations=True)
		ax2.set_xlabel("Single cell deconvolution time", fontsize=FiguresConfig.FIG_LABEL_FONTSIZE,
			labelpad=10)

		# Thicken the spine
		# Or more concisely:
		for ax in [ax1, ax2]:
			for spine in ax.spines.values():
				spine.set_linewidth(1.25)

		ax1.set_ylabel("Replicate 1", fontsize=10,
			labelpad=5)

		ax2.set_ylabel("Replicate 2", fontsize=10,
			labelpad=5)
		
		# Add overall figure labels
		fig.suptitle("Convolution kernel, $\\bf{H}$", 
			fontsize=FiguresConfig.FIG_SUPTITLE_FONTSIZE, y=0.94, fontweight='demi')

		fig.text(0.06, 0.5, "Experiment time", va='center', rotation='vertical', 
			 fontsize=FiguresConfig.FIG_LABEL_FONTSIZE)

		from matplotlib.patches import FancyBboxPatch
		bg_ax = fig.add_axes([0, 0, 1, 1], zorder=-1)
		bg_ax.axis('off')  # Hide axes

		# Add rounded rectangle with light gray background
		# Adjust the parameters as needed for desired appearance
		rect = FancyBboxPatch(
			(0.01, 0.02),                         # (x, y) position
			0.95, 0.96,                             # width, height
			boxstyle="round,pad=0,rounding_size=0.02", # Rounded corners
			facecolor=SUBPANEL_COLOR,
			linewidth=0,
			alpha=1.,
			zorder=-1,
			transform=bg_ax.transAxes,
			clip_on=False
		)

		bg_ax.add_patch(rect)

		return fig, (ax1, ax2)

	def plot_deconvolution(self, should_smooth_data=True):
		fig = self.combined_model.create_deconvolution_plots_abbreviated_flipped(
			ge_model=self.combined_ge_model, vmax=10, should_smooth_data=should_smooth_data)

	def plot_chromatin_profiles_G(self):

		# Let's plot the Raw data for figure 1, as timepoints 1, 2, 3 .. n
		from src.plot_helpers import hide_spines

		img_indices = [0, 1, 2, 3, -2, -1]
		n = len(img_indices)

		fig, axs = plt.subplots(n, 2, figsize=(6, 6))
		axs = np.array(axs).T
		plt.suptitle("Population-level\nchromatin profiles, $\\bf{G}$", 
			fontsize=FiguresConfig.FIG_SUPTITLE_FONTSIZE,
			fontweight='demi', y=1.05)

		# Create a new axes that spans the entire figure for the background
		from matplotlib.patches import FancyBboxPatch
		bg_ax = fig.add_axes([0, 0, 1, 1], zorder=-1)
		bg_ax.axis('off')  # Hide axes

		SUBPANEL_COLOR = '#f5f5f5'

		# Add rounded rectangle with light gray background
		# Adjust the parameters as needed for desired appearance
		rect = FancyBboxPatch(
			(0.01, 0.05),                         # (x, y) position
			0.95, 1.05,                             # width, height
			boxstyle="round,pad=0,rounding_size=0.02", # Rounded corners
			facecolor=SUBPANEL_COLOR,
			linewidth=0,
			alpha=1.,
			zorder=-1,
			transform=bg_ax.transAxes,
			clip_on=False
		)

		bg_ax.add_patch(rect)

		def plot_column_imgs(axs_col, chrom_model, title, show_labels=True):
			imgs = chrom_model.G_imgs
			plt_imgs = imgs[img_indices]

			for i in range(n):
				ax = axs_col[i]
				
				if i == n-3:
					hide_spines(ax)
					ax.set_xlim(0, 1)
					ax.set_ylim(0, 1)

					# Plot dots to signify time series
					ax.scatter([0.5, 0.5, 0.5], [0.3, 0.5, 0.7], c='black', s=5)
					ax.set_facecolor(SUBPANEL_COLOR)
				else:

					if i == 0:
						ax.set_title(title, fontsize=FiguresConfig.FIG_TITLE_FONTSIZE, pad=11)

					plot_img(ax, plt_imgs[i], vmax=25)
					ax.set_xticks([])
					ax.set_yticks([])

					if i == n-1:
						ylabel = '$t_n$'
					elif i == n-2:
						ylabel = '$t_{n-1}$'
					else:
						ylabel = f'$t_{i+1}$'

					if show_labels:
						ax.set_ylabel(ylabel, rotation=0, ha='right', labelpad=8, fontsize=13)

		plot_column_imgs(axs[0], self.chrom_model.chrom1_model, title="Replicate 1")
		plot_column_imgs(axs[1], self.chrom_model.chrom2_model, show_labels=False, title="Replicate 2")


	def plot_deconvolved_phase_annotated(self, chromatin_data_path):

		from src.plot_helpers import plot_rect2
		from src.plot_helpers import hide_spines


		# ------- New loading F code -----------

		# Loading example genomic locus
		from src.GenomeDeconvolutionAnalysis import GenomeDeconvolutionAnalysis
		
		genome_analysis = GenomeDeconvolutionAnalysis(chromatin_data_path)
		span = (10000, 11001)
		chrom = 1
		imgs, loaded_span = genome_analysis.load_mnase_span(chrom, span)

		# ---------------------------------------------------------

		# Plot the deconvolved data as a stack for the diagram of the deconvolution
		chrom_model = self.chrom_model.chrom1_model
		config = chrom_model.config
		rg1_i = config.get_Hpositions_for_phase('RG1')
		cg1_i = config.get_Hpositions_for_phase('CG1')
		dg1_i = config.get_Hpositions_for_phase('DG1')
		s_i = config.get_Hpositions_for_phase('S')
		pg1_i = config.get_Hpositions_for_phase('postG1')

		i = 0

		fig = plt.figure(figsize=(3.5, 5.5))
		ax = plt.gca()

		w, h = 0.7, 0.2
		x, y = 0.5, -0.125
		padding = 0.23

		phases = ['RG1', 'CG1', 'DG1', 'S', 'G2/M']
		phase_rename_mapping = {
			'RG1': 'Recovery G1',
			'CG1': 'Mother G1',
			'DG1': 'Daughter G1',
		}
		img_indices = [rg1_i[0], cg1_i[0], dg1_i[0], s_i[0], pg1_i[0]]
		img_indices = list(reversed(img_indices))

		plt_imgs =  imgs[img_indices]
		n = len(img_indices)

		# Flip the vertical indices such that we are plotting top to bottom
		phases = list(reversed(phases))

		from src.plot_helpers import color_for_key

		for i in range(n):
			x1, x2, y1, y2 = x, x+w, y+i*(h+padding), y+h+i*(h+padding)

			img_data = plt_imgs[i]

			plot_img(ax, img_data, vmax=10, extent=[x1, x2, y1, y2], zorder=100)
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
				rounded_rect = FancyBboxPatch((x1-0.5, y1-0.04), 1.35, 0.37,
					boxstyle='Round, pad=0, rounding_size=0.05', color=color_for_key(phase),
							 alpha = 1., zorder=-1)
				
				rounded_patch = ax.add_patch(rounded_rect)

				phase_name = phase

				if phase in phase_rename_mapping.keys():
					phase_name = phase_rename_mapping[phase]

				ax.text(0.25, (y1+y2)/2+0.02, phase_name, ha='center', color='white')

		xlims = -0.1, 1.4
		ylims = -0.25, 2

		plt.xlim(*xlims)
		plt.ylim(*ylims)
		hide_spines(ax)
		ax.set_facecolor(SUBPANEL_COLOR)

		from matplotlib.patches import FancyBboxPatch
		bg_ax = fig.add_axes([0, 0, 1, 1], zorder=-1)
		bg_ax.axis('off')  # Hide axes

		# Add rounded rectangle with light gray background
		# Adjust the parameters as needed for desired appearance
		rect = FancyBboxPatch(
			(0.1, 0.09), # x, y
			0.83, 0.92, # w, h
			boxstyle="round,pad=0,rounding_size=0.02", # Rounded corners
			facecolor=SUBPANEL_COLOR,
			linewidth=0,
			alpha=1.,
			zorder=-1,
			transform=bg_ax.transAxes,
			clip_on=False
		)
		bg_ax.add_patch(rect)

		ax.set_title("Average single\ncell profile, $\\bf{F}$", fontsize=16, 
			fontweight='demi')


	def plot_mnase_reads_histogram(self):
		from src.DensityScatterPlotter import DensityScatterPlotter

		chrom_model = self.chrom_model.chrom1_model
		reads = chrom_model.locus_reads
		reads = reads[reads['sample'] == 50]
		index = 5

		plt.figure(figsize=(5, 4.5))
		plt.subplot(2, 1, 1)

		ax = plt.gca()
		dsc_plotter = DensityScatterPlotter()
		x, y = reads.mid, reads['length']

		plt.scatter(x, y, s=9, edgecolors='#afafaf', facecolor='none')

		dsc_plotter.set_data(x.values, y.values)
		dsc_plotter.bw = (5, 10)
		dsc_plotter.cmap = 'magma_r'
		dsc_plotter.plot_ax(ax)
		dsc_plotter.s = 7
		ax.set_yticks(np.arange(50, 300, 100))
		plt.ylim(0, 250)

		xlims = chrom_model.bin_extents[0], chrom_model.bin_extents[1]

		plt.ylabel("Fragment length, bp")
		plt.title("MNase-seq reads", fontsize=FiguresConfig.FIG_TITLE_FONTSIZE, pad=7,
			fontweight='demi')
		plt.xticks([])

		plt.subplot(2, 1, 2)
		ax  = plt.gca()
		chrom_model.exact_bins.shape

		img = chrom_model.G_imgs[index]
		plt.imshow(img, origin='lower', cmap='magma_r',
				  aspect='auto', extent=chrom_model.bin_extents, vmax=25)

		# ax.set_xticks(xticks)
		# ax.set_xticklabels(xtick_labels)

		ax.set_yticks(np.arange(50, 300, 100))
		ax.set_xlim(*xlims)
		plt.xlabel("Genomic position, bp")
		plt.ylabel("Fragment length, bp")
		plt.title("2D Histogram", fontsize=FiguresConfig.FIG_TITLE_FONTSIZE, pad=7,
			fontweight='demi')

		plt.subplots_adjust(hspace=0.3)


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


def layout_figure_panel(save_dir):

	from pipeline.figure_composer import FigureCompositor

	image_names = [
	    # A, B
	    'Branching_Diagram',
	    'MNase_2D_Histogram', 
	    
	    # C
	    'Chromatin_profiles_G',
	    'Kernel_H_diagram',
	    'Deconvolved_Profiles_F',
	]

	image_paths = [f"{save_dir}/{name}.png" for name in image_names]

	# Create compositor with a scale factor of 4
	# Logical canvas size is 1024x800, but actual output will be 4096x3200
	compositor = FigureCompositor(1024, 820, debug_mode=True)

	margin = 20

	top_margin = margin+30

	branch_img = compositor.place_image(image_paths[0], margin, top_margin, 640, None, 'branch')
	compositor.add_panel_label_to_image('branch', 'A', offset=(0, -40),
	                                   font_size=36)

	branch_width = branch_img['logical_size'][0]
	branch_height = branch_img['logical_size'][1]
	padding = 20
	hist_img = compositor.place_image(image_paths[1], margin+branch_width+padding, top_margin, 340, None, 
	    'hist')
	compositor.add_panel_label_to_image('hist', 'B', offset=(0, -40),
	                                   font_size=36)

	# ---------- C panels

	vertical_pad = 70
	g_img = compositor.place_image(image_paths[2], margin, 
	    top_margin+branch_height+vertical_pad, 
	    325, None, 
	    'raw')
	compositor.add_panel_label_to_image('raw', 'C', offset=(0, -60),
	    font_size=36)

	g_width = g_img['logical_size'][0]
	padding = 15
	h_img = compositor.place_image(image_paths[3], 
	    margin+g_width+padding, 
	    top_margin+branch_height+vertical_pad-7,
	    410, None, 
	    'H')

	h_width = h_img['logical_size'][0]
	f_img = compositor.place_image(image_paths[4], 
	    margin+g_width+padding+h_width-10, 
	    top_margin+branch_height+vertical_pad, 
	    260, None, 
	    'F')

	compositor.add_panel_label("Cell cycling branching model", margin+40, top_margin-30, 
	    font_size=24,
	    font_type='semi_bold')

	compositor.add_panel_label("MNase data", 
	    hist_img['logical_position'][0]+40, 
	    top_margin-30, 
	    font_size=24,
	    font_type='semi_bold')


	compositor.add_panel_label("Chromatin deconvolution", 
	    margin+40, 
	    g_img['logical_position'][1]-50, 
	    font_size=24,
	    font_type='semi_bold')

	compositor.add_panel_label("=", 
	    h_img['logical_position'][0]-26, 
	    g_img['logical_position'][1]+150, 
	    font_size=36,
	    font_type='semi_bold')

	compositor.add_panel_label("X", 
	    f_img['logical_position'][0]+8,
	    g_img['logical_position'][1]+150, 
	    font_size=20,
	    font_type='semi_bold')

	save_path = f"{save_dir}/Fig1.png"
	compositor.save(save_path)

	print(f"Saved figure panel: {save_path}")
