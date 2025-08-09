
import matplotlib.pyplot as plt
import numpy as np
from src.figure_configs import FiguresConfig
from src.chromatin_model import plot_img
from src.utils import mkdir_safe
from src.figure_configs import save_figure_for_paper
from src.GenomeDeconvolutionAnalysis import GenomeDeconvolutionAnalysis
from src.transcripts_dataset import load_transcripts_sets


SUBPANEL_COLOR = '#f5f5f5'

class FigureDeconvolution(object):
	"""Load and plot figures for the first result figure"""

	def __init__(self, output_dir):

		self.output_dir = output_dir

		self.save_dir = f'{output_dir}/fig_chromatin_deconvolution'
		self.fig_save_dir = f'{output_dir}/Figures/'
		mkdir_safe(self.save_dir)
		mkdir_safe(self.fig_save_dir)

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

		self.genome_deconvolution_analysis = GenomeDeconvolutionAnalysis(self.output_dir)
		self.genes, _ = load_transcripts_sets(self.output_dir)

	def run_and_save_all(self):

		save_dir = self.save_dir

		fig, axs = self.plot_H_matrices()
		save_figure_for_paper(f"{save_dir}/Kernel_H_diagram.png")

		# Load the region for the histogram and profiles
		# The locus plots will override
		self.chrom_model.load_combined_mnase_gene("CLN2")
		self.plot_mnase_reads_histogram()
		save_figure_for_paper(f"{save_dir}/MNase_2D_Histogram.png")

		self.plot_chromatin_profiles_G()
		save_figure_for_paper(f"{save_dir}/Chromatin_profiles_G.png")

		self.plot_deconvolved_phase_annotated()
		save_figure_for_paper(f"{save_dir}/Deconvolved_Profiles_F.png")

		# Save loci to disk
		self.plot_and_save_all_loci()

	def layout_figure_panel(self):
		layout_figure_panel(self.save_dir, self.fig_save_dir)

	def create_panels(self):

		self.layout_figure_panel()

		# Layout supplemental panels
		self.layout_supplemental_raw_locus()
		self.layout_supplemental_raw_deconvolved_thi22_locus()


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
		ax2.set_xlabel("Average single cell deconvolution time", fontsize=FiguresConfig.FIG_LABEL_FONTSIZE,
			labelpad=10)

		# Thicken the spine
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
		ax1.set_facecolor('white')
		ax2.set_facecolor('white')

		return fig, (ax1, ax2)


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


	def plot_deconvolved_phase_annotated(self):

		from src.plot_helpers import plot_rect2
		from src.plot_helpers import hide_spines


		# ------- New loading F code -----------

		# Loading example genomic locus
		from src.GenomeDeconvolutionAnalysis import GenomeDeconvolutionAnalysis
		
		genome_analysis = GenomeDeconvolutionAnalysis(self.output_dir)
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

			plot_img(ax, img_data, vmax=25, extent=[x1, x2, y1, y2], zorder=100)
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

	def plot_deconvolve_locus(self, chrom, span, title):
		"""
		Plot deconvolved locus data.
		
		Parameters:
		-----------
		chrom : int
			Chromosome number
		span : tuple
			(start, end) genomic coordinates
		title : str
			Plot title
			
		Returns:
		--------
		matplotlib.figure.Figure
			The created figure
		"""
		# Load the span data
		_ = self.genome_deconvolution_analysis.load_mnase_span(chrom, span)
		
		# Create the plot
		plotter = self.genome_deconvolution_analysis.plot_loaded_data(
			figsize=(11, 7), 
			title=title
		)
		
		return plt.gcf()

	def plot_raw_data_locus(self, chrom, span, title, replicate):
		"""
		Plot raw data locus for specified replicate.
		
		Parameters:
		-----------
		chrom : int
			Chromosome number
		span : tuple
			(start, end) genomic coordinates
		title : str
			Plot title
		replicate : int
			Replicate number (1 or 2)
			
		Returns:
		--------
		matplotlib.figure.Figure
			The created figure
		"""
		# Load the span data
		self.chrom_model.load_mnase_span(chrom, span)
		
		# Plot based on replicate
		if replicate == 1:
			fig = self.chrom_model.chrom1_model.plot_raw_data(figsize=(11, 16))
		elif replicate == 2:
			fig = self.chrom_model.chrom2_model.plot_raw_data(figsize=(11, 16))
		else:
			raise ValueError("Replicate must be 1 or 2")
		
		# Add title if provided
		if title:
			fig.suptitle(title, fontsize=FiguresConfig.FIG_SUPTITLE_FONTSIZE)
		
		return fig

	def plot_and_save_all_loci(self, plot_gene_name=None, mode='both'):
		"""
		Plot and save all loci (deconvolved and raw data for both replicates).
		Saves files with 'locus_' prefix using save_figure_for_paper function.
		"""
		# Define gene spans relative to TSS
		gene_spans = {
			'CLB5': (-4600, 2400),
			'THI22': (-2300, 2000)
		}
		
		# Define titles for each gene
		gene_titles = {
			'CLB5': "",
			'THI22': "Deconvolved region $\\it{THI22}$"
		}
		
		save_dir = self.save_dir
		
		for gene_name, (start_offset, end_offset) in gene_spans.items():

			if plot_gene_name is not None and not plot_gene_name == gene_name:
				continue

			# Get gene information
			gene = self.genes[self.genes['gene'] == gene_name].iloc[0]
			chrom = gene.chr
			span = (gene.TSS + start_offset, gene.TSS + end_offset)
			title = gene_titles[gene_name]
			
			if mode in ['both', 'deconvolved']:
				# Plot and save deconvolved locus
				fig_deconv = self.plot_deconvolve_locus(chrom, span, title)
				save_figure_for_paper(f"{save_dir}/locus_{gene_name}_deconvolved.png")
				plt.close(fig_deconv)
			
			# Plot and save raw data for both replicates
			if mode in ['both', 'raw']:
				for replicate in [1, 2]:
					raw_title = f"Raw data {title} - Replicate {replicate}"
					fig_raw = self.plot_raw_data_locus(chrom, span, raw_title, replicate)
					save_figure_for_paper(f"{save_dir}/locus_{gene_name}_raw_rep{replicate}.png")
					plt.close(fig_raw)
			
			print(f"Saved locus plots for {gene_name}")
		
		print("All locus plots saved successfully")

	def layout_supplemental_raw_locus(self):
		"""
		Layout CLB5 raw locus images horizontally taking up the full width.
		
		Layouts:
		- locus_CLB5_raw_rep2.png
		- locus_CLB5_raw_rep1.png
		"""
		from pipeline.figure_composer import FigureCompositor
		from pipeline.figure_composer_helpers import layout_images_horizontally, add_panel_labels_to_images

		compositor = FigureCompositor(1024, 790, debug_mode=True)

		image_paths = [
			f'{self.save_dir}/locus_CLB5_raw_rep1.png',
			f'{self.save_dir}/locus_CLB5_raw_rep2.png',
		]

		placed_images = layout_images_horizontally(
			compositor,
			image_paths,
			width_proportions=[1, 1],  # Equal width for both images
			between_padding=20,
			margin=(30, 30),
			image_keys=['CLB5_rep1', 'CLB5_rep2']  # Custom keys for the images
		)

		add_panel_labels_to_images(
			compositor, 
			compositor.placed_images,
			font_size=36,
			offset=(-10, -12)
		)

		compositor.save(f'{self.fig_save_dir}/Supplemental_CLB5_raw_locus.png')


	def layout_supplemental_raw_deconvolved_thi22_locus(self):
		"""
		Layout THI22 raw and deconvolved locus images horizontally taking up the full width.
		
		Layouts:
		- locus_THI22_raw_rep1.png
		- locus_THI22_raw_rep2.png  
		- locus_THI22_deconvolved.png
		"""
		from pipeline.figure_composer import FigureCompositor
		from pipeline.figure_composer_helpers import layout_images_horizontally, add_panel_labels_to_images

		compositor = FigureCompositor(1024, 380, debug_mode=True)

		image_paths = [
			f'{self.save_dir}/locus_THI22_raw_rep1.png',
			f'{self.save_dir}/locus_THI22_raw_rep2.png',
			f'{self.save_dir}/locus_THI22_deconvolved.png',
		]

		placed_images = layout_images_horizontally(
			compositor,
			image_paths,
			width_proportions=[0.45, 0.45, 1],  # Equal width for all three images
			between_padding=40,
			margin=(30, 30),
			image_keys=['THI22_rep1', 'THI22_rep2', 'THI22_deconvolved']  # Custom keys for the images
		)

		add_panel_labels_to_images(
			compositor, 
			compositor.placed_images,
			font_size=24,
			offset=(-20, -12)
		)

		compositor.save(f'{self.fig_save_dir}/Supplemental_THI22_raw_deconvolved_locus.png')


def layout_figure_panel(save_dir, figures_dir):
	from pipeline.figure_composer import FigureCompositor

	image_names = [
		# A, B
		'Branching_Diagram',
		'MNase_2D_Histogram', 
		
		# C
		'Chromatin_profiles_G',
		'Kernel_H_diagram',
		'Deconvolved_Profiles_F',
		
		# D - New panel
		'Deconvolved_Locus'
	]

	image_paths = [f"{save_dir}/{name}.png" for name in image_names]
	
	# Use project pathed branching diagram
	image_paths[0] = "diagrams/Branching_diagram.png"
	# Use the specific filename for the new locus panel
	image_paths[5] = f"{save_dir}/locus_CLB5_deconvolved.png"

	# Create compositor with same canvas size
	compositor = FigureCompositor(1024, 1460, debug_mode=True)

	# Layout parameters
	margin = 20
	top_margin = margin + 30
	canvas_width = 1024
	usable_width = canvas_width - 2 * margin  # 984px
	
	# ABC panels
	# note: Refactoring D to below ABC, so multiply by 1.0 usable width
	abc_width = int(1.0 * usable_width)  # 590px
	section_padding = 30  # padding between ABC and D sections
	
	# Panel D gets the remaining width
	d_width = usable_width
	d_x_position = margin + abc_width + section_padding
	
	# Calculate scaling factor for ABC panels
	# Current layout uses roughly 1000px for max width (A+padding+B or C panels)
	current_max_width = 1000  # Based on your current layout
	scaling_factor = abc_width / current_max_width  # 0.59
	
	print(f"ABC panel width: {abc_width}px")
	print(f"Panel D width: {d_width}px") 
	print(f"Scaling factor: {scaling_factor:.2f}")

	# ========== SCALED ABC PANELS ==========
	
	# A: Branch diagram (scaled)
	branch_width_scaled = int(640 * scaling_factor)  # 378px
	branch_img = compositor.place_image(image_paths[0], margin, top_margin, 
									   branch_width_scaled, None, 'branch')
	compositor.add_panel_label_to_image('branch', 'A', offset=(0, -40), font_size=30)
	
	branch_height = branch_img['logical_size'][1]
	
	# B: MNase histogram (scaled)
	padding_ab_scaled = int(20 * scaling_factor)  # 12px
	hist_width_scaled = int(340 * scaling_factor)  # 201px
	hist_img = compositor.place_image(image_paths[1], 
									 margin + branch_width_scaled + padding_ab_scaled, 
									 top_margin, hist_width_scaled, None, 'hist')
	compositor.add_panel_label_to_image('hist', 'B', offset=(0, -40), font_size=30)

	# C panels (scaled)
	vertical_pad = 50
	c_y_position = top_margin + branch_height + vertical_pad
	
	# C1: Raw profiles (scaled)
	g_width_scaled = int(325 * scaling_factor)  # 192px
	g_img = compositor.place_image(image_paths[2], margin, c_y_position, 
								  g_width_scaled, None, 'raw')
	compositor.add_panel_label_to_image('raw', 'C', offset=(0, -50), font_size=30)

	# C2: Kernel H diagram (scaled)
	padding_gh_scaled = int(15 * scaling_factor)  # 9px
	h_width_scaled = int(410 * scaling_factor)  # 242px
	h_img = compositor.place_image(image_paths[3], 
								  margin + g_width_scaled + padding_gh_scaled, 
								  c_y_position - int(7 * scaling_factor),  # scaled offset
								  h_width_scaled, None, 'H')

	# C3: Deconvolved profiles (scaled)
	f_width_scaled = int(256 * scaling_factor)  # 154px
	f_img = compositor.place_image(image_paths[4], 
								  margin + g_width_scaled + padding_gh_scaled + h_width_scaled - int(0 * scaling_factor), 
								  c_y_position, f_width_scaled, None, 'F')

	# ========== PANEL D: FULL HEIGHT ==========
	
	# Calculate full height for panel D (from top margin to bottom of C panels)
	c_bottom = 480
	
	# Place panel D
	y_position = g_img['logical_position'][1] + g_img['logical_size'][1] + vertical_pad
	d_img = compositor.place_image(image_paths[5], margin, y_position, 
								  width=d_width, name='locus_deconv')
	compositor.add_panel_label_to_image('locus_deconv', 'D', offset=(0, -40), font_size=30)

	# ========== LABELS (scaled positions) ==========
	
	# Main section titles
	compositor.add_panel_label("Cell cycling branching model", margin + 40, top_margin - 34,
							  font_size=22, font_type='semi_bold')

	compositor.add_panel_label("MNase data", 
							  hist_img['logical_position'][0] + int(50 * scaling_factor), 
							  top_margin - 34, font_size=22, font_type='semi_bold')

	compositor.add_panel_label("Chromatin deconvolution", margin + 40, 
							  g_img['logical_position'][1] - 40, 
							  font_size=22, font_type='semi_bold')

	compositor.add_panel_label("Deconvolved Locus", d_img['logical_position'][0] + 40, 
							  d_img['logical_position'][1] - 36, 
							  font_size=22, font_type='semi_bold')

	# Mathematical symbols (scaled positions)
	compositor.add_panel_label("=", 
							  h_img['logical_position'][0] - int(26 * scaling_factor), 
							  g_img['logical_position'][1] + int(150 * scaling_factor), 
							  font_size=32, font_type='semi_bold')

	compositor.add_panel_label("X", 
							  f_img['logical_position'][0] + int(8 * scaling_factor),
							  g_img['logical_position'][1] + int(162 * scaling_factor), 
							  font_size=18, font_type='semi_bold')

	# Save the figure
	save_path = f"{figures_dir}/Figure1_Deconvolution.png"
	compositor.save(save_path)
	print(f"Saved figure panel: {save_path}")
