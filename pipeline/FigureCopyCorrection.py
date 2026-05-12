
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.utils import mkdir_safe

from src.replication_timing import ReplicationTiming
from src.figure_configs import FiguresConfig
from src.figure_configs import save_figure_for_paper
from src.plot_helpers import adjust_lightness_saturation


class FigureCopyCorrection():
	"""Create figures for the replication deconvolution"""

	def __init__(self, output_directory):
		self.output_directory = output_directory
		self.save_dir = f'{self.output_directory}/fig_copy_correction'
		self.figures_dir = f'{self.output_directory}/Figures'

		# Set the math text parameters to computer modern
		plt.rcParams['mathtext.fontset'] = 'cm'

		mkdir_safe(self.save_dir)

		self.replication_timings = ReplicationTiming(output_directory)

	def load_copy_correction(self):
		from pipeline.copy_correction_analysis import CopyCorrectionAnalysis

		self.copy_correction_analysis = CopyCorrectionAnalysis(self.output_directory)
		self.copy_correction_analysis.load_means_all_chromosomes()
		self.copy_correction_analysis.compute_ptrs()
		self.copy_correction_analysis.select_example_windows()
		self.copy_correction_analysis.initialize_replication_time_colormaps()

	def plot_all_ptrs(self):
		self.copy_correction_analysis.plot_ptrs()
		save_figure_for_paper(f"{self.save_dir}/Copy_Correction_PTRs.png")

	def plot_sample_curves(self):
		self.copy_correction_analysis.plot_sample_curves()
		save_figure_for_paper(f"{self.save_dir}/Copy_Correction_Examples.png")

	def plot_replication_results_chr4(self):
		self.replication_timings.plot_chrom_timing(chrom=4)
		save_figure_for_paper(f"{self.save_dir}/Replication_Timing_chr4.png")

		self.replication_timings.plot_muller_correlation()
		save_figure_for_paper(f"{self.save_dir}/Replication_Timing_correlation.png")


	def layout_panel(self, canvas_width=1024, canvas_height=980, margins=20, 
				 column_padding=30, row_padding=25, debug_mode=True):
		"""
		Create a 2x2 composite figure panel with replication timing and copy correction plots.
		"""
		from pipeline.figure_composer import FigureCompositor
		from pipeline.figure_composer_helpers import layout_images_horizontally, layout_images_vertically, add_panel_labels_to_images
		
		# Create compositor 
		compositor = FigureCompositor(canvas_width, canvas_height, debug_mode=debug_mode)

		# ----------- Replication model ------------
		replication_figures_dir = f"{self.output_directory}/fig_replication"

		image_paths = [
			f'{replication_figures_dir}/DNA_replication_diagram.png',
			f'{replication_figures_dir}/Replication_diagram.png',
		]

		placed_images = layout_images_vertically(
			compositor,
			image_paths,
			height_proportions=[0.35, 0.4],
			between_padding=16,
			margin=(30, 30),
			image_keys=['DNA', 'Replication']  # Custom keys for the images
		)

		# --------- Replication and copy correction
		
		# Define file paths for your four figures
		replication_chr4_path = f'{self.save_dir}/Replication_Timing_chr4.png'
		replication_corr_path = f'{self.save_dir}/Replication_Timing_correlation.png'
		copy_ptrs_path = f'{self.save_dir}/Copy_Correction_PTRs.png'
		copy_examples_path = f'{self.save_dir}/Copy_Correction_Examples.png'

		replication_imgs_bottom = compositor.placed_images['Replication']['logical_position'][1] + \
			compositor.placed_images['Replication']['logical_size'][1]+row_padding
		
		# Layout top row (replication timing figures)
		top_row_images = layout_images_horizontally(
			compositor,
			[replication_chr4_path, replication_corr_path],
			width_proportions=[0.75, 0.25],  # Equal width for both top images
			between_padding=column_padding,
			margin=(margins, replication_imgs_bottom),
			image_keys=['replication_chr4', 'replication_corr']
		)
		
		# Calculate starting y position for bottom row
		top_row_height = max([img['logical_size'][1] for img in top_row_images.values()])
		bottom_row_start_y = replication_imgs_bottom + margins + top_row_height
		
		# Layout bottom row (copy correction figures)
		bottom_row_images = layout_images_horizontally(
			compositor,
			[copy_ptrs_path, copy_examples_path],
			width_proportions=[0.24, 0.76],  # Equal width for both bottom images
			between_padding=column_padding+20,
			offsets=[(20, 0), (0, 0)],
			margin=(margins, bottom_row_start_y),
			image_keys=['copy_ptrs', 'copy_examples']
		)
		
		def _retrieve_subset_dict(original_dict, keys_to_extract):
			return {key: original_dict[key] for key in keys_to_extract if key in original_dict}

		ab_imgs = _retrieve_subset_dict(compositor.placed_images,
			['DNA', 'Replication'])

		cdf_imgs = _retrieve_subset_dict(compositor.placed_images,
			['replication_chr4', 'replication_corr', 'copy_examples'])
		e_imgs = _retrieve_subset_dict(compositor.placed_images,
			['copy_ptrs'])
	
		font_size=36

		# Add panel labels
		add_panel_labels_to_images(
			compositor,
			ab_imgs,
			labels='ab',
			font_size=font_size,
			offset=(-10, 23),
			font_type='bold',
			color=(0, 0, 0)
		)

		# Add panel labels
		add_panel_labels_to_images(
			compositor,
			cdf_imgs,
			labels='cdf',
			font_size=font_size,
			offset=(-10, 23),
			font_type='bold',
			color=(0, 0, 0)
		)

		# Add panel labels
		add_panel_labels_to_images(
			compositor,
			e_imgs,
			labels='e',
			font_size=font_size,
			offset=(-20, 23),
			font_type='bold',
			color=(0, 0, 0)
		)
		
		# Save the composite figure
		output_path = f'{self.figures_dir}/Figure2_Replication_Copy_Correction.png'
		compositor.save(output_path)
		
		return compositor

	def plot_increased_ptr(self):
		# Plot the genomic locations for increased PTRs
		self.copy_correction_analysis.plot_genomic_location_ptr_changes()
		save_figure_for_paper(f'{self.save_dir}/increased_ptr_chromosome_location.png')


	def plot_all(self):
		self.plot_replication_results_chr4()
		self.plot_all_ptrs()
		self.plot_sample_curves()

	def layout_supplemental_panel(self):
		from pipeline.figure_composer import FigureCompositor
		from pipeline.figure_composer_helpers import layout_images_horizontally,\
			add_panel_labels_to_images

		compositor = FigureCompositor(1024,460, debug_mode=True)

		# ----------- Replication model ------------
		replication_figures_dir = f"{self.output_directory}/fig_replication"

		image_paths = [
			f'{replication_figures_dir}/Replication_components.png',
		]

		placed_images = layout_images_horizontally(
			compositor,
			image_paths,
			between_padding=16,
			margin=(30, 30),
			image_keys=['Replication_Detail']
		)

		# add_panel_labels_to_images(
		# 	compositor, 
		# 	compositor.placed_images,
		# 	font_size=36,
		# 	offset=(-10, 8)
		# )

		compositor.save(f'{self.figures_dir}/Supplemental2.5_Replication_Detail.png')
