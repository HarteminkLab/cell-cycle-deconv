
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.utils import mkdir_safe

from src.replication_timing import ReplicationTiming
from src.figure_configs import FiguresConfig
from src.figure_configs import save_figure_for_paper
from src.plot_helpers import adjust_lightness_saturation


class Figure4CopyCorrection():
	"""Create figures for the replication deconvolution"""

	def __init__(self, output_directory):
		self.output_directory = output_directory
		self.save_dir = f'{self.output_directory}/copy_correction_figures'

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


	def layout_panel(self, canvas_width=1024, canvas_height=760, margins=20, 
				 column_padding=30, row_padding=30, debug_mode=True):
		"""
		Create a 2x2 composite figure panel with replication timing and copy correction plots.
		"""
		from pipeline.figure_composer import FigureCompositor
		from pipeline.figure_composer_helpers import layout_images_horizontally, layout_images_vertically, add_panel_labels_to_images
		
		# Create compositor 
		compositor = FigureCompositor(canvas_width, canvas_height, debug_mode=debug_mode)
		
		# Define file paths for your four figures
		replication_chr4_path = f'{self.save_dir}/Replication_Timing_chr4.png'
		replication_corr_path = f'{self.save_dir}/Replication_Timing_correlation.png'
		copy_ptrs_path = f'{self.save_dir}/Copy_Correction_PTRs.png'
		copy_examples_path = f'{self.save_dir}/Copy_Correction_Examples.png'
		
		# Layout top row (replication timing figures)
		top_row_images = layout_images_horizontally(
			compositor,
			[replication_chr4_path, replication_corr_path],
			width_proportions=[0.65, 0.35],  # Equal width for both top images
			between_padding=column_padding,
			margin=margins,
			image_keys=['replication_chr4', 'replication_corr']
		)
		
		# Calculate starting y position for bottom row
		top_row_height = max([img['logical_size'][1] for img in top_row_images.values()])
		bottom_row_start_y = margins + top_row_height + row_padding
		
		# Layout bottom row (copy correction figures)
		bottom_row_images = layout_images_horizontally(
			compositor,
			[copy_ptrs_path, copy_examples_path],
			width_proportions=[0.39, 0.61],  # Equal width for both bottom images
			between_padding=column_padding,
			margin=(margins, bottom_row_start_y),
			image_keys=['copy_ptrs', 'copy_examples']
		)
		
		# Add panel labels (A, B, C, D) to each quadrant
		add_panel_labels_to_images(
			compositor,
			compositor.placed_images,
			labels='ABCD',
			font_size=36,
			offset=(-10, -12),
			font_type='bold',
			color=(0, 0, 0)
		)
		
		# Save the composite figure
		output_path = f'{self.save_dir}/Figure4_Copy_Correction.png'
		compositor.save(output_path)
		print(f"Combined 2x2 panel saved to: {output_path}")
		
		return compositor

	def plot_all(self):
		self.plot_replication_results_chr4()
		self.plot_all_ptrs()
		self.plot_sample_curves()