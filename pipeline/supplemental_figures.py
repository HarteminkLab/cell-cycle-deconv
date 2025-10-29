import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests
from src.utils import mkdir_safe
from src.figure_configs import save_figure_for_paper

from src.nucleosome_histone_dataset import HistonesNucleosomesDataset
from src.nucleosome_metrics_processor import NucleosomeDataLoader
from src.peak_to_trough import compute_quantile_ptr_2d


class FigureSupplemental:
	"""
	A class to layout supplemental panels that are not tied to any existing 
	figure panel/story.
	"""
	
	def __init__(self, output_dir="output/draft4_run/"):
		"""
		Initialize the analyzer with configuration parameters.
		"""
		# Configuration parameters
		self.output_dir = output_dir
		self.save_dir = f"{self.output_dir}/supplemental_various"
		self.figures_dir = f'{self.output_dir}/Figures'
		mkdir_safe(self.save_dir)

	def layout_supplemental_flow_cytometry(self):

		from pipeline.figure_composer import FigureCompositor
		from pipeline.figure_composer_helpers import layout_images_horizontally, \
			add_panel_labels_to_images

		# Create compositor with wider dimensions for horizontal layout
		compositor = FigureCompositor(1024, 440, debug_mode=True)

		image_paths = [
			f'data/2019_cloccs_fits/yl_2019_replicate1/rep1.png',
			f'data/2019_cloccs_fits/yl_2019_replicate2/rep2.png',
		]

		placed_images = layout_images_horizontally(
			compositor,
			image_paths,
			between_padding=30,
			margin=(30, 60),
			image_keys=['hm1', 'hm2']  # Custom keys
		)

		# Add panel labels
		add_panel_labels_to_images(
			compositor, 
			compositor.placed_images,
			offset=(-10, -6),
			font_size=36,
		)

		compositor.add_panel_label_to_image('hm1', 'Replicate 1', offset=(170, -6), 
			font_size=24, font_type='semi_bold')

		compositor.add_panel_label_to_image('hm2', 'Replicate 2', offset=(170, -6), 
			font_size=24, font_type='semi_bold')

		# Save the composite figure
		compositor.save(f'{self.figures_dir}/Supplemental1_flow_cytometry.png')

	def layout_supplemental_cloccs_fits(self):

		from pipeline.figure_composer import FigureCompositor
		from pipeline.figure_composer_helpers import layout_images_horizontally, \
			add_panel_labels_to_images

		# Create compositor with wider dimensions for horizontal layout
		compositor = FigureCompositor(1024, 340, debug_mode=True)

		image_paths = [
			f'./data/2019_cloccs_fits/yl_2019_replicate1/fit_curves_rep1.png',
			f'./data/2019_cloccs_fits/yl_2019_replicate2/fit_curves_rep2.png',
		]

		placed_images = layout_images_horizontally(
			compositor,
			image_paths,
			between_padding=30,
			margin=(30, 60),
			image_keys=['fit1', 'fit2']  # Custom keys
		)

		compositor.add_panel_label_to_image('fit1', 'Replicate 1 CLOCCS fit', offset=(90, -20), 
			font_size=24, font_type='semi_bold')

		compositor.add_panel_label_to_image('fit2', 'Replicate 2 CLOCCS fit', offset=(90, -20), 
			font_size=24, font_type='semi_bold')

		# Add panel labels
		add_panel_labels_to_images(
			compositor, 
			compositor.placed_images,
			offset=(-20, -20),
			font_size=36,
		)

		# Save the composite figure
		compositor.save(f'{self.figures_dir}/Supplemental2_fit_curves.png')

