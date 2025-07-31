import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from src.sgd import read_nondubious_genes_dataset, get_orfname
from src.GenomeDeconvolutionAnalysis import GenomeDeconvolutionAnalysis
from src.orf_plotter import load_default_orf_plotter
from src.rna_pileup_plotter import RNASeqPileupPlotter
from src.combined_chromatin_model import CombinedChromatinModel
from src.config import load_default_chrom_configs
from pipeline.chromatin_metrics_processor import ChromatinMetricsProcessor
from src.utils import mkdir_safe
from src.figure_configs import save_figure_for_paper


class FigureDaughterSpecific:
	"""
	A class for analyzing daughter-specific chromatin patterns in yeast cell cycle genes.
	
	Performs analysis and visualization of chromatin metrics (promoter occupancy,
	nucleosome entropy, nucleosome occupancy) for genes involved in daughter cell
	specification and asymmetric cell division.
	"""
	
	def __init__(self, output_dir):
		"""
		Initialize the analyzer with data processors and configuration parameters.
		
		Parameters:
		-----------
		output_dir : str
			Path to the output directory containing analysis results
		"""
		# Store output directory
		self.output_dir = output_dir
		self.save_dir = f"{self.output_dir}/daughter_specific_figures"
		mkdir_safe(self.save_dir)
		self.figures_dir = f"{self.output_dir}/Figures"
		
		# Daughter-specific genes of interest
		self.daughter_specific_genes = [
			'DSE1', 'DSE2', 'DSE3', 'DSE4',  # Daughter-specific expression genes
			'ASH1',  # Asymmetric synthesis of HO
			'EGT2',  # Endoglucanase 
			'AMN1',  # Antagonist of mitotic exit network
			'PRY3',  # Pathogen related yeast protein
			'SCW11', # Cell wall protein
			'CTS1'   # Chitinase
		]
		
		# Gene for detailed locus analysis (can be modified)
		self.detailed_analysis_gene = 'DSE2'
		
		# Chromatin metrics to analyze
		self.chromatin_metrics = ['promoter_occupancy', 'nucleosome_entropy', 'nucleosome_occupancy']
		
		# Y-axis limits for consistent plotting
		self.ylim_config = {
			'promoter_occupancy': (-0.5, 7),
			'nucleosome_entropy': (-0.5, 7), 
			'nucleosome_occupancy': (-0.5, 7)
		}
		
		# Initialize data processors and other components
		self._initialize_processors()
		
		# Load all data upon initialization
		self.load_data()
	
	def _initialize_processors(self):
		"""Initialize all required data processors and analysis components."""
		print("Initializing processors...")

		self.rna_plotter = RNASeqPileupPlotter(self.output_dir)
		self.rna_plotter.ylim = 13
		
		# Initialize chromatin metrics processor
		self.chromatin_metrics_processor = ChromatinMetricsProcessor(self.output_dir)
		
		# Initialize genome deconvolution analysis
		self.genome_deconv_analysis = GenomeDeconvolutionAnalysis(self.output_dir)
		
		# Initialize chromatin configurations and plotters
		config1, config2 = load_default_chrom_configs()
		self.config1 = config1
		self.config2 = config2
		
		# Initialize plotters
		self.orf_plotter = load_default_orf_plotter()
		self.rna_plotter = RNASeqPileupPlotter(self.output_dir)
		self.rna_plotter.ylim = 13
		
		# Initialize combined chromatin model
		self.combined_model = CombinedChromatinModel(config1=config1, config2=config2)
		
		print("Processors initialized.")
	
	def load_data(self):
		"""Load all required data: chromatin metrics and gene annotations."""
		print("Loading chromatin metrics data...")
		
		# Setup and load chromatin metrics
		self.chromatin_metrics_processor.setup_data_loaders()
		self.chromatin_metrics_processor.load_and_assign_saved_metrics()
		
		print("Loading gene annotations...")
		
		# Load gene annotations
		self.genes = read_nondubious_genes_dataset()
		
		print("Data loading complete.")
	
	def get_gene_coordinates(self, gene_name):
		"""
		Get genomic coordinates for a specific gene.
		
		Parameters:
		-----------
		gene_name : str
			Name of the gene
			
		Returns:
		--------
		pd.Series
			Gene annotation data including chromosome, start, stop positions
		"""
		gene_data = self.genes[self.genes['gene'] == gene_name]
		if len(gene_data) == 0:
			raise ValueError(f"Gene {gene_name} not found in gene annotations")
		return gene_data.iloc[0]
	

	def plot_gene_locus(self, gene_name):

		from src.GenomeDeconvolutionAnalysis import GenomeDeconvolutionAnalysis
		from src.transcripts_dataset import load_transcripts_sets

		genes, _ = load_transcripts_sets('output/draft4_run/')
		gene = genes[genes['gene'] == gene_name].iloc[0]
		chrom = gene.chr
		center = gene.TSS

		if gene.strand == '-':
			span = center-1200, center+1000
		else:
			span = center-1000, center+1200

		loaded_data = self.genome_deconv_analysis.load_mnase_span(chrom, span)
		branches = ['mother', 'daughter', 'difference_mother_daughter']
		branch_names = ["Mother", "Daughter", "Difference"]

		for i, branch in enumerate(branches):
			branch_name = branch_names[i]
			self.genome_deconv_analysis.plot_loaded_data(figsize=(7, 11), title=branch_name,
											   branch_type=branch, rna_plotter=self.rna_plotter)
			save_figure_for_paper(f"{self.save_dir}/locus_{gene_name}_{branch}")


	def plot_chromatin_metrics_timecourse(self, gene_name):
		"""
		Plot chromatin metrics timecourse for a gene (reproducing notebook cell 8 logic).
		
		Parameters:
		-----------
		gene_name : str
			Name of the gene to plot
		save_plot : bool, optional
			Whether to save the plot to disk (default: False)
			
		Returns:
		--------
		matplotlib.figure.Figure
			The generated figure object
		"""
		from src.sgd import get_orfname

		orf_name = get_orfname(gene_name)

		b_indices = self.config1.b_indices()
		t_indices = self.config1.t_indices()

		b_tps = self.config1.get_timepoints_for_branch('b')
		t_tps = self.config1.get_timepoints_for_branch('t')
		
		fig = plt.figure(figsize=(7, 2))
		
		# Helper function to plot chromatin metric
		def plot_chromatin_metric(chromatin_key, subplot_pos):
			chromatin_values = self.chromatin_metrics_processor.normalized_deconvolved_metrics[
				chromatin_key].loc[orf_name]
			
			from pipeline.chromatin_metrics_processor import plot_formatting_map
			from src.plot_helpers import color_for_key

			color = plt.get_cmap(plot_formatting_map[chromatin_key]['cmap'])(0.5)

			plt.subplot(1, 3, subplot_pos)
			plt.plot(t_tps, chromatin_values[t_indices].values, label='Mother', 
				color=color,
				lw=2)
			plt.plot(b_tps, chromatin_values[b_indices].values, label='Daughter',
				color=color, lw=2, ls=(0, (1, 1)))

			from src.config import retrieve_phase_ticks

			ax = plt.gca()
			xticks = retrieve_phase_ticks('b', self.config1, self.config1)
			phase_ticks, edge_ticks = xticks

			ax.set_xticks(phase_ticks)
			ax.set_xticklabels(['G1', 'S', 'G2/M'], fontsize=8)

			ax.set_xticks(edge_ticks, minor=True)

			ax.tick_params(axis='x', which='major', length=0)
			ax.tick_params(axis='x', which='minor', length=10) 
			ax.set_xlim(b_tps[0], b_tps[-1])
			
			# Set y-limits
			if chromatin_key in self.ylim_config:
				plt.ylim(self.ylim_config[chromatin_key])
			
			plt.title(chromatin_key.replace('_', ' ').title())
			
			if subplot_pos == 1:  # Only show legend on first subplot
				plt.legend()
			else:
				plt.yticks([])
		
		# Create subplots for each chromatin metric
		for i, metric in enumerate(self.chromatin_metrics):
			plot_chromatin_metric(metric, i + 1)
		
		from src.sgd import get_gene_title_name

		gene_title_name = get_gene_title_name(gene_name)

		plt.suptitle(gene_title_name, fontweight='demi', fontsize=16)
		plt.tight_layout()
		
		# Save plot if requested
		save_figure_for_paper(f"{self.save_dir}/timecourse_{gene_name}")
		
		return fig

	def create_plots(self, plot_all_genes=True):

		if not plot_all_genes:
			genes_to_plot = ['DSE1', 'DSE2', 'AMN1', 'SCW11']
		else:
			genes_to_plot = self.daughter_specific_genes

		for gene_name in genes_to_plot:
			fig = self.plot_chromatin_metrics_timecourse(gene_name)
			
		for gene_name in genes_to_plot:
			fig = self.plot_gene_locus(gene_name)


	def layout_timecourse_and_locus_panel(self, 
										 canvas_width=1024, canvas_height=630, 
										 margins=30, column_padding=38, between_padding=12, 
										 debug_mode=True):
		"""
		Create a composite figure panel with timecourses on left and locus plots on right.
		
		Layout:
		- Left 50%: 4 timecourse plots (DSE1, DSE2, SCW11, AMN1) stacked vertically
		- Right 50% top: 3 DSE1 locus plots (mother, daughter, difference) horizontally
		- Right 50% bottom: 3 SCW11 locus plots (mother, daughter, difference) horizontally
		"""
		
		# Import required modules
		from pipeline.figure_composer import FigureCompositor
		from pipeline.figure_composer_helpers import (layout_images_horizontally, 
													 layout_images_vertically, 
													 add_panel_labels_to_images)
		import os

		save_directory = self.save_dir
		figures_directory = self.figures_dir
		
		# Create compositor
		compositor = FigureCompositor(canvas_width, canvas_height, debug_mode=debug_mode)
		
		# Define file paths for timecourse images (left column)
		timecourse_paths = [
			os.path.join(save_directory, 'timecourse_DSE1.png'),
			os.path.join(save_directory, 'timecourse_DSE2.png'),
			os.path.join(save_directory, 'timecourse_SCW11.png'),
			os.path.join(save_directory, 'timecourse_AMN1.png')
		]
		
		# Define file paths for DSE1 locus images (upper right)
		dse1_locus_paths = [
			os.path.join(save_directory, 'locus_DSE1_mother.png'),
			os.path.join(save_directory, 'locus_DSE1_daughter.png'),
			os.path.join(save_directory, 'locus_DSE1_difference_mother_daughter.png')
		]
		
		# Define file paths for SCW11 locus images (lower right)
		scw11_locus_paths = [
			os.path.join(save_directory, 'locus_SCW11_mother.png'),
			os.path.join(save_directory, 'locus_SCW11_daughter.png'),
			os.path.join(save_directory, 'locus_SCW11_difference_mother_daughter.png')
		]
		
		# Calculate column dimensions
		available_width = canvas_width - (2 * margins)
		left_column_width = int((available_width - column_padding) * 0.506)
		right_column_start_x = margins + left_column_width + column_padding
		right_column_width = canvas_width - right_column_start_x - margins
		
		# Calculate heights for right column sections (split top/bottom)
		available_height = canvas_height - (2 * margins)
		right_section_height = int((available_height - between_padding) * 0.45)  # Each gets ~48%, 4% for padding between
		
		print(f"Layout dimensions:")
		print(f"  Canvas: {canvas_width} x {canvas_height}")
		print(f"  Left column width: {left_column_width}")
		print(f"  Right column width: {right_column_width}")
		print(f"  Right section height: {right_section_height}")
		
		# STEP 1: Layout timecourse images vertically on the left (50% width)
		print("\nPlacing timecourse images on left...")
		left_column_images = layout_images_vertically(
			compositor,
			timecourse_paths,
			between_padding=between_padding,
			margin=(margins, margins),
			x_position=margins,
			widths=[left_column_width] * 4,
			image_keys=['timecourse_DSE1', 'timecourse_DSE2', 'timecourse_SCW11', 'timecourse_AMN1']
		)

		# STEP 2: Layout DSE1 locus images horizontally in upper right
		print("Placing DSE1 locus images in upper right...")
		locus_title_offset = 20
		dse1_images = layout_images_horizontally(
			compositor,
			dse1_locus_paths,
			width_proportions=[1, 1, 1],  # Equal widths
			available_width=right_column_width,
			between_padding=between_padding,
			margin=(right_column_start_x, margins),  # Start at right column position
			y_position=margins+locus_title_offset,
			heights=[right_section_height] * 3,  # All same height
			image_keys=['locus_DSE1_mother', 'locus_DSE1_daughter', 'locus_DSE1_difference']
		)
		
		# STEP 3: Layout SCW11 locus images horizontally in lower right
		print("Placing SCW11 locus images in lower right...")
		# Calculate y position for lower section
		between_padding_vertical = 26
		lower_right_y = margins + right_section_height + between_padding_vertical + locus_title_offset*2
		
		scw11_images = layout_images_horizontally(
			compositor,
			scw11_locus_paths,
			width_proportions=[1, 1, 1],  # Equal widths
			between_padding=between_padding,
			margin=(right_column_start_x, lower_right_y),
			y_position=lower_right_y,
			available_width=right_column_width,
			heights=[right_section_height] * 3,  # All same height
			image_keys=['locus_SCW11_mother', 'locus_SCW11_daughter', 'locus_SCW11_difference']
		)
		
		# STEP 4: Add panel labels to all images
		print("Adding panel labels...")
		# Combine all placed images
		all_images = {}
		all_images.update(left_column_images)
		all_images.update(dse1_images)
		all_images.update(scw11_images)
		
		# Add panel labels A-D
		add_panel_labels_to_images(
			compositor,
			left_column_images,
			labels='ABCD',
			font_size=32,
			offset=(-15, -15),  # Position labels slightly outside and above each image
			font_type='bold',
			color=(0, 0, 0),
			background=(255, 255, 255),  # White background for better visibility
		)

		# Add panel labels E,F
		add_panel_labels_to_images(
			compositor,
			dse1_images,
			labels='E  ',
			font_size=32,
			offset=(-15, -40),  # Position labels slightly outside and above each image
			font_type='bold',
			color=(0, 0, 0),
			background=(255, 255, 255),  # White background for better visibility
		)
		add_panel_labels_to_images(
			compositor,
			scw11_images,
			labels='F  ',
			font_size=32,
			offset=(-15, -40),  # Position labels slightly outside and above each image
			font_type='bold',
			color=(0, 0, 0),
			background=(255, 255, 255),  # White background for better visibility
		)

		compositor.add_panel_label_to_image('locus_DSE1_daughter', 'DSE1',
			offset=(20, -35), font_type='italic', font_size=24)

		compositor.add_panel_label_to_image('locus_SCW11_daughter', 'SCW11',
			offset=(20, -35), font_type='italic', font_size=24)
		
		# STEP 5: Save the composite figure
		print("Saving composite figure...")
		output_path = os.path.join(figures_directory, 'Supplemental5_Daughter_Chromatin.png')
		
		success = compositor.save(output_path, quality=95, dpi=(300, 300))
		
		return compositor

