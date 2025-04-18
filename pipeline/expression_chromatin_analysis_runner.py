
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Custom imports
from src.DG1Analysis import DG1Analysis
from src.figure_configs import save_figure_for_paper
from pipeline.expression_analysis import ExpressionAnalysis
from src.GenomeDeconvolutionAnalysis import GenomeDeconvolutionAnalysis
from src.polar_plotter import create_combined_ptr_data_set, plot_skew, plot_polar_branches_data
from src.timecourse_heatmaps import CellCycleHeatmapPlotter
from src.expression_chromatin_analysis_plots import categorize_genes3, plot_three_category_venn_diagram
from src.heatmap_counts import create_subplot_grid, annotate_top_ax, annotate_left_ax
from src.expression_chromatin_analysis_plots import plot_scatter_timing
from src.chromatin_gene_expression_intersection_2d_analysis import compare_md_and_plot_thresholds as compare_and_plot_thresholds
from src.chromatin_gene_expression_intersection_2d_analysis import plot_expected_intersection_heatmap
from src.plot_helpers import hide_spines


class ExpressionChromatinAnalysis:
	"""
	Class to perform deconvolved expression and chromatin analysis
	"""
	
	def __init__(self, output_directory):
		"""
		Initialize the ExpressionChromatinAnalysis with the output directory
		
		Parameters:
		-----------
		output_directory : str
			Directory where output files will be saved
		"""
		self.output_directory = output_directory
		self.save_plots_dir = os.path.join(output_directory, 'expression_chromatin')
		
		# Create output directory if it doesn't exist
		os.makedirs(self.save_plots_dir, exist_ok=True)
		
		# Initialize analyses
		self.chromatin_analysis = None
		self.expression_analysis = None
		self.genome_analysis = None
		self.config1 = None
		self.config2 = None
		
		# Data frames
		self.expression_combined_polar_data_df = None
		self.small_combined_polar_data_df = None
		self.entropies_combined_polar_data_df = None
		
		# Thresholds
		self.tx_ptr_threshold = None
		self.sm_ptr_threshold = None
		self.entropy_ptr_threshold = None
		
		# Parameters for analysis
		self.skew_threshold = 0.1
		self.tx_qval = 0.9
		self.sm_qval = 0.9
		self.nuc_qval = 0.9
		
	def initialize_analyses(self):
		"""Initialize all analysis objects and load required data"""
		print("Initializing chromatin analysis...")
		self.chromatin_analysis = DG1Analysis(self.output_directory)
		self.chromatin_analysis.load_chromatin_measures()
		
		print("Initializing expression analysis...")
		self.expression_analysis = ExpressionAnalysis(self.output_directory)
		
		print("Initializing genome deconvolution analysis...")
		genome_deconv_dir = os.path.join(self.output_directory, 
										"chromatin_deconvolution/deconvolution_data")
		self.genome_analysis = GenomeDeconvolutionAnalysis(outdir=genome_deconv_dir)
		
		# Store configs for later use
		self.config1 = self.expression_analysis.config1
		self.config2 = self.expression_analysis.config2
		
		print("All analyses initialized successfully")
		
	def create_polar_datasets(self):
		"""Create combined polar datasets for expression and chromatin data"""
		print("Creating polar datasets...")
		
		# Expression data
		self.expression_combined_polar_data_df, self.tx_ptr_threshold = create_combined_ptr_data_set(
			self.expression_analysis.deconvolved_genes_F, 
			self.config1, 
			self.tx_qval, 
			eps=1, 
			assign_p_or_t=True,
			skew_threshold=self.skew_threshold
		)
		
		# Small fragments data (promoter occupancies)
		self.small_combined_polar_data_df, self.sm_ptr_threshold = create_combined_ptr_data_set(
			self.chromatin_analysis.small_promoter_occupancies_df, 
			self.config1,
			self.sm_qval, 
			eps=1
		)
		
		# Nucleosome gene body entropies
		self.entropies_combined_polar_data_df, self.entropy_ptr_threshold = create_combined_ptr_data_set(
			self.chromatin_analysis.nucleosome_genebody_entropies_df, 
			self.config1,
			self.nuc_qval, 
			eps=1
		)
		
		# Filter to common ORFs
		self.filter_to_common_orfs()
		
		print("Polar datasets created successfully")
		
	def filter_to_common_orfs(self):
		"""Ensure analysis uses the same ORFs for all analyses"""
		common_orfs = list(set(self.expression_combined_polar_data_df.index).intersection(
			set(self.small_combined_polar_data_df.index)))
		
		self.expression_combined_polar_data_df = self.expression_combined_polar_data_df.loc[common_orfs]
		self.small_combined_polar_data_df = self.small_combined_polar_data_df.loc[common_orfs]
		self.entropies_combined_polar_data_df = self.entropies_combined_polar_data_df.loc[common_orfs]
		
		print(f"Filtered to {len(common_orfs)} common ORFs across all datasets")
		
	def plot_skew_distribution(self):
		"""Plot skew distribution for expression data"""
		print("Plotting skew distribution...")
		
		plot_skew(self.expression_combined_polar_data_df, self.skew_threshold)
		save_figure_for_paper(f"{self.save_plots_dir}/skew_distribution.png")
		
		print("Skew distribution plot saved")
		
	def plot_ptr_distributions(self):
		"""Plot PTR distributions for all metrics"""
		print("Plotting PTR distributions...")
		
		from src.expression_chromatin_analysis_plots import color_map
		plt.figure(figsize=(11, 3.5))
		
		# Define internal helper function for plotting
		def plot_ptr_hist(data_df, ptr_thresh, subplot_pos, key, title):
			plt.subplot(1, 3, subplot_pos)
			ptrs = np.concatenate([data_df.ptr_t, data_df.ptr_b])
			
			plt.hist(ptrs, bins=30, color=color_map[key+'_only'])
			plt.yscale('log')
			plt.axvline(ptr_thresh, c='red', alpha=0.75)
			
			thresholded_genes = ptrs[ptrs > ptr_thresh]
			num_threshold = len(thresholded_genes)
			plt.xlabel('PTR')
			plt.title(title, y=1.05, fontsize=14)
		
		# Plot expression PTR
		plot_ptr_hist(
			self.expression_combined_polar_data_df, 
			self.tx_ptr_threshold, 
			1, 
			'expression',
			f"Gene expression"
		)
		plt.ylabel("Frequency")
		
		# Plot promoter PTR
		plot_ptr_hist(
			self.small_combined_polar_data_df, 
			self.sm_ptr_threshold, 
			2, 
			'promoter',
			f"Small fragment promoter occupancy"
		)
		
		# Plot entropy PTR
		plot_ptr_hist(
			self.entropies_combined_polar_data_df, 
			self.entropy_ptr_threshold, 
			3, 
			'entropy',
			f"Gene body nucleosome entropy"
		)
		plt.suptitle(f"Peak-to-trough distributions, percentile {self.tx_qval*100:.0f}",
			fontsize=23)

		plt.tight_layout()
		save_figure_for_paper(f"{self.save_plots_dir}/ptr_distributions.png")
		
		print("PTR distribution plots saved")
		
	def plot_cell_cycle_heatmaps(self):
		"""Plot cell cycle heatmaps for gene expression data"""
		print("Plotting cell cycle heatmaps...")
		
		plotter = CellCycleHeatmapPlotter(
			analysis=self.expression_analysis,
			config=self.config1,
			expression_combined_polar_data_df=self.expression_combined_polar_data_df
		)
		
		# Plot with default settings
		heatmap_fig, colorbar_fig = plotter.plot_all(save_directory=self.save_plots_dir)
		
		print("Cell cycle heatmaps saved")
		
	def plot_polar_branches(self):
		"""Plot polar branch data for all metrics"""
		print("Plotting polar branches...")
		
		# Expression polar plot
		plot_polar_branches_data(
			self.expression_combined_polar_data_df,
			self.tx_ptr_threshold, 
			self.tx_qval, 
			"Gene expression", 
			(0, 6),
			plot_skew_cat=False
		)
		save_figure_for_paper(f"{self.save_plots_dir}/expression_polar.png")
		
		# Promoter occupancy polar plot
		plot_polar_branches_data(
			self.small_combined_polar_data_df,
			self.sm_ptr_threshold, 
			self.sm_qval, 
			"Promoter occupancy", 
			(0.1, 2.5)
		)
		save_figure_for_paper(f"{self.save_plots_dir}/promoters_polar.png")
		
		# Nucleosome entropy polar plot
		plot_polar_branches_data(
			self.entropies_combined_polar_data_df,
			self.entropy_ptr_threshold, 
			self.sm_qval, 
			"Nucleosome entropy", 
			(0.5, 2)
		)
		save_figure_for_paper(f"{self.save_plots_dir}/entropy_polar.png")
		
		print("Polar branch plots saved")
		
	def plot_venn_diagrams(self):
		"""Plot Venn diagrams for gene categorization"""
		print("Plotting Venn diagrams...")
		
		fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(7, 14.5))
		
		# Mother branch
		cat3_merged_df, cat3_categories = categorize_genes3(
			self.expression_combined_polar_data_df,
			self.small_combined_polar_data_df,
			self.entropies_combined_polar_data_df, 
			suffix='_t'
		)
		plot_three_category_venn_diagram(ax0, cat3_categories)
		ax0.set_title("Mother branch", fontsize=17, fontweight='demi', y=1)
		
		# Daughter branch
		cat3_merged_df, cat3_categories = categorize_genes3(
			self.expression_combined_polar_data_df,
			self.small_combined_polar_data_df,
			self.entropies_combined_polar_data_df, 
			suffix='_b'
		)
		plot_three_category_venn_diagram(ax1, cat3_categories)
		ax1.set_title("Daughter branch", fontsize=17, fontweight='demi', y=1.0)

		plt.suptitle("Cell cycle expression and\nchromatin concordance",
			fontsize=24, y=1.01)
		
		plt.tight_layout()
		save_figure_for_paper(f"{self.save_plots_dir}/metrics_venn3.png")
		
		print("Venn diagrams saved")
		
	def plot_timing_scatter(self):
		"""Plot scatter timing plots for expression vs. chromatin"""
		print("Plotting timing scatter plots...")
		
		# Mother cell timing
		fig, axs = create_subplot_grid(cbar_ax=False)
		ax = axs['main']
		handles, labels = plot_scatter_timing(
			ax, 
			self.expression_combined_polar_data_df, 
			self.small_combined_polar_data_df, 
			self.entropies_combined_polar_data_df,
			suffix='_t', 
			plot_trough_indices=True
		)
		ax.set_title('Expression vs Promoter/Entropy Timing', y=1.1)
		
		annotate_top_ax(axs['top'], g1_key='MG1')
		annotate_left_ax(axs['left'], g1_key='MG1')
		save_figure_for_paper(f"{self.save_plots_dir}/timing_tx_chrom_mother.png")
		
		# Mother cell timing legend
		plt.figure(figsize=(5, 1))
		ax = plt.gca()
		legend = plt.legend(handles, labels, loc=(0.1, 0))
		ax.add_artist(legend)
		hide_spines(ax)
		save_figure_for_paper(f"{self.save_plots_dir}/timing_tx_chrom_mother_legend.png")
		
		# Daughter cell timing
		fig, axs = create_subplot_grid(cbar_ax=False)
		ax = axs['main']
		plot_scatter_timing(
			ax, 
			self.expression_combined_polar_data_df, 
			self.small_combined_polar_data_df, 
			self.entropies_combined_polar_data_df,
			suffix='_b', 
			plot_trough_indices=True
		)
		ax.set_title('Expression vs Promoter/Entropy Timing', y=1.1)
		
		annotate_top_ax(axs['top'], g1_key='DG1')
		annotate_left_ax(axs['left'], g1_key='DG1')
		
		save_figure_for_paper(f"{self.save_plots_dir}/timing_tx_chrom_daughter.png")
		
		print("Timing scatter plots saved")
		
	def plot_ptr_sensitivity(self):
		"""Plot PTR sensitivity analysis"""
		print("Plotting PTR sensitivity analysis...")
		
		# Expression vs Promoter
		compare_and_plot_thresholds(
			self.expression_combined_polar_data_df,
			self.small_combined_polar_data_df, 
			'Gene expression', 
			'Promoter occupancy'
		)
		save_figure_for_paper(f"{self.save_plots_dir}/ptr_sensitivity_tx_prom.png")
		
		# Expression vs Entropy
		compare_and_plot_thresholds(
			self.expression_combined_polar_data_df,
			self.entropies_combined_polar_data_df, 
			'Gene expression', 
			'Nucleosome entropy'
		)
		save_figure_for_paper(f"{self.save_plots_dir}/ptr_sensitivity_tx_entropy.png")
		
		# Promoter vs Entropy
		interesting_chrom_points = compare_and_plot_thresholds(
			self.small_combined_polar_data_df,
			self.entropies_combined_polar_data_df, 
			'Promoter occupancy', 
			'Nucleosome entropy'
		)
		save_figure_for_paper(f"{self.save_plots_dir}/ptr_sensitivity_prom_entropy.png")
		
		# Expected intersection heatmap
		plot_expected_intersection_heatmap(
			len(self.expression_combined_polar_data_df),
			"Measure 1", 
			"Measure 2", 
			figsize=(5, 4),
			highlight=interesting_chrom_points
		)
		save_figure_for_paper(f"{self.save_plots_dir}/ptr_sensitivity_expected_counts.png")
		
		print("PTR sensitivity plots saved")
		
	# def plot_gene_examples(self, gene_list=None):
	# 	"""
	# 	Plot examples of specific genes
		
	# 	Parameters:
	# 	-----------
	# 	gene_list : list, optional
	# 		List of genes to plot. If None, use default examples.
	# 	"""
	# 	if gene_list is None:
	# 		gene_list = ['ALK1', 'YCR050C', 'SPC110']
			
	# 	print(f"Plotting {len(gene_list)} gene examples...")
		
	# 	for gene in gene_list:
	# 		self.genome_analysis.plot_gene(gene, self.config1, self.expression_analysis)
	# 		save_figure_for_paper(f"{self.save_plots_dir}/gene_{gene}.png")
			
	# 	print("Gene example plots saved")
		
	def run_all_analyses(self):
		"""Run all analysis steps in sequence"""
		print("Starting complete polar analysis...")
		
		# Initialize analyses
		self.initialize_analyses()
		
		# Create and process datasets
		self.create_polar_datasets()
		
		# Generate all plots
		self.plot_skew_distribution()
		self.plot_ptr_distributions()
		self.plot_cell_cycle_heatmaps()
		self.plot_polar_branches()
		self.plot_venn_diagrams()
		self.plot_timing_scatter()
		self.plot_ptr_sensitivity()
		
		print("Complete polar analysis finished!")


def layout_figure_plots(plots_dir, save_dir):
	
	from pipeline.figure_composer import FigureCompositor

	# Define image paths
	image_names = [
		'metrics_venn3',
		'expression_polar',
		'promoters_polar',
		'entropy_polar',
		'timing_tx_chrom_mother',
		'timing_tx_chrom_mother_legend',
		'timing_tx_chrom_daughter'
	]

	image_paths = [f"{plots_dir}/{name}.png" for name in image_names]

	# Create compositor with a scale factor of 4
	# Logical canvas size is 1024x800, but actual output will be 4096x3200
	compositor = FigureCompositor(1024, 740, background_color=(255, 255, 255),
								 debug_mode=True, scale_factor=4.0)

	# Place images individually
	# All coordinates and dimensions are specified in logical pixels
	# but will be rendered at 4x resolution
	margin = 20
	venn_width = 320

	# -------- Venn diagram -------------
	compositor.place_image(image_paths[0], margin, margin, venn_width, None, 'venn3')

	# Example using the new add_panel_label_to_image function
	compositor.add_panel_label_to_image('venn3', 'A', offset=(5, 5), 
									   background=(240, 240, 240), bg_padding=3)

	# -------- Polar plots -------------

	polar_width = 330
	pad_x_polar = 24

	polar_x = venn_width+pad_x_polar+margin
	img = compositor.place_image(image_paths[1], polar_x, margin, polar_width, 
						  None, 'tx_polar')
	polar_height = img['logical_size'][1]  # Use logical_size instead of size

	# Add label to the polar plot
	compositor.add_panel_label_to_image('tx_polar', 'B', offset=(5, 5),
									   background=(240, 240, 240))

	pad_y_polar = 20

	compositor.place_image(image_paths[2], polar_x, margin+polar_height+pad_y_polar, polar_width, 
						  None, 'prom_polar')
	compositor.add_panel_label_to_image('prom_polar', 'C', offset=(5, 5))

	compositor.place_image(image_paths[3], polar_x, margin+polar_height*2+pad_y_polar*2, polar_width,
						  None, 'entropy_polar')
	compositor.add_panel_label_to_image('entropy_polar', 'D', offset=(5, 5))

	# -------- Timing plots -------------
	timing_width = 280
	timing_x = polar_x+polar_width+pad_x_polar
	img = compositor.place_image(image_paths[4], timing_x, margin, timing_width, 
						  None, 'mother_time')
	timing_height = img['logical_size'][1]  # Use logical_size for logical height
	compositor.add_panel_label_to_image('mother_time', 'E', offset=(5, 5))

	pad_time_y = 20
	lgd_img = compositor.place_image(image_paths[5], timing_x, margin+timing_height+pad_time_y, 
						  timing_width, None, 'legend_time')
	lgd_height = lgd_img['logical_size'][1]  # Use logical_size for logical height

	compositor.place_image(image_paths[6], timing_x, margin+timing_height+pad_time_y+lgd_height+\
						  pad_time_y, timing_width, None, 'daughter_time')
	compositor.add_panel_label_to_image('daughter_time', 'F', offset=(5, 5))

	# You can still use the original method for adding labels not attached to images
	padding = 5

	# Save the figure - will be 4x the logical resolution
	compositor.save(f"{save_dir}/Figure_2.png")

def layout_supplemental_1(plots_dir, save_dir):
	from pipeline.figure_composer import FigureCompositor

	# Define image paths
	image_names = [
	    'heatmap_timecourse',
	    'heatmap_timecourse_colorbar',
	]

	image_paths = [f"{plots_dir}/{name}.png" for name in image_names]

	# Create compositor with a scale factor of 4
	# Logical canvas size is 1024x800, but actual output will be 4096x3200
	compositor = FigureCompositor(1024, 740, background_color=(255, 255, 255),
	                             debug_mode=True, scale_factor=4.0)

	# Place images individually
	# All coordinates and dimensions are specified in logical pixels
	# but will be rendered at 4x resolution
	margin = 20
	hm_width = 800

	# -------- Venn diagram -------------
	compositor.place_image(image_paths[0], margin, margin, hm_width, None, 'hm')
	compositor.add_panel_label_to_image('hm', 'A', offset=(5, 5))

	compositor.place_image(image_paths[1], margin+hm_width+30, margin+100, 60, None, 'cbar')

	compositor.save(f"{save_dir}/Supplemental_S1.png")

def layout_supplemental_2(plots_dir, save_dir):
	from pipeline.figure_composer import FigureCompositor

	# Define image paths
	image_names = [
	    'ptr_distributions',
	    'skew_distribution',
	]

	image_paths = [f"{plots_dir}/{name}.png" for name in image_names]

	# Create compositor with a scale factor of 4
	# Logical canvas size is 1024x800, but actual output will be 4096x3200
	scale_factor = 4.0
	compositor = FigureCompositor(1024, 620, background_color=(255, 255, 255),
	                             debug_mode=True, scale_factor=scale_factor)

	# Place images individually
	# All coordinates and dimensions are specified in logical pixels
	# but will be rendered at 4x resolution
	margin = 20
	ptr_width = 800

	# -------- Venn diagram -------------
	img = compositor.place_image(image_paths[0], margin, margin, ptr_width, None, 'ptr')
	compositor.add_panel_label_to_image('ptr', 'A', offset=(5, 5))

	skew_y = img['size'][1]/scale_factor+margin+30
	compositor.place_image(image_paths[1], margin, skew_y, 340, None, 'skew')
	compositor.add_panel_label_to_image('skew', 'B', offset=(5, 5))

	compositor.save(f"{save_dir}/Supplemental_S2.png")


def layout_supplemental_3(plots_dir, save_dir):
	from pipeline.figure_composer import FigureCompositor

	# Define image paths
	image_names = [
	    'ptr_sensitivity_tx_prom',
		'ptr_sensitivity_tx_entropy',
		'ptr_sensitivity_prom_entropy',
		'ptr_sensitivity_expected_counts',
	]

	image_paths = [f"{plots_dir}/{name}.png" for name in image_names]

	scale = 4.0
	compositor = FigureCompositor(1024, 550, scale_factor=scale, debug_mode=True)

	# Place images individually
	# All coordinates and dimensions are specified in logical pixels
	# but will be rendered at 4x resolution
	margin = 20
	sens_width = 480

	# Place sensitivity analysis images
	# A
	img = compositor.place_image(image_paths[0], margin, margin, sens_width, None, 'tx_prom')
	sens_height = img['size'][1]/scale
	compositor.add_panel_label_to_image('tx_prom', 'A', offset=(5, 5))

	# B
	compositor.place_image(image_paths[1], margin+sens_width+30, margin, sens_width, None, 
		'tx_entropy')
	compositor.add_panel_label_to_image('tx_entropy', 'B', offset=(5, 5))

	# C
	compositor.place_image(image_paths[2], margin, margin+sens_height+30, sens_width, None, 
		'prom_entropy')
	compositor.add_panel_label_to_image('prom_entropy', 'C', offset=(5, 5))

	# D
	compositor.place_image(image_paths[3], margin+sens_width+30,
		margin+sens_height+37, 260, None, 
		'control')
	compositor.add_panel_label_to_image('control', 'D', offset=(5, 0))

	compositor.save(f"{save_dir}/Supplemental_S3.png")

	
def main():
	"""Main function to run the analysis"""
	output_directory = 'output/prototype_pipeline_subset'
	
	# Create analysis object
	analyzer = ExpressionChromatinAnalysis(output_directory)
	
	# Run all analyses
	analyzer.run_all_analyses()
	
	print("Analysis completed successfully")


if __name__ == "__main__":
	main()