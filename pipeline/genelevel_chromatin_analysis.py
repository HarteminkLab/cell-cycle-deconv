
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Custom imports
from src.deconvolved_chromatin_analysis import DeconvolvedChromatinAnalysis
from src.figure_configs import save_figure_for_paper
from pipeline.expression_analysis import ExpressionAnalysis
from src.GenomeDeconvolutionAnalysis import GenomeDeconvolutionAnalysis
from src.polar_plotter import create_average_tb_ptr_data_set, plot_skew, plot_polar_plot_polar_mean_branch_data
from src.timecourse_heatmaps import CellCycleHeatmapPlotter
from src.heatmap_counts import create_subplot_grid, annotate_top_ax, annotate_left_ax
from src.expression_chromatin_analysis_plots import plot_scatter_timing
from src.chromatin_gene_expression_intersection_2d_analysis import compare_mean_and_plot_thresholds as compare_and_plot_thresholds
from src.chromatin_gene_expression_intersection_2d_analysis import plot_expected_intersection_heatmap
from src.plot_helpers import hide_spines


class GeneChromatinAnalysis:
	"""
	Class to perform analysis on the chromatin at the gene-level
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
		self.save_plots_dir = os.path.join(output_directory, 'genelevel_chromatin')
		
		# Create output directory if it doesn't exist
		os.makedirs(self.save_plots_dir, exist_ok=True)
		
		# Initialize analyses
		self.chromatin_analysis = None
		self.expression_analysis = None
		self.genome_analysis = None
		self.config1 = None
		self.config2 = None
		
		# Data frames
		self.small_polar_data_df = None
		self.entropies_polar_data_df = None
		self.gb_nucleosome_polar_data_df = None
		
		# Thresholds
		self.sm_ptr_threshold = None
		self.entropy_ptr_threshold = None
		
		# Parameters for analysis, determined from sensitivity analysis
		self.sm_percentile_threshold = None
		self.entropy_percentile_threshold = None
		
	def initialize_analyses(self, copy_correction=True):
		"""Initialize all analysis objects and load required data"""
		print(f"Initializing chromatin analysis... copy correction: {copy_correction}")
		self.chromatin_analysis = DeconvolvedChromatinAnalysis(self.output_directory, 
			copy_correction=copy_correction)
		self.chromatin_analysis.load_chromatin_measures()
		
		print("Initializing genome deconvolution analysis...")
		genome_deconv_dir = os.path.join(self.output_directory, 
										"chromatin_deconvolution/deconvolution_data")
		self.genome_analysis = GenomeDeconvolutionAnalysis(outdir=genome_deconv_dir)
		
		# Store configs for later use
		from src.config import load_default_expression_configs
		self.config1, self.config2 = load_default_expression_configs()

		print("All analyses initialized successfully")
		
	def create_polar_datasets(self):
		"""Create combined polar datasets for expression and chromatin data"""
		print("Creating polar datasets...")
		
		# Small fragments data (promoter occupancies)
		self.small_polar_data_df, self.sm_ptr_threshold = create_average_tb_ptr_data_set(
			self.chromatin_analysis.small_promoter_occupancies_df, 
			self.config1,
			self.sm_percentile_threshold, 
			eps=1
		)

		# Nucleosome fragments data (gene body occupancies)
		self.gb_nucleosome_polar_data_df, self.gb_ptr_threshold = create_average_tb_ptr_data_set(
			self.chromatin_analysis.nucleosome_gene_body_occupancies_df, 
			self.config1,
			self.entropy_percentile_threshold,  # todo: Use the same qvalue threshold as entropy temporarily
			eps=1
		)
		
		# Nucleosome gene body entropies
		self.entropies_polar_data_df, self.entropy_ptr_threshold = create_average_tb_ptr_data_set(
			self.chromatin_analysis.nucleosome_genebody_entropies_df, 
			self.config1,
			self.entropy_percentile_threshold, 
			eps=1
		)
		
		# Filter to common ORFs
		self.filter_to_common_orfs()
		
		print("Polar datasets created successfully")
		
	def filter_to_common_orfs(self):
		"""Ensure analysis uses the same ORFs for all analyses"""
		common_orfs = list(set(self.small_polar_data_df.index))
		
		self.small_polar_data_df = self.small_polar_data_df.loc[common_orfs]
		self.entropies_polar_data_df = self.entropies_polar_data_df.loc[common_orfs]
		
		print(f"Filtered to {len(common_orfs)} common ORFs across all datasets")
		
	def plot_ptr_distributions(self):
		"""Plot PTR distributions for all metrics"""
		print("Plotting PTR distributions...")
		
		from src.expression_chromatin_analysis_plots import color_map
		plt.figure(figsize=(7, 3.5))
		
		# Define internal helper function for plotting
		def plot_ptr_hist(data_df, ptr_thresh, subplot_pos, key, title):
			plt.subplot(1, 2, subplot_pos)
			ptrs = data_df.ptr

			plt.hist(ptrs, bins=30, color=color_map[key+'_only'])
			plt.axvline(ptr_thresh, c='red', alpha=0.75)
			
			thresholded_genes = ptrs[ptrs > ptr_thresh]
			num_threshold = len(thresholded_genes)
			plt.xlabel('PTR')
			plt.title(title, y=1.05, fontsize=14)
		
		# Plot promoter PTR
		plot_ptr_hist(
			self.small_polar_data_df, 
			self.sm_ptr_threshold, 
			1, 
			'promoter',
			f"Small fragment promoter occupancy"
		)
		
		# Plot entropy PTR
		plot_ptr_hist(
			self.entropies_polar_data_df, 
			self.entropy_ptr_threshold, 
			2, 
			'entropy',
			f"Gene body nucleosome entropy"
		)
		plt.suptitle(f"Peak-to-trough distributions, percentile",
			fontsize=23)

		plt.tight_layout()
		save_figure_for_paper(f"{self.save_plots_dir}/ptr_distributions.png")
		
		print("PTR distribution plots saved")
		
	def plot_polar_branches(self):
		"""Plot polar branch data for all metrics"""
		print("Plotting polar branches...")
		
		# Promoter occupancy polar plot
		plot_polar_plot_polar_mean_branch_data(
			self.small_polar_data_df,
			self.sm_ptr_threshold, 
			"Promoter occupancy", 
			(0.5, 2.2)
		)
		save_figure_for_paper(f"{self.save_plots_dir}/promoters_polar.png")

		# Nucleosome entropy polar plot
		plot_polar_plot_polar_mean_branch_data(
			self.entropies_polar_data_df,
			self.entropy_ptr_threshold,
			"Nucleosome entropy", 
			(0.75, 1.75)
		)
		save_figure_for_paper(f"{self.save_plots_dir}/entropy_polar.png")

		# Nucleosome occupancy polar plot
		# plot_polar_plot_polar_mean_branch_data(
		# 	self.gb_nucleosome_polar_data_df,
		# 	self.gb_ptr_threshold,
		# 	"Nucleosome occupancy", 
		# 	(0.5, 2)
		# )
		# save_figure_for_paper(f"{self.save_plots_dir}/gb_polar.png")
		
		print("Polar branch plots saved")
		
	def plot_venn_diagrams(self):
		"""Plot Venn diagrams for gene categorization"""

		from src.expression_chromatin_analysis_plots import categorize_genes, plot_cell_cycle_venn_diagram

		_, category_counts = categorize_genes(self.small_polar_data_df, 
						self.entropies_polar_data_df,
						cat_1_name='promoter_only',
						cat_2_name='entropy_only', suffix='')

		fig, ax = plt.subplots(1, 1, figsize=(6, 4))

		#category_counts
		plot_cell_cycle_venn_diagram(ax, category_counts,
		  category_keys=['promoter_only', 'entropy_only', 'both', 'neither'], 
									 category_names=['Promoter', 'Entropy'])
		ax.set_ylabel("# of genes")
		save_figure_for_paper(f"{self.save_plots_dir}/metrics_venn2.png")
		
		print("Venn diagrams saved")
		
	def plot_ptr_sensitivity(self):
		"""Plot PTR sensitivity analysis"""
		print("Plotting PTR sensitivity analysis...")
		
		# Promoter vs Entropy
		interesting_chrom_points = compare_and_plot_thresholds(
			self.small_polar_data_df,
			self.entropies_polar_data_df, 
			'Promoter occupancy', 
			'Nucleosome entropy'
		)

		
		print("Interesting threshold values:")
		highest_neg_logpval = 0
		thresholds = None
		for row in interesting_chrom_points:
			print(f"{row[0]:0.4f}, {row[1]:0.4f}: p-value {row[2]:0.2f}")
			if row[2] > highest_neg_logpval:
				highest_neg_logpval = row[2]
				thresholds = row[:2]
		print(f"Highest p-value: {thresholds[0]:0.4f}, {thresholds[1]:0.4f}: p-value {highest_neg_logpval:0.2f}")

		# Parameters for analysis, determined from sensitivity analysis
		self.sm_percentile_threshold = thresholds[0]
		self.entropy_percentile_threshold = thresholds[1]

		save_figure_for_paper(f"{self.save_plots_dir}/ptr_sensitivity_prom_entropy.png")
		
		# Expected intersection heatmap
		plot_expected_intersection_heatmap(
			len(self.small_polar_data_df),
			"Measure 1", 
			"Measure 2", 
			figsize=(5, 4),
			highlight=interesting_chrom_points
		)
		save_figure_for_paper(f"{self.save_plots_dir}/ptr_sensitivity_expected_counts.png")
		
		print("PTR sensitivity plots saved")
		
	def load_gene_replication_times(self):
		from src.reference_data import load_p1_gene_regions
		from src.sgd import read_nondubious_genes_dataset
		from src.mnase_10kb_loader import get_bin_for_position
		from src.config import retrieve_replication_timing
		from src.RealDataReplication import load_replication_Fr_df, load_B_df

		output_dir = self.output_directory
		gene_p1s = load_p1_gene_regions().join(read_nondubious_genes_dataset()[['TSS']], how='left')
		gene_p1s = gene_p1s.loc[self.small_polar_data_df.index]

		gene_replication_times = gene_p1s[[]].copy()

		for chrom in range(1, 17):

			_, chrom_replication_indices = load_replication_Fr_df(output_dir, chrom)    
			chrom_genes = gene_p1s[gene_p1s.chr == chrom]

			chrom_bin_indices_starts = [get_bin_for_position(tss, chrom_replication_indices.index) 
						  for tss in chrom_genes.TSS.values]
			chrom_replication_indices = [chrom_replication_indices.loc[bin_start_bp] for bin_idx, bin_start_bp in chrom_bin_indices_starts]

			chrom_replication_times = retrieve_replication_timing(self.config1,
								   self.config2,
								   chrom_replication_indices)

			gene_replication_times.loc[chrom_genes.index, 'replication_index'] = chrom_replication_indices
			gene_replication_times.loc[chrom_genes.index, 'replication_time'] = chrom_replication_times.values

		self.gene_replication_times = gene_replication_times
	
		
	def run_all_analyses(self):
		"""Run all analysis steps in sequence"""
		print("Starting complete polar analysis...")
		
		# Initialize analyses
		self.initialize_analyses()
		
		# Create and process datasets
		self.create_polar_datasets()
		
		# Generate all plots
		self.plot_ptr_distributions()
		self.plot_polar_branches()
		self.plot_venn_diagrams()
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
	analyzer = GeneChromatinAnalysis(output_directory)
	
	# Run all analyses
	analyzer.run_all_analyses()
	
	print("Analysis completed successfully")


if __name__ == "__main__":
	main()