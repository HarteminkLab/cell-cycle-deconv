
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
	
	def __init__(self, output_directory, chromatin_data_directory='chromatin_deconvolution',
		save_directory='genelevel_chromatin'):
		"""
		Initialize the ExpressionChromatinAnalysis with the output directory
		
		Parameters:
		-----------
		output_directory : str
			Directory where output files will be saved
		"""
		self.output_directory = output_directory
		self.save_directory = os.path.join(output_directory, save_directory)
		self.chromatin_data_directory = chromatin_data_directory

		# Create output directory if it doesn't exist
		os.makedirs(self.save_directory, exist_ok=True)
		
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
		self.sm_percentile_threshold = 0.95
		self.entropy_percentile_threshold = 0.95
		
	def initialize_analyses(self, copy_correction=True):
		"""Initialize all analysis objects and load required data"""
		print(f"Initializing chromatin analysis... copy correction: {copy_correction}")
		if not copy_correction:
			print(f" ** todo: need to refactor no copy correction directory loading...")

		genome_deconv_full_path = os.path.join(self.output_directory, 
										f"{self.chromatin_data_directory}/deconvolution_data")
		analysis_path = self.save_directory # save the metrics csv files to the directory with the plots

		self.chromatin_analysis = DeconvolvedChromatinAnalysis(self.output_directory, 
			chromatin_data_path=genome_deconv_full_path,
			analysis_path=analysis_path)

		try:
			print("Loading chromatin measures")
			self.chromatin_analysis.load_chromatin_measures()
		except Exception as e:
			print(f"Failed to load chromatin measures: {e}")
			print("Computing measures...")
			self.chromatin_analysis.compute_chromatin_measures(debug=False)

		print("Initializing genome deconvolution analysis...")

		self.genome_analysis = GenomeDeconvolutionAnalysis(outdir=genome_deconv_full_path)
		
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
		def plot_ptr_hist(data_df, percentile_thresh, ptr_thresh, subplot_pos, key, title):
			plt.subplot(1, 2, subplot_pos)
			ptrs = data_df.ptr

			plt.hist(ptrs, bins=30, color=color_map[key+'_only'])
			plt.axvline(ptr_thresh, c='red', alpha=0.75)
			
			thresholded_genes = ptrs[ptrs > ptr_thresh]
			num_threshold = len(thresholded_genes)
			plt.xlabel('PTR')
			plt.title(title, y=1.05, fontsize=14)

			plt.text(ptr_thresh, 1000, f" Threshold: {ptr_thresh:.2f} ({percentile_thresh*100:.1f}%)", ha='left', c='red')
		
		# Plot promoter PTR
		plot_ptr_hist(
			self.small_polar_data_df, 
			self.sm_percentile_threshold,
			self.sm_ptr_threshold, 
			1, 
			'promoter',
			f"Small fragment promoter occupancy"
		)
		
		# Plot entropy PTR
		plot_ptr_hist(
			self.entropies_polar_data_df, 
			self.entropy_percentile_threshold,
			self.entropy_ptr_threshold, 
			2, 
			'entropy',
			f"Gene body nucleosome entropy"
		)
		plt.suptitle(f"Peak-to-trough distributions, percentile",
			fontsize=23)

		plt.tight_layout()
		save_figure_for_paper(f"{self.save_directory}/ptr_distributions.png")
		
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
		save_figure_for_paper(f"{self.save_directory}/promoters_polar.png")

		# Nucleosome entropy polar plot
		plot_polar_plot_polar_mean_branch_data(
			self.entropies_polar_data_df,
			self.entropy_ptr_threshold,
			"Nucleosome entropy", 
			(0.75, 1.75)
		)
		save_figure_for_paper(f"{self.save_directory}/entropy_polar.png")

		# Nucleosome occupancy polar plot
		# plot_polar_plot_polar_mean_branch_data(
		# 	self.gb_nucleosome_polar_data_df,
		# 	self.gb_ptr_threshold,
		# 	"Nucleosome occupancy", 
		# 	(0.5, 2)
		# )
		# save_figure_for_paper(f"{self.save_directory}/gb_polar.png")
		
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
		save_figure_for_paper(f"{self.save_directory}/metrics_venn2.png")
		
		print("Venn diagrams saved")
		
	def plot_ptr_sensitivity(self):
		"""Plot PTR sensitivity analysis"""
		print("Plotting PTR sensitivity analysis...")
		
		# Promoter vs Entropy
		interesting_chrom_points = compare_and_plot_thresholds(
			self.small_polar_data_df,
			self.entropies_polar_data_df, 
			'Promoter occupancy', 
			'Nucleosome entropy',
			vmax=10,
			plot_pvalue_heatmap=8.5
		)
		
		print("Interesting threshold values:")
		highest_neg_logpval = 0
		thresholds = None

		for row in interesting_chrom_points:
			print(f"{row[0]:0.4f}, {row[1]:0.4f}: p-value {row[2]:0.2f}")
			if row[2] > highest_neg_logpval:
				highest_neg_logpval = row[2]
				thresholds = row[:2]

		# If we found interesting ptr values, update the thresholds
		if len(interesting_chrom_points) > 0:
			print(f"Highest p-value: {thresholds[0]:0.4f}, {thresholds[1]:0.4f}: p-value {highest_neg_logpval:0.2f}")
			# Parameters for analysis, determined from sensitivity analysis
			self.sm_percentile_threshold = thresholds[1]
			self.entropy_percentile_threshold = thresholds[0]

		# Keep thresholds the same for comparison with other analyses
		# may want to keep at 95, for stringent purposes.
		# self.sm_percentile_threshold = 0.95
		# self.entropy_percentile_threshold = 0.95

		save_figure_for_paper(f"{self.save_directory}/ptr_sensitivity_prom_entropy.png")
		
		# Expected intersection heatmap
		plot_expected_intersection_heatmap(
			len(self.small_polar_data_df),
			"Measure 1", 
			"Measure 2", 
			figsize=(5, 4),
			highlight=interesting_chrom_points
		)
		save_figure_for_paper(f"{self.save_directory}/ptr_sensitivity_expected_counts.png")
		
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

		# Plot sensitivity and determine updated thresholds
		self.plot_ptr_sensitivity()

		# Update dataset with new PTR thresholds from sensivity
		# analysis
		self.create_polar_datasets()
		
		# Generate all plots
		self.plot_ptr_distributions()
		self.plot_polar_branches()
		self.plot_venn_diagrams()
		
		print("Complete polar analysis finished!")


def layout_figure_plots(plots_dir, save_dir):
	from pipeline.figure_composer import FigureCompositor

	# Define image paths
	image_names = [
		'metrics_venn2',
		'promoters_polar',
		'entropy_polar'
	]

	image_paths = [f"{plots_dir}/{name}.png" for name in image_names]

	# Create compositor with a scale factor of 4
	# Logical canvas size is 1024x800, but actual output will be 4096x3200
	compositor = FigureCompositor(1024, 340, debug_mode=True, scale_factor=4.0)

	# Place images individually
	# All coordinates and dimensions are specified in logical pixels
	# but will be rendered at 4x resolution
	margin = 20
	venn_width = 320

	# -------- Sensitivity analysis -------------
	compositor.place_image(image_paths[0], margin, margin, venn_width, None, 'venn')
	compositor.add_panel_label_to_image('venn', 'A')

	# -------- Polar plots -------------

	polar_width = 280
	pad_x_polar = 24

	polar_x = venn_width+pad_x_polar+margin
	img = compositor.place_image(image_paths[1], polar_x, margin, polar_width, 
						  None, 'prom_polar')
	compositor.add_panel_label_to_image('prom_polar', 'B')

	polar_x = polar_x+polar_width+pad_x_polar
	img = compositor.place_image(image_paths[2], polar_x, margin, polar_width, 
						  None, 'entropy_polar')
	compositor.add_panel_label_to_image('entropy_polar', 'C')

	# Save the figure - will be 4x the logical resolution
	compositor.save(f"{save_dir}/Figure_2.png")

def layout_supplemental_2(plots_dir, save_dir):
	from pipeline.figure_composer import FigureCompositor

	# Define image paths
	image_names = [
		'ptr_sensitivity_prom_entropy',
		'ptr_sensitivity_expected_counts',
		'ptr_distributions',
	]

	image_paths = [f"{plots_dir}/{name}.png" for name in image_names]

	scale = 4.0
	compositor = FigureCompositor(1024, 840, scale_factor=scale, debug_mode=True)

	# Place images individually
	# All coordinates and dimensions are specified in logical pixels
	# but will be rendered at 4x resolution
	margin = 20
	sens_width = 420
	expected_width = 455

	# A
	compositor.place_image(image_paths[0], margin, margin, sens_width, None, 
		'chrom_sens')
	compositor.add_panel_label_to_image('chrom_sens', 'A', offset=(5, 5))

	# B
	expected_img = compositor.place_image(image_paths[1], margin+24+sens_width, margin, expected_width, None, 
		'expected')
	expected_height = expected_img['logical_size'][1]
	compositor.add_panel_label_to_image('expected', 'B', offset=(5, 5))

	# C
	ptr_width = 700
	img = compositor.place_image(image_paths[2], margin, 
		margin+expected_height+34, ptr_width, None, 'dist')
	compositor.add_panel_label_to_image('dist', 'C', offset=(5, 5))

	compositor.save(f"{save_dir}/Supplemental_S2.png")

	
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