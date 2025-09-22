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

ptr_formatting = {
	'positioning': {
		'xlims': (0.99, 1.5)
	},
	'occupancy': {
		'xlims': (0.95, 2.1)
	},
	'entropy': {
		'xlims': (0.99, 1.25)
	},
}

class FigureNucleosomes:
	"""
	Analyzes cell cycle nucleosome dynamics and histone modification enrichment patterns.
	
	Integrates Chereji nucleosome positioning data with Weiner histone modification data
	to identify cyclicity patterns and statistical enrichments using decile-based analysis.
	"""

	def __init__(self, output_dir="output/draft4_run/", window_size=160, 
				 n_deciles=10, random_seed=123):
		"""
		Initialize the analyzer with configuration parameters.
		"""
		# Configuration parameters
		self.output_dir = output_dir
		self.save_dir = f"{self.output_dir}/nucleosome_metrics"
		self.figures_dir = f'{self.output_dir}/Figures'
		mkdir_safe(self.save_dir)

		self.window_size = window_size
		self.n_deciles = n_deciles
		self.random_seed = random_seed
		
		# Core data processing objects
		self.histones_nucleosomes_dataset = HistonesNucleosomesDataset()
		self.nucleosome_loader = NucleosomeDataLoader(output_dir=output_dir)
		self.expression_processor = None
		
		# Data storage members
		self.chereji_integrated_data = None
		self.weiner_histones = None
		self.histone_cols = None

		# Chromatin metrics storage
		self.plus_one_chromatin_metrics = None
		self.minus_one_chromatin_metrics = None

		# PTRs storage
		self.plus_one_ptrs = None
		self.minus_one_ptrs = None
		
		# Cyclicity measures - now organized by nucleosome type
		self.histones_sorted_by_ptr = {}  # Will store {nucleosome_type: sorted_histone_data}
		
		# Analysis groups - decile-based, organized by nucleosome type
		self.decile_groups = {}  # Will store {nucleosome_type: {metric: {decile_num: group_df}}}
		self.current_nucleosome_type = None
		self.genomic_background = None
		
		# Results storage
		self.enrichment_results = None

	def create_cyclicity_groups(self, metric='occupancy', nucleosome_type='plus_one'):
		"""
		Create decile groups based on specified metric and nucleosome type.
		
		Args:
			metric (str): Metric to use for grouping ('occupancy', 'entropy', 'positioning')
			nucleosome_type (str): Nucleosome type ('plus_one' or 'minus_one')
		"""
		print(f"Creating decile groups based on {metric} metric for {nucleosome_type} nucleosomes...")
		
		# Select the appropriate PTR DataFrame based on nucleosome type
		if nucleosome_type == 'plus_one':
			ptr_source = self.plus_one_ptrs
			nuc_id_column = 'matched_nuc_id_p1'
		elif nucleosome_type == 'minus_one':
			ptr_source = self.minus_one_ptrs
			nuc_id_column = 'matched_nuc_id_m1'
		else:
			raise ValueError(f"Unknown nucleosome_type: {nucleosome_type}. Use 'plus_one' or 'minus_one'")
		
		# Select the appropriate metric
		if metric == 'occupancy':
			ptr_df = ptr_source['occupancy']
			sort_column = 'occupancy_ptr'
		elif metric == 'entropy':
			ptr_df = ptr_source['entropy']
			sort_column = 'entropy_ptr'
		elif metric == 'positioning':
			ptr_df = ptr_source['positioning']
			sort_column = 'positioning_ptr'
		else:
			raise ValueError(f"Unknown metric: {metric}. Use 'occupancy', 'entropy', or 'positioning'")
		
		# Link to histone modification data
		orf_to_nucleosomes = self.chereji_integrated_data[[nuc_id_column]]
		nuc_id_sorted_by_ptr = orf_to_nucleosomes.loc[ptr_df.index]
		
		# Store histone data sorted by PTR for this nucleosome type
		if nucleosome_type not in self.histones_sorted_by_ptr:
			self.histones_sorted_by_ptr[nucleosome_type] = {}
		
		self.histones_sorted_by_ptr[nucleosome_type] = self.weiner_histones.loc[
			nuc_id_sorted_by_ptr.dropna()[nuc_id_column]
		]
		
		# Create decile groups
		df = self.histones_sorted_by_ptr[nucleosome_type].reset_index()
		df.columns = ['nuc_id'] + list(df.columns[1:])
		
		# Initialize storage structure if needed
		if nucleosome_type not in self.decile_groups:
			self.decile_groups[nucleosome_type] = {}
		
		# Create decile groups for this nucleosome type and metric
		self.decile_groups[nucleosome_type][metric] = {}
		decile_size = len(df) // self.n_deciles
		
		for i in range(self.n_deciles):
			start_idx = i * decile_size
			end_idx = (i + 1) * decile_size if i < self.n_deciles - 1 else len(df)
			self.decile_groups[nucleosome_type][metric][i] = df.iloc[start_idx:end_idx]
		
		print(f"Created {self.n_deciles} decile groups with ~{decile_size} nucleosomes each for {nucleosome_type}")
	
	def load_and_integrate_data(self):
		"""Load Chereji nucleosome and Weiner histone data and integrate them."""
		print("Loading and integrating nucleosome and histone data...")
		
		# Load all datasets
		self.histones_nucleosomes_dataset.load_all()
		
		# Store integrated data
		self.chereji_integrated_data = self.histones_nucleosomes_dataset.integrated_data
		self.weiner_histones = self.histones_nucleosomes_dataset.weiner_histones
		self.brogaard_integrated_data = self.load_and_map_brogaard_data()
		
		# Extract histone modification column names
		self.histone_cols = [col for col in self.weiner_histones.columns 
							if col not in ['nuc_id', 'chr', 'start', 'end']]
		
		print(f"Loaded {len(self.chereji_integrated_data)} genes with nucleosome data")
		print(f"Loaded {len(self.weiner_histones)} nucleosomes with histone modifications")
		print(f"Found {len(self.histone_cols)} histone modifications")

	def load_and_map_brogaard_data(self):
		"""
		Load Brogaard nucleosome data and map to Weiner nucleosome IDs using 
		the general matching functionality in HistonesNucleosomesDataset.
		"""
		from src.reference_data import read_brogaard_nucleosomes
		
		print("Loading Brogaard nucleosome data...")
		brogaard_nucleosomes = read_brogaard_nucleosomes()
		print(f"Loaded {len(brogaard_nucleosomes)} Brogaard nucleosomes")
		
		# Use the dataset class's general matching method
		print("Mapping Brogaard nucleosomes to Weiner nucleosome IDs...")
		self.brogaard_integrated_data = self.histones_nucleosomes_dataset.match_brogaard_nucleosomes(
			brogaard_nucleosomes
		)
		
		print("Brogaard-Weiner mapping completed!")
		return self.brogaard_integrated_data

	def setup_processors(self):
		from src.GenomeDeconvolutionAnalysis import GenomeDeconvolutionAnalysis
		from pipeline.transcription_processor import ExpressionAnalysisProcessor
		from src.deconvolved_tpm_plotter import DeconvolvedTPMPlotter

		# For locus plotting
		if self.expression_processor is None:
			self.expression_processor = ExpressionAnalysisProcessor(self.output_dir)
			self.expression_processor.setup_data_loaders()
			self.expression_processor.load_deconvolved_expression()
			self.expression_processor.compute_expression_ptrs()
	
	def compute_nucleosome_metrics(self, force_recompute=False):
		"""Process all gene nucleosomes to compute chromatin metrics across timepoints."""
		print("Computing nucleosome metrics across cell cycle timepoints...")
		
		# Process all metrics for gene-associated nucleosomes
		self.plus_one_chromatin_metrics, self.minus_one_chromatin_metrics = \
			self.nucleosome_loader.process_all_gene_nucleosomes(
				gene_dataset=self.chereji_integrated_data,
				window_size=self.window_size,
				force_recompute=force_recompute,
			)

		self.brogaard_chromatin_metrics = self.nucleosome_loader.process_all_brogaard_nucleosomes(
				self.brogaard_integrated_data,
				window_size=self.window_size,
				force_recompute=force_recompute,
		)
		
		print(f"Computed metrics for {len(self.plus_one_chromatin_metrics['entropy'])} +1 nucleosomes")
		print(f"Computed metrics for {len(self.minus_one_chromatin_metrics['entropy'])} -1 nucleosomes")


	def calculate_cyclicity_measures(self):
		"""Calculate peak-to-trough ratios for all nucleosome metrics."""
		print("Calculating cyclicity measures (peak-to-trough ratios)...")
		
		# Calculate PTRs for +1 nucleosome metrics
		entropy_plus1 = self.plus_one_chromatin_metrics['entropy']
		occupancy_plus1 = self.plus_one_chromatin_metrics['occupancy']
		positioning_plus1 = self.plus_one_chromatin_metrics['positioning']

		# Calculate PTRs for -1 nucleosome metrics
		entropy_minus1 = self.minus_one_chromatin_metrics['entropy']
		occupancy_minus1 = self.minus_one_chromatin_metrics['occupancy']
		positioning_minus1 = self.minus_one_chromatin_metrics['positioning']
		
		# Compute PTRs for +1 nucleosomes
		p1_entropy_ptrs = compute_quantile_ptr_2d(entropy_plus1)
		p1_occupancy_ptrs = compute_quantile_ptr_2d(occupancy_plus1)

		# Positioning is defined difference from the mean, so
		# offset for positioning, ensures non-negative changes
		p1_positioning_ptrs = compute_quantile_ptr_2d(positioning_plus1 + 100)  

		# Compute PTRs for -1 nucleosomes
		m1_entropy_ptrs = compute_quantile_ptr_2d(entropy_minus1)
		m1_occupancy_ptrs = compute_quantile_ptr_2d(occupancy_minus1)
		m1_positioning_ptrs = compute_quantile_ptr_2d(positioning_minus1 + 100)  # Offset for positioning
		
		# Create sorted DataFrames helper function
		def _create_ptr_df(ptrs_data, name, index):
			key = f'{name}_ptr'
			ptrs_df = pd.DataFrame(
					ptrs_data, index=index, columns=[key]
				).sort_values(key, ascending=False)
			return ptrs_df

		# Store +1 nucleosome PTRs
		self.plus_one_ptrs = {
			"entropy": _create_ptr_df(p1_entropy_ptrs, 'entropy', occupancy_plus1.index),
			"occupancy": _create_ptr_df(p1_occupancy_ptrs, 'occupancy', occupancy_plus1.index),
			"positioning": _create_ptr_df(p1_positioning_ptrs, 'positioning', occupancy_plus1.index),
		}

		# Store -1 nucleosome PTRs
		self.minus_one_ptrs = {
			"entropy": _create_ptr_df(m1_entropy_ptrs, 'entropy', occupancy_minus1.index),
			"occupancy": _create_ptr_df(m1_occupancy_ptrs, 'occupancy', occupancy_minus1.index),
			"positioning": _create_ptr_df(m1_positioning_ptrs, 'positioning', occupancy_minus1.index),
		}
		
		print(f"Calculated cyclicity measures for {len(self.plus_one_chromatin_metrics['entropy'])} +1 nucleosomes")
		print(f"Calculated cyclicity measures for {len(self.minus_one_chromatin_metrics['entropy'])} -1 nucleosomes")

		# Calculate PTRs for Brogaard nucleosomes if available
		if hasattr(self, 'brogaard_chromatin_metrics') and self.brogaard_chromatin_metrics is not None:
			print("Calculating cyclicity measures for Brogaard nucleosomes...")
			
			entropy_brogaard = self.brogaard_chromatin_metrics['entropy']
			occupancy_brogaard = self.brogaard_chromatin_metrics['occupancy']
			positioning_brogaard = self.brogaard_chromatin_metrics['positioning']
			
			# Compute PTRs for Brogaard nucleosomes
			brogaard_entropy_ptrs = compute_quantile_ptr_2d(entropy_brogaard)
			brogaard_occupancy_ptrs = compute_quantile_ptr_2d(occupancy_brogaard)
			brogaard_positioning_ptrs = compute_quantile_ptr_2d(positioning_brogaard + 100)  # Offset for positioning
			
			# Store Brogaard nucleosome PTRs
			self.brogaard_ptrs = {
				"entropy": _create_ptr_df(brogaard_entropy_ptrs, 'entropy', entropy_brogaard.index),
				"occupancy": _create_ptr_df(brogaard_occupancy_ptrs, 'occupancy', occupancy_brogaard.index),
				"positioning": _create_ptr_df(brogaard_positioning_ptrs, 'positioning', positioning_brogaard.index),
			}
			
			print(f"Calculated cyclicity measures for {len(entropy_brogaard)} Brogaard nucleosomes")

	def perform_go_on_nucleosome_groups(self):
		from src.gene_ontology import GeneOntology

		gene_ontology = GeneOntology()
		from src.sgd import read_nondubious_genes_dataset

		all_go_results_df = pd.DataFrame()
		for decile_num in range(self.n_deciles):
			for metric in ['positioning', 'occupancy', 'entropy']:
				genes = read_nondubious_genes_dataset()

				selected_orfs = self._retrieve_orfs_for_group(
					metric, decile_num)
				selected_gene_names = genes.loc[selected_orfs]['gene'].values
				gene_ontology.run_go(selected_gene_names)
				results = gene_ontology.results_df.copy()
				results['metric'] = metric
				results['group'] = f'decile_{decile_num}'

				all_go_results_df = pd.concat([all_go_results_df, results])

		results = all_go_results_df

		omit_groups = ['biological_process', 'cellular_component', 'molecular_function']
		all_results = results[~results['name'].isin(omit_groups)].set_index(['group', 'metric'])
		sig_results = all_results[all_results.fdr_bh < 0.05]

		self.all_go_results = all_results
		self.sig_go_results = sig_results

	def create_table_for_go_group(self, group):
		from pipeline.latex_helpers import simple_df_to_latex_table

		self.perform_go_on_nucleosome_groups()

		plot_sig_data = self.sig_go_results[['id', 'name', 'fdr_bh']].loc[group]
		plot_sig_data = plot_sig_data.reset_index()
		# Generate LaTeX table
		plot_sig_data.metric = plot_sig_data.metric.str.title()

		def apply_go_name_capitalization(name):
			if not (name.startswith('rRNA') or name.startswith('tRNA')):
				name = name[0].upper() + name[1:]
			return name
		plot_sig_data['name'] = plot_sig_data['name'].apply(apply_go_name_capitalization)
		plot_sig_data.columns = ['Metric', 'GO ID', 'GO term', 'FDR (BH)']
		
		latex_output = simple_df_to_latex_table(plot_sig_data)
		return latex_output

	def create_latex_go_tables(self):
		# Create tables for extreme deciles
		res = self.create_table_for_go_group(f'decile_{self.n_deciles-1}')  # Lowest cyclicity
		save_path = f'{self.save_dir}/lowest_decile_go_group.txt'
		with open(save_path, 'w') as f:
			f.write(res)
		print(f"Wrote to: ", save_path)

		res = self.create_table_for_go_group('decile_0')  # Highest cyclicity
		save_path = f'{self.save_dir}/highest_decile_go_group.txt'
		with open(save_path, 'w') as f:
			f.write(res)
		print(f"Wrote to: ", save_path)
		
	
	def run_enrichment_analysis(self, metric, nucleosome='plus_one'):
		"""Perform statistical enrichment analysis for all deciles vs background."""
		print("Running enrichment analysis for deciles...")
		
		results = {}
		for decile_num in range(self.n_deciles):
			decile_group = self.decile_groups[nucleosome][metric][decile_num]
			decile_results = self._test_cyclicity_enrichment_zscore(
				decile_group, self.genomic_background, f"Decile_{decile_num}"
			)
			results[f'decile_{decile_num}'] = self._perform_fdr_correction(decile_results)
		
		self.enrichment_results = results
		print(f"Enrichment analysis completed for {self.n_deciles} deciles")

	
	def export_results(self, metric='occupancy', nucleosome='plus_one'):
		"""
		Export analysis results to CSV files.
		
		Args:
			metric (str): Metric used for analysis (for auto-generating filenames)
		"""
		output_prefix = f"{metric}_cyclicity_{nucleosome}"
		print(f"Exporting results with prefix: {output_prefix}")
		
		# Export histone modification data sorted by cyclicity
		histone_export_file = f'{self.save_dir}/histone_mod_values_nucleosomes_sorted_by_{output_prefix}.csv'
		self.histones_sorted_by_ptr[nucleosome].to_csv(histone_export_file)
		print(f"Exported histone data to: {histone_export_file}")
		
		# Export enrichment analysis results for all deciles
		for decile_name, results in self.enrichment_results.items():
			results_file = f'{self.save_dir}/{output_prefix}_enrichment_{decile_name}.csv'
			results.to_csv(results_file, index=False)
			print(f"Exported {decile_name} enrichment results to: {results_file}")
		
		# Export PTR rankings
		ptr_files = {
			'entropy': self.plus_one_ptrs['entropy'],
			'occupancy': self.plus_one_ptrs['occupancy'],
			'positioning': self.plus_one_ptrs['positioning']
		}
		
		for ptr_metric, ptr_df in ptr_files.items():
			ptr_file = f'{self.save_dir}/{output_prefix}_{ptr_metric}_ptr_rankings.csv'
			ptr_df.to_csv(ptr_file)
			print(f"Exported {ptr_metric} PTR rankings to: {ptr_file}")
	
	def generate_visualizations(self, metric='occupancy', vmax=7):
		"""Generate visualization plots for the analysis."""
		print("Generating visualizations...")
		
		# Create heatmap visualization
		plt.figure(figsize=(12, 8))
		
		# Full dataset heatmap
		plt.subplot(1, 3, 1)
		plt.imshow(self.histones_sorted_by_ptr, aspect='auto', cmap='PRGn',
				  interpolation='none', vmin=-vmax, vmax=vmax)
		
		# Add decile boundary lines
		decile_size = len(self.histones_sorted_by_ptr) // self.n_deciles
		for i in range(1, self.n_deciles):
			plt.axhline(i * decile_size, c='red', linewidth=1, alpha=0.7)
		
		plt.title(f"All +1 Nucleosomes\n(sorted by {metric} cyclicity)")
		plt.ylabel("Nucleosomes (ranked by cyclicity)")
		
		# Top decile heatmap
		plt.subplot(1, 3, 2)
		top_decile = self.histones_sorted_by_ptr.head(decile_size)
		plt.imshow(top_decile, aspect='auto', cmap='PRGn',
				  interpolation='none', vmin=-vmax, vmax=vmax)
		plt.title(f"Top Decile (Highest Cyclicity)")
		
		# Bottom decile heatmap
		plt.subplot(1, 3, 3)
		bottom_decile = self.histones_sorted_by_ptr.tail(decile_size)
		plt.imshow(bottom_decile, aspect='auto', cmap='PRGn',
				  interpolation='none', vmin=-vmax, vmax=vmax)
		plt.title(f"Bottom Decile (Lowest Cyclicity)")
		
		plt.tight_layout()
		
		# Save figure
		output_prefix = f"{metric}_cyclicity"
		fig_file = f'{self.save_dir}/{output_prefix}_decile_heatmaps.png'
		save_figure_for_paper(fig_file)
		print(f"Saved heatmap visualization to: {fig_file}")
		
		# Create cyclicity distribution plot
		plt.figure(figsize=(12, 4))
		
		ptr_data = {
			'entropy': self.plus_one_ptrs['entropy']['entropy_ptr'],
			'occupancy': self.plus_one_ptrs['occupancy']['occupancy_ptr'],
			'positioning': self.plus_one_ptrs['positioning']['positioning_ptr']
		}

		for i, (ptr_metric, ptr_values) in enumerate(ptr_data.items(), 1):
			plt.subplot(1, 3, i)
			plt.plot(np.arange(len(ptr_values)), ptr_values.values)
			
			# Add decile boundary lines
			decile_size = len(ptr_values) // self.n_deciles
			for j in range(1, self.n_deciles):
				plt.axvline(j * decile_size, c='red', linewidth=1, alpha=0.7, 
						   label='Decile boundaries' if j == 1 else '')

			plt.xlabel('Nucleosome Rank')
			plt.ylabel(f'{ptr_metric.title()} PTR')
			plt.title(f'{ptr_metric.title()} Cyclicity Distribution')
			if i == 1:
				plt.legend()
		
		plt.tight_layout()
		
		# Save distribution plot
		dist_fig_file = f'{self.save_dir}/nucleosome_cyclicity_decile_distributions.png'
		save_figure_for_paper(dist_fig_file)
		print(f"Saved distribution plot to: {dist_fig_file}")
		plt.show()
	
	def get_summary_stats(self):
		"""Return summary statistics for the loaded datasets."""
		if self.histones_nucleosomes_dataset.integrated_data is not None:
			return self.histones_nucleosomes_dataset.get_summary_stats()
		else:
			return "Data not loaded. Call load_and_integrate_data() first."

	def create_chereji_cyclicity_groups(self):
		metrics = ['occupancy', 'entropy', 'positioning']

		for nucleosome in ['plus_one', 'minus_one']:

			# Initialize stored enrichment and cyclicity groups per nucleosome
			self.all_metrics_enrichment_results[nucleosome] = {}

			for metric in metrics:
				self.create_cyclicity_groups(metric=metric, nucleosome_type=nucleosome)
				enrichment_results = self.run_analysis_for_metric(metric, nucleosome)
				self.all_metrics_enrichment_results[nucleosome][metric] = enrichment_results

	def run_full_analysis(self, force_recompute=False):
		"""
		Execute the complete analysis workflow.
		
		Args:
			force_recompute (bool): Whether to recompute nucleosome metrics
		"""
		self.setup_processors()

		print(f"Starting full cell cycle nucleosome analysis with {self.n_deciles} deciles...")
		print("=" * 60)
		
		# Execute complete pipeline
		self.load_and_integrate_data()
		self.compute_nucleosome_metrics(force_recompute=force_recompute)
		self.calculate_cyclicity_measures()

		self.all_metrics_enrichment_results = {}

		self.create_chereji_cyclicity_groups()
		self.create_brogaard_cyclicity_groups()

		# Create combined visualizations
		self.plot_cyclicity_p1_histograms()
		save_figure_for_paper(f"{self.save_dir}/plus_one_ptr_histograms.png")

		self.plot_all_metrics_enrichment_deciles(plot_key='difference')
		save_figure_for_paper(f"{self.save_dir}/plus_one_decile_enrichment.png")

		self.histone_mod_plotter.plot_colorbar()
		save_figure_for_paper(f"{self.save_dir}/plus_one_heatmap_colorbar.png")

		self.plot_nucleosome_expression_ptrs()
		save_figure_for_paper(f"{self.save_dir}/tx_nucleosome_ptrs.png")

		self.plot_p1_tss_agreement()
		save_figure_for_paper(f"{self.save_dir}/plus_one_tss_comparison.png")

		# Create heatmap of intersection across deciles
		self.plot_heatmap_decile_intersections()
		save_figure_for_paper(f"{self.save_dir}/decile_intersections_heatmap.png")

		# Create LaTeX tables
		self.create_latex_go_tables()


	def create_brogaard_cyclicity_groups(self):
		"""
		Create decile groups based on specified metric for Brogaard nucleosomes.
		
		Args:
			metric (str): Metric to use for grouping ('occupancy', 'entropy', 'positioning')
		"""

		metrics = ['occupancy', 'entropy', 'positioning']

		for metric in metrics:
			
			# Check if Brogaard PTRs are available
			if not hasattr(self, 'brogaard_ptrs') or self.brogaard_ptrs is None:
				raise RuntimeError("Brogaard PTRs not calculated. Call calculate_cyclicity_measures() first.")
			
			# Select the appropriate PTR DataFrame
			ptr_df = self.brogaard_ptrs[metric]
			sort_column = f'{metric}_ptr'
			
			# Get Brogaard nucleosome IDs ranked by PTR (ptr_df is already sorted)
			brogaard_nuc_ids_sorted_by_ptr = ptr_df.index
			
			# Map Brogaard nucleosome IDs to Weiner nucleosome IDs for histone data linkage
			# Assuming brogaard_integrated_data has a column like 'weiner_nuc_id' or similar
			brogaard_to_weiner_mapping = self.brogaard_integrated_data.set_index(self.brogaard_integrated_data.index)
			
			# Identify the correct mapping column (flexible naming)
			mapping_column = 'matched_nuc_id'
			
			# Create mapping series for sorted Brogaard nucleosomes
			weiner_ids_for_sorted_brogaard = []
			valid_brogaard_ids = []
			
			for brogaard_id in brogaard_nuc_ids_sorted_by_ptr:
				if brogaard_id in brogaard_to_weiner_mapping.index:
					weiner_id = brogaard_to_weiner_mapping.loc[brogaard_id, mapping_column]
					if pd.notna(weiner_id):
						weiner_ids_for_sorted_brogaard.append(weiner_id)
						valid_brogaard_ids.append(brogaard_id)
			
			print(f"Found {len(valid_brogaard_ids)} Brogaard nucleosomes with valid Weiner mappings out of {len(brogaard_nuc_ids_sorted_by_ptr)}")
			
			# Get histone modification data for mapped nucleosomes
			valid_weiner_ids = [wid for wid in weiner_ids_for_sorted_brogaard if wid in self.weiner_histones.index]
			histones_sorted_by_brogaard_ptr = self.weiner_histones.loc[valid_weiner_ids]
			
			print(f"Retrieved histone data for {len(histones_sorted_by_brogaard_ptr)} nucleosomes")
			
			# Store histone data sorted by PTR for Brogaard nucleosomes
			if 'brogaard' not in self.histones_sorted_by_ptr:
				self.histones_sorted_by_ptr['brogaard'] = {}
			
			self.histones_sorted_by_ptr['brogaard'] = histones_sorted_by_brogaard_ptr
			
			# Create decile groups
			df = histones_sorted_by_brogaard_ptr.reset_index()
			df.columns = ['nuc_id'] + list(df.columns[1:])
			
			# Initialize storage structure if needed
			if 'brogaard' not in self.decile_groups:
				self.decile_groups['brogaard'] = {}
			
			# Create decile groups for this metric
			self.decile_groups['brogaard'][metric] = {}
			decile_size = len(df) // self.n_deciles
			
			for i in range(self.n_deciles):
				start_idx = i * decile_size
				end_idx = (i + 1) * decile_size if i < self.n_deciles - 1 else len(df)
				self.decile_groups['brogaard'][metric][i] = df.iloc[start_idx:end_idx]

		nucleosome = 'brogaard'
		self.all_metrics_enrichment_results[nucleosome] = {}
		for metric in metrics:
			enrichment_results = self.run_analysis_for_metric(metric, nucleosome)
			self.all_metrics_enrichment_results[nucleosome][metric] = enrichment_results
		
		print(f"Created {self.n_deciles} decile groups with ~{decile_size} nucleosomes each for Brogaard dataset")
			

	def run_analysis_for_metric(self, metric, nucleosome='plus_one'):
		print(f"Running analysis for metric {metric}")

		# Set genomic background to full dataset
		df = self.histones_sorted_by_ptr[nucleosome].reset_index()
		df.columns = ['nuc_id'] + list(df.columns[1:])
		self.genomic_background = df
		
		self.run_enrichment_analysis(metric, nucleosome)
		self.export_results(metric=metric, nucleosome=nucleosome)
		
		print("Analysis completed!")
		print(f"Results exported with prefix: {metric}_{nucleosome}_cyclicity")
		
		return self.enrichment_results

	def _test_cyclicity_enrichment_zscore(self, test_group, full_population, group_name):
		"""Test if cyclicity group differs significantly from population mean using z-test."""
		results = []
		
		for modification in self.histone_cols:
			# Get population parameters
			pop_values = full_population[modification].dropna()
			pop_mean = pop_values.mean()
			pop_std = pop_values.std()
			
			# Get test group values
			test_values = test_group[modification].dropna()
			test_mean = test_values.mean()
			n_test = len(test_values)
			
			# Calculate z-score
			standard_error = pop_std / np.sqrt(n_test)
			z_score = (test_mean - pop_mean) / standard_error
			
			# Two-tailed p-value
			eps = 1e-10 # Very highly enriched values will be zero, so let's
						# set a minimum threshold
			p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))+eps
			
			# Effect size (Cohen's d)
			cohens_d = (test_mean - pop_mean) / pop_std
			
			results.append({
				'group': group_name,
				'modification': modification,
				'test_mean': test_mean,
				'population_mean': pop_mean,
				'difference': test_mean - pop_mean,
				'cohens_d': cohens_d,
				'z_score': z_score,
				'p_value': p_value,
				'test_n': n_test,
				'population_n': len(pop_values)
			})
		
		return results
	
	def _perform_fdr_correction(self, all_results):
		"""Apply FDR correction and format results."""
		# Convert to DataFrame
		all_results = pd.DataFrame(all_results)
		
		# Apply FDR correction
		p_value_fdr = multipletests(all_results.p_value, method='fdr_bh')[1]
		all_results['p_value_fdr'] = p_value_fdr
		
		# Sort by FDR-corrected p-value
		all_results = all_results.sort_values('p_value_fdr')
		
		# Add significance labels
		def get_significance_label(p_val):
			if p_val < 1e-8:
				return '***'
			elif p_val < 1e-6:
				return '**'
			elif p_val < 1e-4:
				return '*'
			else:
				return ''
		
		all_results['significance'] = all_results['p_value_fdr'].apply(get_significance_label)
		
		return all_results

	def plot_all_metrics_enrichment_deciles(self, plot_key='difference', nucleosome='plus_one'):
		"""Plot enrichment across all deciles for all metrics."""
		from src.histone_group_plotter import HistoneModificationGroupedPlotter

		# Initialize the plotter with your enrichment results
		self.histone_mod_plotter = HistoneModificationGroupedPlotter(
			enrichment_results=self.all_metrics_enrichment_results[nucleosome],
			n=len(self.chereji_integrated_data)
		)

		name_mapping = {
			'plus_one': '+1',
			'minus_one': '-1',
			'brogaard': 'Brogaard',
		}

		n = len(self.chereji_integrated_data) if not nucleosome == 'brogaard' else len(self.brogaard_integrated_data)

		nuc_type = name_mapping[nucleosome]

		# Create plot showing gradient across deciles
		fig = self.histone_mod_plotter.plot_decile_enrichment_gradient(
			plot_key=plot_key,
			n_deciles=self.n_deciles,
			title=f"Histone modifications by nucleosome cyclicity,\n{nuc_type} "
				  f"nucleosomes, n={n}"
		)

	def plot_p1_tss_agreement(self):
		from src.transcripts_dataset import load_transcripts_sets
		genes, _ = load_transcripts_sets(self.output_dir)
		joined_tss_p1 = genes[['TSS']].join(self.chereji_integrated_data[['+1 nucleosome']], how='inner').dropna()
		joined_tss_p1['difference'] = joined_tss_p1.TSS-joined_tss_p1['+1 nucleosome']

		plt.figure(figsize=(5, 2))
		plt.hist(joined_tss_p1['difference'], color=plt.cm.Greys(0.5),
		 bins=np.linspace(-600, 600, 100))
		plt.xlim(-600, 600)
		plt.title(f"+1 nucleosomes (Chereji, 2018) vs TSSes,\nn={len(joined_tss_p1)}")
		plt.xlabel("Difference, bp")
		plt.ylabel("Frequency")

	def plot_nucleosome_expression_ptrs(self):
		measures = ['positioning', 'occupancy', 'entropy']

		plt.figure(figsize=(7, 2.75))

		expression_processor = self.expression_processor

		colors = [
			plt.cm.Reds(0.5),
			plt.cm.Blues(0.5),
			plt.cm.Purples(0.5)]

		for i, measure in enumerate(measures):
			joined_chromatin_tx_ptrs = expression_processor.all_transcripts_ptrs[['ptr']]\
				.join(self.plus_one_ptrs[measure],
				how='inner')
			joined_chromatin_tx_ptrs.columns = ['expression_ptr', 'chromatin_ptr']

			plt.subplot(1, 3, i+1)
			plt.scatter(joined_chromatin_tx_ptrs.chromatin_ptr,
				joined_chromatin_tx_ptrs.expression_ptr, color=colors[i],
						alpha=0.25, s=2)
			plt.xlabel(measure.title() + " PTR")

			# Plot vertical lines for decile boundaries
			chromatin_values = joined_chromatin_tx_ptrs.chromatin_ptr.dropna().values.flatten()
			decile_size = len(chromatin_values) // self.n_deciles
			for j in range(1, self.n_deciles):
				q_threshold = np.quantile(chromatin_values, j / self.n_deciles)
				plt.axvline(q_threshold, c='black', lw=0.5, ls='solid', alpha=0.5, zorder=0)

			if i == 0: plt.ylabel('Expression PTR')
			else: plt.yticks([])

			plt.xlim(*ptr_formatting[measure]['xlims'])
			plt.title(measure.title())
			
		plt.suptitle(f"+1 nucleosome vs expression cyclicity, n={len(joined_chromatin_tx_ptrs)}", fontweight='demi', 
					fontsize=16)
		plt.tight_layout()

	def plot_cyclicity_p1_histograms(self):
		def _plot_hist_ptrs(ptrs_data, color, bins=30):
			# Add decile boundary lines
			decile_size = len(ptrs_data) // self.n_deciles
			plt.hist(ptrs_data, color=color, bins=bins)
			for j in range(1, self.n_deciles):
				q_threshold = np.quantile(ptrs_data.dropna().values.flatten(), j / self.n_deciles)
				plt.axvline(q_threshold, c='black', lw=0.5, ls='solid', alpha=0.5)

		colors = [
			plt.cm.Reds(0.5),
			plt.cm.Blues(0.5),
			plt.cm.Purples(0.5)]

		plt.figure(figsize=(9, 3))
		plt.subplot(1, 3, 1)
		_plot_hist_ptrs(self.plus_one_ptrs['positioning'], 
					  color=colors[0], bins=np.linspace(1, 1.2, 30))
		plt.title("Positioning")
		plt.xlabel("Peak-to-Trough Ratio (PTR)")
		plt.xlim(*(ptr_formatting['positioning']['xlims']))

		plt.subplot(1, 3, 2)
		_plot_hist_ptrs(self.plus_one_ptrs['occupancy'], color=colors[1], 
			bins=np.linspace(1, 2.5, 30))
		plt.title("Occupancy")
		plt.xlabel("Peak-to-Trough Ratio (PTR)")
		plt.xlim(*(ptr_formatting['occupancy']['xlims']))

		plt.subplot(1, 3, 3)
		_plot_hist_ptrs(self.plus_one_ptrs['entropy'], color=colors[2], 
			bins=np.linspace(1, 1.4, 30))
		plt.title("Entropy")
		plt.xlabel("Peak-to-Trough Ratio (PTR)")

		decile_size = len(self.plus_one_ptrs['entropy']) // self.n_deciles
		plt.suptitle(f"Cyclicity of +1 Chereji, (2018) nucleosomes,\n"
					f"n={len(self.plus_one_ptrs['entropy'])}, {decile_size} nucleosomes per decile",
					fontweight='demi', fontsize=18)
		plt.tight_layout()
		plt.xlim(*(ptr_formatting['entropy']['xlims']))

	def _retrieve_orfs_for_group(self, metric, decile_num):
		"""Retrieve ORFs for a specific decile."""
		orf_p1s = self.chereji_integrated_data[['matched_nuc_id_p1']].reset_index()
		orf_p1s = orf_p1s.dropna().set_index('matched_nuc_id_p1')
		orf_p1s.index = orf_p1s.index.astype(int)
		
		selected_nucs_histones_mods = self.decile_groups[metric][decile_num].set_index('nuc_id').join(orf_p1s)
		return selected_nucs_histones_mods.ORF.values

	def plot_heatmap_decile_intersections(self):
		"""Plot heatmap showing intersections between deciles across metrics."""
		from src.heatmap_counts import create_set_adjacency_matrix
		
		# Get ORFs for extreme deciles across metrics
		low_occ_nuc_orfs = self._retrieve_orfs_for_group('occupancy', self.n_deciles-1)  # Lowest
		low_ent_nuc_orfs = self._retrieve_orfs_for_group('entropy', self.n_deciles-1)
		low_pos_nuc_orfs = self._retrieve_orfs_for_group('positioning', self.n_deciles-1)

		high_occ_nuc_orfs = self._retrieve_orfs_for_group('occupancy', 0)  # Highest
		high_ent_nuc_orfs = self._retrieve_orfs_for_group('entropy', 0)
		high_pos_nuc_orfs = self._retrieve_orfs_for_group('positioning', 0)

		# Create adjacency matrix
		result = create_set_adjacency_matrix(
			low_pos_nuc_orfs, low_occ_nuc_orfs, low_ent_nuc_orfs,
			high_pos_nuc_orfs, high_occ_nuc_orfs, high_ent_nuc_orfs,
			labels=['Position\nnon-cyclers', 'Occupancy\nnon-cyclers', 'Entropy\nnon-cyclers', 
					'Position\ncyclers', 'Occupancy\ncyclers', 'Entropy\ncyclers'],
			metric='count',
			title="Cycling and non-cycling nucleosome\ndecile intersections"
		)

	def layout_panel(self):
		from pipeline.figure_composer import FigureCompositor
		from pipeline.figure_composer_helpers import layout_images_vertically, \
			add_panel_labels_to_images

		# Create compositor with wider dimensions for horizontal layout
		compositor = FigureCompositor(1024, 900, debug_mode=True)

		image_paths = [
			f'{self.save_dir}/plus_one_decile_enrichment.png',
			f'{self.save_dir}/plus_one_heatmap_colorbar.png',
		]

		placed_images = layout_images_vertically(
			compositor,
			[image_paths[0]],
			heights=[840],
			between_padding=30,
			margin=(30, 30),
			image_keys=['decile_enrichment']
		)

		compositor.add_panel_label_to_image(
			'decile_enrichment', "A", (-10, 47),
			font_size=40,
		)

		compositor.add_panel_label_to_image(
			'decile_enrichment', "B", (-10, 310),
			font_size=40,
		)

		compositor.add_panel_label_to_image(
			'decile_enrichment', "C", (-10, 577),
			font_size=40,
		)

		compositor.place_image(image_paths[1], 940, 80, width=80, name='colorbar')

		# Save the composite figure
		compositor.save(f'{self.figures_dir}/Figure7_Nucleosome_Histones.png')

	def layout_supplemental_panel(self):
		from pipeline.figure_composer import FigureCompositor
		from pipeline.figure_composer_helpers import layout_images_vertically, \
			layout_images_horizontally, add_panel_labels_to_images

		# Create compositor with wider dimensions for horizontal layout
		compositor = FigureCompositor(1024, 640, debug_mode=True)

		image_paths = [
			f'{self.save_dir}/plus_one_tss_comparison.png',
			f'{self.save_dir}/plus_one_ptr_histograms.png',
			f'{self.save_dir}/tx_nucleosome_ptrs.png',
			f'{self.save_dir}/decile_intersections_heatmap.png',
		]

		placed_images = layout_images_vertically(
			compositor,
			image_paths[:3],
			between_padding=30,
			margin=(30, 30),
			widths=[400, 400, 400],
			image_keys=['tsses', 'histograms', 'ptrs']
		)

		placed_images = layout_images_horizontally(
			compositor,
			[image_paths[3]],
			between_padding=30,
			margin=(460, 30),
			heights=[400],
			image_keys=['decile_heatmap'],
			available_width=530
		)

		# Add panel labels
		add_panel_labels_to_images(
			compositor, 
			compositor.placed_images,
			font_size=32,
			offset=(-15, 0)
		)

		# Save the composite figure
		compositor.save(f'{self.figures_dir}/Supplemental9_Nucleosome_metrics.png')