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
		'xlims': (0.99, 1.35)
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
		self.sort_deciles_by_increasing_ptr = True
		
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

	def create_chereji_cyclicity_groups_for_metric(self, metric='occupancy', 
			nucleosome_type='plus_one'):
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
			filter_column = 'filter_p1'
		elif nucleosome_type == 'minus_one':
			ptr_source = self.minus_one_ptrs
			nuc_id_column = 'matched_nuc_id_m1'
			filter_column = 'filter_m1'
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
		
		# Sort by PTR, select unfiltered nuclesomes
		# and retrieve the nucleosome id to analyze
		sort_column = f'{metric}_ptr'
		ptr_df = ptr_df.sort_values(sort_column, ascending=self.sort_deciles_by_increasing_ptr)

		orf_to_nucleosomes = self.chereji_integrated_data.loc[ptr_df.index][~self.chereji_integrated_data[
			filter_column]][[nuc_id_column]]
		nuc_id_sorted_by_ptr = orf_to_nucleosomes
		
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

		# Create filter set for each type of nucleosome
		for nucleosome_type in ['minus_one', 'plus_one', 'brogaard']:
			self.nucleosome_loader.create_occupancy_coverage_plot_and_filter(nucleosome_type)
			save_figure_for_paper(f"{self.save_dir}/nucleosome_coverage_{nucleosome_type}.png")

		# Apply filter and add to the integrated data sets
		filter_m1_orfs = self.nucleosome_loader.filtered_nucleosome_ids['minus_one']
		filter_p1_orfs = self.nucleosome_loader.filtered_nucleosome_ids['plus_one']
		filter_brogaard_ids = self.nucleosome_loader.filtered_nucleosome_ids['brogaard']

		self.chereji_integrated_data['filter_m1'] = False
		self.chereji_integrated_data['filter_p1'] = False
		self.chereji_integrated_data.loc[filter_m1_orfs, 'filter_m1'] = True
		self.chereji_integrated_data.loc[filter_p1_orfs, 'filter_p1'] = True
		self.brogaard_integrated_data['filter_brogaard'] = False
		self.brogaard_integrated_data.loc[filter_brogaard_ids, 'filter_brogaard'] = True
		
		print(f"Computed metrics for {len(self.plus_one_chromatin_metrics['entropy'])} +1 nucleosomes")
		print(f"Computed metrics for {len(self.minus_one_chromatin_metrics['entropy'])} -1 nucleosomes")
		print(f"Computed metrics for {len(self.brogaard_chromatin_metrics['entropy'])} brogaard nucleosomes")


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
				).sort_values(key, ascending=self.sort_deciles_by_increasing_ptr)
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

	def get_orfs_to_chereji_nuc_mapping(self, key='matched_nuc_id_p1', index_on=None):
		orfs_mapping = self.chereji_integrated_data.copy()
		orfs_mapping = orfs_mapping[[key]].dropna()
		orfs_mapping[key] = orfs_mapping[key].astype(int)
		if index_on is None: index_on = key
		orfs_mapping = orfs_mapping.reset_index().set_index(index_on)
		return orfs_mapping

	def _get_nucs_and_orfs_from_group(self, nucleosome, metric, decile_index):
		nuc_to_orf_map = self.get_orfs_to_chereji_nuc_mapping()
		deciles_groups = self.decile_groups['plus_one'][metric]
		nuc_ids = deciles_groups[decile_index].nuc_id.values
		decile_orfs = nuc_to_orf_map.loc[nuc_ids].ORF.values
		return nuc_ids, decile_orfs

	def plot_expression_deciles(self):
		def plot_deciles_of_average_tx(self, metric, summary_func):
			orfs_to_p1 = self.get_orfs_to_chereji_nuc_mapping()

			from src.config import load_default_chrom_configs
			config1, _ = load_default_chrom_configs()

			for i in range(10):
				current_decile = self.decile_groups['plus_one'][metric]\
					[i].nuc_id.values
				current_decile_orfs = orfs_to_p1.loc[current_decile].ORF

				current_decile_avg = summary_func(self\
					.expression_processor.expression_data.loc[
					current_decile_orfs][config1.t_indices()], axis=1)

				plt.boxplot(current_decile_avg, positions=[i+1], vert=False,
						   showfliers=False,
							flierprops=dict(markersize=1, marker='o', markerfacecolor='black'))
			plt.ylim(10.5, 0.5)
			plt.xlabel("Average transcript level")
			plt.title(metric.title())
			
		plt.figure(figsize=(9, 4))
		plt.subplot(1, 3, 1)
		plot_deciles_of_average_tx(self, 'positioning', np.mean)
		plt.ylabel("Cyclicity decile")
			
		plt.subplot(1, 3, 2)
		plot_deciles_of_average_tx(self, 'occupancy', np.mean)
		plt.subplot(1, 3, 3)
		plot_deciles_of_average_tx(self, 'entropy', np.mean)
		plt.suptitle("Average expression level", fontweight='demi')
		plt.tight_layout()

		plt.figure(figsize=(9, 4))
		plt.subplot(1, 3, 1)
		plot_deciles_of_average_tx(self, 'positioning', np.std)
		plt.ylabel("Cyclicity decile")
		plt.xlabel("Average $\\sigma$")

		plt.subplot(1, 3, 2)
		plot_deciles_of_average_tx(self, 'occupancy', np.std)
		plt.xlabel("Average $\\sigma$")

		plt.subplot(1, 3, 3)
		plot_deciles_of_average_tx(self, 'entropy', np.std)
		plt.xlabel("Average $\\sigma$")
		plt.suptitle("Average transcriptional variation", fontweight='demi')
		plt.tight_layout()

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
		# print(f"Exporting results with prefix: {output_prefix}")
		
		# Export histone modification data sorted by cyclicity
		histone_export_file = f'{self.save_dir}/histone_mod_values_nucleosomes_sorted_by_{output_prefix}.csv'
		# self.histones_sorted_by_ptr[nucleosome].to_csv(histone_export_file)
		# print(f"Exported histone data to: {histone_export_file}")
		
		# Export enrichment analysis results for all deciles
		for decile_name, results in self.enrichment_results.items():
			results_file = f'{self.save_dir}/{output_prefix}_enrichment_{decile_name}.csv'
			# results.to_csv(results_file, index=False)
			# print(f"Exported {decile_name} enrichment results to: {results_file}")
		
		# Export PTR rankings
		ptr_files = {
			'entropy': self.plus_one_ptrs['entropy'],
			'occupancy': self.plus_one_ptrs['occupancy'],
			'positioning': self.plus_one_ptrs['positioning']
		}
		
		for ptr_metric, ptr_df in ptr_files.items():
			ptr_file = f'{self.save_dir}/{output_prefix}_{ptr_metric}_ptr_rankings.csv'
			# ptr_df.to_csv(ptr_file)
			# print(f"Exported {ptr_metric} PTR rankings to: {ptr_file}")
	
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
				self.create_chereji_cyclicity_groups_for_metric(metric=metric, nucleosome_type=nucleosome)
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

		# Plot enrichment deciles for each nucleosome type
		self.plot_all_metrics_enrichment_deciles()
		self.plot_all_metrics_enrichment_deciles(nucleosome='brogaard')
		self.plot_all_metrics_enrichment_deciles(nucleosome='minus_one')

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

		# Other plotting metrics
		self.plot_expression_deciles()
		save_figure_for_paper(f"{self.save_dir}/expression_mean_variation_deciles.png")

		self.plot_metrics_by_occupancy_deciles()
		save_figure_for_paper(f"{self.save_dir}/metrics_vs_occupancy_deciles.png")

		self.plot_pos_scatter_w_regression()
		save_figure_for_paper(f"{self.save_dir}/position_scatter_regression.png")

		self.plot_position_expression_deciles()
		

	def plot_pos_scatter_w_regression(self):
		from src.DensityScatterPlotter import DensityScatterPlotter
		expression_ptrs = self.expression_processor.all_transcripts_ptrs
		def plot_ptr_comparision():
			positioning_ptrs = self.plus_one_ptrs['positioning']
			expression_chromatin_ptrs = expression_ptrs.join(positioning_ptrs, how='right')
			expression_chromatin_ptrs = expression_chromatin_ptrs.rename(columns={'ptr': 'expression_ptr'})

			dsc_plotter = DensityScatterPlotter()
			dsc_plotter.plot_outline = True
			dsc_plotter.outline_color = '#f0f0f0'

			np.random.seed(123)
			plot_data = expression_chromatin_ptrs.dropna()
			plot_data = plot_data.loc[np.random.permutation(plot_data.index)]

			# Plot the PTRs
			dsc_plotter.set_data(plot_data.positioning_ptr,
								 plot_data.expression_ptr)
			dsc_plotter.bw = 0.001, 0.001
			dsc_plotter.cmap = 'Blues'
			dsc_plotter.alpha = 1
			dsc_plotter.s = 5
			dsc_plotter.logz = False
			dsc_plotter.plot_ax(plt.gca(), vmin=0, vmax=500, plot_colorbar=False)

			# plt.scatter(expression_chromatin_ptrs.positioning_ptr, 
			#             expression_chromatin_ptrs.expression_ptr, alpha=0.1)
			#plt.title("Positioning PTR vs Expression PTR")

			x = plot_data.positioning_ptr
			y = plot_data.expression_ptr

			# Calculate linear regression
			coefficients = np.polyfit(x, y, 1)
			poly = np.poly1d(coefficients)

			# Create regression line
			x_line = np.linspace(x.min(), x.max(), 100)
			y_line = poly(x_line)

			# Plot regression line
			plt.plot(x_line, y_line, 'r-', linewidth=0.5,
				label=f'y = {coefficients[0]:.3f}x + {coefficients[1]:.3f}')

			plt.xlabel("Positioning PTR")
			plt.ylabel("Expression PTR")

			plt.xlabel('Positioning PTR')
			plt.legend()

			plt.xlim(0.99, 1.2)
			plt.ylim(0.99, 1.5)

		zoomed_xlim = 0.995, 1.15
		zoomed_ylim = (0.99, 1.3)

		plt.figure(figsize=(7, 4))
		plt.subplot(1, 2, 1)
		plot_ptr_comparision()
		plt.xlim(0.975, 1.6)
		plt.ylim(0.975, 7)
		plt.title("Full data")
		from src.plot_helpers import plot_rect2
		plot_rect2(plt.gca(), 
					x1=zoomed_xlim[0],
					x2=zoomed_xlim[1],
					y1=zoomed_ylim[0],
					y2=zoomed_ylim[1],
					facecolor='none', alpha=1,
					edgecolor=plt.cm.Greys(0.5), lw=0.75,
					zorder=30)

		plt.subplot(1, 2, 2)
		plot_ptr_comparision()
		plt.xlim(*zoomed_xlim)
		plt.ylim(*zoomed_ylim)
		plt.yticks([])
		plt.ylabel('')
		plt.title("Zoomed")

		plt.suptitle("Positioning PTR vs Expression PTR", fontweight='demi',
					fontsize=16)

		plt.tight_layout()

	def plot_position_expression_deciles(self, reverse_xlim=False):
		metric = 'positioning'

		orfs_to_p1 = self.get_orfs_to_chereji_nuc_mapping()

		from src.config import load_default_chrom_configs
		config1, _ = load_default_chrom_configs()

		plt.figure(figsize=(6, 5))

		for i in range(10):
			current_decile = self.decile_groups['plus_one'][metric]\
				[i].nuc_id.values
			current_decile_orfs = orfs_to_p1.loc[current_decile].ORF
			current_decile_ptrs = self.expression_processor.all_transcripts_ptrs.loc[
				current_decile_orfs].ptr.values

			box_color = plt.cm.Reds(0.5)

			plt.boxplot(current_decile_ptrs, positions=[i+1], vert=True,
					   showfliers=False,
					   widths=0.3,
					   patch_artist=True,  # Enable filling of boxes
					   showmeans=True,  # Show mean line
					   boxprops=dict(facecolor=box_color, edgecolor=box_color),
					   whiskerprops=dict(color=box_color),
					   capprops=dict(color=box_color),
					   medianprops=dict(color='white'),
					   meanprops=dict(marker='D', markerfacecolor='black', 
					   	markeredgecolor='none', markersize=5),
					   flierprops=dict(markersize=1, marker='o', markerfacecolor='black'))

		if reverse_xlim:
			plt.xlim(10.5, 0.5)
		else:
			plt.xlim(0.5, 10.5)

		plt.ylabel("Average expression PTR", fontsize=14)
		plt.xlabel("Positioning PTR deciles", fontsize=14)
		plt.suptitle("+1 nucleosome position cyclicity \nvs expression cyclicity", 
				  fontweight='demi', fontsize=16)
		plt.tight_layout()
		# plt.subplots_adjust(top=0.9)

		xticks = np.arange(1, 11)
		xtick_labels = [f"{x}" for x in xticks]
		xtick_labels[0] = "1\nStable"
		xtick_labels[-1] = "10\nCyclic"
		plt.xticks(xticks, xtick_labels)
		save_figure_for_paper(f"{self.save_dir}/position_expression_deciles.png")	


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
			ptr_df = ptr_df.sort_values(sort_column, ascending=self.sort_deciles_by_increasing_ptr)
			
			# Get Brogaard nucleosome IDs ranked by PTR (ptr_df is already sorted)
			brogaard_nuc_ids_sorted_by_ptr = ptr_df.index
			
			# Map Brogaard nucleosome IDs to Weiner nucleosome IDs for histone data linkage
			# Assuming brogaard_integrated_data has a column like 'weiner_nuc_id' or similar
			brogaard_to_weiner_mapping = self.brogaard_integrated_data.set_index(self.brogaard_integrated_data.index)
			brogaard_to_weiner_mapping = brogaard_to_weiner_mapping[~brogaard_to_weiner_mapping.filter_brogaard]
			
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

	def plot_all_metrics_enrichment_deciles(self, nucleosome='plus_one'):
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

		if nucleosome == 'plus_one':
			filter_column = 'filter_p1'
		elif nucleosome == 'minus_one':
			filter_column = 'filter_m1'
		elif nucleosome == 'brogaard':
			filter_column = 'filter_brogaard'

		n = len(self.chereji_integrated_data[~self.chereji_integrated_data[filter_column]]) if not nucleosome == 'brogaard' \
			else len(self.brogaard_integrated_data[~self.brogaard_integrated_data[filter_column]])

		nuc_type = name_mapping[nucleosome]

		if nucleosome == 'plus_one':
			# Create plot showing gradient across deciles
			fig = self.histone_mod_plotter.plot_decile_enrichment_gradient(
				metrics=['positioning'],
				show_metric_label=False,
				title=f"Histone modifications by\n{nuc_type} nucleosome position cyclicity, n={n}"
			)
			save_figure_for_paper(f"{self.save_dir}/{nucleosome}_positioning_decile_enrichment.png")

			# Create plot showing gradient across deciles
			fig = self.histone_mod_plotter.plot_decile_enrichment_gradient(
				n_deciles=self.n_deciles,
				metrics=['occupancy', 'entropy'],
				title=f"Histone modifications by {nuc_type} nucleosome cyclicity, n={n}"
			)
			save_figure_for_paper(f"{self.save_dir}/{nucleosome}_occupancy_entropy_decile_enrichment.png")
		else:

			# Create plot showing gradient across deciles
			fig = self.histone_mod_plotter.plot_decile_enrichment_gradient(
				n_deciles=self.n_deciles,
				title=f"Histone modifications by {nuc_type} nucleosome cyclicity, n={n}"
			)
			save_figure_for_paper(f"{self.save_dir}/{nuc_type}_occupancy_entropy_decile_enrichment.png")

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

		plt.figure(figsize=(9, 3.5))

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
						alpha=0.1, s=2)
			plt.xlabel(measure.title() + " PTR")

			# plot regression
			plot_data = joined_chromatin_tx_ptrs.dropna()
			x = plot_data.chromatin_ptr
			y = plot_data.expression_ptr

			# Calculate linear regression
			coefficients = np.polyfit(x, y, 1)
			poly = np.poly1d(coefficients)

			# Create regression line
			x_line = np.linspace(x.min(), x.max(), 100)
			y_line = poly(x_line)

			# Plot regression line
			plt.plot(x_line, y_line, linewidth=0.5, c='black',
				label=f'Best fit, y = {coefficients[0]:.3f}x + {coefficients[1]:.3f}')
			plt.legend()

			# Plot vertical lines for decile boundaries
			chromatin_values = joined_chromatin_tx_ptrs.chromatin_ptr.dropna().values.flatten()
			decile_size = len(chromatin_values) // self.n_deciles
			for j in range(1, self.n_deciles):
				q_threshold = np.quantile(chromatin_values, j / self.n_deciles)
				plt.axvline(q_threshold, c='black', lw=0.35, ls='dotted', alpha=1, zorder=0)

			if i == 0: plt.ylabel('Expression PTR')
			else: plt.yticks([])

			plt.xlim(*ptr_formatting[measure]['xlims'])
			plt.title(measure.title(), fontsize=13)
			plt.ylim(0.99, 4)
			
		plt.suptitle(f"+1 nucleosome vs expression cyclicity, n={len(joined_chromatin_tx_ptrs)}", fontweight='demi', 
					fontsize=18)
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

	def _retrieve_orfs_for_group(self, metric=None, decile_num=None, nucleosome='plus_one'):
		"""Retrieve ORFs for a specific decile."""
		nucs, orfs = self._get_nucs_and_orfs_from_group(nucleosome, metric, decile_num)
		return orfs

		
	def plot_occupancy_for_metric(self, metric):

		occupancy_values = self.nucleosome_loader\
			.plus_one_chromatin_metrics['occupancy']
		occupancy_values.columns = occupancy_values.columns.astype(int)
			
		from src.config import load_default_chrom_configs
		config1, _ = load_default_chrom_configs()

		# Retrieve deciles 
		for decile_index in range(10):
			decile_nuc_ids, decile_orfs = self._get_nucs_and_orfs_from_group(
				'plus_one',  metric, decile_index)
			
			decile_occupancy_values = occupancy_values.loc[decile_orfs]
			decile_average_occupancy = decile_occupancy_values[config1.t_indices()].mean(axis=1)
		
			plt.boxplot(decile_average_occupancy.dropna(), positions=[decile_index+1], 
						vert=False, showfliers=False)
			plt.title(f"{metric.title()}")

		plt.ylim(10.5, 0.5)
		plt.xlim(-0.5, 10)
		
	def plot_metrics_by_occupancy_deciles(self):
		plt.figure(figsize=(9, 4))
		plt.subplot(1, 3, 1)
		self.plot_occupancy_for_metric('positioning')
		plt.ylabel("PTR Decile")

		plt.subplot(1, 3, 2)
		self.plot_occupancy_for_metric('occupancy')

		plt.subplot(1, 3, 3)
		self.plot_occupancy_for_metric('entropy')
		plt.suptitle(f"Average nucleosome occupancy", fontweight='demi', fontsize=16)
		plt.tight_layout()

		# Nothing unusual about the occupancy in the 10th decile, at least on average.
		# Positioning also does not follow a strict monotonic relationship with PTR.


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
			layout_images_horizontally, add_panel_labels_to_images, place_image_below

		# Create compositor with wider dimensions for horizontal layout
		compositor = FigureCompositor(1024, 1060, debug_mode=True)

		image_paths = [
			f'{self.save_dir}/plus_one_tss_comparison.png',
			f'{self.save_dir}/plus_one_ptr_histograms.png',
			f'{self.save_dir}/tx_nucleosome_ptrs.png',
			f'{self.save_dir}/plus_one_occupancy_entropy_decile_enrichment.png',
			f'{self.save_dir}/plus_one_heatmap_colorbar.png'
		]

		placed_images = layout_images_vertically(
			compositor,
			image_paths[:1],
			between_padding=30,
			margin=(30, 30),
			widths=[420],
			image_keys=['tsses'], #'histograms']
		)

		placed_images = layout_images_vertically(
			compositor,
			image_paths[1:3],
			between_padding=30,
			margin=(30, 30),
			offsets=[(440, 0), (440, 0)],
			widths=[530, 530],
			image_keys=['histograms', 'ptrs'],
		)

		place_image_below(compositor, image_paths[3], 'tsses',
			vertical_padding=220, width=900, new_key='deciles_modifications')

		# Add panel labels
		add_panel_labels_to_images(
			compositor, 
			compositor.placed_images,
			font_size=32,
			offset=(-15, 20)
		)

		compositor.place_image(image_paths[4], 940, 520, 
			width=70, name='colorbar')

		# Save the composite figure
		compositor.save(f'{self.figures_dir}/Supplemental9_Nucleosome_metrics.png')

	def layout_combined_origins_nucleosomes(self):

		from pipeline.figure_composer import FigureCompositor
		from pipeline.figure_composer_helpers import layout_images_vertically, \
			add_panel_labels_to_images, layout_images_horizontally

		# Create compositor with wider dimensions for horizontal layout
		compositor = FigureCompositor(1024, 740, debug_mode=True)

		nucleosomes_fig_dir = self.save_dir
		origins_fig_dir = f"{self.output_dir}/figure_origins"

		# ---------- Nucleosomes story -----------

		nucleosomes_image_paths = [
			f'{nucleosomes_fig_dir}/position_expression_deciles.png',
			f'{nucleosomes_fig_dir}/plus_one_positioning_decile_enrichment.png',
			f'{nucleosomes_fig_dir}/plus_one_heatmap_colorbar.png'
		]

		compositor.place_image(nucleosomes_image_paths[2], 930, 50, 
			width=60, name='colorbar')

		placed_images = layout_images_horizontally(
			compositor,
			nucleosomes_image_paths[:2],
			between_padding=32,
			offsets=[(-34, 20), (-34, 20)],
			width_proportions=[0.34, 0.66],
			margin=(70, 0),
			y_position=0,
			image_keys=['position_deciles', 'position_histones']
		)

		# ---------- Origins ------------------

		image_paths = [
			f'{origins_fig_dir}/origin_replication_times.png',
			f'{origins_fig_dir}/early_origin_locus.png',
			f'{origins_fig_dir}/inferred_firing_entropies.png',

			f'{origins_fig_dir}/early_origin_enrichments.png',
			f'{origins_fig_dir}/termination_site_entropy.png',
		]

		y_position = placed_images['position_deciles']['logical_position'][1]+\
			placed_images['position_deciles']['logical_size'][1]+30

		placed_images = layout_images_horizontally(
			compositor,
			image_paths[:3],
			between_padding=32,
			offsets=[(0, 0), (0, 0), (0, 0)],
			width_proportions=[0.37, 0.57, 0.215],
			margin=(33, 30),
			y_position=y_position,
			image_keys=['origins_repl', 'early_locus', 'inferred_entropies']  # Custom keys
		)

		origins_img = placed_images['origins_repl']
		inferred_ent_img = placed_images['inferred_entropies']

		e_img = compositor.place_image(
			image_paths[3], origins_img['logical_position'][0],
			origins_img['logical_position'][1]+origins_img['logical_size'][1]+30,
			width=origins_img['logical_size'][0], name='enrichment'
		)

		y_position = inferred_ent_img['logical_position'][1]+inferred_ent_img['logical_size'][1]+20
		t_img = compositor.place_image(
			image_paths[4], inferred_ent_img['logical_position'][0],
			y_position,
			width=inferred_ent_img['logical_size'][0], name='termination_entropies'
		)

		# ------- Labels ----------

		offsets = [(-20, 20)]*len(compositor.placed_images)
		#offsets[-1] = (-10, 20)

		# Add panel labels
		add_panel_labels_to_images(
			compositor, 
			compositor.placed_images,
			#"acdbe fg",
			" abcefdg",
			font_size=30,
			offsets=offsets
		)

		# Save the composite figure
		compositor.save(f'{self.figures_dir}/Figure6_Nucleosomes_Origins.png')

