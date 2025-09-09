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


class FigureNucleosomes:
	"""
	Analyzes cell cycle nucleosome dynamics and histone modification enrichment patterns.
	
	Integrates Chereji nucleosome positioning data with Weiner histone modification data
	to identify cyclicity patterns and statistical enrichments.
	"""
	
	def __init__(self, output_dir="output/draft4_run/", window_size=160, 
				 subset_qval=0.9, random_seed=123):
		"""
		Initialize the analyzer with configuration parameters.
		"""
		# Configuration parameters
		self.output_dir = output_dir
		self.save_dir = f"{self.output_dir}/nucleosome_metrics"
		self.figures_dir = f'{self.output_dir}/Figures'
		mkdir_safe(self.save_dir)

		self.window_size = window_size
		self.subset_qval = subset_qval
		self.random_seed = random_seed
		
		# Core data processing objects
		self.histones_nucleosomes_dataset = HistonesNucleosomesDataset()
		self.nucleosome_loader = NucleosomeDataLoader(output_dir=output_dir)
		self.expression_processor = None
		
		# Data storage members
		self.integrated_data = None
		self.weiner_histones = None
		self.histone_cols = None

		# Chromatin metrics storage
		self.plus_one_chromatin_metrics = None
		self.minus_one_chromatin_metrics = None

		# PTRs storage
		self.plus_one_ptrs = None
		self.minus_one_ptrs = None
		
		# Cyclicity measures
		self.histones_sorted_by_ptr = None
		
		# Analysis groups
		self.high_cyclicity_group = None
		self.low_cyclicity_group = None
		self.random_group = None
		self.genomic_background = None
		
		# Results storage
		self.enrichment_results = None
	
	def load_and_integrate_data(self):
		"""Load Chereji nucleosome and Weiner histone data and integrate them."""
		print("Loading and integrating nucleosome and histone data...")
		
		# Load all datasets
		self.histones_nucleosomes_dataset.load_all()
		
		# Store integrated data
		self.integrated_data = self.histones_nucleosomes_dataset.integrated_data
		self.weiner_histones = self.histones_nucleosomes_dataset.weiner_histones
		
		# Extract histone modification column names
		self.histone_cols = [col for col in self.weiner_histones.columns 
							if col not in ['nuc_id', 'chr', 'start', 'end']]
		
		print(f"Loaded {len(self.integrated_data)} genes with nucleosome data")
		print(f"Loaded {len(self.weiner_histones)} nucleosomes with histone modifications")
		print(f"Found {len(self.histone_cols)} histone modifications")

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
				gene_dataset=self.integrated_data,
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

		# Calculate PTRs for +1 nucleosome metrics
		entropy_minus1 = self.minus_one_chromatin_metrics['entropy']
		occupancy_minus1 = self.minus_one_chromatin_metrics['occupancy']
		positioning_minus1 = self.minus_one_chromatin_metrics['positioning']
		
		# Compute PTRs
		p1_entropy_ptrs = compute_quantile_ptr_2d(entropy_plus1)
		p1_occupancy_ptrs = compute_quantile_ptr_2d(occupancy_plus1)

		# Positioning is defined differnece from the mean, so
		# offset for positioning, ensures non-negative changes
		p1_positioning_ptrs = compute_quantile_ptr_2d(positioning_plus1 + 100)  

		# Compute PTRs
		m1_entropy_ptrs = compute_quantile_ptr_2d(entropy_minus1)
		m1_occupancy_ptrs = compute_quantile_ptr_2d(occupancy_minus1)
		m1_positioning_ptrs = compute_quantile_ptr_2d(positioning_minus1 + 100)  # Offset for positioning
		
		# Create sorted DataFrames
		def _create_ptr_df(ptrs_data, name):
			index = occupancy_plus1.index
			key = f'{name}_ptr'
			ptrs_df = pd.DataFrame(
					ptrs_data, index=index, columns=[key]
				).sort_values(key, ascending=False)
			return ptrs_df

		self.plus_one_ptrs = {
			"entropy": _create_ptr_df(p1_entropy_ptrs, 'entropy'),
			"occupancy": _create_ptr_df(p1_occupancy_ptrs, 'occupancy'),
			"positioning": _create_ptr_df(p1_positioning_ptrs, 'positioning'),
		}

		self.minus_one_ptrs = {
			"entropy": _create_ptr_df(m1_entropy_ptrs, 'entropy'),
			"occupancy": _create_ptr_df(m1_occupancy_ptrs, 'occupancy'),
			"positioning": _create_ptr_df(m1_positioning_ptrs, 'positioning'),
		}
		
		print(f"Calculated cyclicity measures for {len(self.plus_one_chromatin_metrics['entropy'])} nucleosomes")
	
	def create_cyclicity_groups(self, metric='occupancy'):
		"""
		Create high/low cyclicity groups based on specified metric.
		
		Args:
			metric (str): Metric to use for grouping ('occupancy', 'entropy', 'positioning')
		"""
		print(f"Creating cyclicity groups based on {metric} metric...")
		
		# Select the appropriate PTR DataFrame
		if metric == 'occupancy':
			ptr_df = self.plus_one_ptrs['occupancy']
			sort_column = 'occupancy_ptr'
		elif metric == 'entropy':
			ptr_df = self.plus_one_ptrs['entropy']
			sort_column = 'entropy_ptr'
		elif metric == 'positioning':
			ptr_df = self.plus_one_ptrs['positioning']
			sort_column = 'positioning_ptr'
		else:
			raise ValueError(f"Unknown metric: {metric}. Use 'occupancy', 'entropy', or 'positioning'")
		
		# Link to histone modification data
		orf_to_p1_nucleosomes = self.integrated_data[['matched_nuc_id_p1']]
		nuc_id_sorted_by_ptr = orf_to_p1_nucleosomes.loc[ptr_df.index]
		
		self.histones_sorted_by_ptr = self.weiner_histones.loc[
			nuc_id_sorted_by_ptr.dropna().matched_nuc_id_p1
		]
		
		# Create cyclicity groups
		df = self.histones_sorted_by_ptr.reset_index()
		df.columns = ['nuc_id'] + list(df.columns[1:])
		
		# Define groups
		# subset size defined by the quantile threshold
		self.subset_n = int(len(df)*(1-self.subset_qval))

		self.high_cyclicity_group = df.iloc[:self.subset_n]  # Top (highest cyclicity)
		self.low_cyclicity_group = df.iloc[-self.subset_n:]  # Bottom (lowest cyclicity)
		
		# Create random control group
		np.random.seed(self.random_seed)
		self.random_group = df.iloc[np.random.choice(len(df), size=self.subset_n)]
		
		# Define genomic background (excluding test groups)
		self.genomic_background = df
		
		print(f"High cyclicity: {len(self.high_cyclicity_group)} nucleosomes")
		print(f"Random control: {len(self.random_group)} nucleosomes")
		print(f"Genomic background: {len(self.genomic_background)} nucleosomes")

	def perform_go_on_nucleosome_groups(self):
		from src.gene_ontology import GeneOntology

		gene_ontology = GeneOntology()
		from src.sgd import read_nondubious_genes_dataset

		all_go_results_df = pd.DataFrame()
		for group in ['high', 'low']:
			for metric in ['positioning', 'occupancy', 'entropy']:
				genes = read_nondubious_genes_dataset()

				selected_orfs = self._retrieve_orfs_for_group(
					metric, group)
				selected_gene_names = genes.loc[selected_orfs]['gene'].values
				gene_ontology.run_go(selected_gene_names)
				results = gene_ontology.results_df.copy()
				results['metric'] = metric
				results['group'] = group

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

		res = self.create_table_for_go_group('low')
		save_path = f'{self.save_dir}/low_go_group.txt'
		with open(save_path, 'w') as f:
			f.write(res)
		print(f"Wrote to: ", save_path)

		res = self.create_table_for_go_group('high')
		save_path = f'{self.save_dir}/high_go_group.txt'
		with open(save_path, 'w') as f:
			f.write(res)
		print(f"Wrote to: ", save_path)
		
	
	def run_enrichment_analysis(self):
		"""Perform statistical enrichment analysis comparing cyclicity groups to background."""
		print("Running enrichment analysis...")
		
		# Run enrichment tests for each group
		high_cyclicity_results = self._test_cyclicity_enrichment_zscore(
			self.high_cyclicity_group, self.genomic_background, "High_Cyclicity"
		)
		
		low_cyclicity_results = self._test_cyclicity_enrichment_zscore(
			self.low_cyclicity_group, self.genomic_background, "Low_Cyclicity"
		)
		
		random_results = self._test_cyclicity_enrichment_zscore(
			self.random_group, self.genomic_background, "Random"
		)
		
		# Apply FDR correction and format results
		self.enrichment_results = {
			'high': self._perform_fdr_correction(high_cyclicity_results),
			'low': self._perform_fdr_correction(low_cyclicity_results),
			'random': self._perform_fdr_correction(random_results)
		}
		
		print("Enrichment analysis completed")
	
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
		self.histones_sorted_by_ptr.to_csv(histone_export_file)
		print(f"Exported histone data to: {histone_export_file}")
		
		# Export enrichment analysis results
		for group_name, results in self.enrichment_results.items():
			results_file = f'{self.save_dir}/{output_prefix}_enrichment_{group_name}.csv'
			results.to_csv(results_file, index=False)
			print(f"Exported {group_name} enrichment results to: {results_file}")
		
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
		plt.figure(figsize=(8, 8))
		
		# Full dataset heatmap
		plt.subplot(1, 2, 1)
		plt.imshow(self.histones_sorted_by_ptr, aspect='auto', cmap='PRGn',
				  interpolation='none', vmin=-vmax, vmax=vmax)
		plt.axhline(self.subset_n, c='red', linewidth=2)
		plt.title(f"All +1 Nucleosomes\n(sorted by {metric} cyclicity)")
		plt.ylabel("Nucleosomes (ranked by cyclicity)")
		
		# High cyclicity heatmap
		plt.subplot(1, 2, 2)
		plt.imshow(self.histones_sorted_by_ptr.head(self.subset_n), aspect='auto', cmap='PRGn',
				  interpolation='none', vmin=-vmax, vmax=vmax)
		plt.title(f"Top {self.subset_n} High Cyclicity")
		
		plt.tight_layout()
		
		# Save figure
		output_prefix = f"{metric}_cyclicity"
		fig_file = f'{self.save_dir}/{output_prefix}_heatmaps.png'
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
			plt.axvline(self.subset_n, c='red', linewidth=2, label=f'Top {self.subset_n} '\
				f'({self.subset_qval*100:.0f} percentile)')

			plt.xlabel('Nucleosome Rank')
			plt.ylabel(f'{ptr_metric.title()} PTR')
			plt.title(f'{ptr_metric.title()} Cyclicity Distribution')
		
		plt.tight_layout()
		
		# Save distribution plot
		dist_fig_file = f'{self.save_dir}/nucleosome_cyclicity_distributions.png'
		save_figure_for_paper(dist_fig_file)
		print(f"Saved distribution plot to: {dist_fig_file}")
		plt.show()
	
	def get_summary_stats(self):
		"""Return summary statistics for the loaded datasets."""
		if self.histones_nucleosomes_dataset.integrated_data is not None:
			return self.histones_nucleosomes_dataset.get_summary_stats()
		else:
			return "Data not loaded. Call load_and_integrate_data() first."
	

	def run_full_analysis(self, force_recompute=False):
		"""
		Execute the complete analysis workflow.
		
		Args:
			metric (str): Metric to use for cyclicity analysis ('occupancy', 'entropy', 'positioning')
		"""
		self.setup_processors()

		print(f"Starting full cell cycle nucleosome analysis...")
		print("=" * 60)
		
		# Execute complete pipeline
		self.load_and_integrate_data()
		self.compute_nucleosome_metrics(force_recompute=force_recompute)
		self.calculate_cyclicity_measures()

		metrics = ['occupancy', 'entropy', 'positioning']
		self.all_metrics_enrichment_results = {}
		self.all_cyclicity_groups = {}

		# Intitialize cyclicity groups stored for each metric
		for metric in metrics:
			self.all_cyclicity_groups[metric] = {}

		for metric in metrics:
			enrichment_results = self.run_analysis_for_metric(metric)
			self.all_metrics_enrichment_results[metric] = enrichment_results
			self.all_cyclicity_groups[metric]['high'] = self.high_cyclicity_group
			self.all_cyclicity_groups[metric]['low'] = self.low_cyclicity_group
			self.all_cyclicity_groups[metric]['random'] = self.random_group

		# Create combined visualizations
		self.plot_cyclicity_p1_histograms()
		save_figure_for_paper(f"{self.save_dir}/plus_one_ptr_histograms.png")

		self.plot_all_metrics_enrichment(group='high', plot_key='difference')
		save_figure_for_paper(f"{self.save_dir}/plus_one_high_enrichment.png")

		self.plot_all_metrics_enrichment(group='low', plot_key='difference')
		save_figure_for_paper(f"{self.save_dir}/plus_one_low_enrichment.png")

		self.plot_all_metrics_enrichment(group='random', plot_key='difference')
		save_figure_for_paper(f"{self.save_dir}/plus_one_random_enrichment.png")

		self.histone_mod_plotter.plot_colorbar()
		save_figure_for_paper(f"{self.save_dir}/plus_one_heatmap_colorbar.png")

		self.plot_nucleosome_expression_ptrs()
		save_figure_for_paper(f"{self.save_dir}/tx_nucleosome_ptrs.png")

		self.plot_p1_tss_agreement()
		save_figure_for_paper(f"{self.save_dir}/plus_one_tss_comparison.png")

		# Create venn diagrams
		self.plot_venn_diagrams()

		# Create heatmap of intersection cyclers sets
		self.plot_heatmap_intersections()
		save_figure_for_paper(f"{self.save_dir}/cycler_non_cyclers_adjacency_heatmap.png")

		# Create LaTeX tables
		self.create_latex_go_tables()


	def run_analysis_for_metric(self, metric):

		print(f"Running analysis for metric {metric}")
		self.create_cyclicity_groups(metric=metric)
		self.run_enrichment_analysis()
		self.export_results(metric=metric)
		
		print("Analysis completed!")
		print(f"Results exported with prefix: {metric}_cyclicity")
		
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


	def plot_all_metrics_enrichment(self, group='high', plot_key='p_value_fdr'):
		from src.histone_group_plotter import HistoneModificationGroupedPlotter

		# Initialize the plotter with your enrichment results
		self.histone_mod_plotter = HistoneModificationGroupedPlotter(
			enrichment_results=self.all_metrics_enrichment_results,
			subset_n=self.subset_n  # or whatever your subset size variable is called
		)

		# Create the main grouped plot showing differences for high cycling nucleosomes
		fig = self.histone_mod_plotter.plot_grouped_enrichment(
			group=group, 
			plot_key=plot_key
		)

	def plot_p1_tss_agreement(self):
		from src.transcripts_dataset import load_transcripts_sets
		genes, _ = load_transcripts_sets(self.output_dir)
		joined_tss_p1 = genes[['TSS']].join(self.integrated_data[['+1 nucleosome']], how='inner').dropna()
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

		formatting = {
			'positioning': {
				'xlims': (0.99, 1.5)
			},
			'occupancy': {
				'xlims': (0.95, 3)
			},
			'entropy': {
				'xlims': (0.99, 1.45)
			},
		}
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

			# Plot vertical lines for the threshold values
			chromatin_values = joined_chromatin_tx_ptrs.chromatin_ptr.dropna().values.flatten()
			q_thresholds = np.quantile(chromatin_values, 
				q=[self.subset_qval, ((1-self.subset_qval))])

			for q_threshold in q_thresholds:
				plt.axvline(q_threshold, c='red', lw=1, alpha=0.75)

			if i == 0: plt.ylabel('Expression PTR')
			else: plt.yticks([])

			plt.xlim(*formatting[measure]['xlims'])
			plt.title(measure.title())
			
		plt.suptitle(f"+1 nucleosome vs expression cyclicity, n={len(joined_chromatin_tx_ptrs)}", fontweight='demi', 
					fontsize=16)
		plt.tight_layout()


	def plot_cyclicity_p1_histograms(self):
		def _plot_hist_ptrs(ptrs_data, color, bins=30):
			q_threshold = np.quantile(ptrs_data.dropna().values.flatten(), 
				self.subset_qval)
			plt.hist(ptrs_data, color=color, bins=bins)
			plt.axvline(q_threshold, c='red', lw=1, alpha=0.75)

			q_threshold_lower = np.quantile(ptrs_data.dropna().values.flatten(), 
				(1-self.subset_qval))
			plt.axvline(q_threshold_lower, c='red', lw=1, alpha=0.75)

		colors = [
			plt.cm.Reds(0.5),
			plt.cm.Blues(0.5),
			plt.cm.Purples(0.5)]

		plt.figure(figsize=(9, 3))
		plt.subplot(1, 3, 1)
		_plot_hist_ptrs(self.plus_one_ptrs['positioning'], 
					  color=colors[0], bins=np.linspace(1, 1.4, 30))
		plt.title("Positioning")
		plt.xlabel("Peak-to-Trough Ratio (PTR)")

		plt.subplot(1, 3, 2)
		_plot_hist_ptrs(self.plus_one_ptrs['occupancy'], color=colors[1], 
			bins=np.linspace(1, 2.5, 30))
		plt.title("Occupancy")
		plt.xlabel("Peak-to-Trough Ratio (PTR)")
		plt.xlim(0.9, 2.5)

		plt.subplot(1, 3, 3)
		_plot_hist_ptrs(self.plus_one_ptrs['entropy'], color=colors[2], 
			bins=np.linspace(1, 1.4, 30))
		plt.title("Entropy")
		plt.xlabel("Peak-to-Trough Ratio (PTR)")
		plt.xlim(0.99, 1.35)

		plt.suptitle("Cyclicity of +1 Chereji, (2018) nucleosomes,\n"
					f"n={len(self.plus_one_ptrs['entropy'])}, {self.subset_n} cycling ({self.subset_qval*100:.0f}th perc.) each",
					fontweight='demi', fontsize=18)
		plt.tight_layout()
		plt.xlim(0.99, 1.35)

	def _retrieve_orfs_for_group(self, metric, group_name):
		orf_p1s = self.integrated_data[['matched_nuc_id_p1']].reset_index()
		orf_p1s = orf_p1s.dropna().set_index('matched_nuc_id_p1')
		orf_p1s.index = orf_p1s.index.astype(int)
		selected_nucs_histones_mods = self.all_cyclicity_groups[metric][group_name].set_index('nuc_id').join(orf_p1s)
		return selected_nucs_histones_mods.ORF.values

	def plot_heatmap_intersections(self):
		from src.heatmap_counts import create_set_adjacency_matrix
		low_occ_nuc_orfs = self._retrieve_orfs_for_group('occupancy', 'low')
		low_ent_nuc_orfs = self._retrieve_orfs_for_group('entropy', 'low')
		low_pos_nuc_orfs = self._retrieve_orfs_for_group('positioning', 'low')

		high_occ_nuc_orfs = self._retrieve_orfs_for_group('occupancy', 'high')
		high_ent_nuc_orfs = self._retrieve_orfs_for_group('entropy', 'high')
		high_pos_nuc_orfs = self._retrieve_orfs_for_group('positioning', 'high')

		# Create adjacency matrix
		result = create_set_adjacency_matrix(
			low_pos_nuc_orfs, low_occ_nuc_orfs, low_ent_nuc_orfs,
			high_pos_nuc_orfs, high_occ_nuc_orfs, high_ent_nuc_orfs,
			labels=['Position\nnon-cyclers', 'Occupancy\nnon-cyclers', 'Entropy\nnon-cyclers', 
					'Position\ncyclers', 'Occupancy\ncyclers', 'Entropy\ncyclers'],
			metric='count',
			title="Cycling and non-cycling nucleosome\nintersections counts"
		)

	def plot_venn_diagrams(self):

		low_occ_nuc_orfs = self._retrieve_orfs_for_group('occupancy', 'low')
		low_ent_nuc_orfs = self._retrieve_orfs_for_group('entropy', 'low')
		low_pos_nuc_orfs = self._retrieve_orfs_for_group('positioning', 'low')

		high_occ_nuc_orfs = self._retrieve_orfs_for_group('occupancy', 'high')
		high_ent_nuc_orfs = self._retrieve_orfs_for_group('entropy', 'high')
		high_pos_nuc_orfs = self._retrieve_orfs_for_group('positioning', 'high')

		# What is the overlap between each cell cycle group?
		# Retrieve the set of ORFs for each category tested
		fig, ax, venn, data = create_three_set_venn(
			low_pos_nuc_orfs, low_ent_nuc_orfs, low_occ_nuc_orfs,
			labels=['Positioning', 'Occupancy', 'Entropy'],
			title="Non-cycling nucleosomes",
			alpha=0.7
		)
		save_figure_for_paper(f"{self.save_dir}/noncyclers_venn.png")

		fig, ax, venn, data = create_three_set_venn(
			high_pos_nuc_orfs, high_ent_nuc_orfs, high_occ_nuc_orfs,
			labels=['Positioning', 'Occupancy', 'Entropy'],
			title="Cycling nucleosomes",
			alpha=0.7
		)
		save_figure_for_paper(f"{self.save_dir}/cyclers_venn.png")

		fig, ax, venn, data = create_three_set_venn(
		high_pos_nuc_orfs, low_occ_nuc_orfs, high_ent_nuc_orfs,
			labels=['Cycling positioning', 'Non-cycling occupancy', 'Cycling entropy'],
			title="Non-cycling occupancy vs cyclers",
			alpha=0.7
		)
		save_figure_for_paper(f"{self.save_dir}/noncycler_occupancy_vs_cyclers.png")

	def layout_panel(self):

		from pipeline.figure_composer import FigureCompositor
		from pipeline.figure_composer_helpers import layout_images_vertically, \
			add_panel_labels_to_images

		# Create compositor with wider dimensions for horizontal layout
		compositor = FigureCompositor(1024, 940, debug_mode=True)

		image_paths = [
			f'{self.save_dir}/plus_one_high_enrichment.png',
			f'{self.save_dir}/plus_one_low_enrichment.png',
			f'{self.save_dir}/plus_one_random_enrichment.png',
			f'{self.save_dir}/plus_one_heatmap_colorbar.png',
		]

		placed_images = layout_images_vertically(
			compositor,
			list(np.array(image_paths)[[0, 1, 2]]),
			heights=[280, 278, 282],
			between_padding=30,
			offsets=[(10, 0), (-5, 0), (0, 0)],
			margin=(30, 30),
			image_keys=['high', 'low', 'random']  # Custom keys
		)

		# Add panel labels
		add_panel_labels_to_images(
			compositor, 
			compositor.placed_images,
			"ABC",
			font_size=40,
			offset=(-15, 0)
		)

		compositor.place_image(image_paths[-1], 970, 56, width=50,
			name='colorbar')

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

			# Venn diagrams and adjacency matrices
			f'{self.save_dir}/cyclers_venn.png',
			f'{self.save_dir}/noncyclers_venn.png',
			f'{self.save_dir}/noncycler_occupancy_vs_cyclers.png',

			# Adjacency heatmap
			f'{self.save_dir}/cycler_non_cyclers_adjacency_heatmap.png',
		]

		placed_images = layout_images_vertically(
			compositor,
			image_paths[:3],
			between_padding=30,
			margin=(30, 30),
			widths=[400, 400, 400],
			image_keys=['tsses', 'histograms', 'ptrs']  # Custom keys
		)


		placed_images = layout_images_horizontally(
			compositor,
			image_paths[3:5],
			between_padding=30,
			margin=(460, 30),
			heights=[300, 300],
			image_keys=['cyclers_venn', 'noncyclers_venn'],  # Custom keys
			available_width=530
		)

		placed_images = layout_images_horizontally(
			compositor,
			image_paths[5:7],
			between_padding=30,
			margin=(460, 320),
			heights=[300, 300],
			image_keys=['noncyc_occ_venn', 'adjacency'],  # Custom keys
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


def create_three_set_venn(set1, set2, set3, 
						 labels=None, 
						 colors=None, 
						 alpha=0.6,
						 title="Three-Set Venn Diagram",
						 figsize=(5, 4),
						 print_counts=False,
						 circle_line_width=1,
						 circle_line_color='black'):
	"""
	Create a three-set Venn diagram from three arrays/lists.
	
	Returns:
	--------
	fig, ax : matplotlib figure and axes objects
	venn_diagram : matplotlib_venn object
	intersection_data : dict containing intersection information
	"""
	from matplotlib_venn import venn3, venn3_circles
	from matplotlib.patches import Circle

	# Convert inputs to sets for set operations
	s1 = set(set1)
	s2 = set(set2)
	s3 = set(set3)
	
	# Set default labels if not provided
	if labels is None:
		labels = ['Set 1', 'Set 2', 'Set 3']
	
	# Set default colors if not provided
	if colors is None:
		colors = [
			plt.cm.Reds(0.3),
			plt.cm.Blues(0.25),
			plt.cm.Purples(0.32),
		]
	
	# Create figure and axis
	fig, ax = plt.subplots(figsize=figsize)
	
	# Create the Venn diagram
	venn_diagram = venn3([s1, s2, s3], set_labels=labels, 
		ax=ax, alpha=alpha, 
		set_colors=colors)

	from src.plot_helpers import blend_colors, blend_three_colors

	# 2-way intersections
	if venn_diagram.get_patch_by_id('110'):  # A ∩ B (not C)
		blended_ab = blend_colors(colors[0], colors[1])
		venn_diagram.get_patch_by_id('110').set_facecolor(blended_ab)
	
	if venn_diagram.get_patch_by_id('101'):  # A ∩ C (not B)
		blended_ac = blend_colors(colors[0], colors[2])
		venn_diagram.get_patch_by_id('101').set_facecolor(blended_ac)
	
	if venn_diagram.get_patch_by_id('011'):  # B ∩ C (not A)
		blended_bc = blend_colors(colors[1], colors[2])
		venn_diagram.get_patch_by_id('011').set_facecolor(blended_bc)
	
	# 3-way intersection
	if venn_diagram.get_patch_by_id('111'):  # A ∩ B ∩ C
		blended_abc = blend_three_colors(colors[0], colors[1], colors[2])
		venn_diagram.get_patch_by_id('111').set_facecolor(blended_abc)

	# Add lines around each circle
	for patch in venn_diagram.patches:
	    patch.set_edgecolor(circle_line_color)
	    patch.set_linewidth(circle_line_width)
	
	# Calculate intersection data
	intersection_data = {
		'set1_only': len(s1 - s2 - s3),
		'set2_only': len(s2 - s1 - s3),
		'set3_only': len(s3 - s1 - s2),
		'set1_and_set2_only': len(s1 & s2 - s3),
		'set1_and_set3_only': len(s1 & s3 - s2),
		'set2_and_set3_only': len(s2 & s3 - s1),
		'all_three': len(s1 & s2 & s3),
		'total_unique': len(s1 | s2 | s3),
		'set1_total': len(s1),
		'set2_total': len(s2),
		'set3_total': len(s3)
	}
	
	# Add title
	plt.title(title, fontsize=16, fontweight='bold', pad=20)
	
	# Print intersection summary if show_counts is True
	if print_counts:
		print("Intersection Summary:")
		print(f"Total unique items: {intersection_data['total_unique']}")
		print(f"{labels[0]} only: {intersection_data['set1_only']}")
		print(f"{labels[1]} only: {intersection_data['set2_only']}")
		print(f"{labels[2]} only: {intersection_data['set3_only']}")
		print(f"{labels[0]} ∩ {labels[1]} only: {intersection_data['set1_and_set2_only']}")
		print(f"{labels[0]} ∩ {labels[2]} only: {intersection_data['set1_and_set3_only']}")
		print(f"{labels[1]} ∩ {labels[2]} only: {intersection_data['set2_and_set3_only']}")
		print(f"All three sets: {intersection_data['all_three']}")
	
	# Adjust layout and show
	plt.tight_layout()
	
	return fig, ax, venn_diagram, intersection_data
