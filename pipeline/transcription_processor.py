import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.config import load_default_chrom_configs
from src.utils import print_fl, mkdir_safe
from src.figure_configs import save_figure_for_paper


class ExpressionAnalysisProcessor:
	"""
	Process deconvolved gene expression data to compute PTR values and identify
	top cycling genes.
	
	This class handles loading deconvolved expression data, computing peak-to-trough
	ratios, filtering to genic transcripts, and identifying highly cycling genes
	based on PTR thresholds.
	"""
	
	def __init__(self, output_dir: str):
		"""
		Initialize the ExpressionAnalysisProcessor.
		
		Parameters
		----------
		output_dir : str
			Directory where output files will be saved and loaded from
		"""
		self.output_dir = output_dir
		self.config1 = None
		self.config2 = None
		self.all_transcripts_set = None
		
		# Data storage
		self.expression_data = None
		self.all_transcripts_ptrs = None
		self.genic_ptrs = None
		self.top_cycling_genes = None
		self.threshold_value = None
	
	def setup_data_loaders(self):
		"""
		Initialize the data loading components.
		
		Sets up:
		- Configuration objects for PTR computation
		- Transcript set for filtering genic vs non-genic transcripts
		"""
		# Load configuration objects
		self.config1, self.config2 = load_default_chrom_configs()
		
		# Load transcript sets
		from src.transcripts_dataset import load_transcripts_sets
		self.all_transcripts_set = load_transcripts_sets(
			output_dir=self.output_dir, combined=True
		)
		
		print_fl(f"Loaded {len(self.all_transcripts_set)} transcripts")
		genic_count = len(self.all_transcripts_set[
			self.all_transcripts_set.transcript_class == 'genic'
		])
		print_fl(f"  {genic_count} genic transcripts")
		print_fl(f"  {len(self.all_transcripts_set) - genic_count} non-genic transcripts")

	def load_deconvolved_expression(self, include_nongenic=True):
		"""
		Load deconvolved gene expression data.
		
		Parameters
		----------
		include_nongenic : bool, optional
			Whether to include non-genic transcripts. Default is True.
			
		Returns
		-------
		pd.DataFrame
			Expression data with transcript names as rows and timepoints as columns
		"""
		from pipeline.expression_analysis import load_deconvolved_gene_expression
		
		print_fl("Loading deconvolved gene expression data...")
		
		self.expression_data = load_deconvolved_gene_expression(
			self.output_dir, include_nongenic=include_nongenic
		)
		
		print_fl(f"Loaded expression data: {self.expression_data.shape[0]} transcripts, "
				f"{self.expression_data.shape[1]} timepoints")
		
		return self.expression_data


	def load_raw_expression(self):
		from src.gene_expression import load_gene_and_nongenic_transcription_data
		self.raw_rep1_expression_data = load_gene_and_nongenic_transcription_data(1)
		self.raw_rep2_expression_data = load_gene_and_nongenic_transcription_data(2)

	
	def compute_expression_ptrs(self, expression_data=None, ptr_lo=0.1, ptr_hi=0.9):
		"""
		Compute peak-to-trough ratios for all transcripts in expression data.
		
		Parameters
		----------
		expression_data : pd.DataFrame, optional
			Expression data with transcripts as rows, timepoints as columns.
			If None, uses self.expression_data
		ptr_lo : float, optional
			Lower quantile for PTR computation. Default is 0.1.
		ptr_hi : float, optional
			Upper quantile for PTR computation. Default is 0.9.
			
		Returns
		-------
		pd.DataFrame
			DataFrame with transcript names as index and 'ptr' column
		"""
		from src.peak_to_trough import compute_ptr_tb
		
		if expression_data is None:
			if self.expression_data is None:
				raise ValueError("No expression data available. Call load_deconvolved_expression() first.")
			expression_data = self.expression_data
		
		if self.config1 is None:
			raise ValueError("Configuration not loaded. Call setup_data_loaders() first.")
		
		print_fl(f"Computing PTR values with quantiles {ptr_lo}-{ptr_hi}...")
		
		# Compute PTR for each transcript (row)
		ptr_values = np.apply_along_axis(
			lambda row: compute_ptr_tb(self.config1, row, lo=ptr_lo, hi=ptr_hi),
			axis=1,
			arr=expression_data
		)
		
		# Create DataFrame with named index
		self.all_transcripts_ptrs = pd.DataFrame(
			ptr_values,
			columns=['ptr'],
			index=expression_data.index
		)
		self.all_transcripts_ptrs.index.name = 'transcript_name'
		
		print_fl(f"Computed PTR values for {len(self.all_transcripts_ptrs)} transcripts")
		print_fl(f"PTR range: {self.all_transcripts_ptrs.ptr.min():.3f} - "
				f"{self.all_transcripts_ptrs.ptr.max():.3f}")
		
		return self.all_transcripts_ptrs
	
	def filter_to_genic_transcripts(self, ptr_data=None):
		"""
		Filter PTR data to include only genic transcripts.
		
		Parameters
		----------
		ptr_data : pd.DataFrame, optional
			PTR data for all transcripts. If None, uses self.all_transcripts_ptrs
			
		Returns
		-------
		pd.DataFrame
			PTR data filtered to genic transcripts only
		"""
		if ptr_data is None:
			if self.all_transcripts_ptrs is None:
				raise ValueError("No PTR data available. Call compute_expression_ptrs() first.")
			ptr_data = self.all_transcripts_ptrs
		
		if self.all_transcripts_set is None:
			raise ValueError("Transcript set not loaded. Call setup_data_loaders() first.")
		
		# Get genic transcript names
		genic_transcripts = self.all_transcripts_set[
			self.all_transcripts_set.transcript_class == 'genic'
		].index
		
		# Filter PTR data to genic transcripts
		intersection_set = list(set(ptr_data.index).intersection(set(genic_transcripts)))
		self.genic_ptrs = ptr_data.loc[intersection_set]
		
		print_fl(f"Filtered to {len(self.genic_ptrs)} genic transcripts")
		print_fl(f"Genic PTR range: {self.genic_ptrs.ptr.min():.3f} - "
				f"{self.genic_ptrs.ptr.max():.3f}")
		
		return self.genic_ptrs
	
	def plot_ptr_histogram(self, ptr_data=None, quantile_cutoff=0.9, 
						  xlim=(0.95, 2), bins=200, figsize=(4, 3)):
		"""
		Plot histogram of PTR distribution with quantile threshold.
		
		Parameters
		----------
		ptr_data : pd.DataFrame, optional
			PTR data to plot. If None, uses self.genic_ptrs
		quantile_cutoff : float, optional
			Quantile for threshold line. Default is 0.9 (90th percentile).
		xlim : tuple, optional
			X-axis limits for plot. Default is (0.95, 2).
		bins : int, optional
			Number of histogram bins. Default is 200.
		figsize : tuple, optional
			Figure size. Default is (4, 2).
			
		Returns
		-------
		tuple
			(fig, ax, threshold_value) - matplotlib figure, axis, and threshold value
		"""
		if ptr_data is None:
			if self.genic_ptrs is None:
				raise ValueError("No genic PTR data available. Call filter_to_genic_transcripts() first.")
			ptr_data = self.genic_ptrs
		
		# Calculate threshold
		threshold_value = np.quantile(ptr_data.ptr, quantile_cutoff)
		
		# Create plot
		fig, ax = plt.subplots(figsize=figsize)
		
		# Plot histogram
		ax.hist(ptr_data.ptr, bins=bins, color=plt.cm.Greens(0.75), alpha=0.75, edgecolor='black')
		ax.set_xlim(xlim)
		
		# Add threshold line
		ax.axvline(threshold_value, color='red', linestyle='--', linewidth=2, 
				  label=f'{quantile_cutoff*100:.0f}th percentile')
		
		# Labels and title
		ax.set_xlabel('Peak-to-Trough Ratio (PTR)')
		ax.set_ylabel('Number of genes')
		ax.set_title(f'Deconvolved gene expression PTR distribution\n'
					f'Cell cycling threshold = {threshold_value:.2f} '
					f'({quantile_cutoff*100:.0f}th percentile)')
		ax.legend()
		plt.tight_layout()
		
		# Count genes above threshold
		n_above_threshold = np.sum(ptr_data.ptr > threshold_value)
		print_fl(f"Number of transcripts in {quantile_cutoff*100:.0f}th percentile: {n_above_threshold}")
		
		return fig, ax, threshold_value
	
	def identify_top_cycling_genes(self, ptr_data=None, quantile_cutoff=0.9):
		"""
		Identify genes in the top quantile for cycling based on PTR values.
		
		Parameters
		----------
		ptr_data : pd.DataFrame, optional
			PTR data for genic transcripts. If None, uses self.genic_ptrs
		quantile_cutoff : float, optional
			Quantile cutoff for top cycling genes. Default is 0.9.
			
		Returns
		-------
		tuple
			(top_cycling_genes_df, threshold_value, n_genes)
			- DataFrame of top cycling genes
			- Threshold PTR value used
			- Number of genes selected
		"""
		if ptr_data is None:
			if self.genic_ptrs is None:
				raise ValueError("No genic PTR data available. Call filter_to_genic_transcripts() first.")
			ptr_data = self.genic_ptrs
		
		# Calculate threshold
		self.threshold_value = np.quantile(ptr_data.ptr, quantile_cutoff)
		
		# Select top cycling genes
		self.top_cycling_genes = ptr_data[ptr_data.ptr > self.threshold_value]
		
		# Sort by PTR value (highest first)
		self.top_cycling_genes = self.top_cycling_genes.sort_values('ptr', ascending=False)
		
		n_genes = len(self.top_cycling_genes)
		
		print_fl(f"Identified {n_genes} top cycling genes")
		print_fl(f"PTR threshold: {self.threshold_value:.3f}")
		print_fl(f"Top cycling PTR range: {self.top_cycling_genes.ptr.min():.3f} - "
				f"{self.top_cycling_genes.ptr.max():.3f}")
		
		return self.top_cycling_genes, self.threshold_value, n_genes
	
	def run_full_analysis(self, quantile_cutoff=0.9, 
						 ptr_lo=0.1, ptr_hi=0.9, plot=True):
		"""
		Run the complete expression analysis workflow.
		
		Parameters
		----------
		include_nongenic : bool, optional
			Whether to include non-genic transcripts in initial loading. Default is True.
		quantile_cutoff : float, optional
			Quantile cutoff for identifying top cycling genes. Default is 0.9.
		ptr_lo : float, optional
			Lower quantile for PTR computation. Default is 0.1.
		ptr_hi : float, optional
			Upper quantile for PTR computation. Default is 0.9.
		plot : bool, optional
			Whether to create histogram plot. Default is True.
			
		Returns
		-------
		dict
			Dictionary containing:
			- 'expression_data': Raw expression data
			- 'all_transcripts_ptrs': PTR values for all transcripts
			- 'genic_ptrs': PTR values for genic transcripts only
			- 'top_cycling_genes': Top cycling genes DataFrame
			- 'threshold': PTR threshold value
			- 'n_top_genes': Number of top cycling genes
		"""
		print_fl("Starting full expression analysis workflow...")
		
		# Step 1: Setup
		self.setup_data_loaders()
		
		# Step 2: Load expression data
		self.load_deconvolved_expression(include_nongenic=True)
		self.load_raw_expression()
		
		# Step 3: Compute PTRs
		self.compute_expression_ptrs(ptr_lo=ptr_lo, ptr_hi=ptr_hi)
		
		# Step 4: Filter to genic transcripts
		self.filter_to_genic_transcripts()
		
		# Step 5: Identify top cycling genes
		top_genes, threshold, n_genes = self.identify_top_cycling_genes(
			quantile_cutoff=quantile_cutoff
		)
		
		# Compile results
		results = {
			'expression_data': self.expression_data,
			'all_transcripts_ptrs': self.all_transcripts_ptrs,
			'genic_ptrs': self.genic_ptrs,
			'top_cycling_genes': self.top_cycling_genes,
			'threshold': self.threshold_value,
			'n_top_genes': len(self.top_cycling_genes),
		}
		
		print_fl("Full analysis complete!")
		print_fl(f"Summary: {results['n_top_genes']} top cycling genes identified "
				f"from {len(self.genic_ptrs)} genic transcripts")
		
		return results
	
	def create_summary_plots(self, results_dict=None, save_plots=False):
		"""
		Create summary plots for the expression analysis.
		
		Parameters
		----------
		results_dict : dict, optional
			Results dictionary from run_full_analysis(). If None, uses current state.
		save_plots : bool, optional
			Whether to save plots to disk. Default is False.
			
		Returns
		-------
		dict
			Dictionary of matplotlib figure objects
		"""
		if results_dict is None:
			# Use current state
			if self.genic_ptrs is None:
				raise ValueError("No analysis results available. Run run_full_analysis() first.")
			
			results_dict = {
				'genic_ptrs': self.genic_ptrs,
				'top_cycling_genes': self.top_cycling_genes,
				'threshold': self.threshold_value,
				'n_top_genes': len(self.top_cycling_genes) if self.top_cycling_genes is not None else 0
			}
		
		figures = {}
		
		# Create histogram plot
		fig, ax, threshold = self.plot_ptr_histogram(
			ptr_data=results_dict['genic_ptrs'],
			quantile_cutoff=0.9
		)
		figures['ptr_histogram'] = fig
		
		if save_plots:
			save_dir = f"{self.output_dir}/expression_analysis/figures"
			mkdir_safe(save_dir)
			save_figure_for_paper(f"{save_dir}/ptr_histogram.png")
			print_fl(f"Saved histogram to {save_dir}/ptr_histogram.png")
		
		return figures
