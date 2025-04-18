import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

class ChromatinExpressionIntersectionAnalysis:
	"""
	A class for analyzing and finding optimal thresholds for cycling genes,
	comparing gene expression and promoter occupancy PTR values.
	"""
	
	def __init__(self, gene_expression_PTR, promoter_occupancy_PTR, gene_names=None):
		"""
		Initialize the analyzer with gene expression and promoter occupancy data.
		
		Parameters:
		-----------
		gene_expression_PTR : numpy array or pandas Series
			PTR values for gene expression
		promoter_occupancy_PTR : numpy array or pandas Series
			PTR values for promoter occupancy
		gene_names : list or pandas Index, optional
			Names of genes corresponding to the PTR values
		"""
		# Convert pandas Series to numpy arrays if needed
		self.gene_names = None
		if isinstance(gene_expression_PTR, pd.Series):
			if gene_names is None:
				self.gene_names = gene_expression_PTR.index.tolist()
			self.gene_expression_PTR = gene_expression_PTR.values
		else:
			self.gene_expression_PTR = gene_expression_PTR
			
		if isinstance(promoter_occupancy_PTR, pd.Series):
			if gene_names is None and self.gene_names is None:
				self.gene_names = promoter_occupancy_PTR.index.tolist()
			self.promoter_occupancy_PTR = promoter_occupancy_PTR.values
		else:
			self.promoter_occupancy_PTR = promoter_occupancy_PTR
		
		# Use passed gene_names if provided
		if gene_names is not None:
			self.gene_names = gene_names
			
		# If no gene names are available, create placeholder names
		if self.gene_names is None:
			self.gene_names = [f"Gene_{i}" for i in range(len(self.gene_expression_PTR))]
			
		# Check that we have the correct number of gene names
		if len(self.gene_names) != len(self.gene_expression_PTR):
			raise ValueError("Number of gene names must match the number of gene expression values")
			
		self.total_genes = len(self.gene_expression_PTR)
		self.results_df = None
		self.cycling_expr_genes = None
		self.expr_cycling_count = None
		self.threshold_gene_tracking = None
		
	def set_expression_threshold(self, gene_expr_percentile=0.95):
		"""
		Set the threshold for cycling expression genes.
		
		Parameters:
		-----------
		gene_expr_percentile : float
			Percentile threshold for gene expression (default: 0.95)
			
		Returns:
		--------
		float
			Expression threshold value
		"""
		expr_threshold = np.percentile(self.gene_expression_PTR, gene_expr_percentile * 100)
		self.cycling_expr_genes = np.where(self.gene_expression_PTR >= expr_threshold)[0]
		self.expr_cycling_count = len(self.cycling_expr_genes)
		
		return expr_threshold
	
	def analyze_thresholds(self, gene_expr_percentile=0.95, 
						   promoter_percentiles=np.arange(0.0, 1.01, 0.01)):
		"""
		Analyze a range of promoter occupancy thresholds.
		
		Parameters:
		-----------
		gene_expr_percentile : float
			Percentile threshold for gene expression (default: 0.95)
		promoter_percentiles : numpy array
			Range of percentiles to test for promoter occupancy
			
		Returns:
		--------
		pd.DataFrame
			DataFrame with results for each threshold
		"""
		# Set expression threshold if not already set
		if self.cycling_expr_genes is None:
			self.set_expression_threshold(gene_expr_percentile)
		
		# Initialize results list
		results = []
		
		# Sort percentiles to ensure they're in ascending order
		sorted_percentiles = np.sort(promoter_percentiles)
		
		# First, get the intersection at the lowest threshold to start with all genes
		lowest_percentile = sorted_percentiles[0]
		lowest_threshold = np.percentile(self.promoter_occupancy_PTR, lowest_percentile * 100)
		cycling_promoter_genes = np.where(self.promoter_occupancy_PTR >= lowest_threshold)[0]
		initial_intersection = np.intersect1d(self.cycling_expr_genes, cycling_promoter_genes)
		
		# Initialize gene tracking DataFrame and previous intersection set
		gene_tracking = {}
		for gene_idx in initial_intersection:
			gene_name = self.gene_names[gene_idx]
			gene_tracking[gene_name] = {
				'removal_percentile': None,  # Will be filled when gene is removed
				'removal_ptr_threshold': None,
				'gene_idx': gene_idx,
				'expression_ptr': self.gene_expression_PTR[gene_idx],
				'promoter_ptr': self.promoter_occupancy_PTR[gene_idx]
			}
		
		prev_intersection = set(initial_intersection)
		
		# Sweep through promoter occupancy thresholds (skipping the first one since we used it for initialization)
		for percentile in sorted_percentiles[1:]:
			promoter_threshold = np.percentile(self.promoter_occupancy_PTR, percentile * 100)
			cycling_promoter_genes = np.where(self.promoter_occupancy_PTR >= promoter_threshold)[0]
			n = len(cycling_promoter_genes)
			
			# Find intersection
			intersection = np.intersect1d(self.cycling_expr_genes, cycling_promoter_genes)
			k = len(intersection)
			
			# Track genes that were removed at this threshold
			current_intersection = set(intersection)
			removed_genes = prev_intersection - current_intersection
			
			# Store which genes were removed at this threshold
			for gene_idx in removed_genes:
				gene_name = self.gene_names[gene_idx]
				if gene_name in gene_tracking:
					gene_tracking[gene_name]['removal_percentile'] = percentile
					gene_tracking[gene_name]['removal_ptr_threshold'] = promoter_threshold
			
			prev_intersection = current_intersection
			
			# Expected overlap under null hypothesis
			expected = (self.expr_cycling_count * n) / self.total_genes
			
			# Hypergeometric test - probability of >= k successes
			p_value = 1 - stats.hypergeom.cdf(k-1, self.total_genes, self.expr_cycling_count, n)
			
			# Store results
			results.append({
				'percentile': percentile,
				'ptr_threshold': promoter_threshold,
				'promoter_cycling_genes': n,
				'intersection_size': k,
				'expected_overlap': expected,
				'enrichment_ratio': k / expected if expected > 0 else np.inf,
				'p_value': p_value,
				'-log10_p_value': -np.log10(p_value) if p_value > 0 else np.inf
			})
		
		# For genes still in the intersection at the highest threshold, mark as not removed
		highest_percentile = sorted_percentiles[-1]
		highest_threshold = np.percentile(self.promoter_occupancy_PTR, highest_percentile * 100)
		for gene_name, gene_data in gene_tracking.items():
			if gene_data['removal_percentile'] is None:
				gene_data['removal_percentile'] = float('inf')  # Or some marker for "not removed"
				gene_data['removal_ptr_threshold'] = float('inf')
		
		# Convert to DataFrame
		self.results_df = pd.DataFrame(results)
		
		# Create gene tracking DataFrame and sort by removal threshold
		self.threshold_gene_tracking = pd.DataFrame.from_dict(gene_tracking, orient='index')
		self.threshold_gene_tracking = self.threshold_gene_tracking.sort_values(by='removal_percentile')
		
		return self.results_df
	
	def get_genes_by_threshold(self, threshold_percentile):
		"""
		Get genes added up to a specific threshold percentile.
		
		Parameters:
		-----------
		threshold_percentile : float
			Threshold percentile to get genes for
			
		Returns:
		--------
		pandas.DataFrame
			DataFrame with genes that are in the intersection at the given threshold
		"""
		if self.threshold_gene_tracking is None:
			raise ValueError("No gene tracking data available. Run analyze_thresholds() first.")
			
		return self.threshold_gene_tracking[self.threshold_gene_tracking['percentile_threshold'] <= threshold_percentile]
	
	def get_gene_sets_with_names(self, promoter_percentile):
		"""
		Get gene sets at a specific promoter percentile threshold with gene names.
		
		Parameters:
		-----------
		promoter_percentile : float
			Percentile threshold for promoter occupancy
			
		Returns:
		--------
		dict
			Dictionary with gene sets and their names
		"""
		if self.cycling_expr_genes is None:
			raise ValueError("Expression threshold not set. Run set_expression_threshold() first.")
			
		# Get gene sets by index
		gene_sets = self.get_gene_sets(promoter_percentile)
		
		# Convert indices to gene names
		gene_sets_with_names = {}
		for set_name, gene_indices in gene_sets.items():
			gene_sets_with_names[set_name] = [self.gene_names[idx] for idx in gene_indices]
			
		return gene_sets_with_names
		
	def plot_results(self):
		"""
		Plot the results of the threshold optimization.
		
		Returns:
		--------
		matplotlib.figure.Figure
			Figure containing the plots
		"""
		if self.results_df is None:
			raise ValueError("No results to plot. Run analyze_thresholds() first.")
			
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
		
		# Left plot: Proportion of cycling genes with twin axis for total gene count
		ax1.plot(self.results_df['percentile'], 
				 self.results_df['intersection_size']/self.expr_cycling_count * 100, 'b-', 
				 label='% of Expression Cycling Genes')
		ax1.plot(self.results_df['percentile'], 
				 self.results_df['expected_overlap']/self.expr_cycling_count * 100, 'r--', 
				 label='% Expected Overlap')
		ax1.set_xlabel('Promoter Occupancy Percentile Threshold')
		ax1.set_ylabel('% of Cycling Expression Genes')
		ax1.set_title('Overlap as % of Cycling Expression Genes')
		ax1.legend(loc='upper left')
		ax1.grid(True)

		max_intersection_perc = self.results_df['intersection_size'].max()/self.expr_cycling_count*100
		ax1.set_ylim(0, max_intersection_perc*1.1)

		# Twin axis for number of genes in intersection
		ax1_twin = ax1.twinx()
		ax1_twin.plot(self.results_df['percentile'], self.results_df['intersection_size'], 'g-.')
		ax1_twin.set_ylabel('Number of genes in intersection', color='g')
		ax1_twin.tick_params(axis='y', labelcolor='g')
		ax1_twin.set_ylim(0, max_intersection_perc/100*self.expr_cycling_count*1.1)
			
		# Right plot: Statistical significance
		ax2.plot(self.results_df['percentile'], self.results_df['-log10_p_value'], 'g-')
		ax2.set_xlabel('Promoter Occupancy Percentile Threshold')
		ax2.set_ylabel('-log10(p-value)')
		ax2.set_title('Statistical Significance')
		ax2.grid(True)
		
		# Add enrichment ratio as a twin axis
		ax2_twin = ax2.twinx()
		ax2_twin.plot(self.results_df['percentile'], self.results_df['enrichment_ratio'], 'm-.', 
					 label='Enrichment Ratio')
		ax2_twin.set_ylabel('Enrichment Ratio (Observed/Expected)', color='m')
		ax2_twin.tick_params(axis='y', labelcolor='m')
		ax2_twin.legend(loc='upper right')
		
		plt.tight_layout()
		return fig
		
	def run_complete_analysis(self, gene_expr_percentile=0.95, 
							 promoter_percentiles=np.arange(0.0, 1.01, 0.01)):
		"""
		Run the complete analysis workflow and output results.
		
		Parameters:
		-----------
		gene_expr_percentile : float
			Percentile threshold for gene expression (default: 0.95)
		promoter_percentiles : numpy array
			Range of percentiles to test for promoter occupancy
			
		Returns:
		--------
		tuple
			(results_df, figure, gene_tracking_df)
		"""
		# Run analysis
		self.analyze_thresholds(gene_expr_percentile, promoter_percentiles)
		
		# Create plots
		fig = self.plot_results()
		
		return self.results_df, fig
	
	def get_gene_sets(self, promoter_percentile):
		"""
		Get gene sets at a specific promoter percentile threshold.
		
		Parameters:
		-----------
		promoter_percentile : float
			Percentile threshold for promoter occupancy
			
		Returns:
		--------
		dict
			Dictionary with gene sets
		"""
		if self.cycling_expr_genes is None:
			raise ValueError("Expression threshold not set. Run set_expression_threshold() first.")
			
		promoter_threshold = np.percentile(self.promoter_occupancy_PTR, promoter_percentile * 100)
		cycling_promoter_genes = np.where(self.promoter_occupancy_PTR >= promoter_threshold)[0]
		
		# Find intersection
		intersection = np.intersect1d(self.cycling_expr_genes, cycling_promoter_genes)
		
		# Find expression-only cycling genes
		expr_only = np.setdiff1d(self.cycling_expr_genes, intersection)
		
		# Find promoter-only cycling genes
		promoter_only = np.setdiff1d(cycling_promoter_genes, intersection)
		
		return {
			'expression_cycling': self.cycling_expr_genes,
			'promoter_cycling': cycling_promoter_genes,
			'intersection': intersection,
			'expression_only': expr_only,
			'promoter_only': promoter_only
		}
	
	