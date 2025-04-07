import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union, Any


# Colors of positive and negative correlation
# thought of as activators or repressors
pos_color = plt.cm.Greens(0.75)
neg_color = plt.cm.Reds(0.75)
grey_color = plt.cm.Greys(0.75)

class ExpressionCorrelationAnalyzer:
	"""
	A class to analyze optimal time shifts between expression and occupancy data
	for maximum positive and negative correlations in cyclical time series.
	"""
	
	def __init__(self, 
				 expressions_df: pd.DataFrame, 
				 occupancies_df: pd.DataFrame, 
				 max_shift: int = 3,
				 correlation_threshold: float = 0.5):
		"""
		Initialize the ExpressionCorrelationAnalyzer.
		
		Parameters:
		-----------
		expressions_df : pandas.DataFrame
			DataFrame containing gene expression data
		occupancies_df : pandas.DataFrame
			DataFrame containing small fragment occupancy data
		max_shift : int, default=3
			Maximum time shift to consider in both directions
		correlation_threshold : float, default=0.5
			Absolute correlation threshold to classify a gene as an activator or repressor
		"""
		self.expressions_df = expressions_df
		self.occupancies_df = occupancies_df
		self.max_shift = max_shift
		self.shifts = list(range(-max_shift, max_shift + 1))
		self.correlation_threshold = correlation_threshold
		
		# Store results for different analyses
		self.results = {}
		self.correlation_data = {}
		
		# Get common genes between expression and occupancy data
		self.common_genes = expressions_df.index.intersection(occupancies_df.index)
	
	def compute_shifts(self, 
					   indices: List[int], 
					   branch_name: str = 'default', 
					   debug: bool = False, 
					   debug_genes: List[str] = None) -> pd.DataFrame:
		"""
		Compute optimal time shifts between expression and occupancy data.
		
		Parameters:
		-----------
		indices : list
			Column indices to use for the correlation calculation
		branch_name : str, default='default'
			Name to identify this analysis branch
		debug : bool, default=False
			If True, print debug information
		debug_genes : list, default=None
			List of specific genes to debug
			
		Returns:
		--------
		DataFrame with gene indices and columns for:
			max_corr: maximum positive correlation value
			max_corr_shift: shift that gave maximum positive correlation
			min_corr: maximum negative correlation value (minimum correlation)
			min_corr_shift: shift that gave maximum negative correlation
			regulation_type: classification as 'activator', 'repressor', or 'unclear'
		"""
		# Initialize results containers
		results = {
			'gene': [],
			'max_corr': [],
			'max_corr_shift': [],
			'min_corr': [],
			'min_corr_shift': [],
			'regulation_type': [],
			'abs_max_corr': [],
			'abs_max_corr_shift': [],
			'is_positive_correlation': []
		}
		
		# Store correlations for each gene and shift
		self.correlation_data[branch_name] = {}
		
		if debug:
			print(f"Shift range: {self.shifts}")
			print(f"Time series length: {len(indices)}")
			print(f"First few indices: {indices[:5]}...")
			print(f"Number of common genes: {len(self.common_genes)}")
		
		# Set default debug genes if not provided
		if debug and debug_genes is None:
			debug_genes = list(self.common_genes)[:3]  # Take first 3 genes
		
		# Keep track of all shift distributions for a sanity check
		all_max_shifts = []
		all_min_shifts = []
		
		# Process each gene
		for gene in self.common_genes:
			# Get expression and occupancy time series for this gene
			expr_series = self.expressions_df.loc[gene, indices].values
			occ_series = self.occupancies_df.loc[gene, indices].values
			
			# Length of time series
			n = len(indices)
			
			# Calculate correlation for each potential shift
			gene_corrs = self._calculate_correlations(gene, expr_series, occ_series, n, debug, debug_genes)
			
			# Store gene correlation data for later analysis
			self.correlation_data[branch_name][gene] = {
				'shifts': self.shifts,
				'correlations': gene_corrs,
				'expression': expr_series,
				'occupancy': occ_series,
				'indices': indices
			}
			
			# Handle case where all correlations are NaN
			if np.all(np.isnan(gene_corrs)):
				max_corr = min_corr = np.nan
				max_corr_shift = min_corr_shift = np.nan
			else:
				# Find maximum correlation
				max_idx = np.nanargmax(gene_corrs)
				max_corr = gene_corrs[max_idx]
				max_corr_shift = self.shifts[max_idx]
				
				# Find minimum correlation
				min_idx = np.nanargmin(gene_corrs)
				min_corr = gene_corrs[min_idx]
				min_corr_shift = self.shifts[min_idx]
				
				# Add to our tracking lists
				all_max_shifts.append(max_corr_shift)
				all_min_shifts.append(min_corr_shift)
			
			# Determine regulation type based on correlation values
			regulation_type = self._classify_gene_regulation(max_corr, min_corr)
			
			# Determine absolute maximum correlation (whether positive or negative)
			abs_max_corr, abs_max_corr_shift, is_positive = self._determine_absolute_max_correlation(
				max_corr, max_corr_shift, min_corr, min_corr_shift
			)
			
			# Store results for this gene
			results['gene'].append(gene)
			results['max_corr'].append(max_corr)
			results['max_corr_shift'].append(max_corr_shift)
			results['min_corr'].append(min_corr)
			results['min_corr_shift'].append(min_corr_shift)
			results['regulation_type'].append(regulation_type)
			results['abs_max_corr'].append(abs_max_corr)
			results['abs_max_corr_shift'].append(abs_max_corr_shift)
			results['is_positive_correlation'].append(is_positive)
		
		# Debug overall shift distributions
		if debug:
			self._plot_shift_distributions(all_max_shifts, all_min_shifts)
		
		# Create DataFrame from results
		results_df = pd.DataFrame(results).set_index('gene')
		
		# Store results for this branch
		self.results[branch_name] = results_df
		
		return results_df
	
	def _calculate_correlations(self, 
							   gene: str, 
							   expr_series: np.ndarray, 
							   occ_series: np.ndarray, 
							   n: int, 
							   debug: bool, 
							   debug_genes: List[str]) -> np.ndarray:
		"""
		Calculate correlations for all shifts for a specific gene.
		
		Parameters:
		-----------
		gene : str
			Gene identifier
		expr_series : numpy.ndarray
			Expression data for the gene
		occ_series : numpy.ndarray
			Occupancy data for the gene
		n : int
			Length of time series
		debug : bool
			Whether to print debug information
		debug_genes : list
			List of genes to debug
			
		Returns:
		--------
		numpy.ndarray of correlation values for each shift
		"""
		gene_corrs = []
		
		# For debugging: store correlation for each shift
		if debug and gene in debug_genes:
			print(f"\nDEBUG - Gene: {gene}")
			print(f"Expression data: {expr_series}")
			print(f"Occupancy data: {occ_series}")
			
			# Create a debug plot for this gene
			self._create_debug_plots(gene, expr_series, occ_series)
			
		# Calculate correlation for each potential shift
		for shift in self.shifts:
			# For cyclical data, we can use modulo arithmetic to wrap around
			shifted_indices = [(i + shift) % n for i in range(n)]
			
			# Rearrange the occupancy data based on the shifted indices
			shifted_occ = np.array([occ_series[i] for i in shifted_indices])
			
			# Calculate correlation using the full time series length
			corr, p_value = pearsonr(expr_series, shifted_occ)
			gene_corrs.append(corr)
			
			# Debug specific genes
			if debug and gene in debug_genes:
				print(f"Shift {shift}: correlation = {corr:.4f}, p-value = {p_value:.4f}")
				if shift in [-self.max_shift, -self.max_shift//2, 0, self.max_shift//2, self.max_shift]:
					plt.subplot(212)
					plt.plot(expr_series, 'r-', label=f'Expression' if shift == -self.max_shift else "")
					plt.plot(shifted_occ, '.-', label=f'Occupancy (shift={shift})')
		
		# Complete the debug plot
		if debug and gene in debug_genes:
			self._complete_debug_plots(gene, gene_corrs)
			
		return np.array(gene_corrs)
	
	def _create_debug_plots(self, gene: str, expr_series: np.ndarray, occ_series: np.ndarray) -> None:
		"""Create initial debug plots for a gene."""
		plt.figure(figsize=(15, 10))
		plt.subplot(211)
		plt.plot(expr_series, 'r-', label='Expression')
		plt.plot(occ_series, 'b-', label='Occupancy')
		plt.legend()
		plt.title(f"Original data for gene {gene}")
		
		plt.subplot(212)
	
	def _complete_debug_plots(self, gene: str, gene_corrs: np.ndarray) -> None:
		"""Complete debug plots with correlation data."""
		plt.subplot(212)
		plt.legend()
		plt.title(f"Expression vs. shifted occupancy for gene {gene}")
		
		plt.figure(figsize=(10, 5))
		plt.plot(self.shifts, gene_corrs, '.-')
		plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
		plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
		plt.grid(True, alpha=0.3)
		plt.xlabel('Shift')
		plt.ylabel('Correlation')
		plt.title(f"Correlation vs. shift for gene {gene}")
		plt.tight_layout()
		plt.show()
	
	def _determine_absolute_max_correlation(self, max_corr: float, max_corr_shift: float, 
										 min_corr: float, min_corr_shift: float) -> Tuple[float, float, bool]:
		"""
		Determine the absolute maximum correlation and corresponding shift.
		
		Parameters:
		-----------
		max_corr : float
			Maximum positive correlation value
		max_corr_shift : float
			Shift that gave maximum positive correlation
		min_corr : float
			Maximum negative correlation value (minimum correlation)
		min_corr_shift : float
			Shift that gave maximum negative correlation
			
		Returns:
		--------
		Tuple[float, float, bool]
			Absolute maximum correlation, corresponding shift, and whether it's positive
		"""
		# Handle NaN cases
		if np.isnan(max_corr) and np.isnan(min_corr):
			return np.nan, np.nan, False
		
		if np.isnan(max_corr):
			return abs(min_corr), min_corr_shift, False
			
		if np.isnan(min_corr):
			return max_corr, max_corr_shift, True
		
		# Compare absolute values to find the stronger correlation
		abs_max = max_corr
		abs_min = abs(min_corr)  # Take absolute value of negative correlation
		
		if abs_max >= abs_min:
			return abs_max, max_corr_shift, True
		else:
			return abs_min, min_corr_shift, False

	def _classify_gene_regulation(self, max_corr: float, min_corr: float) -> str:
		"""
		Classify a gene as an activator, repressor, or unclear based on its correlation values.
		
		Parameters:
		-----------
		max_corr : float
			Maximum positive correlation value
		min_corr : float
			Maximum negative correlation value
			
		Returns:
		--------
		str
			Classification as 'activator', 'repressor', or 'unclear'
		"""
		# Handle NaN values
		if np.isnan(max_corr) or np.isnan(min_corr):
			return 'unclear'
		
		# Get absolute values
		abs_max = abs(max_corr)
		abs_min = abs(min_corr)
		
		# Check if correlations exceed threshold
		if abs_max < self.correlation_threshold and abs_min < self.correlation_threshold:
			return 'unclear'
		
		# If positive correlation is stronger than negative, classify as activator
		if abs_max > abs_min and max_corr > 0:
			return 'activator'
		
		# If negative correlation is stronger than positive, classify as repressor
		if abs_min > abs_max and min_corr < 0:
			return 'repressor'
		
		# Default case
		return 'unclear'
	
	def _plot_shift_distributions(self, all_max_shifts: List[int], all_min_shifts: List[int]) -> None:
		"""Plot distributions of optimal shifts."""
		print("\nShift distribution summary:")
		print(f"Max correlation shifts: {np.bincount(np.array(all_max_shifts) + self.max_shift)}")
		print(f"Min correlation shifts: {np.bincount(np.array(all_min_shifts) + self.max_shift)}")
		
		plt.figure(figsize=(12, 5))
		plt.subplot(121)
		plt.hist(all_max_shifts, bins=len(self.shifts))
		plt.title("Distribution of shifts with maximum positive correlation")
		plt.xlabel("Shift")
		plt.ylabel("Count")
		
		plt.subplot(122)
		plt.hist(all_min_shifts, bins=len(self.shifts))
		plt.title("Distribution of shifts with maximum negative correlation")
		plt.xlabel("Shift")
		plt.ylabel("Count")
		plt.tight_layout()
		plt.show()
	
	def plot_gene_correlation(self, gene: str, branch_name: str = 'default') -> plt.Figure:
		"""
		Plot correlation vs. shift for a specific gene.
		
		Parameters:
		-----------
		gene : str
			Gene identifier
		branch_name : str, default='default'
			Name of the analysis branch
			
		Returns:
		--------
		matplotlib.figure.Figure
			The created figure
		"""
		from src.sgd import get_orfname

		orfname = get_orfname(gene)

		if branch_name not in self.correlation_data or orfname not in self.correlation_data[branch_name]:
			raise ValueError(f"No correlation data available for gene {gene} in branch {branch_name}")
		
		gene_data = self.correlation_data[branch_name][orfname]
		shifts = gene_data['shifts']
		correlations = gene_data['correlations']
		expr_series = gene_data['expression']
		occ_series = gene_data['occupancy']
		
		# Create a figure with two subplots
		fig = plt.figure(figsize=(14, 10))
		
		# Plot 1: Original time series
		ax1 = fig.add_subplot(2, 1, 1)
		tx_line = ax1.plot(expr_series, 'r-', label='Expression', lw=3)

		occ_ax = ax1.twinx()
		occ_line = occ_ax.plot(occ_series, 'b-', label='Occupancy', lw=3)

		ax1.set_title(f"Original data for gene {gene}")
		ax1.set_xlabel("Time point")
		ax1.set_ylabel("Expression")
		occ_ax.set_ylabel("Promoter occupancy")
		ax1.grid(True, alpha=0.3)
		
		# Plot 2: Correlation vs. shift
		ax2 = fig.add_subplot(2, 1, 2)
		ax2.plot(shifts, correlations, '.-')
		ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
		ax2.axvline(x=0, color='k', linestyle='-', alpha=0.3)
		
		# Highlight maximum and minimum correlation points
		max_idx = np.nanargmax(correlations)
		min_idx = np.nanargmin(correlations)
		
		ax2.plot(shifts[max_idx], correlations[max_idx], 'go', markersize=10, 
				label=f'Max Corr: {correlations[max_idx]:.3f} at shift {shifts[max_idx]}')
		ax2.plot(shifts[min_idx], correlations[min_idx], 'ro', markersize=10,
				label=f'Min Corr: {correlations[min_idx]:.3f} at shift {shifts[min_idx]}')
		
		ax2.grid(True, alpha=0.3)
		ax2.set_xlabel('Shift')
		ax2.set_ylabel('Correlation')
		ax2.set_title(f"Correlation vs. shift for gene {gene}")
		ax2.legend()
		
		plt.tight_layout()
		return fig
	
	def plot_best_shift_alignment(self, gene: str, branch_name: str = 'default', shift_type: str = 'max_corr') -> plt.Figure:
		"""
		Plot the original and optimally shifted data for a specific gene.
		
		Parameters:
		-----------
		gene : str
			Gene identifier
		branch_name : str, default='default'
			Name of the analysis branch
		shift_type : str, default='max_corr'
			Type of correlation to use for optimal shift ('max_corr' or 'min_corr')
			
		Returns:
		--------
		matplotlib.figure.Figure
			The created figure
		"""
		if branch_name not in self.correlation_data or gene not in self.correlation_data[branch_name]:
			raise ValueError(f"No correlation data available for gene {gene} in branch {branch_name}")
		
		if branch_name not in self.results or gene not in self.results[branch_name].index:
			raise ValueError(f"No results available for gene {gene} in branch {branch_name}")
		
		gene_data = self.correlation_data[branch_name][gene]
		expr_series = gene_data['expression']
		occ_series = gene_data['occupancy']
		n = len(expr_series)
		
		# Get optimal shift from results
		if shift_type == 'max_corr':
			optimal_shift = self.results[branch_name].loc[gene, 'max_corr_shift']
			correlation = self.results[branch_name].loc[gene, 'max_corr']
			title_suffix = "Maximum Positive Correlation"
		elif shift_type == 'min_corr':
			optimal_shift = self.results[branch_name].loc[gene, 'min_corr_shift']
			correlation = self.results[branch_name].loc[gene, 'min_corr']
			title_suffix = "Maximum Negative Correlation"
		else:
			raise ValueError("shift_type must be 'max_corr' or 'min_corr'")
		
		# Calculate shifted occupancy
		shifted_indices = [(i + int(optimal_shift)) % n for i in range(n)]
		shifted_occ = np.array([occ_series[i] for i in shifted_indices])
		
		# Create figure
		fig = plt.figure(figsize=(12, 8))
		
		# Original data
		ax1 = fig.add_subplot(2, 1, 1)
		ax1.plot(expr_series, 'r-', label='Expression')
		ax1.plot(occ_series, 'b-', label='Occupancy (original)')
		ax1.legend()
		ax1.set_title(f"Original data for gene {gene}")
		ax1.set_xlabel("Time point")
		ax1.set_ylabel("Value")
		ax1.grid(True, alpha=0.3)
		
		# Shifted data
		ax2 = fig.add_subplot(2, 1, 2)
		ax2.plot(expr_series, 'r-', label='Expression')
		ax2.plot(shifted_occ, 'b-', label=f'Occupancy (shift={int(optimal_shift)})')
		ax2.legend()
		ax2.set_title(f"Shifted data for gene {gene} - {title_suffix} (r={correlation:.3f})")
		ax2.set_xlabel("Time point")
		ax2.set_ylabel("Value")
		ax2.grid(True, alpha=0.3)
		
		plt.tight_layout()
		return fig
	
	def analyze_branches(self, config: Any, max_shift: int = None, correlation_threshold: float = None) -> Dict[str, pd.DataFrame]:
		"""
		Analyze optimal shifts between expression and occupancy for both mother and daughter branches.
		
		Parameters:
		-----------
		config : object
			Configuration object with methods t_indices() and b_indices()
		max_shift : int, optional
			Maximum time shift to consider in both directions. Overrides the class attribute if provided.
		correlation_threshold : float, optional
			Correlation threshold for classifying genes. Overrides the class attribute if provided.
			
		Returns:
		--------
		Dictionary containing DataFrames with correlation results for mother and daughter branches
		"""
		if max_shift is not None:
			self.max_shift = max_shift
			self.shifts = list(range(-max_shift, max_shift + 1))
			
		if correlation_threshold is not None:
			self.correlation_threshold = correlation_threshold
		
		# Get indices for top and bottom branches
		t_indices = config.t_indices()
		b_indices = config.b_indices()
		
		# Compute optimal shifts for top branch (mother)
		self.compute_shifts(t_indices, branch_name='mother')
		
		# Compute optimal shifts for bottom branch (daughter)
		self.compute_shifts(b_indices, branch_name='daughter')
		
		return {
			'mother': self.results['mother'],
			'daughter': self.results['daughter']
		}
		
	def get_regulation_summary(self, branch_name: str = 'default', gene_subset: List[str] = None) -> Dict[str, int]:
		"""
		Get a summary of gene regulation types for a branch.
		
		Parameters:
		-----------
		branch_name : str, default='default'
			Name of the analysis branch
		gene_subset : list, optional
			Subset of genes to consider, if None, use all genes
			
		Returns:
		--------
		dict
			Dictionary with counts for each regulation type
		"""
		if branch_name not in self.results:
			raise ValueError(f"No results available for branch {branch_name}")
		
		# Get results for this branch
		results_df = self.results[branch_name]
		
		# Filter to gene subset if provided
		if gene_subset is not None:
			common_genes = results_df.index.intersection(gene_subset)
			if len(common_genes) == 0:
				raise ValueError(f"No genes from the provided subset found in results for branch {branch_name}")
			results_df = results_df.loc[common_genes]
		
		# Count regulation types
		reg_counts = results_df['regulation_type'].value_counts().to_dict()
		
		# Ensure all categories are present
		for reg_type in ['activator', 'repressor', 'unclear']:
			if reg_type not in reg_counts:
				reg_counts[reg_type] = 0
				
		return reg_counts
	
	def plot_regulation_distribution(self, branch_name: str = 'default', 
							   gene_subsets: Dict[str, List[str]] = None,
							   figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
		"""
		Plot the distribution of activators vs repressors for different gene subsets.
		
		Parameters:
		-----------
		branch_name : str, default='default'
			Name of the analysis branch
		gene_subsets : dict, optional
			Dictionary mapping subset names to lists of genes
		figsize : tuple, default=(12, 8)
			Figure size
			
		Returns:
		--------
		matplotlib.figure.Figure
			The created figure
		"""
		if branch_name not in self.results:
			raise ValueError(f"No results available for branch {branch_name}")
		
		if gene_subsets is None:
			# Use all genes if no subsets are provided
			gene_subsets = {'All genes': list(self.results[branch_name].index)}
		
		# Create figure
		fig, axes = plt.subplots(1, len(gene_subsets), figsize=figsize)
		
		# Convert to array if only one subset
		if len(gene_subsets) == 1:
			axes = np.array([axes])
		
		# Prepare colors for bar chart
		colors = {'activator': pos_color, 'repressor': neg_color, 'unclear': 'gray'}
		
		# Plot each subset
		for i, (subset_name, genes) in enumerate(gene_subsets.items()):
			# Get genes in this subset that are also in the results
			common_genes = self.results[branch_name].index.intersection(genes)
			
			if len(common_genes) == 0:
				print(f"Warning: No genes from subset '{subset_name}' found in results")
				continue
			
			# Count regulation types
			reg_counts = self.results[branch_name].loc[common_genes, 'regulation_type'].value_counts()
			
			# Ensure all categories are present
			for reg_type in ['activator', 'repressor', 'unclear']:
				if reg_type not in reg_counts:
					reg_counts[reg_type] = 0
			
			# Plot bar chart
			ax = axes[i]
			bars = ax.bar(reg_counts.index, reg_counts.values, color=[colors[t] for t in reg_counts.index])
			
			# Add count labels on top of bars
			for bar in bars:
				height = bar.get_height()
				ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
						f'{int(height)}', ha='center', va='bottom')
			
			# Add percentage labels inside or beside bars
			total = reg_counts.sum()
			for j, (reg_type, count) in enumerate(reg_counts.items()):
				percentage = 100 * count / total
				if percentage > 5:  # Only add label if enough space
					y_pos = count / 2  # Position in middle of bar
					ax.text(j, y_pos, f'{percentage:.1f}%', ha='center', va='center')
			
			# Set title and labels
			ax.set_title(f"{subset_name}\n(n={len(common_genes)})")
			ax.set_ylabel("Number of genes")
			
			# Add grid for readability
			ax.grid(axis='y', alpha=0.3)
		
		plt.tight_layout()
		plt.suptitle(f"Distribution of Gene Regulation Types - {branch_name.capitalize()} Branch", 
					 y=1.05, fontsize=14)
		
		return fig
	
	def plot_correlation_scatter(self, branch_name: str = 'default', 
								gene_subsets: Dict[str, List[str]] = None,
								figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
		"""
		Create a scatter plot of maximum positive vs negative correlations to visualize
		gene regulation patterns.
		
		Parameters:
		-----------
		branch_name : str, default='default'
			Name of the analysis branch
		gene_subsets : dict, optional
			Dictionary mapping subset names to lists of genes
		figsize : tuple, default=(12, 10)
			Figure size
			
		Returns:
		--------
		matplotlib.figure.Figure
			The created figure
		"""
		if branch_name not in self.results:
			raise ValueError(f"No results available for branch {branch_name}")
		
		if gene_subsets is None:
			# Use all genes if no subsets are provided
			gene_subsets = {'All genes': list(self.results[branch_name].index)}
		
		# Create a colormap for regulation types
		regulation_colors = {
			'activator': 'green',
			'repressor': 'red',
			'unclear': 'gray'
		}
		
		# Create figure
		fig = plt.figure(figsize=figsize)
		
		# Main scatter plot
		ax = fig.add_subplot(111)
		
		# Add diagonal line
		ax.plot([-1, 1], [-1, 1], 'k--', alpha=0.5)
		
		# Add threshold lines
		threshold = self.correlation_threshold
		ax.axhline(-threshold, color='gray', linestyle=':', alpha=0.5)
		ax.axhline(threshold, color='gray', linestyle=':', alpha=0.5)
		ax.axvline(-threshold, color='gray', linestyle=':', alpha=0.5)
		ax.axvline(threshold, color='gray', linestyle=':', alpha=0.5)
		
		# Plot each subset
		subset_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']
		
		for i, (subset_name, genes) in enumerate(gene_subsets.items()):
			# Get common genes
			common_genes = self.results[branch_name].index.intersection(genes)
			
			if len(common_genes) == 0:
				print(f"Warning: No genes from subset '{subset_name}' found in results")
				continue
			
			# Get data for this subset
			subset_data = self.results[branch_name].loc[common_genes]
			
			# Plot each regulation type separately for the legend
			for reg_type in ['activator', 'repressor', 'unclear']:
				mask = subset_data['regulation_type'] == reg_type
				if mask.any():
					ax.scatter(
						subset_data.loc[mask, 'max_corr'], 
						subset_data.loc[mask, 'min_corr'],
						c=regulation_colors[reg_type],
						marker=subset_markers[i % len(subset_markers)],
						alpha=0.7,
						label=f'{subset_name} - {reg_type}'
					)
		
		# Add labels and grid
		ax.set_xlabel('Maximum Positive Correlation')
		ax.set_ylabel('Maximum Negative Correlation')
		ax.set_xlim(-1.05, 1.05)
		ax.set_ylim(-1.05, 1.05)
		ax.grid(True, alpha=0.3)
		
		# Add legend
		ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
		
		# Add regions labels
		ax.text(0.9, -0.9, 'Repressor\nRegion', ha='center', va='center', 
				bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
		ax.text(0.9, 0.9, 'Activator\nRegion', ha='center', va='center',
				bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
		
		plt.tight_layout()
		plt.suptitle(f"Correlation Pattern Analysis - {branch_name.capitalize()} Branch", 
					  y=1.02, fontsize=14)
		
		return fig
	
	def plot_absolute_max_shift_distribution(self, branch_name: str = 'default', 
									  gene_subsets: Dict[str, List[str]] = None,
									  figsize: Tuple[int, int] = (8, 8)) -> plt.Figure:
		"""
		Plot histogram of absolute maximum correlation shifts, colored by whether they're positive or negative.
		
		Parameters:
		-----------
		branch_name : str, default='default'
			Name of the analysis branch
		gene_subsets : dict, optional
			Dictionary mapping subset names to lists of genes
		figsize : tuple, default=(12, 8)
			Figure size
			
		Returns:
		--------
		matplotlib.figure.Figure
			The created figure
		"""
		if branch_name not in self.results:
			raise ValueError(f"No results available for branch {branch_name}")
		
		if gene_subsets is None:
			# Use all genes if no subsets are provided
			gene_subsets = {'All genes': list(self.results[branch_name].index)}
		
		# Create figure
		fig, axes = plt.subplots(len(gene_subsets), 1, figsize=figsize)
		
		# Convert to array if only one subset
		if len(gene_subsets) == 1:
			axes = np.array([axes])
		
		# Determine shift range for consistent x-axis
		all_shifts = self.results[branch_name]['abs_max_corr_shift'].dropna().values
		min_shift = min(all_shifts) if len(all_shifts) > 0 else -self.max_shift
		max_shift = max(all_shifts) if len(all_shifts) > 0 else self.max_shift
		
		# Create bins for histogram
		bins = np.arange(min_shift - 0.5, max_shift + 1.5)
		
		# Plot histograms for each subset
		for i, (subset_name, genes) in enumerate(gene_subsets.items()):
			ax = axes[i]
			
			# Get common genes between subset and results
			common_genes = self.results[branch_name].index.intersection(genes)
			
			if len(common_genes) == 0:
				ax.text(0.5, 0.5, f"No genes from subset '{subset_name}' found in results",
					   ha='center', va='center', transform=ax.transAxes)
				continue
			
			# Get data for this subset
			subset_data = self.results[branch_name].loc[common_genes]
			
			# Separate positive and negative correlations
			pos_mask = subset_data['is_positive_correlation']
			neg_mask = ~pos_mask

			ax.hist([subset_data.loc[pos_mask, 'abs_max_corr_shift'], 
					 subset_data.loc[neg_mask, 'abs_max_corr_shift']], 
					bins=bins, stacked=True,  alpha=0.7,
					color=[pos_color, neg_color],
					label=['Positive Correlation', 'Negative Correlation'])
			
			# Add a vertical line at 0 shift
			ax.axvline(0, color='black', linestyle='--', alpha=0.5)
			
			# Set title and labels
			ax.set_title(f"{subset_name} (n={len(common_genes)})")
			ax.set_xlabel("Optimal Shift")
			ax.set_ylabel("Number of Genes")
			
			# Set consistent x-axis limits
			ax.set_xlim(min_shift - 1, max_shift + 1)
			
			# Add grid and legend
			ax.grid(True, alpha=0.3)
			ax.legend()
		
		plt.tight_layout()
		plt.suptitle(f"Distribution of Absolute Maximum Correlation Shifts - {branch_name.capitalize()} Branch", 
					 y=1.02, fontsize=14)
		
		return fig

	def plot_optimal_shift_distributions(self, gene_subsets: Dict[str, List[str]] = None, 
			figsize: Tuple[int, int] = (14, 6)) -> Tuple[plt.Figure, np.ndarray]:
		"""
		Plot histograms of optimal shift distributions for different gene subsets.
		
		Parameters:
		-----------
		gene_subsets : dict, optional
			Dictionary mapping subset names to lists of genes
		figsize : tuple, default=(14, 10)
			Figure size
			
		Returns:
		--------
		tuple
			Figure and axes objects
		"""
		if 'mother' not in self.results or 'daughter' not in self.results:
			raise ValueError("Both 'mother' and 'daughter' branches must be analyzed first using analyze_branches()")
		
		if gene_subsets is None:
			# Use all genes if no subsets are provided
			gene_subsets = {'All genes': list(self.results['mother'].index)}
		
		branches = ['mother', 'daughter']
		correlation_types = ['max_corr_shift', 'min_corr_shift']
		labels = ['Positive', 'Negative']
		
		fig, axes = plt.subplots(len(branches) * len(correlation_types), len(gene_subsets),
								figsize=figsize)
		
		# Flatten axes if there's only one gene subset
		if len(gene_subsets) == 1:
			axes = axes.reshape(1, -1)
		
		# Determine shift range for consistent x-axis
		all_shifts = []
		for branch in branches:
			for corr_type in correlation_types:
				all_shifts.extend(self.results[branch][corr_type].dropna().values)
		
		min_shift = min(all_shifts) if all_shifts else -self.max_shift
		max_shift = max(all_shifts) if all_shifts else self.max_shift
		shift_range = np.arange(min_shift - 0.5, max_shift + 1.5)

		corr_colors = {
			'max_corr_shift': pos_color,
			'min_corr_shift': neg_color
		}
		
		# Plot histograms
		for i, (subset_name, genes) in enumerate(gene_subsets.items()):
			row = 0
			for j, corr_type in enumerate(correlation_types):
				for branch in branches:
					ax = axes[row, i]
					
					# Get shifts for genes in this subset
					common_genes = self.results[branch].index.intersection(genes)
					shifts = self.results[branch].loc[common_genes, corr_type].dropna()
					
					# Plot histogram
					ax.hist(shifts, bins=shift_range, alpha=0.7, color=corr_colors[corr_type])
					ax.set_xlim(min_shift - 1, max_shift + 1)
					ax.axvline(0, color='red', linestyle='--', linewidth=1)

					if i == len(gene_subsets)-1:
						ax.set_ylim(0, 100)
					else:
						ax.set_ylim(0, 14)
					
					# Add title and labels
					if i == 0:
						branch_name = 'Mother Branch' if branch == 'mother' else 'Daughter Branch'
						ax.set_ylabel(f"{branch_name}\n{labels[j]}")
					
					if row == 0:
						ax.set_title(f"{subset_name}, n={len(common_genes)}")
					
					row += 1
		
		plt.tight_layout()
		plt.suptitle("Distribution of Optimal Time Shifts between Expression and Occupancy", 
			y=1.05, fontsize=16)
		return fig, axes


	def plot_regulation_type_bar_counts(self, branch_name: str = 'default',
							  gene_subsets: Dict[str, List[str]] = None,
							  figsize: Tuple[int, int] = (15, 3),
							  colors: Dict[str, str] = None) -> plt.Figure:
		"""
		Plot bar counts of regulation types (activator, repressor, unclear) for each gene subset
		using separate subplots for each subset.
		
		Parameters:
		-----------
		branch_name : str, default='default'
			Name of the analysis branch
		gene_subsets : dict, optional
			Dictionary mapping subset names to lists of genes
		figsize : tuple, default=(15, 10)
			Figure size
		colors : dict, optional
			Dictionary mapping regulation types to colors.
			Default: {'activator': 'green', 'repressor': 'red', 'unclear': 'gray'}
			
		Returns:
		--------
		matplotlib.figure.Figure
			The created figure
		"""
		if branch_name not in self.results:
			raise ValueError(f"No results available for branch {branch_name}")
		
		if gene_subsets is None:
			# Use all genes if no subsets are provided
			gene_subsets = {'All genes': list(self.results[branch_name].index)}
		
		# Set default colors if not provided
		if colors is None:
			colors = {'activator': pos_color, 'repressor': neg_color, 'unclear': grey_color}
		
		# Calculate number of subplots needed
		n_subplots = len(gene_subsets)
		
		# Calculate grid dimensions for subplots
		n_cols = min(5, n_subplots)
		n_rows = (n_subplots + n_cols - 1) // n_cols  # Ceiling division
		
		# Create figure and axes
		fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
		
		# Handle the case of a single subplot
		if n_subplots == 1:
			axs = np.array([axs])
		
		# Flatten axes array for easier indexing
		axs = axs.flatten()
		
		# Define regulation types
		reg_types = ['activator', 'repressor', 'unclear']
		
		# Process each gene subset
		for i, (subset_name, genes) in enumerate(gene_subsets.items()):
			ax = axs[i]
			
			# Get common genes between subset and results
			common_genes = self.results[branch_name].index.intersection(genes)
			
			if len(common_genes) == 0:
				ax.text(0.5, 0.5, f"No genes from subset '{subset_name}' found in results",
					   ha='center', va='center', transform=ax.transAxes)
				continue
			
			# Count regulation types
			reg_counts = self.results[branch_name].loc[common_genes]['regulation_type'].value_counts()
			
			# Make sure all regulation types are represented
			for reg_type in reg_types:
				if reg_type not in reg_counts:
					reg_counts[reg_type] = 0
			
			# Sort counts by regulation type for consistent ordering
			reg_counts = reg_counts.reindex(reg_types)
			
			# Plot bars for this subset
			bars = ax.bar(reg_types, reg_counts.values, color=[colors[t] for t in reg_types], 
						  linewidth=1, alpha=0.7)
			
			# Add count labels on top of bars
			for bar in bars:
				height = bar.get_height()
				if height > 0:  # Only add label if bar has non-zero height
					ax.text(bar.get_x() + bar.get_width()/2., height + 0.05 * ax.get_ylim()[1],
							f'{int(height)}', ha='center', va='bottom', fontsize=9)
			
			# Set title and labels
			ax.set_title(f"{subset_name} (n={len(common_genes)})")
			ax.set_ylabel("Number of Genes")
			
			# Auto-scale y-axis for this specific subplot
			max_count = reg_counts.max()
			ax.set_ylim(0, max_count * 1.25)  # Add 15% headroom for labels
			
			# Add grid for readability
			ax.grid(axis='y', alpha=0.3)
		
		# Hide any unused subplots
		for j in range(i + 1, len(axs)):
			axs[j].set_visible(False)
		
		# Create a common legend
		# handles = [plt.Rectangle((0,0), 1, 1, color=colors[t]) for t in reg_types]
		# labels = [t.capitalize() for t in reg_types]
		# fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02),
		# 		  fancybox=True, shadow=True, ncol=len(reg_types))
		
		plt.suptitle(f'Regulation Type Distribution - {branch_name.capitalize()} Branch', 
					 fontsize=16, y=0.98)
		plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Leave space for suptitle and legend
		
		return fig