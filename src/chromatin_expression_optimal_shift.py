import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union, Any


# Colors of positive and negative correlation
# thought of as activators or repressors

REGULATION_TYPES = {
	'ACTIVATION_IMMEDIATE': {
		'name': "Activation, immediate",
		'primary_corr': 'positive',
		'primary_shift_min': -5,
		'primary_shift_max': 5,
		'secondary_conditions': None,
		'description': 'Direct recruitment of transcriptional machinery',
		'color': plt.cm.Greens(0.75)
	},
	'ACTIVATION_PIONEER': {
		'name': "Activation, pioneer",
		'primary_corr': 'positive',
		'primary_shift_min': -15,
		'primary_shift_max': -5,
		'secondary_conditions': None,
		'description': 'Pioneer factors preparing chromatin',
		'color': plt.cm.Greens(0.55)
	},
	'REPRESSION_IMMEDIATE': {
		'name': "Repression, immediate",
		'primary_corr': 'negative',
		'primary_shift_min': -5,
		'primary_shift_max': 5,
		'secondary_conditions': None,
		'description': 'Direct interference with transcriptional machinery',
		'color': plt.cm.Reds(0.75)
	},
	'REPRESSION_DELAYED': {
		'name': "Repression, delayed",
		'primary_corr': 'positive',
		'primary_shift_min': 10,
		'primary_shift_max': 30,
		'secondary_conditions': None,
		'description': 'Feedback repression following initial transcriptional activity',
		'color': plt.cm.Reds(0.6)
	},
	'UNCLEAR': {
		'color': plt.cm.Greys(0.75)
	}
}


# Set default colors based on regulation types
def reg_color_for_key(key):
	color = REGULATION_TYPES[key]['color']
	return color

def reg_title_for_key(key):
	reg_entry = REGULATION_TYPES[key]
	title = (f"{reg_entry['name']}")
	return title


class ExpressionCorrelationAnalyzer:
	"""
	A class to analyze optimal time shifts between expression and occupancy data
	for maximum positive and negative correlations in cyclical time series.
	"""
	
	def __init__(self, 
				 expressions_df: pd.DataFrame, 
				 occupancies_df: pd.DataFrame, 
				 max_shift: int = 3,
				 correlation_threshold: float = 0.75):
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

	def _determine_local_optima_correlation(self, gene_corrs: np.ndarray) -> \
		Tuple[float, float, float, float]:
		"""
		Determine optimal time shifts by finding the first local maximum and minimum
		when moving away from zero shift in both directions.
		
		Parameters:
		-----------
		gene_corrs : np.ndarray
			Array of correlation values for each shift
			
		Returns:
		--------
		Tuple[float, float, float, float]
			Local maximum correlation, corresponding shift, local minimum correlation, corresponding shift
		"""
		# Handle NaN cases
		if np.all(np.isnan(gene_corrs)):
			return np.nan, np.nan, np.nan, np.nan
		
		# Find the index of zero shift
		zero_idx = np.where(np.array(self.shifts) == 0)[0][0]
		zero_corr = gene_corrs[zero_idx]
		
		# Initialize best values for each direction
		max_corr_left = zero_corr
		max_shift_left = 0
		max_corr_right = zero_corr
		max_shift_right = 0
		
		min_corr_left = zero_corr
		min_shift_left = 0
		min_corr_right = zero_corr
		min_shift_right = 0
		
		# Search left for maximum (stop at first decrease)
		prev_corr = zero_corr
		for i in range(zero_idx - 1, -1, -1):
			curr_corr = gene_corrs[i]
			curr_shift = self.shifts[i]
			
			if curr_corr > prev_corr:
				# Still increasing, update maximum
				if curr_corr > max_corr_left:
					max_corr_left = curr_corr
					max_shift_left = curr_shift
			else:
				# Found a decrease, stop searching
				break
				
			prev_corr = curr_corr
		
		# Search right for maximum (stop at first decrease)
		prev_corr = zero_corr
		for i in range(zero_idx + 1, len(gene_corrs)):
			curr_corr = gene_corrs[i]
			curr_shift = self.shifts[i]
			
			if curr_corr > prev_corr:
				# Still increasing, update maximum
				if curr_corr > max_corr_right:
					max_corr_right = curr_corr
					max_shift_right = curr_shift
			else:
				# Found a decrease, stop searching
				break
				
			prev_corr = curr_corr
		
		# Search left for minimum (stop at first increase)
		prev_corr = zero_corr
		for i in range(zero_idx - 1, -1, -1):
			curr_corr = gene_corrs[i]
			curr_shift = self.shifts[i]
			
			if curr_corr < prev_corr:
				# Still decreasing, update minimum
				if curr_corr < min_corr_left:
					min_corr_left = curr_corr
					min_shift_left = curr_shift
			else:
				# Found an increase, stop searching
				break
				
			prev_corr = curr_corr
		
		# Search right for minimum (stop at first increase)
		prev_corr = zero_corr
		for i in range(zero_idx + 1, len(gene_corrs)):
			curr_corr = gene_corrs[i]
			curr_shift = self.shifts[i]
			
			if curr_corr < prev_corr:
				# Still decreasing, update minimum
				if curr_corr < min_corr_right:
					min_corr_right = curr_corr
					min_shift_right = curr_shift
			else:
				# Found an increase, stop searching
				break
				
			prev_corr = curr_corr
		
		# Select the better maximum between left and right
		if max_corr_left > max_corr_right:
			max_corr = max_corr_left
			max_corr_shift = max_shift_left
		else:
			max_corr = max_corr_right
			max_corr_shift = max_shift_right
		
		# Select the better minimum between left and right
		if min_corr_left < min_corr_right:
			min_corr = min_corr_left
			min_corr_shift = min_shift_left
		else:
			min_corr = min_corr_right
			min_corr_shift = min_shift_right
		
		return max_corr, max_corr_shift, min_corr, min_corr_shift

	def classify_regulation(self, max_corr, max_shift, min_corr, min_shift):
		"""
		Classify regulation type based on prioritized criteria.
		
		Parameters:
		-----------
		max_corr : float
			Maximum (positive) correlation value
		max_shift : float
			Shift corresponding to maximum correlation
		min_corr : float
			Minimum (negative) correlation value
		min_shift : float
			Shift corresponding to minimum correlation
			
		Returns:
		--------
		str
			Classification label based on priority order
		"""
		# Priority order of mechanisms (biological relevance)
		priority_order = [
			'REPRESSION_IMMEDIATE',   # Check direct repression first (anti-correlation)
			'ACTIVATION_IMMEDIATE',   # Then direct activation
			'ACTIVATION_PIONEER',  # Then pioneer activation
			'REPRESSION_DELAYED'  # Finally indirect repression
		]
		
		for reg_type in priority_order:
			criteria = REGULATION_TYPES[reg_type]
			
			# Select appropriate correlation and shift based on type
			if criteria['primary_corr'] == 'positive':
				corr = max_corr
				shift = max_shift
			else:  # negative
				corr = min_corr
				shift = min_shift
			
			# Check if correlation exceeds threshold
			if abs(corr) < self.correlation_threshold:
				continue
				
			# Check if shift is within range
			if criteria['primary_shift_min'] <= shift <= criteria['primary_shift_max']:
				# Found a match - return immediately
				return reg_type
		
		# No criteria matched
		return 'UNCLEAR'

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
			local_max_corr: local maximum positive correlation within window
			local_max_shift: shift for local maximum correlation
			local_min_corr: local maximum negative correlation within window
			local_min_shift: shift for local minimum correlation
			local_regulation_type: classification based on local optima
		"""
		# Initialize results containers
		results = {
			'gene': [],
			'local_max_corr': [],
			'local_max_shift': [],
			'local_min_corr': [],
			'local_min_shift': [],
			'local_regulation_type': [],
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
		all_local_max_shifts = []
		all_local_min_shifts = []

		# Process each gene
		for gene in self.common_genes:
			# Get expression and occupancy time series for this gene
			expr_series = self.expressions_df.loc[gene, indices].values
			occ_series = self.occupancies_df.loc[gene, indices].values
			
			# Length of time series
			n = len(indices)
			
			# Calculate correlation for each potential shift
			gene_corrs = self._calculate_correlations(gene, expr_series, occ_series, n, debug, debug_genes)
			occ_std = occ_series.std()

			occ_std_threshold = 0.1

			# Store gene correlation data for later analysis
			self.correlation_data[branch_name][gene] = {
				'shifts': self.shifts,
				'correlations': gene_corrs,
				'expression': expr_series,
				'occupancy': occ_series,
				'occupancy_std': occ_std,
				'indices': indices
			}
			
			# Handle case where all correlations are NaN
			if np.all(np.isnan(gene_corrs)):
				max_corr_shift = min_corr_shift = local_max_shift = local_min_shift = np.nan
			else:
				
				# Find local optima within window around zero
				local_max_corr, local_max_shift, local_min_corr, local_min_shift = \
					self._determine_local_optima_correlation(
					gene_corrs)
				
				# Add to our tracking lists
				all_local_max_shifts.append(local_max_shift)
				all_local_min_shifts.append(local_min_shift)
			
			# Determine regulation type based on local correlation values
			local_regulation_type = self.classify_regulation(
				local_max_corr, local_max_shift,
				local_min_corr, local_min_shift)

			if occ_std < occ_std_threshold:
				local_regulation_type = 'UNCLEAR'
			
			# Store results for this gene
			results['gene'].append(gene)
			results['local_max_corr'].append(local_max_corr)
			results['local_max_shift'].append(local_max_shift)
			results['local_min_corr'].append(local_min_corr)
			results['local_min_shift'].append(local_min_shift)
			results['local_regulation_type'].append(local_regulation_type)
		
		# Debug overall shift distributions
		if debug:
			self._plot_shift_distributions(all_max_shifts, all_min_shifts)
			self._plot_local_shift_distributions(all_local_max_shifts, all_local_min_shifts)
		
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
		
		constant_check = (expr_series == expr_series[0]).all() or (occ_series == occ_series[0]).all()

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

			# If an input is constant, assume no correlation
			if constant_check:
				corr, p_value = 0, 1
			else:
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
	
	def plot_gene_correlation_with_local_optima(self, gene: str) -> \
		plt.Figure:
		"""
		Plot correlation vs. shift for a specific gene, highlighting both global and local optima.
		
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
		
		# Create a figure with two subplots
		fig, axs = plt.subplots(2, 2, figsize=(10, 6))

		def plot_shift_plots(ax1, ax2, branch_name):

			gene_data = self.correlation_data[branch_name][orfname]
			shifts = gene_data['shifts']
			correlations = gene_data['correlations']
			expr_series = gene_data['expression']
			occ_series = gene_data['occupancy']
			
			# Get global and local optima from results
			gene_results = self.results[branch_name].loc[orfname]
			local_max_shift = gene_results['local_max_shift']
			local_min_shift = gene_results['local_min_shift']

			# Plot 1: Original time series
			tx_line = ax1.plot(expr_series, 'r-', label='Expression', lw=3)

			occ_ax = ax1.twinx()
			occ_line = occ_ax.plot(occ_series, 'b-', label='Occupancy', lw=3)

			ax1.set_title(f"{branch_name.title()} branch")
			ax1.set_xlabel("Time point")
			ax1.set_ylabel("Expression")
			occ_ax.set_ylabel("Promoter occupancy")
			occ_ax.set_ylim(0.5, 1.8)
			ax1.grid(True, alpha=0.3)
			ax1.legend()
			
			# Plot 2: Correlation vs. shift
			ax2.plot(shifts, correlations, '.-')
			ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
			ax2.axvline(x=0, color='k', linestyle='-', alpha=0.3)
			
			# Highlight local optima
			ax2.plot(local_max_shift, gene_results['local_max_corr'], 'g*', markersize=12, 
					label=f'Local Max: {gene_results["local_max_corr"]:.3f} at shift {local_max_shift}')
			ax2.plot(local_min_shift, gene_results['local_min_corr'], 'r*', markersize=12,
					label=f'Local Min: {gene_results["local_min_corr"]:.3f} at shift {local_min_shift}')
			
			ax2.grid(True, alpha=0.3)
			ax2.set_xlabel('Shift')
			ax2.set_ylabel('Correlation')
			ax2.legend()
		
		plot_shift_plots(axs[0][0], axs[1][0], 'mother')
		plot_shift_plots(axs[0][1], axs[1][1], 'daughter')

		plt.suptitle(f"Shift and correlation for {gene}")
		plt.tight_layout()

		return fig

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
		ax1.legend()
		
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
	
	def analyze_branches(self, config: Any, max_shift: int = None, 
			correlation_threshold: float = None) -> Dict[str, pd.DataFrame]:
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

	def plot_optimal_shift_distributions(self, gene_subsets: Dict[str, List[str]] = None, 
			figsize: Tuple[int, int] = (14, 7)) -> Tuple[plt.Figure, np.ndarray]:
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
		correlation_types = ['local_max_shift', 'local_min_shift']
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
		shift_range_bins = np.linspace(min_shift - 0.5, max_shift + 1.5, 20)

		from src.plot_helpers import color_for_key
		
		# Plot histograms
		for i, (subset_name, genes) in enumerate(gene_subsets.items()):
			row = 0
			for j, corr_type in enumerate(correlation_types):
				for branch in branches:
					ax = axes[row, i]
					
					# Get shifts for genes in this subset
					common_genes = self.results[branch].index.intersection(genes)
					shifts = self.results[branch].loc[common_genes, corr_type].dropna()
					
					if 'MG1 and DG1' == subset_name:
						color = color_for_key('MG1')
					elif 'MG1' in subset_name:
						color = color_for_key('MG1')
					elif 'DG1' in subset_name:
						color = color_for_key('DG1')
					elif 'Post G1' in subset_name:
						color = color_for_key('postG1')
					else:
						color = plt.cm.Greys(0.7)

					# Plot histogram
					ax.hist(shifts, bins=shift_range_bins, alpha=0.7, 
						color=color)
					ax.set_xlim(min_shift - 1, max_shift + 1)
					ax.axvline(0, color='red', linestyle='--', linewidth=1)

					if i == len(gene_subsets)-1:
						ax.set_ylim(0, 1000)
					else:
						ax.set_ylim(0, 90)
					
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
		Plot stacked bar counts of regulation types (activator, repressor, unclear) for each gene subset
		using separate subplots for each subset with 270-degree rotated x-labels.
		
		Parameters:
		-----------
		branch_name : str, default='default'
			Name of the analysis branch
		gene_subsets : dict, optional
			Dictionary mapping subset names to lists of genes
		figsize : tuple, default=(15, 3)
			Figure size
		colors : dict, optional
			Dictionary mapping regulation types to colors.
			Default: uses REGULATION_TYPES color mapping
			
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
		
		# Group regulation types for stacking
		activation_types = ['ACTIVATION_IMMEDIATE', 'ACTIVATION_PIONEER']
		repression_types = ['REPRESSION_IMMEDIATE', 'REPRESSION_DELAYED']
		other_types = ['UNCLEAR']
		
		# X-axis labels
		x_labels = ['Act.', 'Repr.', 'Unclear']
		
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
			reg_counts = self.results[branch_name].loc[common_genes]['local_regulation_type'].value_counts()
			
			# Initialize counts for each group
			activation_counts = np.zeros(2)  # Direct, Pioneer
			repression_counts = np.zeros(2)  # Direct, Indirect
			unclear_count = 0
			
			# Fill in counts for each regulation type
			for j, reg_type in enumerate(activation_types):
				if reg_type in reg_counts:
					activation_counts[j] = reg_counts[reg_type]
					
			for j, reg_type in enumerate(repression_types):
				if reg_type in reg_counts:
					repression_counts[j] = reg_counts[reg_type]
					
			if 'UNCLEAR' in reg_counts:
				unclear_count = reg_counts['UNCLEAR']
			
			# Create stacked bars
			# Plot activators (stacked)
			activation_bottom = 0
			activation_bars = []
			for j, (count, reg_type) in enumerate(zip(activation_counts, activation_types)):
				bar = ax.bar(0, count, bottom=activation_bottom, 
						   color=reg_color_for_key(reg_type), linewidth=1, alpha=0.7)
				activation_bottom += count
				activation_bars.append(bar)
			
			# Plot repressors (stacked)
			repression_bottom = 0
			repression_bars = []
			for j, (count, reg_type) in enumerate(zip(repression_counts, repression_types)):
				bar = ax.bar(1, count, bottom=repression_bottom, 
						   color=reg_color_for_key(reg_type), linewidth=1, alpha=0.7)
				repression_bottom += count
				repression_bars.append(bar)
			
			# Plot unclear
			unclear_bar = ax.bar(2, unclear_count, color=reg_color_for_key('UNCLEAR'), 
								linewidth=1, alpha=0.7)
			
			# Set title and labels
			ax.set_title(f"{subset_name} (n={len(common_genes)})",
			 font='Open Sans', fontweight='demi', fontsize=16)
			ax.set_ylabel("# of genes")
			
			# Set x-axis labels with 270-degree rotation
			ax.set_xticks([0, 1, 2])
			ax.set_xticklabels(x_labels, rotation=0)

			if i == 4:
				ax.set_ylim(0, 3000)
			else:
				ax.set_ylim(0, 250)
			
			# Add grid for readability
			ax.grid(axis='y', alpha=0.3)
		
		# Hide any unused subplots
		for j in range(i + 1, len(axs)):
			axs[j].set_visible(False)
		
		plt.suptitle(f'Promoter binding classification - {branch_name.capitalize()} Branch', 
					 fontsize=21, y=0.98, font='Open Sans', fontweight='demi')
		plt.tight_layout(rect=[0, 0.08, 1, 0.95])  # Leave space for suptitle and legend
		
		return fig


def plot_binding_modalities(config1):
	from scipy.stats.distributions import norm

	fig, axs = plt.subplots(2, 2, figsize=(6, 4))
	axs = np.array(axs).T.flatten()

	def plot_shift_ax(ax, shift, key, flip=False, ls='solid'):
		centered = 30
		std = 10

		xs = np.arange(len(config1.t_indices()))
		tx = norm.pdf(xs, centered, std)
		sm_prom = norm.pdf(xs, centered+shift, std)

		if flip:
			sm_prom = sm_prom*-1
			sm_prom = sm_prom-sm_prom.min()+0.005

		from src.plot_helpers import plot_rect2

		color = reg_color_for_key(key)

		entry = REGULATION_TYPES[key]
		shift_min = entry['primary_shift_min']
		shift_max = entry['primary_shift_max']
		l = centered+shift_min
		r = centered+shift_max

		plot_rect2(ax, l, 0, r, 1, 
			color=color, fill_alpha=0.15,
			zorder=10)

		ax.axvline(centered, c='black', lw=0.75, ls='solid', zorder=11)
		ax.axvline(centered+shift, c=color, lw=0.75, ls='solid', zorder=11)

		ax.fill_between(xs, tx, 0, color='#d3d7db', zorder=0)

		ax.plot(xs, sm_prom, lw=3, color=color, ls=ls, zorder=11)
		ax.set_ylim(-0.0001, 0.05)
		ax.set_xlim(0, xs[-1])
		ax.set_xticks([l, r])
		ax.set_xticklabels([shift_min, shift_max])
		ax.set_yticks([])
		ax.set_title(reg_title_for_key(key))

	plot_shift_ax(axs[0], -1, 'ACTIVATION_IMMEDIATE')
	plot_shift_ax(axs[1], -10, 'ACTIVATION_PIONEER', ls=(0, (2, 1)))
	plot_shift_ax(axs[2], -1, 'REPRESSION_IMMEDIATE', flip=True)
	plot_shift_ax(axs[3], 20, 'REPRESSION_DELAYED', ls=(0, (2, 1)))

	axs[1].set_xlabel("Promoter shift window")
	axs[1].set_ylabel("Expression/\nOccupancy")

	plt.tight_layout()
	plt.suptitle("Promoter binding classifications", fontsize=16)
	plt.subplots_adjust(top=0.85, hspace=0.6, wspace=0.1)