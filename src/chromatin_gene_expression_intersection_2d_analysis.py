import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


class TwoDimensionalPTRAnalysis:
	"""
	A class for performing two-dimensional sensitivity analysis between two cycling metrics' PTR thresholds.
	"""
	
	def __init__(self, metric1_name, metric2_name, metric1_PTR, metric2_PTR, gene_names=None,
		pval_vmax=8, enrichment_vmax=2):
		"""
		Initialize the analysis with two metrics.
		
		Parameters:
		-----------
		metric1_name : str
			Name of the first metric
		metric2_name : str
			Name of the second metric
		metric1_PTR : array-like
			PTR values for first metric
		metric2_PTR : array-like
			PTR values for second metric
		gene_names : list, optional
			List of gene names corresponding to the PTR values
		"""
		self.metric1_name = metric1_name
		self.metric2_name = metric2_name
		self.pval_vmax = pval_vmax
		self.enrichment_vmax = enrichment_vmax

		# Convert pandas Series to numpy arrays if needed
		self.gene_names = None
		if isinstance(metric1_PTR, pd.Series):
			if gene_names is None:
				self.gene_names = metric1_PTR.index.tolist()
			self.metric1_PTR = metric1_PTR.values
		else:
			self.metric1_PTR = metric1_PTR
			
		if isinstance(metric2_PTR, pd.Series):
			if gene_names is None and self.gene_names is None:
				self.gene_names = metric2_PTR.index.tolist()
			self.metric2_PTR = metric2_PTR.values
		else:
			self.metric2_PTR = metric2_PTR
		
		# Use passed gene_names if provided
		if gene_names is not None:
			self.gene_names = gene_names
			
		# If no gene names are available, create placeholder names
		if self.gene_names is None:
			self.gene_names = [f"Gene_{i}" for i in range(len(self.metric1_PTR))]
			
		# Check that we have the correct number of gene names
		if len(self.gene_names) != len(self.metric1_PTR):
			raise ValueError("Number of gene names must match the number of metric1 values")
			
		self.total_genes = len(self.metric1_PTR)
		self.results_df = None
		self.heatmap_data = None
		self.gene_sets = {}
		
	def run_2d_analysis(self, metric1_percentiles=np.arange(0.5, 1.01, 0.05), 
						metric2_percentiles=np.arange(0.5, 1.01, 0.05)):
		"""
		Run a two-dimensional analysis across various thresholds for both metrics.
		"""
		# Sort percentiles to ensure they're in ascending order
		metric1_percentiles = np.sort(metric1_percentiles)
		metric2_percentiles = np.sort(metric2_percentiles)
		
		# Initialize results container
		results = []
		
		# Initialize data structures for heatmaps
		p_values = np.zeros((len(metric1_percentiles), len(metric2_percentiles)))
		log10_p_values = np.zeros((len(metric1_percentiles), len(metric2_percentiles)))
		enrichment_ratios = np.zeros((len(metric1_percentiles), len(metric2_percentiles)))
		intersection_sizes = np.zeros((len(metric1_percentiles), len(metric2_percentiles)))
		
		# Store gene sets for key threshold combinations
		self.gene_sets = {}
		
		# Loop through all combinations of thresholds
		for i, metric1_percentile in enumerate(metric1_percentiles):
			metric1_threshold = np.percentile(self.metric1_PTR, metric1_percentile * 100)
			cycling_metric1_genes = np.where(self.metric1_PTR >= metric1_threshold)[0]
			metric1_cycling_count = len(cycling_metric1_genes)
			
			for j, metric2_percentile in enumerate(metric2_percentiles):
				metric2_threshold = np.percentile(self.metric2_PTR, metric2_percentile * 100)
				cycling_metric2_genes = np.where(self.metric2_PTR >= metric2_threshold)[0]
				n = len(cycling_metric2_genes)
				
				# Find intersection
				intersection = np.intersect1d(cycling_metric1_genes, cycling_metric2_genes)
				k = len(intersection)
				
				# Expected overlap under null hypothesis
				expected = (metric1_cycling_count * n) / self.total_genes
				
				# Hypergeometric test - probability of >= k successes
				p_value = 1 - stats.hypergeom.cdf(k-1, self.total_genes, metric1_cycling_count, n)
				
				# Store results
				results.append({
					'metric1_percentile': metric1_percentile,
					'metric1_threshold': metric1_threshold,
					'metric1_cycling_count': metric1_cycling_count,
					'metric2_percentile': metric2_percentile,
					'metric2_threshold': metric2_threshold,
					'metric2_cycling_count': n,
					'intersection_size': k,
					'expected_overlap': expected,
					'enrichment_ratio': k / expected if expected > 0 else np.inf,
					'p_value': p_value,
					'-log10_p_value': -np.log10(p_value) if p_value > 0 else np.inf
				})
				
				# Store values for heatmaps
				p_values[i, j] = p_value
				log10_p_values[i, j] = -np.log10(p_value) if p_value > 0 else np.inf
				enrichment_ratios[i, j] = k / expected if expected > 0 else np.inf
				intersection_sizes[i, j] = k
				
				# Store gene sets for key combinations
				key = (metric1_percentile, metric2_percentile)
				
				# Store every combination for later retrieval
				self.gene_sets[key] = {
					'intersection': [self.gene_names[idx] for idx in intersection],
					'metric1_only': [self.gene_names[idx] for idx in np.setdiff1d(cycling_metric1_genes, intersection)],
					'metric2_only': [self.gene_names[idx] for idx in np.setdiff1d(cycling_metric2_genes, intersection)]
				}
		
		# Convert to DataFrame
		self.results_df = pd.DataFrame(results)
		
		# Store heatmap data for plotting
		self.heatmap_data = {
			'metric1_percentiles': metric1_percentiles,
			'metric2_percentiles': metric2_percentiles,
			'p_values': p_values,
			'log10_p_values': log10_p_values,
			'enrichment_ratios': enrichment_ratios,
			'intersection_sizes': intersection_sizes
		}
		
		return self.results_df
	
	def plot_pvalue_heatmap(self, ax):
		"""
		Plot heatmaps of the analysis results.
		
		Returns:
		--------
		matplotlib.figure.Figure
			Figure containing the heatmap plots
		"""

		# Extract data
		x = self.heatmap_data['metric2_percentiles']
		y = self.heatmap_data['metric1_percentiles']
		
		# Create meshgrid for contour plots
		X, Y = np.meshgrid(x, y)

		pvalues = self.heatmap_data['p_values']
		# adjusted_p_values = apply_benjamini_hochberg(pvalues)

		# num_tests = pvalues.size
		# adjusted_p_values = np.minimum(pvalues * num_tests, 1.0)

		log_pvalues = -np.log10(pvalues)
		
		# Plot 1: p-values heatmap
		im = ax.pcolormesh(X, Y, log_pvalues, cmap='viridis',
			vmin=0, vmax=self.pval_vmax)
		ax.set_xlabel(f'{self.metric2_name} percentile threshold')
		ax.set_ylabel(f'{self.metric1_name} percentile threshold')

		# Plot 3: Intersection size heatmap
		# im = ax.pcolormesh(X, Y, self.heatmap_data['intersection_sizes'], cmap='Blues')
		ax.set_xlabel(f'{self.metric2_name} Percentile Threshold')
		ax.set_ylabel(f'{self.metric1_name} Percentile Threshold')
		ax.set_title('Number of Genes in Intersection')

		# Add intersection size text to each cell in the third heatmap
		for i, plot_y_position in enumerate(y):
			for j, plot_x_position in enumerate(x):

				intersection_size = int(self.heatmap_data['intersection_sizes'][i, j])

				lp_value = log_pvalues[j, i]
				text = str(intersection_size)

				ax.text(plot_y_position, plot_x_position, text,
					   ha='center', va='center', 
					   color='white' if lp_value < \
					   self.pval_vmax*0.75 else 'black',
					   fontsize=8, fontweight='demi')

		return im
		
	def plot_heatmaps(self):
		"""
		Plot heatmaps of the analysis results.
		
		Returns:
		--------
		matplotlib.figure.Figure
			Figure containing the heatmap plots
		"""
		if self.heatmap_data is None:
			raise ValueError("No heatmap data available. Run run_2d_analysis() first.")
		
		# Create 3 subplots
		fig, axes = plt.subplots(1, 3, figsize=(26, 8))
		
		# Extract data
		x = self.heatmap_data['metric2_percentiles']
		y = self.heatmap_data['metric1_percentiles']
		
		# Create meshgrid for contour plots
		X, Y = np.meshgrid(x, y)
		
		# Plot 3: Intersection size heatmap
		ax = axes[2]
		im = ax.pcolormesh(X, Y, self.heatmap_data['intersection_sizes'], cmap='Blues')
		ax.set_xlabel(f'{self.metric2_name} Percentile Threshold')
		ax.set_ylabel(f'{self.metric1_name} Percentile Threshold')
		ax.set_title('Number of Genes in Intersection')
		plt.colorbar(im, ax=ax, label='Number of genes')
		
		# Add intersection size text to each cell in the third heatmap
		for i, plot_y_position in enumerate(y):
			for j, plot_x_position in enumerate(x):

				intersection_size = int(self.heatmap_data['intersection_sizes'][i, j])
				ax.text(plot_y_position, plot_x_position, str(intersection_size),
					   ha='center', va='center', 
					   color='white' if intersection_size > \
						np.max(self.heatmap_data['intersection_sizes'])/2 else 'black',
					   fontsize=8, fontweight='bold')		

		return fig
	
	def find_optimal_thresholds(self, criteria='p_value', top_n=5):
		"""
		Find optimal threshold combinations based on a given criteria.
		
		Parameters:
		-----------
		criteria : str
			Criteria to optimize ('p_value', 'enrichment_ratio', or 'intersection_size')
		top_n : int
			Number of top combinations to return
			
		Returns:
		--------
		pandas.DataFrame
			DataFrame with top threshold combinations
		"""
		if self.results_df is None:
			raise ValueError("No results available. Run run_2d_analysis() first.")
			
		# Define sorting key based on criteria
		if criteria == 'p_value':
			sort_key = 'p_value'
			ascending = True
		elif criteria == 'enrichment_ratio':
			sort_key = 'enrichment_ratio'
			ascending = False
		elif criteria == 'intersection_size':
			sort_key = 'intersection_size'
			ascending = False
		else:
			raise ValueError("Invalid criteria. Choose 'p_value', 'enrichment_ratio', or 'intersection_size'")
			
		# Sort and return top combinations
		top_combinations = self.results_df.sort_values(by=sort_key, ascending=ascending).head(top_n)
		return top_combinations
	
	def characterize_gene_set(self, metric1_percentile, metric2_percentile):
		"""
		Characterize gene sets at a specific threshold combination.
		
		Parameters:
		-----------
		metric1_percentile : float
			Percentile threshold for metric 1
		metric2_percentile : float
			Percentile threshold for metric 2
			
		Returns:
		--------
		dict
			Dictionary with gene sets and statistics
		"""
		# Find closest thresholds if exact match not available
		key = (metric1_percentile, metric2_percentile)
		if key not in self.gene_sets:
			metric1_thresholds = self.heatmap_data['metric1_percentiles']
			metric2_thresholds = self.heatmap_data['metric2_percentiles']
			
			closest_metric1 = metric1_thresholds[np.abs(metric1_thresholds - metric1_percentile).argmin()]
			closest_metric2 = metric2_thresholds[np.abs(metric2_thresholds - metric2_percentile).argmin()]
			key = (closest_metric1, closest_metric2)
			
			print(f"Exact thresholds not found. Using closest available: ({closest_metric1}, {closest_metric2})")
		
		# Get gene sets
		gene_sets = self.gene_sets[key]
		
		# Add statistics
		result = gene_sets.copy()
		result['intersection_size'] = len(gene_sets['intersection'])
		result['metric1_only_size'] = len(gene_sets['metric1_only'])
		result['metric2_only_size'] = len(gene_sets['metric2_only'])
		
		return result
	
	def compare_threshold_combinations(self, combinations):
		"""
		Compare gene sets across different threshold combinations.
		
		Parameters:
		-----------
		combinations : list of tuples
			List of (metric1_percentile, metric2_percentile) combinations to compare
			
		Returns:
		--------
		dict
			Dictionary with comparison results
		"""
		if not combinations:
			raise ValueError("No combinations provided")
			
		# Get gene sets for each combination
		gene_sets = {}
		for combo in combinations:
			metric1_perc, metric2_perc = combo
			key = (metric1_perc, metric2_perc)
			
			# Find closest thresholds if exact match not available
			if key not in self.gene_sets:
				metric1_thresholds = self.heatmap_data['metric1_percentiles']
				metric2_thresholds = self.heatmap_data['metric2_percentiles']
				
				closest_metric1 = metric1_thresholds[np.abs(metric1_thresholds - metric1_perc).argmin()]
				closest_metric2 = metric2_thresholds[np.abs(metric2_thresholds - metric2_perc).argmin()]
				key = (closest_metric1, closest_metric2)
				
				print(f"Exact thresholds not found for {combo}. Using closest available: {key}")
			
			gene_sets[combo] = self.gene_sets[key]
		
		# Initialize comparison results
		comparison = {
			'threshold_combos': combinations,
			'intersection_sizes': {},
			'unique_genes': {},
			'shared_genes': {},
			'all_intersection_genes': set()
		}
		
		# Get intersection sizes and unique genes for each combo
		for combo in combinations:
			intersection_genes = set(gene_sets[combo]['intersection'])
			comparison['intersection_sizes'][combo] = len(intersection_genes)
			comparison['all_intersection_genes'].update(intersection_genes)
		
		# Find genes unique to each combination
		for i, combo1 in enumerate(combinations):
			unique_genes = set(gene_sets[combo1]['intersection'])
			for j, combo2 in enumerate(combinations):
				if i != j:
					unique_genes -= set(gene_sets[combo2]['intersection'])
			comparison['unique_genes'][combo1] = list(unique_genes)
		
		# Find shared genes between pairs of combinations
		for i, combo1 in enumerate(combinations):
			for j, combo2 in enumerate(combinations):
				if i < j:
					shared = set(gene_sets[combo1]['intersection']) & set(gene_sets[combo2]['intersection'])
					comparison['shared_genes'][(combo1, combo2)] = list(shared)
		
		# Create a Venn diagram-like summary (up to 3 combinations)
		if len(combinations) <= 3:
			venn_data = {}
			all_regions = []
			
			if len(combinations) == 2:
				combo1, combo2 = combinations
				set1 = set(gene_sets[combo1]['intersection'])
				set2 = set(gene_sets[combo2]['intersection'])
				
				# Unique to combo1
				venn_data['only_' + str(combo1)] = list(set1 - set2)
				all_regions.append(len(set1 - set2))
				
				# Unique to combo2
				venn_data['only_' + str(combo2)] = list(set2 - set1)
				all_regions.append(len(set2 - set1))
				
				# Shared
				venn_data['shared'] = list(set1 & set2)
				all_regions.append(len(set1 & set2))
				
			elif len(combinations) == 3:
				combo1, combo2, combo3 = combinations
				set1 = set(gene_sets[combo1]['intersection'])
				set2 = set(gene_sets[combo2]['intersection'])
				set3 = set(gene_sets[combo3]['intersection'])
				
				# Unique to each combo
				venn_data['only_' + str(combo1)] = list(set1 - set2 - set3)
				all_regions.append(len(set1 - set2 - set3))
				
				venn_data['only_' + str(combo2)] = list(set2 - set1 - set3)
				all_regions.append(len(set2 - set1 - set3))
				
				venn_data['only_' + str(combo3)] = list(set3 - set1 - set2)
				all_regions.append(len(set3 - set1 - set2))
				
				# Shared between pairs
				venn_data['shared_' + str(combo1) + '_' + str(combo2)] = list((set1 & set2) - set3)
				all_regions.append(len((set1 & set2) - set3))
				
				venn_data['shared_' + str(combo1) + '_' + str(combo3)] = list((set1 & set3) - set2)
				all_regions.append(len((set1 & set3) - set2))
				
				venn_data['shared_' + str(combo2) + '_' + str(combo3)] = list((set2 & set3) - set1)
				all_regions.append(len((set2 & set3) - set1))
				
				# Shared among all
				venn_data['shared_all'] = list(set1 & set2 & set3)
				all_regions.append(len(set1 & set2 & set3))
			
			comparison['venn_data'] = venn_data
			comparison['venn_counts'] = all_regions
		
		return comparison


def apply_benjamini_hochberg(p_values, fdr=0.05):
	"""
	Apply Benjamini-Hochberg correction to p-values
	
	Parameters:
	-----------
	p_values : numpy.ndarray
		2D array of p-values
	fdr : float, optional
		False discovery rate threshold
		
	Returns:
	--------
	numpy.ndarray
		Array of adjusted p-values
	"""
	# Flatten the 2D array
	p_flat = p_values.flatten()
	
	# Get the indices that would sort p_flat
	sorted_indices = np.argsort(p_flat)
	
	# Get ranks (add 1 to make it 1-based)
	ranks = np.empty_like(sorted_indices)
	ranks[sorted_indices] = np.arange(1, len(p_flat) + 1)
	
	# Calculate Benjamini-Hochberg adjusted p-values
	p_adjusted_flat = p_flat * len(p_flat) / ranks
	
	# Cap at 1.0
	p_adjusted_flat = np.minimum(p_adjusted_flat, 1.0)
	
	# Reshape back to original dimensions
	p_adjusted = p_adjusted_flat.reshape(p_values.shape)
	
	return p_adjusted


def plot_expected_intersection_heatmap(total_genes, metric1_name, metric2_name, 
									   percentile_thresholds=np.linspace(0.25, 0.99, 10),
									   ax=None, cmap='Reds', include_text=True,
									  figsize=(5, 4)):
	"""
	"""
	# Create figure and axes if not provided
	if ax is None:
		fig, ax = plt.subplots(figsize=figsize)
	else:
		fig = ax.figure
	
	# Create the grid of percentile thresholds
	X, Y = np.meshgrid(percentile_thresholds, percentile_thresholds)
	
	# Calculate the expected gene counts for each threshold combination
	expected_counts = np.zeros((len(percentile_thresholds), len(percentile_thresholds)))
	
	for i, metric1_percentile in enumerate(percentile_thresholds):
		# Number of genes in first set based on percentile threshold
		metric1_gene_count = int(total_genes * (1 - metric1_percentile))
		
		for j, metric2_percentile in enumerate(percentile_thresholds):
			# Number of genes in second set based on percentile threshold
			metric2_gene_count = int(total_genes * (1 - metric2_percentile))
			
			# Expected intersection under random sampling (hypergeometric expected value)
			expected_counts[i, j] = (metric1_gene_count * metric2_gene_count) / total_genes
	
	# Plot the heatmap
	im = ax.pcolormesh(X, Y, expected_counts, cmap=cmap)
	
	# Add colorbar
	plt.colorbar(im, ax=ax, label='Expected number of genes')
	
	# Set labels and title
	ax.set_xlabel(f'Percentile Threshold')
	ax.set_ylabel(f'Percentile Threshold')
	ax.set_title(f'Expected Intersection Size',
				pad=13)
	
	# Add text annotations if requested
	if include_text:
		for i, y_val in enumerate(percentile_thresholds):
			for j, x_val in enumerate(percentile_thresholds):
				expected_count = expected_counts[i, j]
				# Format as integer
				text = f"{int(expected_count)}"
				
				# Determine text color based on heatmap color intensity
				ax.text(x_val, y_val, text,
					   ha='center', va='center',
					   color='white' if expected_count > np.max(expected_counts)/2 
						else 'black',
					   fontsize=8, fontweight='demi')
	
	return fig, im

def compare_md_and_plot_thresholds(metric_1, metric_2, metric_1_name, metric_2_name,
							   vmax=12):
	"""Plot and compare the threshold sweep for mothers and daughters"""
	from src.chromatin_gene_expression_intersection_2d_analysis import TwoDimensionalPTRAnalysis
	import matplotlib.pyplot as plt
	import numpy as np
	from mpl_toolkits.axes_grid1 import make_axes_locatable

	# Create the plot with a bit more space on the right for the colorbar
	ptr_linspace = np.linspace(0.25, 0.99, 10)
	fig, (ax_m, ax_d) = plt.subplots(1, 2, figsize=(9, 4)) 

	# First subplot (Mother)
	analyzer = TwoDimensionalPTRAnalysis(metric_1_name, metric_2_name, 
		metric_1['ptr_t'], metric_2['ptr_t'], pval_vmax=vmax)
	analyzer.run_2d_analysis(metric1_percentiles=ptr_linspace, 
		metric2_percentiles=ptr_linspace)
	analyzer.plot_pvalue_heatmap(ax_m)
	ax_m.set_title("Mother")

	# Second subplot (Daughter)
	analyzer = TwoDimensionalPTRAnalysis(metric_1_name, metric_2_name, 
			metric_1['ptr_b'], metric_2['ptr_b'], pval_vmax=vmax)
	analyzer.run_2d_analysis(metric1_percentiles=ptr_linspace, 
		metric2_percentiles=ptr_linspace)
	im = analyzer.plot_pvalue_heatmap(ax_d)
	ax_d.set_title("Daughter")
	ax_d.set_yticks([])
	ax_d.set_ylabel('')

	cax = fig.add_axes([ax_d.get_position().x1 + 0.05,
		ax_d.get_position().y0, 
		0.02, ax_d.get_position().height])

	# Add the colorbar using the returned im object
	cbar = fig.colorbar(im, cax=cax)
	cbar.set_label('-log10(p-value)')

	plt.suptitle(f"{metric_1_name} and {metric_2_name}\nGene set intersection, "
				 f"PTR threshold sensitivity", 
		fontsize=16, y=1.1)
