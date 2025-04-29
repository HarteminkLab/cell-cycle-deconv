import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from typing import Tuple, Optional, List, Union, Dict

class OriginConsensusCorrelationAnalysis:
	"""
	Class for analyzing correlations between a consensus origin profile and individual origins,
	with functionality to find optimal time shifts to maximize correlation.
	"""
	
	def __init__(self, deconv_origin_analysis):
		"""
		Initialize the Origin Correlation Analysis class.
		
		Parameters:
		-----------
		composite_data : numpy.ndarray
			Consensus data representing average origin behavior
		origin_histogram_data : numpy.ndarray
			Histogram data for individual origins
		origin_footprint_region : Tuple[int, int, int, int]
			Region defining the origin footprint (x_min, x_max, y_min, y_max)
		origins_df : pd.DataFrame
			DataFrame containing origin metadata
		time_indices : Optional[List[int]]
			Indices of timepoints to use for analysis, if None all timepoints are used
		"""

		self.deconv_origin_analysis = deconv_origin_analysis

	def initialize_data(self):

		# Extract the consensus bins from the composite data
		self.consensus_bins = self.deconv_origin_analysis.subselect_bins_on_centered_region(
			self.deconv_origin_analysis.composite_data,
			self.deconv_origin_analysis.origin_footprint_region
		)
		
		# Extract individual origin bins
		self.origin_bins = self.deconv_origin_analysis.subselect_bins_on_centered_region(
			self.deconv_origin_analysis.origin_histogram_data,
			self.deconv_origin_analysis.origin_footprint_region
		)
		
		# Calculate the mean consensus histogram (averaging over fragment lengths and positions)
		self.consensus_mean = self.consensus_bins.mean((1, 2))
		
		# Initialize results storage
		self.correlations = None
		self.optimal_shifts = None
		self.max_correlations = None

	def plot_distributions(self):
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

		q_threshold_corr = 0.5
		corr_cutoff = np.quantile(self.correlations.correlation, q=q_threshold_corr)
		num_above = np.sum(self.correlations.correlation > corr_cutoff)
		n = len(self.correlations)

		title = f'Consensus correlations\n{num_above} of {n} origins above perc. {q_threshold_corr*100:.0f}'
		self.plot_correlation_histogram(ax1, cutoff=corr_cutoff, title=title)
		self.plot_optimal_shifts_histogram(ax2, threshold=corr_cutoff)
		ax1.set_ylim(0, 21)
		ax2.set_ylim(0, 21)

		plt.tight_layout()
		plt.subplots_adjust(wspace=0.3)

		return fig


	def retrieve_high_correlation_examples(self, cutoff):
		origins = self.deconv_origin_analysis.origins
		shifts = self.optimal_shifts.join(self.correlations).sort_values('optimal_shift').join(origins[['activation_time', 'ars_name']])
		return shifts[shifts.correlation > cutoff]


	def calculate_correlations_3d(self, array1: np.ndarray, array2: np.ndarray, axis: int = 0) -> np.ndarray:
		"""
		Calculate Pearson correlation coefficients between two 3D arrays along specified axis.
		
		Parameters:
		-----------
		array1 : numpy.ndarray
			First 3D input array
		array2 : numpy.ndarray
			Second 3D input array with same shape as array1
		axis : int, default=0
			Axis along which to calculate correlations
		
		Returns:
		--------
		numpy.ndarray
			Array of correlation coefficients
		"""
		# Check that arrays have same shape
		if array1.shape != array2.shape:
			raise ValueError("Input arrays must have the same shape")
		
		# Check that arrays are 3D
		if len(array1.shape) != 3:
			raise ValueError("Input arrays must be 3D")
		
		# Move the correlation axis to the beginning for easier iteration
		array1_swapped = np.moveaxis(array1, axis, 0)
		array2_swapped = np.moveaxis(array2, axis, 0)
		
		# Get dimensions of the result array
		dim1, dim2 = array1.shape[(axis+1) % 3], array1.shape[(axis+2) % 3]
		
		# Initialize result array
		result = np.zeros((dim1, dim2))
		
		# Iterate through x and y dimensions
		for i in range(dim1):
			for j in range(dim2):
				# Extract vectors along the correlation axis
				vec1 = array1_swapped[:, i, j]
				vec2 = array2_swapped[:, i, j]
				
				# Check for zero standard deviation to avoid division by zero
				# Using np.isclose to handle floating point precision issues
				if np.isclose(np.std(vec1), 0) or np.isclose(np.std(vec2), 0):
					result[i, j] = 0
				else:
					result[i, j] = np.corrcoef(vec1, vec2)[0, 1]
		
		return result
	
	def compute_correlations(self, 
							shift_range: Tuple[int, int] = (-10, 10),
							find_optimal_shift: bool = True, time_indices=None) -> Dict[str, Union[np.ndarray, pd.DataFrame]]:
		"""
		Compute correlations between consensus and individual origins,
		optionally finding the optimal shift to maximize correlation.
		
		Parameters:
		-----------
		shift_range : Tuple[int, int], default=(-10, 10)
			Range of time shifts to try (min_shift, max_shift) in minutes
		find_optimal_shift : bool, default=True
			Whether to find the optimal shift for each origin
			
		Returns:
		--------
		Dict containing:
			'correlations': pd.DataFrame of correlation values for each origin
			'optimal_shifts': pd.DataFrame of optimal shift values (if find_optimal_shift=True)
			'max_correlations': pd.DataFrame of maximum correlation values (if find_optimal_shift=True)
		"""

		# Get the shape of the correlation matrix
		n_origins = self.origin_bins.shape[0]
		corr_shape = (self.origin_bins.shape[2], self.origin_bins.shape[3])
		
		# Initialize storage for results
		all_correlations = np.zeros((n_origins, corr_shape[0], corr_shape[1]))
		
		if find_optimal_shift:
			shifts = range(shift_range[0], shift_range[1] + 1)
			optimal_shifts = np.zeros(n_origins, dtype=int)
			max_correlations = np.zeros((n_origins, corr_shape[0], corr_shape[1]))
			
			# For each origin, try different shifts and find the optimal one
			for origin_idx in range(n_origins):
				best_corr = None
				best_shift = 0
				
				for shift in shifts:
					# Roll the origin data along the time axis
					rolled_origin_bins = np.roll(self.origin_bins[origin_idx], shift, axis=0)
					
					# Extract the time indices for both arrays
					consensus_t = self.consensus_bins[time_indices]
					rolled_t = rolled_origin_bins[time_indices]
					
					# Calculate correlation
					corr_mat = self.calculate_correlations_3d(consensus_t, rolled_t)
					
					# Check if this is the best correlation so far
					if best_corr is None or np.mean(corr_mat) > np.mean(best_corr):
						best_corr = corr_mat
						best_shift = shift
				
				# Store the results
				optimal_shifts[origin_idx] = best_shift
				max_correlations[origin_idx] = best_corr
		
		else:
			# Just calculate correlations with no shift
			for origin_idx in range(n_origins):
				consensus_t = self.consensus_bins[time_indices]
				origin_t = self.origin_bins[origin_idx, time_indices]
				
				all_correlations[origin_idx] = self.calculate_correlations_3d(consensus_t, origin_t)
		
		# Convert results to DataFrames
		corr_df = pd.DataFrame(
			[np.mean(c) for c in (max_correlations if find_optimal_shift else all_correlations)],
			index=self.deconv_origin_analysis.origins.index,
			columns=['correlation']
		)
		
		# Store the results
		self.correlations = corr_df
		
		result = {'correlations': corr_df}
		
		if find_optimal_shift:
			self.optimal_shifts = pd.DataFrame(
				optimal_shifts,
				index=self.deconv_origin_analysis.origins.index,
				columns=['optimal_shift']
			)
			self.max_correlations = max_correlations
			
			result['optimal_shifts'] = self.optimal_shifts
			result['max_correlations'] = self.max_correlations
		
		return result
	
	def plot_origin_correlation_heatmap(self, 
									  sort_by_correlation: bool = False, 
									  figsize: Tuple[int, int] = (4, 4),
									  time_indices=None) -> None:
		"""
		Plot a heatmap of origins sorted by their correlation with the consensus.
		
		Parameters:
		-----------
		sort_by_correlation : bool, default=True
			Whether to sort origins by their correlation values
		figsize : Tuple[int, int], default=(10, 10)
			Figure size
		"""
		if self.correlations is None:
			raise ValueError("Correlations have not been computed yet. Call compute_correlations() first.")
		
		# Calculate the mean occupancy over time for each origin
		mean_occupancy = self.origin_bins.mean((2, 3))[:, time_indices]
		
		# Create DataFrame
		origins = self.deconv_origin_analysis.origins.join(self.correlations)
		occupancy_df = pd.DataFrame(mean_occupancy, index=origins.index)

		plot_data = occupancy_df
		normalized_data = plot_data.copy()
		normalized_data.loc[:] = plot_data.values-plot_data.values.mean(1)[:, None]
		early_indices = origins[origins.activation_time == 'early'].index
		late_indices = origins[origins.activation_time == 'late'].index

		import matplotlib.gridspec as gridspec

		figsize = (4, 6)

		n = len(origins)
		row_heights = [len(early_indices)/n, len(late_indices)/n]
		nrows = 2
		ncols = 2

		# Create figure
		fig = plt.figure(figsize=figsize)

		# Create GridSpec with specified row heights
		gs = gridspec.GridSpec(nrows, ncols, 
			height_ratios=row_heights)

		def plot_im(plot_data, vlims, cmap):
			plt.imshow(plot_data, interpolation='none', cmap=cmap, vmin=vlims[0], 
				vmax=vlims[1],aspect='auto')
			plt.xticks([])
			plt.yticks([])

		plot_cols = [plot_data, normalized_data]
		row_indices = [early_indices, late_indices]
		plt_col_specs = [('magma_r', (0, 7)), ('RdBu_r', (-1.5, 1.5))]

		for row_idx in range(nrows):
			for col_idx in range(ncols):
				cmap, vlims = plt_col_specs[col_idx]
				plt.subplot(gs[row_idx, col_idx])
				plot_im(plot_cols[col_idx].loc[row_indices[row_idx]],
					vlims=vlims, cmap=cmap)

				if row_idx == 0:
					if col_idx == 0:
						plt.title("Deconvolved\noccupancy")
						plt.ylabel(f"Early firing, n={len(early_indices)}")
					elif col_idx == 1:
						plt.title("Normalized\noccupancy")
				elif row_idx == 1:
					plt.xlabel('Average single cell time')

				if col_idx == 0:
					if row_idx == 1:
						plt.ylabel(f"Late firing, n={len(late_indices)}")

		plt.suptitle('Origin occupancy over time', fontsize=16, fontweight='demi')
		plt.tight_layout()
		
	def plot_correlation_histogram(self, ax=None, bins: int = 12, cutoff=None,
		title=None) -> None:
		"""
		Plot a histogram of correlation values with early and late side-by-side.
		"""
		if self.correlations is None:
			raise ValueError("Correlations have not been computed yet. Call compute_correlations() first.")

		if ax is None:
			plt.figure(figsize=(6, 3))  # Wider figure to accommodate side-by-side
			ax = plt.gca()
		
		# Join with origins to get activation time
		origins = self.deconv_origin_analysis.origins
		correlations = self.correlations.copy()
		correlations = correlations.join(origins[['activation_time']])
		
		# Split by activation time
		early_corrs = correlations[correlations.activation_time == 'early']
		late_corrs = correlations[correlations.activation_time == 'late']
		
		# Get correlation values
		early_corr_values = early_corrs['correlation'].values
		late_corr_values = late_corrs['correlation'].values
		
		# Define bin edges for both datasets
		all_corrs = correlations['correlation'].values
		bin_edges = np.linspace(min(all_corrs), max(all_corrs), bins+1)
		
		# Calculate histogram values
		early_counts, _ = np.histogram(early_corr_values, bins=bin_edges)
		late_counts, _ = np.histogram(late_corr_values, bins=bin_edges)
		
		# Calculate bar positions
		bar_positions = (bin_edges[:-1] + bin_edges[1:]) / 2  # Center positions
		bar_width = 0.8 * (bin_edges[1] - bin_edges[0]) / 2  # Bar width for side-by-side
		early_positions = bar_positions - bar_width/2
		late_positions = bar_positions + bar_width/2
		
		# Plot early and late bars side by side
		ax.bar(early_positions, early_counts, width=bar_width, color='red', alpha=0.6,
			   label=f'Early, n={len(early_corrs)}')
		ax.bar(late_positions, late_counts, width=bar_width, color='blue', alpha=0.6,
			   label=f'Late, n={len(late_corrs)}')
		
		ax.legend()
		ax.set_xlabel('Correlation with consensus')
		ax.set_ylabel('Number of origins')
		
		# Use provided title or create default
		if title is None:
			title = f'Correlations, n={len(correlations)}'
		ax.set_title(title, y=1.04, fontweight='demi', fontsize=13)

		# Add cutoff line if provided
		if cutoff is not None:
			ax.axvline(cutoff, c='orange', lw=1.5, alpha=1, linestyle='solid', label=f'Cutoff: {cutoff:.2f}')
			
	def plot_optimal_shifts_histogram(self, ax=None, threshold=None) -> None:
		"""
		Plot a histogram of optimal shift values with early and late side-by-side.
		
		Parameters:
		-----------
		figsize : Tuple[int, int], default=(8, 6)
			Figure size
		"""
		if self.optimal_shifts is None:
			raise ValueError("Optimal shifts have not been computed yet. Call compute_correlations() with find_optimal_shift=True first.")

		if ax is None:
			plt.figure(figsize=(6, 3))  # Wider figure to accommodate side-by-side
			ax = plt.gca()

		origins = self.deconv_origin_analysis.origins
		optimal_shifts = self.correlations.join(self.optimal_shifts)
		optimal_shifts = optimal_shifts[optimal_shifts.correlation > threshold]
		optimal_shifts = optimal_shifts.join(origins[['activation_time']])

		early_shifts = optimal_shifts[optimal_shifts.activation_time == 'early']
		late_shifts = optimal_shifts[optimal_shifts.activation_time == 'late']
		
		# Define bins consistently for both histograms
		shift_vals = optimal_shifts.optimal_shift
		bins = np.arange(min(shift_vals)-0.5, max(shift_vals)+1.5)
		
		# Set the bar width for side-by-side
		bar_width = 0.4
		
		# Calculate histogram values but don't plot yet
		early_counts, edges = np.histogram(early_shifts.optimal_shift, bins=bins)
		late_counts, _ = np.histogram(late_shifts.optimal_shift, bins=bins)
		
		# Calculate bar positions
		bar_positions = (edges[:-1] + edges[1:]) / 2  # Center positions
		early_positions = bar_positions - bar_width/2
		late_positions = bar_positions + bar_width/2
		
		# Plot early and late bars side by side
		ax.bar(early_positions, early_counts, width=bar_width, color='red', alpha=0.6,
			   label=f'Early firing, n={len(early_shifts)}')
		ax.bar(late_positions, late_counts, width=bar_width, color='blue', alpha=0.6,
			   label=f'Late firing, n={len(late_shifts)}')
		
		plt.legend()

		ax.set_xlabel('Optimal shift')
		ax.set_ylabel('Number of origins')
		ax.set_title(f'High correlation shifts, n={len(optimal_shifts)}', 
			y=1.06, fontweight='demi', fontsize=13)
		
		# Set x-ticks to be at the bin centers
		ax.set_xticks(bar_positions)
		ax.set_xticklabels([int(pos) for pos in bar_positions])

		
	def get_top_correlated_origins(self, n: int = 10) -> pd.DataFrame:
		"""
		Get the top n origins with highest correlation to the consensus.
		
		Parameters:
		-----------
		n : int, default=10
			Number of top origins to return
			
		Returns:
		--------
		pd.DataFrame
			DataFrame containing the top correlated origins
		"""
		if self.correlations is None:
			raise ValueError("Correlations have not been computed yet. Call compute_correlations() first.")
		
		# Sort by correlation and return top n
		return self.correlations.sort_values('correlation', ascending=False).head(n)
	
	def plot_individual_origin(self, 
							 origin_idx: Union[int, str], 
							 timepoint: Optional[int] = None,
							 figsize: Tuple[int, int] = (10, 5)) -> None:
		"""
		Plot the heatmap for an individual origin at a specific timepoint.
		
		Parameters:
		-----------
		origin_idx : Union[int, str]
			Index or name of the origin to plot
		timepoint : Optional[int], default=None
			Timepoint to plot, if None plots the average over all timepoints
		figsize : Tuple[int, int], default=(10, 5)
			Figure size
		"""
		# Convert string index to position if needed
		if isinstance(origin_idx, str):
			if origin_idx in self.deconv_origin_analysis.origins.index:
				idx = self.deconv_origin_analysis.origins.index.get_loc(origin_idx)
			else:
				raise ValueError(f"Origin '{origin_idx}' not found in origins DataFrame")
		else:
			idx = origin_idx
			
		# Get the origin data
		if timepoint is not None:
			origin_data = self.origin_bins[idx, timepoint]
			title = f"Origin {self.deconv_origin_analysis.origins.index[idx]} at Timepoint {timepoint}"
		else:
			origin_data = self.origin_bins[idx].mean(axis=0)
			title = f"Origin {self.deconv_origin_analysis.origins.index[idx]} (Mean Over Time)"
			
		# Plot
		plt.figure(figsize=figsize)
		plt.imshow(origin_data, cmap='magma_r', origin='lower')
		plt.colorbar(label='Occupancy')
		plt.title(title)
		plt.xlabel('Position')
		plt.ylabel('Fragment Length')
		plt.tight_layout()
		
	def plot_consensus_vs_origin(self, 
							  origin_id, figsize: Tuple[int, int] = (6, 3), 
							  time_indices=None, title=None):
		"""
		Plot the time series of consensus vs. an individual origin.
		"""

		# Convert string index to position if needed
		origins = self.deconv_origin_analysis.origins.copy()
		origins['np_index'] = np.arange(len(origins))
		origin = origins.loc[origin_id]
		idx = origin.np_index
			
		# Get the time series data
		consensus_ts = self.consensus_mean[time_indices]
		origin_ts = self.origin_bins[idx].mean(axis=(1, 2))[time_indices]
		
		# Apply optimal shift if requested
		shift = self.optimal_shifts.iloc[idx]['optimal_shift']
		rolled_origin_ts = np.roll(origin_ts, shift)
		full_title = "$\\it{" + origins.iloc[idx].ars_name + "}$, " + title
			
		# Plot
		plt.figure(figsize=figsize)
		plt.plot(consensus_ts, label='Consensus', linewidth=2)
		plt.xlabel('Average single cell time')
		plt.ylabel('Consensus Occupancy')
		plt.title(full_title, y=1.05, fontsize=12, fontweight='demi')
		plt.legend()

		ax = plt.gca()
		twin_ax = ax.twinx()

		twin_ax.plot(origin_ts, label=f'Origin occupancy', linewidth=2, alpha=0.7,
			c='orange')
		twin_ax.plot(rolled_origin_ts, label=f'Optimal shift, {shift}', 
			linewidth=2, alpha=0.7, c='orange', ls='dotted')
		twin_ax.set_ylabel("Origin occupancy")

		plt.legend()

		plt.tight_layout()

