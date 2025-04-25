import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from src.timer import Timer
from src.mnase_10kb_loader import MNase10kbLoader


class PrecomputationOriginAnalysis:
	"""
	A class to analyze origins of replication using MNase-seq data, with a focus on
	small fragment footprints versus nucleosomal occupancy.
	"""
	
	def __init__(self, window_size=200):
		"""
		Initialize the PrecomputationOriginAnalysis
		
		Parameters:
		-----------
		window_size : int
			Size of the window around each origin (in bp)
		"""
		self.timer = Timer()
		self.mnase_loader = MNase10kbLoader()
		self.window_size = window_size
		self.origins = None
		self.origin_metrics = None
		self.small_fragment_curve = None
		
	def load_origin_reference_dataset(self, origin_file=None):
		"""
		Load the origin reference dataset
		
		Parameters:
		-----------
		origin_file : str or None
			Path to the origin reference file. If None, uses default.
		
		Returns:
		--------
		self : PrecomputationOriginAnalysis
			Returns self for method chaining
		"""
		# This part will depend on your specific data format
		# Placeholder implementation
		if origin_file is None:
			# Load default origins file (implementation depends on your data)
			# Example:
			# from src.reference_data import read_origins
			# self.origins = read_origins()
			raise NotImplementedError("Please specify an origin file or implement default loading")
		else:
			# Load the specified origin file
			self.origins = pd.read_csv(origin_file)
			
			# Ensure required columns exist
			required_cols = ['chr', 'position', 'name']
			for col in required_cols:
				if col not in self.origins.columns:
					raise ValueError(f"Origin dataset must contain column: {col}")
			
			# Add mid point if not present
			if 'mid' not in self.origins.columns:
				self.origins['mid'] = self.origins['position']
				
		print(f"Loaded {len(self.origins)} origins of replication")
		return self
	
	def load_small_fragment_curve(self, curve_file=None):
		"""
		Load the small fragment Gaussian curve parameters from file or 
		use predefined parameters
		
		Parameters:
		-----------
		curve_file : str or None
			Path to the file containing Gaussian parameters for small fragments
			
		Returns:
		--------
		self : PrecomputationOriginAnalysis
			Returns self for method chaining
		"""
		if curve_file is None:
			# Use predefined parameters from previous analysis
			# For example, from the ABF1 analysis you completed
			self.small_fragment_loc = 80.0  # Example value from your ABF1 analysis
			self.small_fragment_scale = 19.5  # Example value from your ABF1 analysis
		else:
			# Load parameters from file
			params = pd.read_csv(curve_file)
			self.small_fragment_loc = params['loc'][0]
			self.small_fragment_scale = params['scale'][0]
			
		# Create the selection curve for small fragments
		fragment_lengths = np.arange(0, 250)  # Adjust range as needed
		weights = norm.pdf(fragment_lengths, loc=self.small_fragment_loc, scale=self.small_fragment_scale)
		self.small_fragment_curve = pd.DataFrame({
			'length': fragment_lengths,
			'weight': weights
		})
		
		# Zero out values outside the small fragment range
		small_range = (0, 100)  # Adjust as needed
		self.small_fragment_curve.loc[self.small_fragment_curve['length'] < small_range[0], 'weight'] = 0
		self.small_fragment_curve.loc[self.small_fragment_curve['length'] > small_range[1], 'weight'] = 0
		
		# Normalize weights
		non_zero_mean = self.small_fragment_curve.loc[self.small_fragment_curve['weight'] > 0, 'weight'].mean()
		if non_zero_mean > 0:
			self.small_fragment_curve['weight'] = self.small_fragment_curve['weight'] / non_zero_mean
			
		print(f"Loaded small fragment curve (μ={self.small_fragment_loc:.1f}, σ={self.small_fragment_scale:.1f})")
		return self
		
	def compute_origin_metrics(self, replicates=None):
		"""
		Compute metrics for each origin of replication
		
		Parameters:
		-----------
		replicates : list or None
			List of replicates to use. If None, uses all available replicates.
			
		Returns:
		--------
		self : PrecomputationOriginAnalysis
			Returns self for method chaining
		"""
		if self.origins is None:
			raise ValueError("Origin reference dataset not loaded. Call load_origin_reference_dataset first.")
			
		if self.small_fragment_curve is None:
			raise ValueError("Small fragment curve not loaded. Call load_small_fragment_curve first.")
		
		if replicates is None:
			replicates = [1, 2]  # Default to using both replicates
			
		# Initialize metrics DataFrame
		self.origin_metrics = self.origins.copy()
		
		# Add columns for metrics
		metrics_columns = [
			'total_reads', 'coverage', 'small_reads', 'nucleosomal_reads',
			'small_proportion', 'peak_position', 'peak_height'
		]
		
		for col in metrics_columns:
			self.origin_metrics[col] = np.nan
			
		# Add histograms column (will store as lists)
		self.origin_metrics['read_histogram'] = [[] for _ in range(len(self.origin_metrics))]
		
		# Process each origin
		total_origins = len(self.origin_metrics)
		self.timer.start()
		
		print(f"Computing metrics for {total_origins} origins...")
		
		for i, (_, origin) in enumerate(self.origin_metrics.iterrows()):
			if i % 100 == 0:
				print(f"Processing origin {i+1}/{total_origins} - {self.timer.elapsed_time()}")
				
			# Get chromosome and position
			chrom = origin['chr']
			mid = origin['mid']
			
			# Define window around origin
			half_window = self.window_size // 2
			window_start = mid - half_window
			window_end = mid + half_window
			
			# Initialize counters for this origin
			total_reads = 0
			small_reads = 0
			nucleosomal_reads = 0
			
			# Initialize histogram for position distribution
			position_hist = np.zeros(self.window_size + 1)
			
			# Process each replicate
			for replicate in replicates:
				# Load MNase data for this chromosome
				mnase_data = self.mnase_loader.load_mnase_data(replicate, chrom)
				
				# Filter reads in the window
				window_reads = mnase_data[
					(mnase_data['mid'] >= window_start) & 
					(mnase_data['mid'] <= window_end)
				]
				
				if len(window_reads) == 0:
					continue
				
				# Count total reads
				total_reads += len(window_reads)
				
				# Classify reads as small or nucleosomal based on their length
				for _, read in window_reads.iterrows():
					read_length = read['length']
					read_pos = read['mid'] - window_start  # Position relative to window start
					
					# Update position histogram
					if 0 <= read_pos <= self.window_size:
						position_hist[read_pos] += 1
					
					# Get weight from small fragment curve
					if read_length in self.small_fragment_curve['length'].values:
						weight = self.small_fragment_curve.loc[
							self.small_fragment_curve['length'] == read_length, 
							'weight'
						].values[0]
					else:
						weight = 0
						
					# Count small and nucleosomal reads
					if weight > 0:
						small_reads += weight
						nucleosomal_reads += (1 - weight)
					else:
						nucleosomal_reads += 1
			
			# Skip origins with no reads
			if total_reads == 0:
				continue
				
			# Calculate metrics
			coverage = total_reads / self.window_size
			small_proportion = small_reads / total_reads if total_reads > 0 else 0
			
			# Find peak position
			if np.sum(position_hist) > 0:
				peak_position = np.argmax(position_hist)
				peak_height = position_hist[peak_position]
				
				# Convert peak position to be relative to origin midpoint
				peak_position = peak_position - half_window
			else:
				peak_position = 0
				peak_height = 0
				
			# Update metrics in DataFrame
			self.origin_metrics.loc[i, 'total_reads'] = total_reads
			self.origin_metrics.loc[i, 'coverage'] = coverage
			self.origin_metrics.loc[i, 'small_reads'] = small_reads
			self.origin_metrics.loc[i, 'nucleosomal_reads'] = nucleosomal_reads
			self.origin_metrics.loc[i, 'small_proportion'] = small_proportion
			self.origin_metrics.loc[i, 'peak_position'] = peak_position
			self.origin_metrics.loc[i, 'peak_height'] = peak_height
			
			# Store histogram
			self.origin_metrics.at[i, 'read_histogram'] = position_hist.tolist()
			
		self.timer.print_time(f"Completed metrics calculation for {total_origins} origins")
		
		# Remove origins with no reads
		no_reads_mask = self.origin_metrics['total_reads'].isna() | (self.origin_metrics['total_reads'] == 0)
		print(f"Removing {no_reads_mask.sum()} origins with no reads")
		self.origin_metrics = self.origin_metrics[~no_reads_mask].reset_index(drop=True)
		
		return self
	
	def plot_total_occupancy_distribution(self):
		"""
		Plot the distribution of total occupancies to determine a threshold
		for low coverage origins
		
		Returns:
		--------
		fig : matplotlib.figure.Figure
			The figure containing the plot
		"""
		if self.origin_metrics is None:
			raise ValueError("Origin metrics not computed. Call compute_origin_metrics first.")
			
		fig, ax = plt.subplots(figsize=(10, 6))
		
		# Plot histogram on log scale
		ax.hist(self.origin_metrics['coverage'], bins=50)
		ax.set_xlabel('Coverage (reads per bp)')
		ax.set_ylabel('Number of origins')
		ax.set_title('Distribution of origin coverage')
		
		# Add log scale for coverage
		ax.set_xscale('log')
		
		return fig
	
	def plot_small_fragment_proportion_distribution(self):
		"""
		Plot the distribution of small fragment proportions to determine
		a threshold for origin footprint identification
		
		Returns:
		--------
		fig : matplotlib.figure.Figure
			The figure containing the plot
		"""
		if self.origin_metrics is None:
			raise ValueError("Origin metrics not computed. Call compute_origin_metrics first.")
			
		fig, ax = plt.subplots(figsize=(10, 6))
		
		# Plot histogram
		ax.hist(self.origin_metrics['small_proportion'], bins=50)
		ax.set_xlabel('Proportion of small fragments')
		ax.set_ylabel('Number of origins')
		ax.set_title('Distribution of small fragment proportions')
		
		return fig
	
	def determine_thresholds(self, coverage_quantile=0.25, small_prop_quantile=0.75):
		"""
		Determine thresholds for coverage and small fragment proportion
		based on their distributions
		
		Parameters:
		-----------
		coverage_quantile : float
			Quantile to use for coverage threshold (e.g., 0.25 for bottom 25%)
		small_prop_quantile : float
			Quantile to use for small fragment proportion threshold (e.g., 0.75 for top 25%)
			
		Returns:
		--------
		thresholds : dict
			Dictionary with the determined thresholds
		"""
		if self.origin_metrics is None:
			raise ValueError("Origin metrics not computed. Call compute_origin_metrics first.")
			
		# Determine coverage threshold
		coverage_threshold = self.origin_metrics['coverage'].quantile(coverage_quantile)
		
		# Determine small fragment proportion threshold
		small_prop_threshold = self.origin_metrics['small_proportion'].quantile(small_prop_quantile)
		
		thresholds = {
			'coverage_threshold': coverage_threshold,
			'small_proportion_threshold': small_prop_threshold
		}
		
		print(f"Determined thresholds:")
		print(f"  Coverage threshold: {coverage_threshold:.2f} reads per bp")
		print(f"  Small fragment proportion threshold: {small_prop_threshold:.2f}")
		
		return thresholds

	def compute_small_fragment_peak(self):
		"""
		Compute the position of the peak small fragment signal for each origin
		
		This method analyzes reads based on their length and the small fragment
		weight curve to identify where small fragments are most concentrated.
		
		Returns:
		--------
		self : PrecomputationOriginAnalysis
			Returns self for method chaining
		"""
		if self.origin_metrics is None:
			raise ValueError("Origin metrics not computed. Call compute_origin_metrics first.")
			
		if self.small_fragment_curve is None:
			raise ValueError("Small fragment curve not loaded. Call load_small_fragment_curve first.")
		
		# Add columns for small fragment peak metrics
		self.origin_metrics['small_fragment_peak_position'] = np.nan
		self.origin_metrics['small_fragment_peak_height'] = np.nan
		self.origin_metrics['small_fragment_histogram'] = [[] for _ in range(len(self.origin_metrics))]
		
		# Process each origin
		total_origins = len(self.origin_metrics)
		
		print(f"Computing small fragment peak positions for {total_origins} origins...")
		
		for i, (_, origin) in enumerate(self.origin_metrics.iterrows()):
			if i % 100 == 0:
				print(f"Processing origin {i+1}/{total_origins}")
				
			# Skip origins with no reads
			if origin['total_reads'] == 0 or pd.isna(origin['total_reads']):
				continue
				
			# Get chromosome and position
			chrom = origin['chr']
			mid = origin['mid']
			
			# Define window around origin
			half_window = self.window_size // 2
			window_start = mid - half_window
			window_end = mid + half_window
			
			# Initialize histogram for small fragment position distribution
			small_fragment_hist = np.zeros(self.window_size + 1)
			
			# Process each replicate
			for replicate in [1, 2]:  # Use both replicates by default
				# Load MNase data for this chromosome
				mnase_data = self.mnase_loader.load_mnase_data(replicate, chrom)
				
				# Filter reads in the window
				window_reads = mnase_data[
					(mnase_data['mid'] >= window_start) & 
					(mnase_data['mid'] <= window_end)
				]
				
				if len(window_reads) == 0:
					continue
				
				# Process each read
				for _, read in window_reads.iterrows():
					read_length = read['length']
					read_pos = int(read['mid'] - window_start)  # Position relative to window start
					
					# Skip reads outside the histogram range
					if read_pos < 0 or read_pos >= len(small_fragment_hist):
						continue
					
					# Get weight from small fragment curve
					if read_length in self.small_fragment_curve['length'].values:
						weight = self.small_fragment_curve.loc[
							self.small_fragment_curve['length'] == read_length, 
							'weight'
						].values[0]
					else:
						weight = 0
						
					# Only count reads with positive weight (small fragments)
					if weight > 0:
						small_fragment_hist[read_pos] += weight
			
			# Find peak in the small fragment histogram
			if np.sum(small_fragment_hist) > 0:
				# Apply smoothing to reduce noise
				from scipy.ndimage import gaussian_filter1d
				smoothed_hist = gaussian_filter1d(small_fragment_hist, sigma=3)
				
				peak_position = np.argmax(smoothed_hist)
				peak_height = smoothed_hist[peak_position]
				
				# Convert peak position to be relative to origin midpoint
				peak_position = peak_position - half_window
			else:
				peak_position = 0
				peak_height = 0
				
			# Update metrics in DataFrame
			self.origin_metrics.loc[i, 'small_fragment_peak_position'] = peak_position
			self.origin_metrics.loc[i, 'small_fragment_peak_height'] = peak_height
			self.origin_metrics.at[i, 'small_fragment_histogram'] = small_fragment_hist.tolist()
		
		# Calculate absolute offset from origin center
		self.origin_metrics['small_fragment_offset'] = np.abs(self.origin_metrics['small_fragment_peak_position'])
		
		print(f"Completed small fragment peak analysis")
		
		# Summary statistics
		mean_offset = self.origin_metrics['small_fragment_offset'].mean()
		median_offset = self.origin_metrics['small_fragment_offset'].median()
		
		print(f"Small fragment peak summary:")
		print(f"  Mean absolute offset from origin center: {mean_offset:.2f} bp")
		print(f"  Median absolute offset from origin center: {median_offset:.2f} bp")
		
		return self
		