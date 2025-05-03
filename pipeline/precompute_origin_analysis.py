import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.timer import Timer
from src.config import load_default_chrom_configs
from src.origins import load_origins
from src.combined_chromatin_model import CombinedChromatinModel

class OriginFootprintAnalysis:
	"""
	A class to analyze origins of replication using MNase-seq data,
	leveraging the CombinedChromatinModel for normalization.
	"""
	
	def __init__(self, window_size=1000, chromosomes=np.arange(1, 17)):
		"""
		Initialize the OriginFootprintAnalysis
		
		Parameters:
		-----------
		window_size : int
			Size of the window around each origin (in bp)
		chromosomes : array-like
			List of chromosomes to process
		"""
		self.timer = Timer()
		self.chromosomes = chromosomes
		self.window_size = window_size
		self.half_window = window_size // 2
		
		# Load default configurations
		self.config1, self.config2 = load_default_chrom_configs()
		
		# Initialize combined model with configurations
		self.combined_model = CombinedChromatinModel(
			config1=self.config1, 
			config2=self.config2
		)
		
		# Get all timepoints for each replicate from the model
		self.wt1_timepoints = self.config1.timepoints
		self.wt2_timepoints = self.config2.timepoints
		
		# Default timepoints for early S-phase
		self.default_s_phase_timepoints = {
			1: 30,
			2: 20
		}
		
		self.origins = None
		
		# Coverage tracking DataFrame
		self.origin_coverage = None
		
		# Member variables to store data
		# Small fragment data: indexed by replicate -> timepoint -> orf -> fragment counts
		self.small_fragments_data = {
			1: {},  # For replicate 1
			2: {}   # For replicate 2
		}
		
		# Composite data: indexed by replicate -> timepoint -> 2D histogram
		self.composite_data = {
			1: {},  # For replicate 1
			2: {}   # For replicate 2
		}
		
		# Separate storage for composite counts
		self.composite_counts = {
			1: {},  # For replicate 1
			2: {}   # For replicate 2
		}
	
	def load_origin_reference_dataset(self):
		"""Load the origin reference dataset"""
		self.origins = load_origins(full=True)
		
		# Initialize coverage DataFrame
		self.origin_coverage = pd.DataFrame(
			index=self.origins.index,
			columns=["coverage_rep1", "coverage_rep2"]
		)
		self.origin_coverage.fillna(0, inplace=True)
		
		return self.origins
	
	def get_normalized_histogram_for_origin(self, origin, replicate):
		"""
		Get normalized MNase histogram data for a specific origin using the combined model
		
		Parameters:
		-----------
		origin : pandas.Series
			Origin data from the origins DataFrame
		replicate : int
			Replicate (1 or 2)
			
		Returns:
		--------
		numpy.ndarray
			3D array of normalized histogram data [timepoint, length, position]
		"""
		chrom = origin['chr']
		mid = origin['pos']
		mnase_span = (mid - self.half_window, mid + self.half_window+1)
		
		# Load MNase data for this span
		self.combined_model.load_mnase_span(chrom, mnase_span, verbose=False)
		
		# Get the appropriate model based on replicate
		model = self.combined_model.chrom1_model if replicate == 1 else self.combined_model.chrom2_model
		
		# Return the normalized histogram data
		return model.exact_bins
	
	def calculate_origin_coverage(self, origin_idx, replicate):
		"""
		Calculate the coverage for a specific origin and replicate
		
		Parameters:
		-----------
		origin_idx : int
			Index of the origin in the origins DataFrame
		replicate : int
			Replicate (1 or 2)
			
		Returns:
		--------
		int
			Number of bases covered by at least one read
		"""
		origin = self.origins.loc[origin_idx]
		chrom = origin['chr']
		mid = origin['pos']
		
		# Define the span for this origin
		span = (mid - self.half_window, mid + self.half_window + 1)
		
		# Get the appropriate raw data from the model
		model = self.combined_model.chrom1_model if replicate == 1 else self.combined_model.chrom2_model
		
		# Create a coverage array for the window
		window_size = span[1] - span[0]-1
		coverage_array = np.zeros(window_size, dtype=bool)
		
		# For each timepoint, update the coverage array
		for timepoint_idx in range(len(model.exact_bins)):
			# Get 2D histogram for this timepoint
			hist_2d = model.exact_bins[timepoint_idx]
			
			# If any fragment covers a position, mark it as covered
			position_coverage = np.any(hist_2d > 0, axis=0)
			coverage_array = np.logical_or(coverage_array, position_coverage)
		
		# Count the number of covered bases
		coverage = np.sum(coverage_array)
		
		return coverage
	
	def generate_small_fragments_for_chromosome(self, chrom, replicate, frag_sel=(0, 100), strand_correct=True):
		"""
		Generate small fragment summaries for all origins on a chromosome, for all timepoints
		
		Parameters:
		-----------
		chrom : str
			Chromosome name
		replicate : int
			Replicate (1 or 2)
		frag_sel : tuple
			Fragment size selection range (min, max)
		strand_correct : bool
			Whether to flip the data for origins on the Crick strand
			
		Returns:
		--------
		None (updates member variables in place)
		"""
		if self.origins is None:
			self.load_origin_reference_dataset()
			
		# Get all origins on this chromosome
		origins_on_chrom = self.origins[self.origins['chr'] == chrom]
		
		if len(origins_on_chrom) == 0:
			return
			
		print(f"  Processing chromosome {chrom} ({len(origins_on_chrom)} origins) - {self.timer.get_time()}")
		
		# Get all timepoints for this replicate
		timepoints = self.wt1_timepoints if replicate == 1 else self.wt2_timepoints
		
		# Process each origin on this chromosome
		for idx, origin in origins_on_chrom.iterrows():
			# Get strand for strand correction
			strand = origin['strand']
			
			# Get normalized histogram data for this origin
			normalized_histograms = self.get_normalized_histogram_for_origin(origin, replicate)
			
			# Calculate coverage for this origin and update the coverage DataFrame
			coverage = self.calculate_origin_coverage(idx, replicate)
			coverage_col = f"coverage_rep{replicate}"
			self.origin_coverage.loc[idx, coverage_col] = coverage

			# Skip origins with low coverage from being included in the dataset
			if coverage < 0.9: continue
			
			# For each timepoint
			for i, timepoint in enumerate(timepoints):
				# Skip if already processed
				if (timepoint in self.small_fragments_data[replicate] and 
					idx in self.small_fragments_data[replicate][timepoint]):
					continue
					
				# Initialize timepoint dictionary if needed
				if timepoint not in self.small_fragments_data[replicate]:
					self.small_fragments_data[replicate][timepoint] = {}
				
				# Initialize composite data and counts if needed
				if timepoint not in self.composite_data[replicate]:
					self.composite_data[replicate][timepoint] = None
					self.composite_counts[replicate][timepoint] = 0
				
				# Get histogram for this timepoint
				hist_2d = normalized_histograms[i]
				
				# Calculate small fragment mean
				small_fragment_mean = hist_2d[frag_sel[0]:frag_sel[1], :].mean(axis=0)
				
				# Apply strand correction if needed
				if strand_correct and strand == '-':
					small_fragment_mean = np.flip(small_fragment_mean)
				
				# Store in member variable
				self.small_fragments_data[replicate][timepoint][idx] = small_fragment_mean
				
				# For composite data, flip the histogram if on Crick strand
				hist_for_composite = hist_2d.copy()
				if strand_correct and strand == '-':
					hist_for_composite = np.flip(hist_for_composite, axis=1)
				
				# Add to composite (will normalize later)
				if self.composite_data[replicate][timepoint] is None:
					self.composite_data[replicate][timepoint] = hist_for_composite
				else:
					self.composite_data[replicate][timepoint] += hist_for_composite
				
				self.composite_counts[replicate][timepoint] += 1

	def finalize_composite_data(self):
		"""
		Normalize all composite histograms by the number of origins
		"""
		for replicate in [1, 2]:
			for timepoint in self.composite_data[replicate].keys():
				# Skip if no data for this timepoint
				if self.composite_data[replicate][timepoint] is None:
					continue
				
				# Normalize by count
				count = self.composite_counts[replicate][timepoint]
				if count > 0:
					self.composite_data[replicate][timepoint] /= count
	
	def generate_summary_histogram_data(self, frag_sel=(0, 100), debug=False, strand_correct=True):
		"""
		Generate small fragment summaries and composite data for all origins, timepoints, and replicates
		
		Parameters:
		-----------
		frag_sel : tuple
			Fragment size selection range (min, max)
		debug : bool
			If True, only process chromosomes 1-4 for faster testing
		strand_correct : bool
			Whether to flip the data for origins on the Crick strand
			
		Returns:
		--------
		tuple
			(small_fragments_data, composite_data) - references to the member variables
		"""
		if self.origins is None:
			self.load_origin_reference_dataset()
			
		self.timer.start()

		print(f"Generating summary histogram data for all timepoints and replicates...")
		
		if debug:
			print(f"Debug mode, only loading chromosomes 1-4")
		
		# Process replicate 1
		print(f"Processing replicate 1...")
		for chrom in self.chromosomes:
			self.generate_small_fragments_for_chromosome(chrom, 1, frag_sel, strand_correct)
			if debug and chrom == 4: break
		
		# Process replicate 2
		print(f"Processing replicate 2...")
		for chrom in self.chromosomes:
			self.generate_small_fragments_for_chromosome(chrom, 2, frag_sel, strand_correct)
			if debug and chrom == 4: break

		# Normalize composite data
		self.finalize_composite_data()
		
		self.timer.print_time(f"Completed summary histogram data generation for all timepoints and replicates")
		
		return (self.small_fragments_data, self.composite_data)
	
	def get_coverage_df(self):
		"""
		Get the coverage DataFrame
		
		Returns:
		--------
		pandas.DataFrame
			DataFrame with origins as rows and coverage for each replicate as columns
		"""
		if self.origin_coverage is None:
			raise ValueError("Coverage data not available. Generate summary histogram data first.")
		
		return self.origin_coverage
	
	def filter_origins_by_coverage(self, min_coverage=500):
		"""
		Filter origins by coverage threshold
		
		Parameters:
		-----------
		min_coverage : int
			Minimum number of bases that must be covered
			
		Returns:
		--------
		pandas.DataFrame
			Filtered origin reference dataset
		"""
		if self.origin_coverage is None:
			raise ValueError("Coverage data not available. Generate summary histogram data first.")
		
		# Create a mask for origins that meet the coverage threshold in both replicates
		mask = (self.origin_coverage['coverage_rep1'] >= min_coverage) & \
			   (self.origin_coverage['coverage_rep2'] >= min_coverage)
		
		# Get the filtered origins
		filtered_origins = self.origins.loc[mask]
		
		return filtered_origins
	
	def plot_coverage_histogram(self, bins=30, figsize=(10, 6)):
		"""
		Plot histograms of coverage for both replicates
		
		Parameters:
		-----------
		bins : int
			Number of bins for the histogram
		figsize : tuple
			Figure size
			
		Returns:
		--------
		matplotlib.figure.Figure
			The created figure
		"""
		if self.origin_coverage is None:
			raise ValueError("Coverage data not available. Generate summary histogram data first.")
		
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
		
		# Plot histograms
		ax1.hist(self.origin_coverage['coverage_rep1'], bins=bins, alpha=0.7)
		ax1.set_xlabel('Coverage (bases)')
		ax1.set_ylabel('Number of origins')
		ax1.set_title('Replicate 1 Coverage')
		
		ax2.hist(self.origin_coverage['coverage_rep2'], bins=bins, alpha=0.7)
		ax2.set_xlabel('Coverage (bases)')
		ax2.set_ylabel('Number of origins')
		ax2.set_title('Replicate 2 Coverage')
		
		plt.tight_layout()
		return fig
	
	# Keep all the existing methods for getting data and plotting
	def get_small_fragments_df(self, replicate, timepoint):
		"""
		Convert the small fragments data for a specific replicate and timepoint to a DataFrame
		
		Parameters:
		-----------
		replicate : int
			Replicate (1 or 2)
		timepoint : int
			Timepoint
			
		Returns:
		--------
		pandas.DataFrame
			DataFrame with origins as rows and positions as columns
		"""
		if replicate not in self.small_fragments_data or timepoint not in self.small_fragments_data[replicate]:
			raise ValueError(f"No data available for replicate {replicate}, timepoint {timepoint}")
		
		# Get the data dictionary
		data_dict = self.small_fragments_data[replicate][timepoint]
		
		# Create a list of arrays
		data_arrays = []
		indices = []
		
		for idx, array in data_dict.items():
			data_arrays.append(array)
			indices.append(idx)
		
		# Create positions for columns
		win_2 = self.window_size // 2
		positions = np.arange(-win_2, win_2)
		
		# Create DataFrame
		df = pd.DataFrame(
			data_arrays,
			index=indices,
			columns=positions
		)
		
		return df
	
	def get_composite_histogram(self, replicate, timepoint):
		"""
		Get the composite histogram for a specific replicate and timepoint
		
		Parameters:
		-----------
		replicate : int
			Replicate (1 or 2)
		timepoint : int
			Timepoint
			
		Returns:
		--------
		numpy.ndarray
			2D composite histogram
		"""
		if replicate not in self.composite_data or timepoint not in self.composite_data[replicate]:
			raise ValueError(f"No composite data available for replicate {replicate}, timepoint {timepoint}")
		
		return self.composite_data[replicate][timepoint]
	
	#--------------------------------
	# Plotting Methods
	#--------------------------------
	
	def plot_small_fragment_heatmap(self, replicate, timepoint, origins_sorted=None, max_value=0.02):
		"""
		Plot heatmap of small fragment means for a specific replicate and timepoint
		
		Parameters:
		-----------
		replicate : int
			Replicate (1 or 2)
		timepoint : int
			Timepoint
		origins_sorted : pandas.DataFrame or None
			Pre-sorted origins to plot
			If None, uses the index order from the DataFrame
		max_value : float
			Maximum value for color scale
			
		Returns:
		--------
		matplotlib.figure.Figure
			The created figure
		"""
		# Get the data
		df = self.get_small_fragments_df(replicate, timepoint)
		
		# Use provided sorting if available
		if origins_sorted is not None:
			df = df.reindex(origins_sorted.index)
		
		# Create figure
		fig, ax = plt.subplots(figsize=(3, 4))
		
		# Window half-size for plot extents
		win_2 = self.window_size // 2
		
		# Plot heatmap
		n = len(df)
		ax.imshow(df, vmax=max_value, aspect='auto',
				 extent=[-win_2, win_2, 0, n], origin='lower', interpolation='none',
				 cmap='Blues')
		ax.set_ylim(n, 0)
		
		# Add title
		ax.set_title(f"WT{replicate} @ {timepoint}")
		
		# Add dividing line if there's a classification in the origins
		if origins_sorted is not None and 'footprint_class' in origins_sorted.columns:
			num_footprint = sum(origins_sorted.footprint_class == 'g1_and_g2_footprint')
			if num_footprint > 0:
				ax.axhline(num_footprint, c='black', lw=1, ls='dashed')
		
		plt.tight_layout()
		return fig
	
	def plot_small_fragment_heatmaps(self, replicates, timepoints, origins_sorted=None, max_value=0.02):
		"""
		Plot multiple heatmaps of small fragment means
		
		Parameters:
		-----------
		replicates : list
			List of replicates to plot
		timepoints : list or dict
			List of timepoints to plot (one per replicate) or dict mapping replicate to timepoint
		origins_sorted : pandas.DataFrame or None
			Pre-sorted origins to plot
			If None, uses the index order from the DataFrame
		max_value : float
			Maximum value for color scale
			
		Returns:
		--------
		matplotlib.figure.Figure
			The created figure
		"""
		# Create figure
		n_plots = len(replicates)
		fig, axs = plt.subplots(1, n_plots, figsize=(3 * n_plots, 4))
		if n_plots == 1:
			axs = [axs]
		
		# Window half-size for plot extents
		win_2 = self.window_size // 2
		
		# Plot each heatmap
		for i, rep in enumerate(replicates):
			# Determine timepoint
			if isinstance(timepoints, dict):
				tp = timepoints[rep]
			else:
				tp = timepoints[i]
			
			# Get the data
			df = self.get_small_fragments_df(rep, tp)
			
			# Use provided sorting if available
			if origins_sorted is not None:
				df = df.reindex(origins_sorted.index)
			
			# Plot heatmap
			n = len(df)
			axs[i].imshow(df, vmax=max_value, aspect='auto',
						 extent=[-win_2, win_2, 0, n], origin='lower', interpolation='none',
						 cmap='Blues')
			axs[i].set_ylim(n, 0)
			
			# Add title
			axs[i].set_title(f"WT{rep} @ {tp}")
			
			# Add dividing line if there's a classification in the origins
			if origins_sorted is not None and 'footprint_class' in origins_sorted.columns:
				num_footprint = sum(origins_sorted.footprint_class == 'g1_and_g2_footprint')
				if num_footprint > 0:
					axs[i].axhline(num_footprint, c='black', lw=1, ls='dashed')
			
			# Remove y-ticks for all but the first plot
			if i > 0:
				axs[i].set_yticks([])
		
		plt.tight_layout()
		return fig
	
	def plot_composite_histogram(self, replicate, timepoint, max_value=0.1, cmap='magma_r'):
		"""
		Plot composite histogram for a specific replicate and timepoint
		
		Parameters:
		-----------
		replicate : int
			Replicate (1 or 2)
		timepoint : int
			Timepoint
		max_value : float
			Maximum value for color scale
		cmap : str
			Colormap to use
			
		Returns:
		--------
		matplotlib.figure.Figure
			The created figure
		"""
		# Get the data
		hist = self.get_composite_histogram(replicate, timepoint)
		
		# Create figure
		fig, ax = plt.subplots(figsize=(3, 1.5))
		
		# Window half-size for plot extents
		win_2 = self.window_size // 2
		
		# Plot composite histogram
		ax.imshow(hist, cmap=cmap, origin='lower', vmax=max_value,
				 extent=[-win_2, win_2, 0, 250], aspect='auto', interpolation='none')
		
		# Add title
		ax.set_title(f"WT{replicate} @ {timepoint}")
		
		plt.tight_layout()
		return fig
	
	def plot_composite_histograms(self, replicates, timepoints, max_value=0.1, cmap='magma_r'):
		"""
		Plot multiple composite histograms with timepoints as rows
		
		Parameters:
		-----------
		replicates : list
			List of replicates to plot as columns
		timepoints : list or dict
			List of timepoints to plot as rows (one per replicate) or dict mapping replicate to timepoint
		max_value : float
			Maximum value for color scale
		cmap : str
			Colormap to use
			
		Returns:
		--------
		matplotlib.figure.Figure
			The created figure
		"""
		# Determine number of unique timepoints
		
		n_timepoints = len(timepoints)
		n_replicates = len(replicates)
		
		# Create figure with timepoints as rows and replicates as columns
		fig, axs = plt.subplots(n_timepoints, n_replicates, 
							   figsize=(3. * n_replicates, 1 * n_timepoints))
		
		# Window half-size for plot extents
		win_2 = self.window_size // 2
		
		# Plot each composite histogram
		for row_idx, tp in enumerate(timepoints):
			for col_idx, rep in enumerate(replicates):
				
				# Get the data
				hist = self.get_composite_histogram(rep, tp)
				
				# Plot composite histogram
				axs[row_idx, col_idx].imshow(hist, cmap=cmap, origin='lower', vmax=max_value,
											extent=[-win_2, win_2, 0, 250], aspect='auto', 
											interpolation='none')
				
				# Add titles only to the top row
				if row_idx == 0:
					axs[row_idx, col_idx].set_title(f"WT{rep}")
				
				# Add timepoint labels to the leftmost column
				if col_idx == 0:
					axs[row_idx, col_idx].set_ylabel(f"{tp}'")
				
				# Remove y-ticks for all but the leftmost column
				axs[row_idx, col_idx].set_yticks([])

				if row_idx < n_timepoints-1:
					axs[row_idx, col_idx].set_xticks([])
		
		plt.tight_layout()
		plt.subplots_adjust(wspace=0.1, hspace=0.2)
		return fig
	
	def plot_timecourse_heatmaps(self, replicate, timepoints, origins_sorted=None, max_value=0.02):
		"""
		Plot small fragment heatmaps for a timecourse
		
		Parameters:
		-----------
		replicate : int
			Replicate to plot
		timepoints : list
			List of timepoints to plot
		origins_sorted : pandas.DataFrame or None
			Pre-sorted origins to plot
			If None, uses the index order from the DataFrame
		max_value : float
			Maximum value for color scale
			
		Returns:
		--------
		matplotlib.figure.Figure
			The created figure
		"""
		# Create figure
		n_plots = len(timepoints)
		fig, axs = plt.subplots(1, n_plots, figsize=(3 * n_plots, 4))
		if n_plots == 1:
			axs = [axs]
		
		# Window half-size for plot extents
		win_2 = self.window_size // 2
		
		# Plot each heatmap
		for i, tp in enumerate(timepoints):
			# Get the data
			df = self.get_small_fragments_df(replicate, tp)
			
			# Use provided sorting if available
			if origins_sorted is not None:
				df = df.reindex(origins_sorted.index)
			
			# Plot heatmap
			n = len(df)
			axs[i].imshow(df, vmax=max_value, aspect='auto',
						 extent=[-win_2, win_2, 0, n], origin='lower', interpolation='none',
						 cmap='Blues')
			axs[i].set_ylim(n, 0)
			
			# Add title
			axs[i].set_title(f"WT{replicate} @ {tp}")
			
			# Add dividing line if there's a classification in the origins
			if origins_sorted is not None and 'footprint_class' in origins_sorted.columns:
				num_footprint = sum(origins_sorted.footprint_class == 'g1_and_g2_footprint')
				if num_footprint > 0:
					axs[i].axhline(num_footprint, c='black', lw=1, ls='dashed')
			
			# Remove y-ticks for all but the first plot
			if i > 0:
				axs[i].set_yticks([])
		
		plt.tight_layout()
		return fig
	
	def plot_timecourse_composites(self, replicate, timepoints, max_value=0.1, cmap='magma_r'):
		"""
		Plot composite histograms for a timecourse vertically
		
		Parameters:
		-----------
		replicate : int
			Replicate to plot
		timepoints : list
			List of timepoints to plot
		max_value : float
			Maximum value for color scale
		cmap : str
			Colormap to use
			
		Returns:
		--------
		matplotlib.figure.Figure
			The created figure
		"""
		# Create figure with vertical layout
		n_plots = len(timepoints)
		fig, axs = plt.subplots(n_plots, 1, figsize=(1.5, 0.75 * n_plots))
		if n_plots == 1:
			axs = [axs]
		
		# Window half-size for plot extents
		win_2 = self.window_size // 2
		
		# Plot each composite histogram
		for i, tp in enumerate(timepoints):
			# Get the data
			hist = self.get_composite_histogram(replicate, tp)
			
			# Plot composite histogram
			axs[i].imshow(hist, cmap=cmap, origin='lower', vmax=max_value,
						 extent=[-win_2, win_2, 0, 250], aspect='auto', interpolation='none')
			
			axs[i].set_ylabel(f"{tp}'")
			axs[i].set_yticks([])
			
			# Remove x-ticks for all but the last plot
			if i < n_plots - 1:
				axs[i].set_xticks([])
		
		plt.suptitle(f"WT{replicate}", y=0.98)
		plt.tight_layout()
		plt.subplots_adjust(hspace=0)  # Remove vertical spacing between plots
		return fig


	def plot_origin_read_coverage(self):
		plt.figure(figsize=(8, 2))
		plt.subplot(1, 2, 1)
		plt.hist(self.origin_coverage[self.origin_coverage.coverage_rep1 > 0]\
			.coverage_rep1 / self.window_size, bins=20)
		plt.axvline(0.95, c='red')
		plt.subplot(1, 2, 2)
		plt.hist(self.origin_coverage[self.origin_coverage.coverage_rep2 > 0]\
			 .coverage_rep1 / self.window_size, bins=20)
		plt.axvline(0.95, c='red')
		plt.suptitle("Origin read coverage distribution")
