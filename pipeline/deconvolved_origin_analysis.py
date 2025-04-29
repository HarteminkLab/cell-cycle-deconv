
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.timer import Timer
from src.global_config import GlobalConstants


class DeconvolvedOriginAnalysis:
	"""
	A class to analyze origins of replication using deconvolved MNase-seq data,
	using numpy arrays for efficient storage and processing
	"""

	def __init__(self, chromatin_data_path, window_size=1000, bin_size=10, chromosomes=np.arange(1, 17)):
		"""
		Initialize the DeconvolvedOriginAnalysis
		
		Parameters:
		-----------
		chromatin_data_path : str
			Path to the chromatin deconvolution data
		window_size : int
			Size of the window around each origin (in bp)
		bin_size : int
			Size of each bin (in bp)
		chromosomes : array-like
			List of chromosomes to process
		"""
		self.timer = Timer()
		self.chromosomes = chromosomes
		self.window_size = window_size
		self.bin_size = bin_size
		self.origins = None
		self.chromatin_data_path = chromatin_data_path
		
		# Calculate highlight box coordinates if needed (80bp window)
		self.origin_footprint_region = (-20, 100, 40, 120)
		self.origin_plus_nucleosomes = (-170, 240, 40, 220)

		from src.deconvolved_chromatin_analysis import GenomeDeconvolutionAnalysis
		self.genome_analysis = GenomeDeconvolutionAnalysis(chromatin_data_path)
		
		# Will be initialized after loading origins and first data sample
		self.n_origins = 0
		self.n_timepoints = 0
		self.n_positions = window_size // bin_size + 1
		
		# NumPy arrays to store data
		self.small_fragments_data = None  # Will be shape (n_origins, n_timepoints, n_positions)
		self.composite_data = None  # Will be shape (n_timepoints, n_fragment_lengths, n_positions)
		self.origin_histogram_data = None  # Will be shape (n_origins, n_timepoints, n_fragment_lengths, n_positions)
		
		# Dictionary to map origin index to row in small_fragments_data
		self.origin_idx_map = {}

	def load_origin_reference_dataset(self):
			"""Load the origin reference dataset"""
			from src.origins import load_origins
			self.origins = load_origins(full=False)
			self.n_origins = len(self.origins)
			return self.origins
	
	def load_mnase_span(self, chromosome, span):
		"""
		Load MNase data for a specific genomic span
		
		Parameters:
		-----------
		chromosome : str or int
			Chromosome name or number
		span : tuple
			(start, end) genomic positions
			
		Returns:
		--------
		numpy.ndarray
			3D array of shape (timepoints, heights, widths)
		"""
		F, loaded_span = self.genome_analysis.load_mnase_span(chromosome, span)
		return F, loaded_span
	
	def initialize_data_arrays(self, sample_data):
		"""
		Initialize the data arrays based on a sample of the deconvolved data
		
		Parameters:
		-----------
		sample_data : numpy.ndarray
			Sample of deconvolved data with shape (timepoints, fragment_lengths, positions)
		"""
		self.n_timepoints = sample_data.shape[0]
		n_fragment_lengths = sample_data.shape[1]
		self.n_positions = sample_data.shape[2]
		
		print(f"Initializing data arrays with: {self.n_origins} origins, {self.n_timepoints} timepoints, {self.n_positions} positions")
		
		# Initialize small fragments data array - set to NaN initially
		self.small_fragments_data = np.full((self.n_origins, self.n_timepoints, self.n_positions), np.nan)
		
		# Initialize composite data array - set to zeros initially
		self.composite_data = np.zeros((self.n_timepoints, n_fragment_lengths, self.n_positions))

		# Initialize the new origin histogram data array - set to zeros initially
		self.origin_histogram_data = np.zeros((self.n_origins, self.n_timepoints, n_fragment_lengths, self.n_positions))
		
		# Create mapping from origin index to row in array
		for i, idx in enumerate(self.origins.index):
			self.origin_idx_map[idx] = i
	
	def process_origin(self, idx, origin_data, strand, small_frag_sel=(0, 10), strand_correct=True):
		"""
		Process data for a single origin
		
		Parameters:
		-----------
		idx : int or str
			Origin ID
		origin_data : numpy.ndarray
			Deconvolved data for this origin with shape (timepoints, fragment_lengths, positions)
		strand : str
			Strand of the origin ('+' or '-')
		small_frag_sel : tuple
			Fragment size selection range (min, max)
		strand_correct : bool
			Whether to flip the data for origins on the Crick strand
		"""
		# Get the array index for this origin
		origin_array_idx = self.origin_idx_map[idx]
		
		# Make a copy of origin_data to avoid modifying the original
		origin_data_processed = origin_data.copy()
		
		# Apply strand correction if needed
		if strand_correct and strand == '-':
			# Flip each timepoint's histogram for strand correction
			for tp in range(self.n_timepoints):
				origin_data_processed[tp] = np.flip(origin_data_processed[tp], axis=1)
		
		# Calculate small fragment means for all timepoints at once
		small_fragment_means = origin_data_processed[:, small_frag_sel[0]:small_frag_sel[1], :].mean(axis=1)
		
		# Store in small fragments array
		self.small_fragments_data[origin_array_idx] = small_fragment_means
		
		# Store in origin histogram data array
		self.origin_histogram_data[origin_array_idx] = origin_data_processed
		
		# Add to composite data
		self.composite_data += origin_data_processed

	def generate_summary_histogram_data(self, small_frag_sel=(0, 10), strand_correct=True):
		"""
		Generate small fragment summaries and composite data for all origins, timepoints
		
		Parameters:
		-----------
		small_frag_sel : tuple
			Fragment size selection range (min, max)
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
		
		print(f"Generating summary histogram data for all timepoints...")
		
		# Process first origin to get dimensions for array initialization
		first_origin = None
		for chrom in self.chromosomes:
			origins_on_chrom = self.origins[self.origins['chr'] == chrom]
			if len(origins_on_chrom) > 0:
				first_origin = origins_on_chrom.iloc[0]
				mid = first_origin['pos']
				half_window = self.window_size // 2
				window_start = mid - half_window
				window_end = mid + half_window
				
				# Load data for first origin
				first_origin_data, _ = self.load_mnase_span(first_origin['chr'], (window_start, window_end))
				
				# Initialize arrays based on this sample
				self.initialize_data_arrays(first_origin_data)
				break
		
		if first_origin is None:
			raise ValueError("No origins found in specified chromosomes")
		
		# Process each chromosome
		for chrom in self.chromosomes:
			# Get all origins on this chromosome
			origins_on_chrom = self.origins[self.origins['chr'] == chrom]
			
			if len(origins_on_chrom) == 0:
				continue
				
			print(f"  Processing chromosome {chrom} ({len(origins_on_chrom)} origins) - {self.timer.get_time()}")
			
			# Process each origin on this chromosome
			for idx, origin in origins_on_chrom.iterrows():
				# Define window around origin
				mid = origin['pos']
				half_window = self.window_size // 2
				window_start = mid - half_window
				window_end = mid + half_window
				
				try:
					# Load deconvolved data for this origin - all timepoints at once
					origin_data, _ = self.load_mnase_span(chrom, (window_start, window_end))
					
					# Process this origin
					self.process_origin(idx, origin_data, origin['strand'], small_frag_sel, strand_correct)
				except Exception as e:
					print(f"Error processing origin {idx}: {e}")
					continue
		
		# Normalize composite data by the number of origins
		self.composite_data /= self.n_origins
		
		self.timer.print_time(f"Completed summary histogram data generation for all timepoints")
		
		return (self.small_fragments_data, self.composite_data)

	def plot_origin_heatmap(self, origin_id, branch_type):
		# origin_id = 'oridb_817'
		from src.config import get_average_timepoints_for_branch, load_default_chrom_configs

		config1, config2 = load_default_chrom_configs()
		branch_type = 't'
		timepoints = get_average_timepoints_for_branch(config1, config2, branch_type)

		# Subsample to reduce number of plots
		indices = config1.t_indices()
		max_value = 1500.
		subset_indexes = np.linspace(0, len(indices)-1, 8).astype(int)
		subset_indices = indices[subset_indexes]
		subset_timepoints = timepoints[subset_indexes]
		origins = self.origins.copy()
		origins['np_index'] = np.arange(len(origins))
		origin = origins.loc[origin_id]

		# Offset to start at 0
		subset_timepoints = subset_timepoints - subset_timepoints[0]

		data = self.origin_histogram_data[origin.np_index]

		# Create plot
		fig = self.plot_heatmap(
			subset_indices,
			tp_labels=subset_timepoints,
			max_value=max_value / len(data),
			data=data,
			highlight_center=True
		)
		plt.suptitle(f"{origin.ars_name}", y=1.02)


	def plot_heatmap(self, timepoints, tp_labels, data=None, max_value=0.1, cmap='magma_r', 
			highlight_center=False):
		"""
		Plot multiple composite histograms
		Returns:
		--------
		matplotlib.figure.Figure
			The created figure
		"""

		if data is None:
			data = self.composite_data

		# Create figure
		n_plots = len(timepoints)
		fig, axs = plt.subplots(n_plots, 1, figsize=(2, 1 * n_plots))
		if n_plots == 1:
			axs = [axs]
		
		# Window half-size for plot extents
		half_window = self.window_size // 2

		# Plot all fragments
		min_frag, max_frag = 0, 26

		# Plot each composite histogram
		for i, tp in enumerate(timepoints):

			# Get the data
			hist = data[tp]
			hist = hist[min_frag:max_frag, :]

			frag_extent = (min_frag*self.bin_size, max_frag*self.bin_size)

			extent = [-half_window, half_window, frag_extent[0], frag_extent[1]]
			
			# Plot composite histogram
			im = axs[i].imshow(hist, cmap=cmap, origin='lower', vmax=max_value,
							  extent=extent, 
							  aspect='auto', interpolation='none')
			
			# Add highlight box if requested
			if highlight_center:

				x1, x2, y1, y2 = self.origin_footprint_region

				# Create a rectangle patch for the 80bp window
				from matplotlib.patches import Rectangle
				rect = Rectangle((x1, y1), 
								width=x2-x1, 
								height=y2-y1,
								linewidth=1, edgecolor='#00dd00', facecolor='none', 
								linestyle='solid', alpha=0.5)
				axs[i].add_patch(rect)

			axs[i].set_ylabel(f"{tp_labels[i]:.1f}")
			
			axs[i].set_xticks([])
			axs[i].set_yticks([])
		
		plt.tight_layout()
		plt.subplots_adjust(hspace=0.0)
		return fig


	def plot_composite_histograms(self, branch_type='t', subsample_step=10, max_value=800,
		highlight_center=False):
		"""
		Plot composite histograms for either mother ('t') or daughter ('b') branch.
		
		Parameters:
		-----------
		branch_type : str
			Type of branch to plot: 't' for mother branch, 'b' for daughter branch
		subsample_step : int
			Step size for subsampling indices (higher values mean fewer plots)
		max_value : int
			Maximum value for the histogram plots
		highlight_center : bool
			If True, highlights an 80bp window around the center
			
		Returns:
		--------
		fig : matplotlib.figure.Figure
			The figure containing the composite histograms
		"""
		from src.config import get_average_timepoints_for_branch, load_default_chrom_configs

		# Validate branch type
		if branch_type not in ['i', 't', 'b']:
			raise ValueError("branch_type must be either 't' (mother) or 'b' (daughter)")
		
		# Load configurations
		config1, config2 = load_default_chrom_configs()
		
		# Get indices based on branch type
		if branch_type == 'i':
			indices = config1.i_indices()
			branch_name = "recovery branch"
		elif branch_type == 't':
			indices = config1.t_indices()
			branch_name = "mother branch"
		else:  # branch_type == 'b'
			indices = config1.b_indices()
			branch_name = "daughter branch"
		
		# Get timepoints
		timepoints = get_average_timepoints_for_branch(config1, config2, branch_type)
		
		# Subsample to reduce number of plots
		subset_indexes = np.linspace(0, len(indices)-1, 8).astype(int)
		subset_indices = indices[subset_indexes]
		subset_timepoints = timepoints[subset_indexes]

		# Offset to start at 0
		subset_timepoints = subset_timepoints - subset_timepoints[0]
		
		# Create plot
		fig = self.plot_heatmap(
			subset_indices,
			tp_labels=subset_timepoints,
			max_value=max_value / len(self.origins),
			highlight_center=highlight_center
		)
		
		# Add title
		plt.suptitle(f"Composite origins,\n{branch_name}", y=1.03)
		
		return fig


	def plot_small_fragment_occupancy_comparison(self, figsize=(4, 3)):
		"""
		Plot the average occupancy of small fragments for mother and daughter branches over time.
		Predefined from the origin_footprint_region member variable
		
		Parameters:
		-----------
		figsize : tuple
			Figure size (width, height) in inches
			
		Returns:
		--------
		fig : matplotlib.figure.Figure
			The figure containing the occupancy comparison plot
		"""
		from src.config import get_average_timepoints_for_branch, load_default_chrom_configs
		
		# Load configurations
		config1, config2 = load_default_chrom_configs()
		
		# Get indices for both branches
		t_indices = config1.t_indices()
		b_indices = config1.b_indices()
		
		# Get timepoints for both branches
		t_tps = get_average_timepoints_for_branch(config1, config2, 't')
		b_tps = get_average_timepoints_for_branch(config1, config2, 'b')

		# Highlighted region bin definition
		selected_bins = self.subselect_bins_on_centered_region(self.composite_data, self.origin_footprint_region)
		
		# Calculate global mean occupancy for each timepoint by averaging over:
		# 1. Selected fragment lengths
		# 2. Positions within the central window
		
		# Mother branch occupancy
		t_occupancy = np.zeros(len(t_indices))
		for i, idx in enumerate(t_indices):
			# Get data for this timepoint, selected fragments only and center window
			timepoint_data = selected_bins[idx]
			# Average over both fragment lengths and positions
			t_occupancy[i] = np.mean(timepoint_data)
		
		# Daughter branch occupancy
		b_occupancy = np.zeros(len(b_indices))
		for i, idx in enumerate(b_indices):
			# Get data for this timepoint, selected fragments only and center window
			timepoint_data = selected_bins[idx]
			# Average over both fragment lengths and positions
			b_occupancy[i] = np.mean(timepoint_data)
		
		# Create figure
		fig, ax = plt.subplots(figsize=figsize)
		
		# Plot occupancy over time for both branches
		ax.plot(t_tps, t_occupancy, '-', linewidth=2, label='Mother branch')
		ax.plot(t_tps, b_occupancy, '-', linewidth=2,label='Daughter branch')
		
		# Add labels and title
		ax.set_xlabel('Time (min)')
		ax.set_ylabel(f'Mean occupancy small\nfragments at origin footprint')
		ax.set_title('Origin occupancy, Mother vs Daughter', fontweight='demi', fontsize=12, y=1.01)
		ax.set_ylim(0, 2.0)
		
		ax.legend(loc='lower right')
		
		plt.tight_layout()

		return fig

	def plot_small_fragment_heatmap(self, timepoint, origins_sorted=None):
		"""
		Plot heatmap of small fragment means for a specific timepoint
		
		Parameters:
		-----------
		timepoint : int
			Timepoint index
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
		df = pd.DataFrame(self.small_fragments_data[:, timepoint, :], 
			index=self.origins.index)
		
		# Use provided sorting if available
		if origins_sorted is not None:
			df = df.reindex(origins_sorted.index)
		
		# Create figure
		fig, ax = plt.subplots(figsize=(3, 4))
		
		# Window half-size for plot extents
		half_window = self.window_size // 2
		
		# Plot heatmap
		n = len(df)
		im = ax.imshow(df, aspect='auto',
					   extent=[-half_window, half_window, 0, n], origin='lower', interpolation='none',
					   cmap='magma_r')
		ax.set_ylim(n, 0)
		
		# Add title
		ax.set_title(f"Timepoint {timepoint}")
		
		# Add colorbar
		plt.colorbar(im, ax=ax, label='Small fragment density')
		
		plt.tight_layout()
		return fig

	def subselect_bins_on_centered_region(self, data, region_bp):
		# Highlighted region bin definition

		# Compute the plot region in bins, identify the center for offsetting
		# the highlighted region
		region_width = self.composite_data.shape[2]*self.bin_size
		region_width_2 = region_width//2

		bin_size = self.bin_size

		# Convert the highlight region into the bins to select:
		# x1, x2, y1, y2: e.g. [48, 60, 4, 12] -> region: (480-600) for lengths (40-120)
		# in the 1000 bp window

		bin_highlight_region = ((region_width_2+region_bp[0])//bin_size,
								(region_width_2+region_bp[1])//bin_size,
								region_bp[2]//bin_size,
								region_bp[3]//bin_size)

		start_bin, end_bin = bin_highlight_region[0], bin_highlight_region[1]
		small_frag_range = bin_highlight_region[2], bin_highlight_region[3]

		# 3D array, subset axes 1 and 2
		if len(data.shape) == 3:
			selected_bins = data[:, small_frag_range[0]:small_frag_range[1], start_bin:end_bin]
		# 4D array, subset axes 2 and 3
		elif len(data.shape) == 4:
			selected_bins = data[:, :, small_frag_range[0]:small_frag_range[1], start_bin:end_bin]

		return selected_bins
