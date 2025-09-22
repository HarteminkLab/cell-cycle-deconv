import os
import json
import numpy as np
import pandas as pd
from src.utils import mkdir_safe
from src.global_config import GlobalConstants
from matplotlib import pyplot as plt

class OriginReplicationForkProcessor:
	"""
	Processor for analyzing replication fork progression at origins.
	
	Analyzes 30kb regions around origins using sliding 3kb windows with 500bp stride
	to track chromatin dynamics during replication fork progression.
	"""
	
	def __init__(self, output_dir: str, copy_correct=True):
		"""
		Initialize the origin footprint processor.
		
		Parameters
		----------
		output_dir : str
			Directory containing the deconvolved chromatin data
		"""
		self.output_dir = output_dir
		self.save_dir = f"{self.output_dir}/figure_origins/fork_progression"
		mkdir_safe(self.save_dir)
		
		# Fork progression analysis parameters - sliding window approach
		self.fork_window_size = 50000  # 50kb total region around origin
		self.individual_window_size = 3000  # 2kb window size to capture neighboring effects
		self.window_stride = 500 # 1000bp stride between windows

		from src.config import load_default_chrom_configs
		self.config1, _ = load_default_chrom_configs()

		#                            center window, 500 bp wide
		#           15000 upstream                15000 downstream
		#       150 bins upstream, 150 bins downstream
		#
		#       301 bins total
		#
		# bin centered on origin
		#
		#   number of positions upstream: window_stride * fork_window_size / 2
		#   number of positions downstream: window_stride * fork_window_size / 2
		# 
		# total resulting size: u + d + 1

		
		# Calculate number of windows for sliding approach
		# Formula: (total_size - window_size) / stride + 1
		self.n_fork_windows = (self.fork_window_size + self.window_stride) // self.window_stride
		
		self.nucleosome_fragment_lengths = GlobalConstants.WIDER_NUCLEOSOME_FRAGMENT_LENGTHS
		
		# Footprint analysis parameters (kept for compatibility)
		self.footprint_fragment_lengths = (0, 100)
		self.footprint_window_size = 200
		
		self.deconvolved_loader = None
		self._setup_loader(copy_correct)
		
		# Matrix storage for fork progression analysis
		self.fork_entropy_matrix = None
		self.fork_occupancy_matrix = None
		self.origin_index_mapping = None
		self.processed_origins = None

	def _setup_loader(self, copy_correct):
		"""Initialize the deconvolved chromatin data loader."""
		from src.deconvolved_chromatin_loader import DeconvolvedChromatinDataLoader
		self.deconvolved_loader = DeconvolvedChromatinDataLoader(self.output_dir, copy_correction=copy_correct)

	def load_origin_fork_region(self, chrom: int, origin_center: int, 
								window_size: int = None, strand: str = None, log: bool = False):
		"""
		Load deconvolved chromatin data for fork progression analysis around origin.
		
		Parameters
		----------
		chrom : int
			Chromosome number
		origin_center : int
			Center position of the origin
		window_size : int, optional
			Total window size around origin center (default: self.fork_window_size)
		strand : str, optional
			Origin strand ('+' or '-'). If '-', data will be flipped horizontally
		log : bool, optional
			Enable logging of padding operations
			
		Returns
		-------
		np.ndarray
			Fork region data array with shape (timepoints, fragment_lengths, positions)
			Always returns expected size by padding with zeros at boundaries
			Data is strand-oriented (crick strand data is flipped)
		"""

		window_size = self.fork_window_size

		self.current_origin_center = origin_center
		self.strand = strand
			
		# Calculate genomic span, use half of the stride size to offset
		# to center a window on the origin
		half_stride_size = self.window_stride //2
		half_window = (window_size) // 2

		# Select the region, plus padding to handle the edge effects
		# pad based on the window size
		padding = self.individual_window_size
		requested_start = int(origin_center - half_window - padding)
		requested_end = int(origin_center + half_window + padding)

		fork_span = (requested_start, requested_end)

		if log:
			print(f"Loading reads around origin: chr{chrom}, {origin_center}")
			print(f"Window is +- {half_window} + padding: {padding}")
			print(f"Loaded span: {requested_start} to {requested_end}")
		
		# Load the chromosome region
		self.deconvolved_loader.load_mnase_span(chrom, fork_span)

	def subset_into_windows(self, log: bool = False):
		"""
		Subset the loaded fork region data into sliding windows.
		"""
		spans_to_load = self.define_window_subsets_to_load()

		window_spans_to_load = self.define_window_subsets_to_load()

		# Define the indices in the loaded histogram data to load
		# offset by the start of the loaded span, and divide by the
		# bin size (10 bp)
		loaded_span = self.deconvolved_loader.loaded_subset_span
		loaded_span_indices = []
		for i in range(len(window_spans_to_load)):
			start = (window_spans_to_load[i][0] - loaded_span[0])//GlobalConstants.BIN_WIDTH
			end = (window_spans_to_load[i][1] - loaded_span[0])//GlobalConstants.BIN_WIDTH
			loaded_span_indices.append((start, end))

		nuc_frag_length_start, nuc_frag_length_end = self.nucleosome_fragment_lengths
		nuc_frag_length_start = nuc_frag_length_start//GlobalConstants.BIN_HEIGHT
		nuc_frag_length_end = nuc_frag_length_end//GlobalConstants.BIN_HEIGHT

		loaded_data = self.deconvolved_loader.loaded_subset_data
		loaded_windows = []
		for start, end in loaded_span_indices:
			loaded_window = loaded_data[:, nuc_frag_length_start:nuc_frag_length_end, start:end]

			# Flip if the origin strand is crick
			if self.strand == '-':
				loaded_window = np.flip(loaded_window, axis=2)

			loaded_windows.append(loaded_window)

		return loaded_windows


	def define_window_subsets_to_load(self):
		"""Define the windows to load using the window size, fork size,
		and the stride.
		
		The window centers are fixed to offset from a center window. Then
		translated forward based on the currently loaded origin center.
		
		Use in conjuction with the deconvolved data loader to load the actual
		subset windows
		"""
		stride, fork_window_size, individual_window_size = self.window_stride, self.fork_window_size, \
			self.individual_window_size

		# Define the strided windows using the window size
		# and half of a stride, the half stride centers a window on the origin
		window_centers = range(-fork_window_size//2-stride//2, 
							  fork_window_size//2+stride//2,
							 stride)

		spans = []
		for i, window_center in enumerate(window_centers):
			start = self.current_origin_center+window_center-individual_window_size//2
			end = self.current_origin_center+window_center+individual_window_size//2
			spans.append((start, end))
			
		return spans


	def compute_window_metrics(self, window_data):
		"""
		Compute entropy and occupancy for a single window.
		
		Parameters
		----------
		window_data : np.ndarray
			Shape (timepoints, fragment_lengths, positions_in_window)
			
		Returns
		-------
		dict
			Dictionary with 'entropy' and 'occupancy' arrays of shape (timepoints,)
		"""
		
		# Compute entropy per timepoint
		entropy = self._compute_entropy_per_timepoint(window_data)
		
		# Compute occupancy - mean over fragment lengths and positions
		if window_data.size == 0:
			occupancy = np.zeros_like(entropy)
		else:
			occupancy = np.nanmean(window_data, axis=(1, 2))

		return {
			'entropy': entropy,
			'occupancy': occupancy
		}

	def process_single_origin_fork_progression(self, chrom: int, origin_center: int, 
											   origin_name: str = None, strand: str = None,
											   log: bool = False):
		"""
		Process a single origin to compute fork progression metrics.
		"""

		if log:
			print(f"Processing origin {origin_name if origin_name else f'chr{chrom}:{origin_center}'}")
			if strand:
				print(f"Origin strand: {strand}")
		
		# Load fork region around origin with strand-aware processing
		self.load_origin_fork_region(chrom, origin_center, strand=strand, log=log)
		
		# Subset into sliding windows
		windows = self.subset_into_windows(log=log)
		
		# Initialize result arrays
		n_timepoints = len(self.config1.timepoints_df)
		entropy_results = np.zeros((n_timepoints, self.n_fork_windows), dtype=np.float32)
		occupancy_results = np.zeros((n_timepoints, self.n_fork_windows), dtype=np.float32)
		
		# Process each window
		for window_idx, window_data in enumerate(windows):
			if window_data is not None:
				metrics = self.compute_window_metrics(window_data)
				
				if metrics['entropy'] is not None:
					entropy_results[:, window_idx] = metrics['entropy']
				if metrics['occupancy'] is not None:
					occupancy_results[:, window_idx] = metrics['occupancy']
		
		if log:
			print(f"Results shapes - Entropy: {entropy_results.shape}, Occupancy: {occupancy_results.shape}")
		
		return {
			'entropy': entropy_results,
			'occupancy': occupancy_results
		}


	def process_all_origins_fork_progression(self, origins_dataset, force_recompute=False):
		"""
		Process all origins for fork progression analysis.
		
		Parameters
		----------
		origins_dataset : pd.DataFrame
			DataFrame with origin names as index and 'chr', 'pos', 'strand' columns
		force_recompute : bool, optional
			Force recomputation even if cached files exist
			
		Returns
		-------
		dict
			Dictionary with 'entropy_matrix', 'occupancy_matrix', 'origin_index_mapping'
		"""
		from src.utils import print_fl
		
		self.processed_origins = origins_dataset
		
		# Check for cached data unless force_recompute is True
		if not force_recompute and self._check_fork_cache_exists():
			print_fl("Loading cached fork progression matrices from disk...")
			return self._load_fork_matrices()
		
		print_fl(f"Starting sliding window fork progression analysis of {len(origins_dataset)} origins...")
		print_fl(f"Window parameters: {self.individual_window_size}bp windows, {self.window_stride}bp stride, "
				f"{self.n_fork_windows} windows per origin")
		
		# Determine number of timepoints from first successful origin
		n_timepoints = len(self.config1.timepoints_df)
		
		# Initialize result matrices
		n_origins = len(origins_dataset)
		entropy_matrix = np.full((n_origins, n_timepoints, self.n_fork_windows), np.nan, dtype=np.float32)
		occupancy_matrix = np.full((n_origins, n_timepoints, self.n_fork_windows), np.nan, dtype=np.float32)
		
		# Create origin index mapping
		origin_names = origins_dataset.index.tolist()
		origin_index_mapping = {name: idx for idx, name in enumerate(origin_names)}
		
		from src.timer import Timer
		timer = Timer()
		
		# Process each origin
		successful_origins = 0
		for origin_idx, (origin_name, origin) in enumerate(origins_dataset.iterrows()):
			chrom = int(origin['chr'])
			pos = int(origin['pos'])
			strand = origin.get('strand', '+')  # Default to '+' if strand not specified
			
			try:
				# Process single origin with strand information
				results = self.process_single_origin_fork_progression(chrom, pos, origin_name, strand)
				
				if results['entropy'] is not None and results['occupancy'] is not None:
					entropy_matrix[origin_idx] = results['entropy']
					occupancy_matrix[origin_idx] = results['occupancy']
					successful_origins += 1
				else:
					print_fl(f"Warning: No data for origin {origin_name}")
					
			except Exception as e:
				print_fl(f"Error processing origin {origin_name}: {e}")
				continue
			
			# Progress tracking
			if origin_idx % 100 == 0:
				print_fl(f"{origin_idx + 1}/{n_origins} ({successful_origins} successful) {timer.get_time()}")
		
		print_fl(f"Completed processing {n_origins} origins! ({successful_origins} successful)")
		
		# Save results to disk
		print_fl("Saving fork progression matrices to disk...")
		self._save_fork_matrices(entropy_matrix, occupancy_matrix, origin_index_mapping)
		print_fl(f"Results saved to {self.save_dir}/")
		
		# Store in class attributes
		self.fork_entropy_matrix = entropy_matrix
		self.fork_occupancy_matrix = occupancy_matrix
		self.origin_index_mapping = origin_index_mapping
		
		return {
			'entropy_matrix': entropy_matrix,
			'occupancy_matrix': occupancy_matrix,
			'origin_index_mapping': origin_index_mapping,
			'n_successful': successful_origins
		}

	def _save_fork_matrices(self, entropy_matrix, occupancy_matrix, origin_index_mapping):
		"""
		Save fork progression matrices and metadata to disk.
		
		Parameters
		----------
		entropy_matrix : np.ndarray
			Shape (n_origins, n_timepoints, n_windows)
		occupancy_matrix : np.ndarray
			Shape (n_origins, n_timepoints, n_windows)
		origin_index_mapping : dict
			Mapping from origin names to matrix row indices
		"""
		# Save matrices as numpy arrays
		np.save(f"{self.save_dir}/fork_entropy_matrix{self._copy_correct_file_suffix()}.npy", 
			entropy_matrix)
		np.save(f"{self.save_dir}/fork_occupancy_matrix{self._copy_correct_file_suffix()}.npy", 
			occupancy_matrix)
		
		# Save origin index mapping as CSV
		mapping_df = pd.DataFrame([
			{'origin_name': name, 'matrix_index': idx} 
			for name, idx in origin_index_mapping.items()
		])
		mapping_df.to_csv(f"{self.save_dir}/origin_index_mapping.csv", index=False)
		
		# Save processing metadata as JSON
		metadata = {
			'fork_window_size': self.fork_window_size,
			'individual_window_size': self.individual_window_size,
			'window_stride': self.window_stride,
			'n_fork_windows': self.n_fork_windows,
			'nucleosome_fragment_lengths': self.nucleosome_fragment_lengths,
			'window_approach': 'sliding_overlapping',
			'matrix_shapes': {
				'entropy': entropy_matrix.shape,
				'occupancy': occupancy_matrix.shape
			},
			'n_origins': len(origin_index_mapping)
		}
		
		with open(f"{self.save_dir}/processing_metadata.json", 'w') as f:
			json.dump(metadata, f, indent=2, default=str)

	def _load_fork_matrices(self):
		"""
		Load fork progression matrices and metadata from disk.
		
		Returns
		-------
		dict
			Dictionary with loaded matrices and mapping
		"""
		# Load matrices
		entropy_matrix = np.load(f"{self.save_dir}/fork_entropy_matrix.npy")
		occupancy_matrix = np.load(f"{self.save_dir}/fork_occupancy_matrix.npy")
		
		# Load origin index mapping
		mapping_df = pd.read_csv(f"{self.save_dir}/origin_index_mapping.csv")
		origin_index_mapping = dict(zip(mapping_df['origin_name'], mapping_df['matrix_index']))
		
		# Store in class attributes
		self.fork_entropy_matrix = entropy_matrix
		self.fork_occupancy_matrix = occupancy_matrix
		self.origin_index_mapping = origin_index_mapping
		
		return {
			'entropy_matrix': entropy_matrix,
			'occupancy_matrix': occupancy_matrix,
			'origin_index_mapping': origin_index_mapping
		}

	def _copy_correct_file_suffix(self):
		return '' if self.deconvolved_loader.copy_correction else '_no_copy'

	def _check_fork_cache_exists(self):
		"""
		Check if all required cache files exist.
		
		Returns
		-------
		bool
			True if all cache files exist, False otherwise
		"""
		required_files = [
			f"fork_entropy_matrix{self._copy_correct_file_suffix()}.npy",
			f"fork_occupancy_matrix{self._copy_correct_file_suffix()}.npy", 
			"origin_index_mapping.csv",
			"processing_metadata.json"
		]
		
		return all(os.path.exists(f"{self.save_dir}/{filename}") for filename in required_files)

	def get_origin_fork_data(self, origin_name, metric='entropy'):
		"""
		Get fork progression data for a specific origin.
		
		Parameters
		----------
		origin_name : str
			Name of the origin
		metric : str, optional
			Metric to retrieve ('entropy' or 'occupancy')
			
		Returns
		-------
		np.ndarray
			Array of shape (timepoints, windows) for the specified origin
		"""
		if self.origin_index_mapping is None:
			raise ValueError("No fork progression data loaded. Run process_all_origins_fork_progression() first.")
		
		if origin_name not in self.origin_index_mapping:
			raise ValueError(f"Origin {origin_name} not found in processed data")
		
		origin_idx = self.origin_index_mapping[origin_name]
		
		if metric == 'entropy':
			return self.fork_entropy_matrix[origin_idx]
		elif metric == 'occupancy':
			return self.fork_occupancy_matrix[origin_idx]
		else:
			raise ValueError(f"Invalid metric: {metric}. Choose 'entropy' or 'occupancy'")

	def analyze_origin_class_fork_progression(self, origins_dataset, class_column, class_value):
		"""
		Analyze fork progression for a specific class of origins.
		
		Parameters
		----------
		origins_dataset : pd.DataFrame
			Full origins dataset
		class_column : str
			Column name for classification
		class_value : str
			Value to filter by
			
		Returns
		-------
		dict
			Dictionary containing filtered data and summary statistics
		"""
		# Filter origins by class
		class_origins = origins_dataset[origins_dataset[class_column] == class_value]
		
		print(f"Found {len(class_origins)} origins with {class_column} = {class_value}")
		
		if self.fork_entropy_matrix is None:
			print("Warning: No fork progression data loaded")
			return {'origins': class_origins, 'count': len(class_origins)}
		
		# Get matrix indices for this class
		class_indices = [self.origin_index_mapping[name] for name in class_origins.index 
						if name in self.origin_index_mapping]
		
		if not class_indices:
			print("Warning: No processed data found for this origin class")
			return {'origins': class_origins, 'count': len(class_origins)}
		
		# Extract data for this class
		class_entropy = self.fork_entropy_matrix[class_indices]
		class_occupancy = self.fork_occupancy_matrix[class_indices]
		
		return {
			'origins': class_origins,
			'count': len(class_origins),
			'entropy_data': class_entropy,
			'occupancy_data': class_occupancy,
			'matrix_indices': class_indices
		}

	def _compute_entropy_per_timepoint(self, data):
		"""
		Compute entropy for each timepoint.
		
		Parameters
		----------
		data : np.ndarray
			Shape (timepoints, fragment_lengths, positions)
			
		Returns
		-------
		np.ndarray
			Entropy values for each timepoint, shape (timepoints,)
		"""
		from src.helpers import calc_entropy
		
		# Reshape to (timepoints, flattened_fragments_x_positions)
		n_timepoints = data.shape[0]
		reshaped = data.reshape(n_timepoints, -1)
		
		# Calculate entropy for each timepoint
		entropies = []
		for t in range(n_timepoints):
			# Add small value to avoid log(0)
			entropy = calc_entropy(reshaped[t] + 0.001)
			entropies.append(entropy)
		
		return np.array(entropies)

	# Legacy methods kept for compatibility with footprint analysis
	def load_origin_footprint_data(self, chrom: int, origin_center: int, 
								 window_size: int = None):
		"""
		Load deconvolved chromatin data for origin footprint analysis.
		
		Parameters
		----------
		chrom : int
			Chromosome number
		origin_center : int
			Center position of the origin
		window_size : int, optional
			Window size around origin center (default: self.footprint_window_size)
			
		Returns
		-------
		np.ndarray
			Footprint data array with shape (timepoints, fragment_lengths, positions)
		"""
		if window_size is None:
			window_size = self.footprint_window_size
			
		# Calculate genomic span
		half_window = window_size // 2
		footprint_span = (
			int(origin_center - half_window),
			int(origin_center + half_window)
		)
		
		# Load the chromosome region
		self.deconvolved_loader.load_mnase_span(chrom, footprint_span)
		
		# Subset for footprint fragments only
		footprint_data = self.deconvolved_loader.subset_loaded_data(
			genomic_region=footprint_span,
			fragment_lengths=self.footprint_fragment_lengths
		)

		return footprint_data

	def compute_footprint_summaries(self, data):
		"""
		Compute summary metrics for origin footprint data.
		
		Parameters
		----------
		data : np.ndarray
			Shape (timepoints, fragment_lengths, positions)
			
		Returns
		-------
		dict
			Dictionary with 'occupancy', 'positioning', 'entropy' arrays of shape (timepoints,)
		"""
		if data is None:
			return None
		
		# 1. Occupancy - mean signal over fragment lengths and positions
		occupancy = np.mean(data, axis=(1, 2))
		
		# 2. Positioning - weighted average position relative to origin center
		positioning = self._compute_weighted_positioning(data)

		# If computing positional entropy, collapse fragment dim
		positional_signal = np.mean(data, axis=1)

		# If computing fragment length entropy, collapse positional dim
		fragment_signal = np.mean(data, axis=2)

		# 3. Entropy
		entropy = self._compute_entropy_per_timepoint(fragment_signal)
		
		return {
			'occupancy': occupancy,
			'positioning': positioning,
			'entropy': entropy
		}

	def _compute_weighted_positioning(self, data):
		"""
		Compute weighted average position relative to center.
		
		Parameters
		----------
		data : np.ndarray
			Shape (timepoints, fragment_lengths, positions)
			
		Returns
		-------
		np.ndarray
			Weighted position relative to center for each timepoint, shape (timepoints,)
		"""
		# Sum over fragment lengths: (timepoints, positions)
		position_signal = np.sum(data, axis=1)
		
		# Create position coordinates relative to center
		n_positions = data.shape[2]
		bp_per_position = GlobalConstants.BIN_WIDTH
		window_size = bp_per_position * n_positions
		
		# Position indices: 0, 1, 2, ..., n_positions-1
		# Convert to bp relative to center: -half_window to +half_window
		position_indices = np.arange(n_positions)
		half_window = window_size / 2
		position_coords = (position_indices * bp_per_position) - half_window + (bp_per_position / 2)
		
		# Calculate weighted average for each timepoint
		weighted_positions = []
		for t in range(position_signal.shape[0]):
			weights = position_signal[t]
			if np.sum(weights) > 0:  # Avoid division by zero
				weighted_pos = np.average(position_coords, weights=weights)
			else:
				weighted_pos = 0.0  # Default to center if no signal
			weighted_positions.append(weighted_pos)
		
		return np.array(weighted_positions)

	def _plot_heatmaps(self, ax1, metric, matrix, normalize, branch, 
		plot_phase_labels=True, plot_postg1_peak_locations=False):
		
		from src.config import load_default_chrom_configs

		config1, config2 = load_default_chrom_configs()

		def _normalize(origin_metrics_data, by_indices=None):

			# Average by all origins for a shape of
			# time point, genomic coordinate
			average_origin_data = np.nanmean(origin_metrics_data, axis=0)

			if by_indices is None:
				by_indices = np.arange(average_origin_data.shape[0])

			g1_indices = config1.get_Hpositions_for_phase('CG1')
			mean_indices = g1_indices[:(3*len(g1_indices))//4]

			mean = average_origin_data[mean_indices].mean(axis=0)[None, :]
			std = average_origin_data[by_indices].std(axis=0)[None, :]
			normalized_data = (average_origin_data - mean) / std

			return normalized_data

		if branch == 'i':
			by_indices = config1.i_indices()
		elif branch == 't':
			by_indices = config1.t_indices()
		elif branch == 'b':
			by_indices = config1.b_indices()

		if normalize:
			data_to_plot = _normalize(matrix, by_indices=by_indices)
			vmin, vmax = -3.5, 3.5
		else:
			data_to_plot = np.nanmean(matrix, axis=0)
			vmin, vmax = 0.75, 3.5

		# Set the extents to the window size + the extra bin centered on the origin
		# add another half window to center the heatmap on the exact window positions
		half_fork_window_size = self.fork_window_size//2+self.window_stride//2+self.window_stride//2
		
		if metric == 'occupancy':
			cmap = 'Blues'
			if not normalize:
				cmap = 'plasma'
		else:
			cmap = 'RdBu_r'

		def _plot_im_by_phases(ax, image_data, branch):
			from src.config import load_default_chrom_configs, retrieve_phase_ticks
			config1, config2 = load_default_chrom_configs()

			# Plot the G1 and PostG1 phases separately
			if branch == 't':
				g1_indices = config1.get_Hpositions_for_phase('CG1')
			elif branch == 'b':
				g1_indices = config1.get_Hpositions_for_phase('DG1')
			elif branch == 'i':
				g1_indices = config1.get_Hpositions_for_phase('RG1')

			postg1_indices = config1.get_Hpositions_for_phase('postG1')

			g1_tps = config1.get_timepoints_for_branch

			phase_ticks, edge_ticks, tick_labels = retrieve_phase_ticks(branch, config1, config2, True)

			g1_extent = [-half_fork_window_size, half_fork_window_size, edge_ticks[0], 0]
			postg1_extent = [-half_fork_window_size, half_fork_window_size, 0, edge_ticks[3]]

			# Plot the G1 and postG1 indices separately
			ax.imshow(image_data[g1_indices], aspect='auto', interpolation='none',
					   extent=g1_extent, origin='lower', vmin=vmin, vmax=vmax, cmap=cmap)
			im = ax.imshow(image_data[postg1_indices], aspect='auto', interpolation='none', 
					   extent=postg1_extent, origin='lower', vmin=vmin, vmax=vmax, cmap=cmap)

			if plot_postg1_peak_locations:
				# Plot the peak positions
				t_indices = config1.t_indices()
				peak_positions = image_data[t_indices].argmax(axis=0)
				t_peak_indices = t_indices[peak_positions]
				t_branch_timepoints = config1.timepoints_df.loc[['CG1', 'postG1']].set_index('Hpos')
				peak_timepoints = t_branch_timepoints.loc[t_peak_indices].timepoint_start
				xs = np.arange(-half_fork_window_size+250, half_fork_window_size-250, 500)
				select_postg1_indices = t_peak_indices > g1_indices[-1]

				def _plot_lowess(ax, x, y, frac=0.25):
					from statsmodels.nonparametric.smoothers_lowess import lowess
					smoothed = lowess(y, x, frac=frac)
					ax.plot(smoothed[:, 0], smoothed[:, 1], color=plt.cm.Oranges(0.5), lw=1)

				select_window_near_origin = (xs > -18000) & (xs < 18000)

				x = xs[select_postg1_indices & select_window_near_origin]
				y = peak_timepoints[select_postg1_indices & select_window_near_origin]

				ax.scatter(x, y, color=plt.cm.Oranges(0.5), marker='D', s=5)

				if len(x) > 0:
					_plot_lowess(ax, x, y)

			# Plot the phase labels
			ax.set_ylim(edge_ticks[3], edge_ticks[0])

			right_ax = ax.twinx()
			right_ax.set_ylim(edge_ticks[3], edge_ticks[0])
			right_ax.set_yticks(edge_ticks, minor=True)
			right_ax.tick_params(axis='y', which='major', length=0, pad=7)
			right_ax.tick_params(axis='y', which='minor', length=3)
			right_ax.set_yticks([])

			if plot_phase_labels:
				right_ax.set_yticks(phase_ticks)
				right_ax.set_yticklabels(tick_labels, fontsize=8, 
					rotation=270, ha='center', va='center', fontweight='regular')
				right_ax.tick_params(axis='y', which='minor', length=16)

			return im

		im = _plot_im_by_phases(ax1, data_to_plot, branch)

		for x in range(-self.fork_window_size//2, self.fork_window_size//2, 5000):
			ax1.axvline(x, c='white', lw=1.5, alpha=0.2, ls='solid')

		ax1.set_xticks([])
		ax1.set_yticks([])
		
		return data_to_plot, im
			
		
	def plot_all_origin_percentiles(self, metric='occupancy', normalize=True,
								inferred=True, branch='t',
								sort_key='replication_time', figsize=(7, 6), num_percentiles=5,
								percentiles_to_plot=None, plot_phase_labels=True,
								plotting_xlim=(-15000, 15000), plot_postg1_peak_locations=None):

		if percentiles_to_plot is None:
			percentiles_to_plot = range(num_percentiles)
		
		if sort_key.endswith('_r'):
			sort_key = sort_key.replace('_r', '')
			sorted_origins = self.processed_origins.sort_values(sort_key, ascending=False)
		else:
			sorted_origins = self.processed_origins.sort_values(sort_key)

		if inferred:
			# sorted_origins = sorted_origins[(sorted_origins.inferred_firing)]
			sorted_origins = sorted_origins[sorted_origins.derived_origin_efficiency_from_mcguffee_et_al_2013 > 0.5]

		index_spacing = len(sorted_origins) // num_percentiles
		origin_mapping = self.origin_index_mapping

		fig, axes = plt.subplots(len(percentiles_to_plot), 1, figsize=figsize)

		if metric == 'occupancy':
			metric_matrix = self.fork_occupancy_matrix
		elif metric == 'entropy':
			metric_matrix = self.fork_entropy_matrix

		for i, percentile_index in enumerate(percentiles_to_plot):

			if num_percentiles == 1:
				ax1 = axes
			else:
				ax1 = axes[i]

			origin_indices_span = percentile_index*index_spacing, (percentile_index+1)*index_spacing

			origin_names_to_plot = sorted_origins[origin_indices_span[0]
				:origin_indices_span[1]].index
			origin_indices_to_plot = [origin_mapping[origin_name] \
				for origin_name in origin_names_to_plot]

			matrix = metric_matrix[origin_indices_to_plot]

			if plot_postg1_peak_locations is None:
				plot_postg1_peak_locations = (normalize and i==0) and (metric == 'occupancy')

			normalized_data, im = self._plot_heatmaps(ax1, metric, matrix, 
				normalize=normalize, branch=branch, plot_phase_labels=plot_phase_labels,
				plot_postg1_peak_locations=plot_postg1_peak_locations)
			
			from src.helpers import _to_ordinal_from_array

			names = _to_ordinal_from_array(np.arange(1, num_percentiles+1, 1))
			names = [f"Earliest,\n{names[0]}"] + names[1:-1]+\
					[f"Latest,\n{names[-1]}"]

			if num_percentiles == 10:
				suffix = 'decile'
			elif num_percentiles == 5:
				suffix = 'quintile'
			else:
				suffix = 'percentile'

			ax1.set_ylabel(f"{names[percentile_index]} {suffix}", rotation=0, 
				fontsize=12, ha='right', va='center')

			if percentile_index == percentiles_to_plot[-1]:

				half_fork_window_size = self.fork_window_size//2
				half_stride = self.window_stride//2

				xticks = np.arange(-half_fork_window_size, half_fork_window_size+half_stride, 
					5000)
				xtick_labels = [f"{x/1000:.0f} kb" if x <=0 else f"+{x/1000:.0f} kb"\
								for x in xticks]

				ax1.set_xticks(xticks)
				ax1.set_xticklabels(xtick_labels)
				ax1.set_xlabel("Genomic position relative to origin center")

			ax1.set_xlim(*plotting_xlim)

		if not self.deconvolved_loader.copy_correction:
			title = f"Replication fork progression\nwithout copy correction, n={len(sorted_origins)}"
		elif inferred:
			title = f"Replication fork progression by\nnucleosome {metric}, "\
				f"inferred firing, n={len(sorted_origins)}"			
		else:
			title = f"Replication fork progression by\nnucleosome {metric}, n={len(sorted_origins)}"

		plt.suptitle(title, fontsize=20, fontweight='demi')
		plt.tight_layout()

		# Create custom axis for colorbar (adjust position/size as needed)
		if len(percentiles_to_plot) == 5:
			cbar_ax = fig.add_axes([0.92, 0.55, 0.015, 0.3])  # [left, bottom, width, height]
		else:
			cbar_ax = fig.add_axes([0.92, 0.3, 0.015, 0.3])  # [left, bottom, width, height]

		# Get the mappable from the last heatmap for colorbar reference
		# You'll need to modify _plot_heatmaps to return the mappable object
		# OR create colorbar from data range:
		cbar = fig.colorbar(im, cax=cbar_ax)
		colorbar_title = f"Average {metric}\nnormalized to G1" if normalize else f"Average {metric}"
		cbar.set_label(colorbar_title, rotation=270, labelpad=5, va='bottom')

		plt.subplots_adjust(hspace=0.075, right=0.85)

		return normalized_data
		
	