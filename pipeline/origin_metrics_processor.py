import os
import numpy as np
import pandas as pd
from src.utils import mkdir_safe
from src.global_config import GlobalConstants
from matplotlib import pyplot as plt

class OriginFootprintProcessor:
	"""
	Processor for analyzing chromatin dynamics at replication origins.
	
	Analyzes origin footprints and flanking nucleosome organization,
	including occupancy, entropy, and positioning metrics across timepoints.
	"""
	
	def __init__(self, output_dir: str):
		"""
		Initialize the origin footprint processor.
		
		Parameters
		----------
		output_dir : str
			Directory containing the deconvolved chromatin data
		"""
		self.output_dir = output_dir
		self.save_dir = f"{self.output_dir}/figure_origins/metrics"
		mkdir_safe(self.save_dir)
		
		# Fragment length ranges for different analyses
		self.footprint_fragment_lengths = (0, 100)  # Smaller fragments for footprints
		self.nucleosome_fragment_lengths = GlobalConstants.WIDER_NUCLEOSOME_FRAGMENT_LENGTHS
		
		# Window sizes
		self.footprint_window_size = 200  # Larger window to handle footprint/ACS positioning variability
		self.nucleosome_window_size = 500  # Collect three flanking nucleosomes on each side
		
		# Nucleosome positioning relative to origin (approximate distances)
		self.nucleosome_offset = 100  # bp offset for downstream or upstream of origin center
		
		self.deconvolved_loader = None
		self._setup_loader()
		
		# Storage for computed metrics
		self.origin_footprint_metrics = None
		self.upstream_nucleosome_metrics = None
		self.downstream_nucleosome_metrics = None
	
	def _setup_loader(self):
		"""Initialize the deconvolved chromatin data loader."""
		from src.deconvolved_chromatin_loader import DeconvolvedChromatinDataLoader
		self.deconvolved_loader = DeconvolvedChromatinDataLoader(self.output_dir)
	
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
	
	def load_flanking_nucleosome_data(self, chrom: int, origin_center: int, 
									nucleosome_type, strand,
									window_size: int = None):
		"""
		Load nucleosome data for upstream or downstream flanking nucleosomes.
		
		Parameters
		----------
		chrom : int
			Chromosome number
		origin_center : int
			Center position of the origin
		nucleosome_type : str
			Either 'upstream' or 'downstream'
		window_size : int, optional
			Window size around nucleosome center (default: self.nucleosome_window_size)
			
		Returns
		-------
		np.ndarray
			Nucleosome data array with shape (timepoints, fragment_lengths, positions)
		"""
		if window_size is None:
			window_size = self.nucleosome_window_size

		# Calculate nucleosome center based on type
		if strand == '+':
			if nucleosome_type == 'upstream':
				nucleosome_center = origin_center - self.nucleosome_offset
			elif nucleosome_type == 'downstream':
				nucleosome_center = origin_center + self.nucleosome_offset
			else: raise ValueError()
		elif strand == '-':
			if nucleosome_type == 'downstream':
				nucleosome_center = origin_center - self.nucleosome_offset
			elif nucleosome_type == 'upstream':
				nucleosome_center = origin_center + self.nucleosome_offset
			else: raise ValueError()
		else: raise ValueError()
		
		# Calculate genomic span
		half_window = window_size // 2
		nucleosome_span = (
			int(nucleosome_center - half_window),
			int(nucleosome_center + half_window)
		)
		
		# Load the chromosome region
		self.deconvolved_loader.load_mnase_span(chrom, nucleosome_span)
		
		# Subset for nucleosome fragments only
		nucleosome_data = self.deconvolved_loader.subset_loaded_data(
			genomic_region=nucleosome_span,
			fragment_lengths=self.nucleosome_fragment_lengths
		)

		return nucleosome_data
	
	def load_origin_and_flanking_data(self, chrom, origin_center, strand):
		"""
		Load data for origin footprint and both flanking nucleosomes.
		
		Parameters
		----------
		chrom : int
			Chromosome number
		origin_center : int
			Center position of the origin
			
		Returns
		-------
		dict
			Dictionary with 'footprint', 'upstream', 'downstream' data arrays
		"""
		results = {}
		
		results['footprint'] = self.load_origin_footprint_data(chrom, origin_center)
	
		results['upstream'] = self.load_flanking_nucleosome_data(
			chrom, origin_center, 'upstream', strand)
		
		results['downstream'] = self.load_flanking_nucleosome_data(
			chrom, origin_center, 'downstream', strand)
		
		return results
	
	def _save_origin_metrics(self, footprint_metrics, upstream_metrics, downstream_metrics):
		"""
		Save origin metrics DataFrames to CSV files.
		
		Parameters
		----------
		footprint_metrics : dict
			Dictionary with 'occupancy', 'positioning' DataFrames (no entropy for footprints)
		upstream_metrics : dict
			Dictionary with 'entropy', 'occupancy', 'positioning' DataFrames
		downstream_metrics : dict
			Dictionary with 'entropy', 'occupancy', 'positioning' DataFrames
		"""
		# Save footprint metrics (no entropy computed for footprints)
		footprint_metrics['occupancy'].to_csv(f"{self.save_dir}/footprint_occupancy.csv")
		footprint_metrics['positioning'].to_csv(f"{self.save_dir}/footprint_positioning.csv")
		
		# Save upstream nucleosome metrics
		upstream_metrics['entropy'].to_csv(f"{self.save_dir}/upstream_nucleosome_entropy.csv")
		upstream_metrics['occupancy'].to_csv(f"{self.save_dir}/upstream_nucleosome_occupancy.csv")
		upstream_metrics['positioning'].to_csv(f"{self.save_dir}/upstream_nucleosome_positioning.csv")
		
		# Save downstream nucleosome metrics
		downstream_metrics['entropy'].to_csv(f"{self.save_dir}/downstream_nucleosome_entropy.csv")
		downstream_metrics['occupancy'].to_csv(f"{self.save_dir}/downstream_nucleosome_occupancy.csv")
		downstream_metrics['positioning'].to_csv(f"{self.save_dir}/downstream_nucleosome_positioning.csv")
	
	def _load_origin_metrics(self):
		"""
		Load origin metrics DataFrames from CSV files.
		
		Returns
		-------
		tuple
			(footprint_metrics, upstream_metrics, downstream_metrics) dictionaries
		"""

		def _load_integer_columns_csv(path):
			df = pd.read_csv(path, index_col=0)
			df.columns = df.columns.astype(int)
			return df

		footprint_metrics = {
			'occupancy': _load_integer_columns_csv(f"{self.save_dir}/footprint_occupancy.csv"),
			'positioning': _load_integer_columns_csv(f"{self.save_dir}/footprint_positioning.csv")
		}
		
		upstream_metrics = {
			'entropy': _load_integer_columns_csv(f"{self.save_dir}/upstream_nucleosome_entropy.csv"),
			'occupancy': _load_integer_columns_csv(f"{self.save_dir}/upstream_nucleosome_occupancy.csv"),
			'positioning': _load_integer_columns_csv(f"{self.save_dir}/upstream_nucleosome_positioning.csv")
		}
		
		downstream_metrics = {
			'entropy': _load_integer_columns_csv(f"{self.save_dir}/downstream_nucleosome_entropy.csv"),
			'occupancy': _load_integer_columns_csv(f"{self.save_dir}/downstream_nucleosome_occupancy.csv"),
			'positioning': _load_integer_columns_csv(f"{self.save_dir}/downstream_nucleosome_positioning.csv")
		}
		
		self.origin_footprint_metrics = footprint_metrics
		self.upstream_nucleosome_metrics = upstream_metrics
		self.downstream_nucleosome_metrics = downstream_metrics
		
		return footprint_metrics, upstream_metrics, downstream_metrics
	
	def _check_cached_files_exist(self):
		"""
		Check if all required CSV files exist.
		
		Returns
		-------
		bool
			True if all files exist, False otherwise
		"""
		required_files = [
			"footprint_occupancy.csv",
			"footprint_positioning.csv",
			"upstream_nucleosome_entropy.csv",
			"upstream_nucleosome_occupancy.csv", 
			"upstream_nucleosome_positioning.csv",
			"downstream_nucleosome_entropy.csv",
			"downstream_nucleosome_occupancy.csv",
			"downstream_nucleosome_positioning.csv"
		]
		
		return all(os.path.exists(f"{self.save_dir}/{filename}") for filename in required_files)
	
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
	
	def _compute_weighted_positioning(self, data, window_size):
		"""
		Compute weighted average position relative to center.
		
		Parameters
		----------
		data : np.ndarray
			Shape (timepoints, fragment_lengths, positions)
		window_size : int
			Total window size in bp
			
		Returns
		-------
		np.ndarray
			Weighted position relative to center for each timepoint, shape (timepoints,)
		"""
		# Sum over fragment lengths: (timepoints, positions)
		position_signal = np.sum(data, axis=1)
		
		# Create position coordinates relative to center
		n_positions = data.shape[2]
		bp_per_position = window_size / n_positions
		
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
			Dictionary with 'occupancy', 'positioning' arrays of shape (timepoints,)
			Note: No entropy computed for footprints (less meaningful with smaller fragments)
		"""
		if data is None:
			return None
		
		# 1. Occupancy - mean signal over fragment lengths and positions
		occupancy = np.mean(data, axis=(1, 2))
		
		# 2. Positioning - weighted average position relative to origin center
		positioning = self._compute_weighted_positioning(data, self.footprint_window_size)
		
		return {
			'occupancy': occupancy,
			'positioning': positioning
		}
	
	def compute_nucleosome_summaries(self, data):
		"""
		Compute summary metrics for flanking nucleosome data.
		
		Parameters
		----------
		data : np.ndarray
			Shape (timepoints, fragment_lengths, positions)
			
		Returns
		-------
		dict
			Dictionary with 'entropy', 'occupancy', 'positioning' arrays of shape (timepoints,)
		"""
		if data is None:
			return None
		
		# 1. Entropy - captures chromatin organization disorder
		entropy = self._compute_entropy_per_timepoint(data)
		
		# 2. Occupancy - mean signal over fragment lengths and positions
		occupancy = np.mean(data, axis=(1, 2))
		
		# 3. Positioning - weighted average position relative to nucleosome center
		positioning = self._compute_weighted_positioning(data, self.nucleosome_window_size)
		
		return {
			'entropy': entropy,
			'occupancy': occupancy, 
			'positioning': positioning
		}
	
	def process_all_origins(self, origins_dataset, force_recompute=False):
		"""
		Process all origins and create summary DataFrames.
		
		Parameters
		----------
		origins_dataset : pd.DataFrame
			DataFrame with origin names as index and 'chr', 'pos' columns
		force_recompute : bool, optional
			Force recomputation even if cached files exist
			
		Returns
		-------
		tuple
			(footprint_metrics, upstream_metrics, downstream_metrics)
			Each is a dict with DataFrames having origin names as rows and timepoints as columns
		"""
		from src.utils import print_fl
		
		self.processed_origins = origins_dataset
		
		# Check for cached data unless force_recompute is True
		if not force_recompute and self._check_cached_files_exist():
			print_fl("Loading cached origin metrics from disk...")
			return self._load_origin_metrics()
		
		print_fl(f"Starting batch processing of {len(origins_dataset)} origins...")
		
		# Determine number of timepoints from first successful origin
		n_timepoints = None
		for origin_name, origin in origins_dataset.iterrows():
			try:
				chrom = int(origin['chr'])
				pos = int(origin['pos'])
				test_data = self.load_origin_footprint_data(chrom, pos)
				n_timepoints = test_data.shape[0]
				print_fl(f"Detected {n_timepoints} timepoints from sample origin {origin_name}")
				break
			except Exception as e:
				continue
		
		if n_timepoints is None:
			raise RuntimeError("Could not determine number of timepoints from any origin")
		
		# Initialize result DataFrames
		origin_names = origins_dataset.index.tolist()
		timepoint_columns = list(range(n_timepoints))
		
		# Create DataFrames for footprint metrics (no entropy)
		footprint_metrics = {
			'occupancy': pd.DataFrame(index=origin_names, columns=timepoint_columns, dtype=float),
			'positioning': pd.DataFrame(index=origin_names, columns=timepoint_columns, dtype=float)
		}
		
		# Create DataFrames for nucleosome metrics (with entropy)
		def create_nucleosome_metric_dfs():
			return {
				'entropy': pd.DataFrame(index=origin_names, columns=timepoint_columns, dtype=float),
				'occupancy': pd.DataFrame(index=origin_names, columns=timepoint_columns, dtype=float),
				'positioning': pd.DataFrame(index=origin_names, columns=timepoint_columns, dtype=float)
			}
		
		upstream_metrics = create_nucleosome_metric_dfs()
		downstream_metrics = create_nucleosome_metric_dfs()
		
		from src.timer import Timer
		timer = Timer()
		
		# Process each origin
		for i, (origin_name, origin) in enumerate(origins_dataset.iterrows()):
			chrom = int(origin['chr'])
			pos = int(origin['pos'])
			strand = origin.strand
			
			# Load all data for this origin
			origin_data = self.load_origin_and_flanking_data(chrom, pos, strand)
			
			try:
				# Process footprint data
				if origin_data['footprint'] is not None:
					footprint_summaries = self.compute_footprint_summaries(origin_data['footprint'])
					footprint_metrics['occupancy'].loc[origin_name] = footprint_summaries['occupancy']
					footprint_metrics['positioning'].loc[origin_name] = footprint_summaries['positioning']
				
				# Process upstream nucleosome
				if origin_data['upstream'] is not None:
					upstream_summaries = self.compute_nucleosome_summaries(origin_data['upstream'])
					upstream_metrics['entropy'].loc[origin_name] = upstream_summaries['entropy']
					upstream_metrics['occupancy'].loc[origin_name] = upstream_summaries['occupancy']
					upstream_metrics['positioning'].loc[origin_name] = upstream_summaries['positioning']
				
				# Process downstream nucleosome
				if origin_data['downstream'] is not None:
					downstream_summaries = self.compute_nucleosome_summaries(origin_data['downstream'])
					downstream_metrics['entropy'].loc[origin_name] = downstream_summaries['entropy']
					downstream_metrics['occupancy'].loc[origin_name] = downstream_summaries['occupancy']
					downstream_metrics['positioning'].loc[origin_name] = downstream_summaries['positioning']
					
			except ZeroDivisionError:
				print(f"Skipping: {origin_name}")
				continue
			
			# Progress tracking
			if i % 100 == 0:
				print_fl(f"{i + 1}/{len(origins_dataset)} {timer.get_time()}")
		
		print_fl(f"Completed processing {len(origins_dataset)} origins!")
		
		# Save results to disk
		print_fl("Saving results to disk...")
		self._save_origin_metrics(footprint_metrics, upstream_metrics, downstream_metrics)
		print_fl(f"Results saved to {self.save_dir}/")
		
		self.origin_footprint_metrics = footprint_metrics
		self.upstream_nucleosome_metrics = upstream_metrics
		self.downstream_nucleosome_metrics = downstream_metrics
		
		return footprint_metrics, upstream_metrics, downstream_metrics
	
	def get_origins_by_class(self, origins_dataset, class_column, class_value):
		"""
		Filter origins by classification.
		
		Parameters
		----------
		origins_dataset : pd.DataFrame
			Full origins dataset
		class_column : str
			Column name for classification (e.g., 'footprint_class', 'activation_time')
		class_value : str
			Value to filter by (e.g., 'g1_and_g2_footprint', 'early')
			
		Returns
		-------
		pd.DataFrame
			Filtered origins dataset
		"""
		return origins_dataset[origins_dataset[class_column] == class_value]
	
	def analyze_origin_class(self, origins_dataset, class_column, class_value):
		"""
		Analyze a specific class of origins.
		
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
			Dictionary containing filtered datasets and summary statistics
		"""
		# Filter origins by class
		class_origins = self.get_origins_by_class(origins_dataset, class_column, class_value)
		
		print(f"Found {len(class_origins)} origins with {class_column} = {class_value}")
		
		# Get corresponding metrics if already computed
		results = {'origins': class_origins, 'count': len(class_origins)}
		
		if self.origin_footprint_metrics is not None:
			# Filter metrics to this class
			class_indices = class_origins.index
			results['footprint_metrics'] = {
				k: df.loc[class_indices] for k, df in self.origin_footprint_metrics.items()
			}
			results['upstream_metrics'] = {
				k: df.loc[class_indices] for k, df in self.upstream_nucleosome_metrics.items()
			}
			results['downstream_metrics'] = {
				k: df.loc[class_indices] for k, df in self.downstream_nucleosome_metrics.items()
			}
		
		return results

	def plot_origin_metrics_heatmaps(self, sorted_index=None, highlight_column=None):

		from src.config import load_default_chrom_configs
		config1, config2 = load_default_chrom_configs()

		if sorted_index is None:
			sorted_index = self.processed_origins.sort_values('replication_time').index

		# In case we need any information on the plotted origins
		sorted_origins_data = self.processed_origins.loc[sorted_index]

		from src.config import load_default_chrom_configs, get_average_timepoints_for_phase, \
			load_mean_dg1_mg1_length, retrieve_phase_ticks, get_average_timepoints_for_branch

		def add_phase_ticks(ax, config1, config2):
			xticks = retrieve_phase_ticks('tb', config1, config2)
			phase_ticks, edge_ticks = xticks

			ax.set_xticks(phase_ticks)
			ax.set_xticklabels(['mean G1', 'S', 'G2/M'], fontsize=8)

			ax.set_xticks(edge_ticks, minor=True)

			ax.tick_params(axis='x', which='major', length=0)
			ax.tick_params(axis='x', which='minor', length=10) 
			config1, config2 = load_default_chrom_configs()
		
		footprint_occupancy = self.origin_footprint_metrics['occupancy'].loc[sorted_index]
		upstream_positioning = self.upstream_nucleosome_metrics['occupancy'].loc[sorted_index]
		downstream_positioning = self.downstream_nucleosome_metrics['occupancy'].loc[sorted_index]

		mean_normed_upstream_pos = (upstream_positioning - upstream_positioning.mean(1).values[:, None])/\
			(upstream_positioning.std(1).values[:, None])
		mean_normed_downstream_pos = (downstream_positioning - downstream_positioning.mean(1).values[:, None])/\
			(downstream_positioning.std(1).values[:, None])
		mean_normed_footprint_occupancy = footprint_occupancy - \
			footprint_occupancy[config1.t_indices()].mean(1).values[:, None]

		def get_tb_data(df):
			"""Average the mother and daughter branches"""
			tb_dat = (df[config1.t_indices()].values+df[config1.b_indices()].values)/2
			return tb_dat
		
		def _plot_replication():
			ys = np.arange(len(sorted_index))
			mean_g1 = load_mean_dg1_mg1_length()
			plt.scatter(self.processed_origins.loc[sorted_index].replication_time\
				-mean_g1, ys, c='black', s=0.5)
			
		def plot_branch_im(data, vmin, vmax, cmap):
			"""Plot the G1 and S/G2/M phases separately for proper time scale"""
			num_g1 = len(config1.get_Hpositions_for_phase('CG1'))
			g1_length = -get_average_timepoints_for_branch(config1, config2, 'tb')[0]
			post_g1_time = get_average_timepoints_for_branch(config1, config2, 'tb')[-1]
			
			# Plot g1 data
			extent = [-g1_length, 0, 0, len(data)]
			plt.imshow(data[:, :num_g1], aspect='auto', cmap=cmap,
				  vmin=vmin, vmax=vmax, extent=extent, origin='upper', 
				  interpolation='none')
			
			# Plot S/G2M data
			extent = [0, post_g1_time, 0, len(data)]
			plt.imshow(data[:, num_g1:], aspect='auto', cmap=cmap,
				  vmin=vmin, vmax=vmax, extent=extent, origin='upper',
				  interpolation='none')
			plt.xlim(-g1_length, post_g1_time)
			add_phase_ticks(plt.gca(), config1, config2)
			
			plt.ylim(len(data)+0.5, -0.5)
			plt.yticks([])
				   
		entropy_vmax = 4

		plt.figure(figsize=(5.5, 6))
		plt.subplot(1, 3, 1)
		plot_branch_im(get_tb_data(mean_normed_upstream_pos), cmap='RdBu_r',
				  vmin=-entropy_vmax, vmax=entropy_vmax)
		_plot_replication()
		plt.title("Upstream entropy")
		plt.ylabel("Origin sorted by replication time")

		plt.subplot(1, 3, 2)
		plot_branch_im(get_tb_data(mean_normed_footprint_occupancy), cmap='Oranges',
				  vmin=0, vmax=0.25)
		_plot_replication()
		plt.title("Footprint occupancy")

		plt.subplot(1, 3, 3)
		plot_branch_im(get_tb_data(mean_normed_downstream_pos), cmap='RdBu_r',
				  vmin=-entropy_vmax, vmax=entropy_vmax)
		_plot_replication()

		plt.title("Downstream entropy")

		plt.suptitle(f"Origin of replication metrics, n={len(mean_normed_downstream_pos)}",
					fontweight='demi', fontsize=18)
		plt.tight_layout()
		plt.subplots_adjust(wspace=0.15)
		