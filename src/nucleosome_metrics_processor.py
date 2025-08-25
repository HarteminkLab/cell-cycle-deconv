
import os
import numpy as np
import pandas as pd
from src.utils import mkdir_safe

class NucleosomeDataLoader:
	"""
	Simplified loader for deconvolved chromatin data at nucleosome positions.
	
	Focuses specifically on loading nucleosome-sized fragments (130-200 bp)
	from deconvolved MNase data for individual nucleosome positions.
	"""
	
	def __init__(self, output_dir: str):
		"""
		Initialize the nucleosome data loader.
		
		Parameters
		----------
		output_dir : str
			Directory containing the deconvolved chromatin data
		"""
		self.output_dir = output_dir
		self.save_dir = f"{self.output_dir}/nucleosome_metrics"
		mkdir_safe(self.save_dir)

		self.nucleosome_fragment_lengths = (130, 200)
		self.deconvolved_loader = None
		self._setup_loader()

		self.minus_one_chromatin_metrics = None
		self.plus_one_chromatin_metrics = None
	
	def _setup_loader(self):
		"""Initialize the deconvolved chromatin data loader."""
		from src.deconvolved_chromatin_loader import DeconvolvedChromatinDataLoader
		self.deconvolved_loader = DeconvolvedChromatinDataLoader(self.output_dir)
	
	def load_nucleosome_data(self, chrom: int, nucleosome_center: int, 
						   window_size: int = 160):
		"""
		Load deconvolved MNase data for a single nucleosome position.
		
		Parameters
		----------
		chrom : int
			Chromosome number
		nucleosome_center : int
			Dyad position of the nucleosome
		window_size : int, optional
			Total window size around the nucleosome center (default: 160bp)
			
		Returns
		-------
		np.ndarray
			Nucleosome data array with shape (timepoints, fragment_lengths, positions)
		"""
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

	def _save_nucleosome_metrics(self, plus_one_metrics, minus_one_metrics):
		"""
		Save nucleosome metrics DataFrames to CSV files.
		
		Parameters
		----------
		plus_one_metrics : dict
			Dictionary with 'entropy', 'occupancy', 'positioning' DataFrames
		minus_one_metrics : dict
			Dictionary with 'entropy', 'occupancy', 'positioning' DataFrames
		"""
		
		# Save plus one nucleosome metrics
		plus_one_metrics['entropy'].to_csv(f"{self.save_dir}/plus_one_entropy.csv")
		plus_one_metrics['occupancy'].to_csv(f"{self.save_dir}/plus_one_occupancy.csv")
		plus_one_metrics['positioning'].to_csv(f"{self.save_dir}/plus_one_positioning.csv")
		
		# Save minus one nucleosome metrics
		minus_one_metrics['entropy'].to_csv(f"{self.save_dir}/minus_one_entropy.csv")
		minus_one_metrics['occupancy'].to_csv(f"{self.save_dir}/minus_one_occupancy.csv")
		minus_one_metrics['positioning'].to_csv(f"{self.save_dir}/minus_one_positioning.csv")

		
	def _load_nucleosome_metrics(self):
		"""
		Load nucleosome metrics DataFrames from CSV files.
		
		Returns
		-------
		tuple
			(plus_one_metrics, minus_one_metrics) dictionaries
		"""
		plus_one_metrics = {
			'entropy': pd.read_csv(f"{self.save_dir}/plus_one_entropy.csv", index_col=0),
			'occupancy': pd.read_csv(f"{self.save_dir}/plus_one_occupancy.csv", index_col=0),
			'positioning': pd.read_csv(f"{self.save_dir}/plus_one_positioning.csv", index_col=0)
		}
		
		minus_one_metrics = {
			'entropy': pd.read_csv(f"{self.save_dir}/minus_one_entropy.csv", index_col=0),
			'occupancy': pd.read_csv(f"{self.save_dir}/minus_one_occupancy.csv", index_col=0),
			'positioning': pd.read_csv(f"{self.save_dir}/minus_one_positioning.csv", index_col=0)
		}

		self.plus_one_chromatin_metrics = plus_one_metrics
		self.minus_one_chromatin_metrics = minus_one_metrics
		
		return plus_one_metrics, minus_one_metrics

	def _check_cached_files_exist(self):
		"""
		Check if all 6 required CSV files exist.
		
		Returns
		-------
		bool
			True if all files exist, False otherwise
		"""
		required_files = [
			"plus_one_entropy.csv",
			"plus_one_occupancy.csv", 
			"plus_one_positioning.csv",
			"minus_one_entropy.csv",
			"minus_one_occupancy.csv",
			"minus_one_positioning.csv"
		]
		
		return all(os.path.exists(f"{self.save_dir}/{filename}") for filename in required_files)


	def load_multiple_nucleosomes(self, nucleosome_positions: list, 
								window_size: int = 160):
		"""
		Load data for multiple nucleosomes efficiently.
		
		Parameters
		----------
		nucleosome_positions : list of tuples
			List of (chrom, nucleosome_center) tuples
		window_size : int, optional
			Window size around each nucleosome center
			
		Returns
		-------
		dict
			Dictionary with (chrom, position) as keys and data arrays as values
		"""
		results = {}
		
		for chrom, nucleosome_center in nucleosome_positions:
			try:
				data = self.load_nucleosome_data(chrom, nucleosome_center, window_size)
				results[(chrom, nucleosome_center)] = data
			except Exception as e:
				print(f"Failed to load nucleosome at chr{chrom}:{nucleosome_center}: {e}")
				results[(chrom, nucleosome_center)] = None
		
		return results
	
	def process_all_gene_nucleosomes(self, gene_dataset, 
								   nucleosome_keys=['+1 nucleosome', '-1 nucleosome'],
								   window_size=160, force_recompute=False):
		"""
		Process all genes and create summary DataFrames for each nucleosome type.
		
		Parameters
		----------
		gene_dataset : pd.DataFrame
			DataFrame with ORF names as index and nucleosome position columns
		nucleosome_keys : list, optional
			Column names for nucleosome positions (default: ['+1 nucleosome', '-1 nucleosome'])
		window_size : int, optional
			Window size around each nucleosome center
			
		Returns
		-------
		tuple
			(plus_one_chromatin_metrics, minus_one_chromatin_metrics)
			Each is a dict with keys: 'entropy', 'occupancy', 'positioning'
			Each value is a DataFrame with ORF names as rows and timepoints as columns
		"""
		from src.utils import print_fl
	
		# Check for cached data unless force_recompute is True
		if not force_recompute and self._check_cached_files_exist():
			print_fl("Loading cached nucleosome metrics from disk...")
			return self._load_nucleosome_metrics()
		
		print_fl(f"Starting batch processing of {len(gene_dataset)} genes...")
		
		# Determine number of timepoints from first successful gene
		n_timepoints = None
		for orf_name, gene in gene_dataset.iterrows():
			try:
				chrom = int(gene['Chr'])
				nuc_center = int(gene[nucleosome_keys[0]])  # Use first nucleosome type
				test_data = self.load_nucleosome_data(chrom, nuc_center, window_size)
				n_timepoints = test_data.shape[0]
				print_fl(f"Detected {n_timepoints} timepoints from sample gene {orf_name}")
				break
			except Exception as e:
				continue
				
		if n_timepoints is None:
			raise RuntimeError("Could not determine number of timepoints from any gene")
		
		# Initialize result DataFrames
		orf_names = gene_dataset.index.tolist()
		timepoint_columns = list(range(n_timepoints))
		
		# Create DataFrames for each nucleosome type and metric
		def create_metric_dfs():
			return {
				'entropy': pd.DataFrame(index=orf_names, columns=timepoint_columns, dtype=float),
				'occupancy': pd.DataFrame(index=orf_names, columns=timepoint_columns, dtype=float),
				'positioning': pd.DataFrame(index=orf_names, columns=timepoint_columns, dtype=float)
			}
		
		plus_one_metrics = create_metric_dfs()
		minus_one_metrics = create_metric_dfs()
		
		# Map nucleosome keys to result dictionaries
		nucleosome_results = {
			'+1 nucleosome': plus_one_metrics,
			'-1 nucleosome': minus_one_metrics
		}

		from src.timer import Timer
		timer = Timer()
		
		# Process each gene
		for i, (orf_name, gene) in enumerate(gene_dataset.iterrows()):
			chrom = int(gene['Chr'])
			
			# Process each nucleosome type for this gene
			for nuc_key in nucleosome_keys:
				nuc_center = int(gene[nuc_key])
				
				# Load data and compute summaries
				data = self.load_nucleosome_data(chrom, nuc_center, window_size)

				try:
					summaries = self.compute_nucleosome_summaries(data, window_size)
					# Store results in appropriate DataFrames
					target_dfs = nucleosome_results[nuc_key]
					target_dfs['entropy'].loc[orf_name] = summaries['entropy']
					target_dfs['occupancy'].loc[orf_name] = summaries['occupancy']
					target_dfs['positioning'].loc[orf_name] = summaries['positioning']
				except ZeroDivisionError:
					print(f"Skipping: ", orf_name)
					continue
			
			# Progress tracking
			if (i) % 200 == 0:
				print_fl(f"{i + 1}/{len(gene_dataset)} {timer.get_time()}")
		
		print_fl(f"Completed processing {len(gene_dataset)} genes!")
		
		print_fl(f"Completed processing {len(gene_dataset)} genes!")
		
		# Save results to disk
		print_fl("Saving results to disk...")
		self._save_nucleosome_metrics(plus_one_metrics, minus_one_metrics)
		print_fl(f"Results saved to {self.save_dir}/")

		self.plus_one_chromatin_metrics = plus_one_metrics
		self.minus_one_chromatin_metrics = minus_one_metrics
		
		return plus_one_metrics, minus_one_metrics

	
	def load_gene_nucleosomes(self, gene_nucleosomes: pd.Series, 
							nucleosome_keys: list = ['+1 nucleosome', '-1 nucleosome'],
							window_size: int = 160):
		"""
		Load nucleosome data for a specific gene.
		
		Parameters
		----------
		gene_nucleosomes : pd.Series
			Gene row containing chromosome and nucleosome positions
		nucleosome_keys : list, optional
			Column names for nucleosome positions to load
		window_size : int, optional
			Window size around each nucleosome
			
		Returns
		-------
		dict
			Dictionary with nucleosome_key as keys and data arrays as values
		"""
		chrom = int(gene_nucleosomes['Chr'])
		results = {}
		
		for nuc_key in nucleosome_keys:
			if nuc_key in gene_nucleosomes and pd.notna(gene_nucleosomes[nuc_key]):
				nucleosome_center = int(gene_nucleosomes[nuc_key])
				try:
					data = self.load_nucleosome_data(chrom, nucleosome_center, window_size)
					results[nuc_key] = data
				except Exception as e:
					print(f"Failed to load {nuc_key} for gene: {e}")
					results[nuc_key] = None
			else:
				results[nuc_key] = None
		
		return results
	
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
	
	def _compute_weighted_positioning(self, data, window_size=160):
		"""
		Compute weighted average position relative to nucleosome center.
		
		Parameters
		----------
		data : np.ndarray
			Shape (timepoints, fragment_lengths, positions)
		window_size : int, optional
			Total window size in bp (default: 160)
			
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
	
	def compute_nucleosome_summaries(self, data, window_size=160):
		"""
		Compute summary metrics for nucleosome data.
		
		Parameters
		----------
		data : np.ndarray
			Shape (timepoints, fragment_lengths, positions)
		window_size : int, optional
			Window size in bp for positioning calculation
			
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
		positioning = self._compute_weighted_positioning(data, window_size)
		
		return {
			'entropy': entropy,
			'occupancy': occupancy, 
			'positioning': positioning
		}
	
	def load_and_summarize_nucleosome(self, chrom: int, nucleosome_center: int, 
									window_size: int = 160):
		"""
		Load nucleosome data and compute summary metrics in one step.
		
		Parameters
		----------
		chrom : int
			Chromosome number
		nucleosome_center : int
			Dyad position of the nucleosome
		window_size : int, optional
			Window size around nucleosome center
			
		Returns
		-------
		dict
			Dictionary with summary metrics and raw data
		"""
		# Load the raw data
		data = self.load_nucleosome_data(chrom, nucleosome_center, window_size)
		
		# Compute summaries
		summaries = self.compute_nucleosome_summaries(data, window_size)
		
		return {
			'data': data,
			'summaries': summaries,
			'metadata': {
				'chrom': chrom,
				'center': nucleosome_center,
				'window_size': window_size,
				'data_shape': data.shape
			}
		}
	

