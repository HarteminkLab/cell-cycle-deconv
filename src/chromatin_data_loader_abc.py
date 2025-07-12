from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Optional

from src.WindowCache import WindowCache
from src.global_config import GlobalConstants


class ChromatinDataLoader(ABC):
	"""
	Abstract base class for loading chromatin data with window-based caching.
	
	Handles common logic for:
	- Window calculation and caching
	- Data concatenation across windows
	- Subsetting to exact genomic spans
	
	Subclasses must implement _load_window_data() to handle specific data formats.
	"""
	
	def __init__(self, window_size: int = 10000, cache_size: int = 3):
		"""
		Initialize the chromatin data loader.
		
		Parameters
		----------
		window_size : int
			Size of windows for data loading (default: 10000 bp)
		cache_size : int
			Number of windows to keep in cache (default: 3)
		"""
		self.window_cache = WindowCache(max_size=cache_size)
		self.window_size = window_size
		self.bin_width = GlobalConstants.BIN_WIDTH
		self.bin_height = GlobalConstants.BIN_HEIGHT
		
		# Store current loaded data
		self.chrom = None
		self.loaded_subset_data = None
		self.loaded_subset_span = None
		
	def load_mnase_span(self, chrom: int, mnase_span: Tuple[int, int]) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int]]]:
		"""
		Load chromatin data for a given span with caching.
		
		Parameters
		----------
		chrom : int
			Chromosome number
		mnase_span : tuple of (int, int)
			Genomic span (start, end) to load
			
		Returns
		-------
		tuple of (np.ndarray, tuple)
			Loaded data array and actual span loaded, or (None, None) if loading fails
		"""

		# Determine which windows need to be loaded
		load_spans = get_load_spans(chrom, mnase_span, window_size=self.window_size)
		
		# Load and concatenate data from each window
		loaded_data_arr = []
		
		for load_span in load_spans:
			# Create cache key for this window
			cache_key = (chrom, load_span[0], load_span[1])
			
			# Check cache first
			cached_data = self.window_cache.get(cache_key)
			
			if cached_data is not None:
				loaded_data_arr.append(cached_data)
			else:
				# Load window data using subclass implementation
				try:
					window_data = self._load_window_data(chrom, load_span)
					
					# Validate data shape
					if window_data is None:
						raise ValueError(f"No data returned for window {load_span}")
					
					# Add to cache
					self.window_cache.put(cache_key, window_data)
					loaded_data_arr.append(window_data)
					
				except Exception as e:
					print(f"Error loading window {load_span} on chr{chrom}: {e}")
					return None, None
		
		if not loaded_data_arr:
			return None, None
			
		# Concatenate along position axis (axis=2)
		loaded_data = np.concatenate(loaded_data_arr, axis=2)
		
		# Calculate the full span that was loaded
		full_loaded_span = (load_spans[0][0], load_spans[-1][1])
		
		# Subset to exact requested span
		loaded_subset_data, loaded_subset_span = subset_data_to_span(
			loaded_data,
			full_loaded_span,
			mnase_span,
			self.bin_width,
			chrom
		)
		
		# Store results
		self.chrom = chrom
		self.loaded_subset_data = loaded_subset_data
		self.loaded_subset_span = loaded_subset_span
		
		return loaded_subset_data, loaded_subset_span

	def subset_loaded_data(self, 
						   genomic_region: Optional[Tuple[int, int]] = None,
						   fragment_lengths: Optional[Tuple[int, int]] = (0, 260)) -> np.ndarray:
		"""
		Subset the currently loaded chromatin data by genomic position and/or fragment length.
		
		Parameters
		----------
		genomic_region : tuple of (int, int), optional
			Genomic region to extract (start_bp, end_bp). If None, keep all positions.
		fragment_lengths : tuple of (int, int), optional
			Fragment length range to extract (min_bp, max_bp). If None, keep all lengths.
			
		Returns
		-------
		np.ndarray
			Subsetted data array
			
		Raises
		------
		RuntimeError
			If no data has been loaded yet
		"""
		if self.loaded_subset_data is None or self.loaded_subset_span is None:
			raise RuntimeError("No data loaded. Call load_mnase_span() first.")
		
		result = self.loaded_subset_data.copy()
		
		# Subset by genomic position if requested
		if genomic_region is not None:
			start_bp, end_bp = genomic_region
			loaded_start_bp = self.loaded_subset_span[0]
			
			start_idx = max(0, (start_bp - loaded_start_bp) // self.bin_width)
			end_idx = (end_bp - loaded_start_bp) // self.bin_width
			
			# Ensure indices are within bounds
			max_positions = result.shape[2]
			start_idx = min(start_idx, max_positions)
			end_idx = min(end_idx, max_positions)
			
			result = result[:, :, start_idx:end_idx]
		
		# Subset by fragment length if requested
		if fragment_lengths is not None:
			min_len, max_len = fragment_lengths
			
			start_frag_idx = min_len // self.bin_height
			end_frag_idx = max_len // self.bin_height
			
			# Ensure indices are within bounds
			max_frags = result.shape[1]
			start_frag_idx = min(start_frag_idx, max_frags)
			end_frag_idx = min(end_frag_idx, max_frags)
			
			result = result[:, start_frag_idx:end_frag_idx, :]
		
		return result
	
	@abstractmethod
	def _load_window_data(self, chrom: int, window_span: Tuple[int, int]) -> np.ndarray:
		"""
		Load data for a single window. Must be implemented by subclasses.
		
		Parameters
		----------
		chrom : int
			Chromosome number
		window_span : tuple of (int, int)
			Window boundaries (start, end)
			
		Returns
		-------
		np.ndarray
			Data array with shape (timepoints, fragment_lengths, positions)
		"""
		pass
	
	def clear_cache(self):
		"""Clear the window cache to free memory."""
		self.window_cache.clear()
		
	def get_cache_info(self) -> dict:
		"""Get information about cache usage."""
		return {
			'cache_size': self.window_cache.max_size,
			'current_keys': list(self.window_cache.cache.keys()),
			'window_size': self.window_size
		}



def get_load_spans(chrom, span, window_size=10000):
	"""
	Get the load spans that cover the given genomic span.
	
	Args:
		chrom: Chromosome name/number
		span: Tuple of (start, end) positions to load
		window_size: Size of the windows (default: 10000)
	

		List of (start, end) tuples for each window to load
	"""
	from src.sgd import get_chromosome_length
	import math

	max_bp = get_chromosome_length(chrom)
	start, end = span
	
	# Calculate which windows contain the start and end positions
	start_window = int(start // window_size)
	end_window = int(math.ceil(end / window_size))
	
	# Ensure we don't go beyond chromosome boundaries
	start_window = max(0, start_window)
	end_window = min(end_window, math.ceil(max_bp / window_size))
	
	# Create a list of all windows to load
	spans = []
	for window in range(start_window, end_window):
		window_start = window * window_size
		window_end = min((window + 1) * window_size, max_bp)
		# Add one to include the last bp
		# formatting will be e.g. 10000, 20001
		spans.append((window_start, window_end+1))
	
	return spans


def subset_data_to_span(data, loaded_span, desired_span, bin_width, chrom):
	"""
	Subset the loaded data to the desired genomic span.
	
	Args:
		data: The loaded data with shape (timepoints, fragment_lengths, positions)
		loaded_span: The actual span that was loaded (start_bp, end_bp)
		desired_span: The span that is desired (start_bp, end_bp)
		bin_width: The width of each bin in base pairs
		chrom: Chromosome name/number for boundary checking
	
	Returns:
		Tuple of (subset_data, actual_span_loaded)
	"""
	from src.sgd import get_chromosome_length
	import math

	max_bp = get_chromosome_length(chrom)
	first_bp = loaded_span[0]
	
	# Calculate bin indices for the desired span
	start_idx = (desired_span[0] - first_bp) // bin_width
	end_idx = (desired_span[1] - first_bp) // bin_width
	
	# Calculate the actual base pair positions corresponding to these indices
	actual_start_bp = first_bp + start_idx * bin_width
	actual_end_bp = first_bp + end_idx * bin_width
	
	# Handle cases where the desired span is at chromosome boundaries
	start_padding = 0
	end_padding = 0
	
	if start_idx < 0:
		start_padding = -start_idx
		start_idx = 0
		actual_start_bp = first_bp
	
	if actual_end_bp > max_bp:
		end_padding = (actual_end_bp - max_bp) // bin_width + 1
		actual_end_bp = max_bp
	
	# Convert to integers to use as indices
	start_idx, end_idx = int(start_idx), int(end_idx)
	
	# Subset the data
	subset_data = data[:, :, start_idx:end_idx]
	
	# Add padding if necessary
	if start_padding > 0:
		pad_shape = (subset_data.shape[0], subset_data.shape[1], start_padding)
		subset_data = np.concatenate([np.zeros(pad_shape), subset_data], axis=2)
	
	if end_padding > 0:
		pad_shape = (subset_data.shape[0], subset_data.shape[1], end_padding)
		subset_data = np.concatenate([subset_data, np.zeros(pad_shape)], axis=2)
	
	return subset_data, (actual_start_bp, actual_end_bp)
