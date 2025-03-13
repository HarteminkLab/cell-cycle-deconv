import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.global_config import GlobalConstants
from src.sgd import get_chromosome_length
from src.WindowCache import WindowCache

# Global cache instance
WINDOW_CACHE = WindowCache(max_size=3)

class GenomeDeconvolutionAnalysis():
	"""
	Class to perform analysis on genome-wide deconvolution results.
	
	Functions:
	load_mnase_span - Load the genomic data for a given chromosome and span,
					 e.g., loading a gene's chromatin context.
	"""
	def __init__(self, outdir):
		self.outdir = outdir
	
	def load_mnase_span(self, chrom, mnase_span, window_size=10000):
		"""
		Load the MNase data for a given span with caching.
		
		Args:
			chrom: Chromosome name/number
			mnase_span: Tuple of (start, end) positions to load
			window_size: Size of the windows (default: 10000)
		
		Returns:
			Tuple of (loaded_data, actual_span_loaded) or (None, None) if no data is loaded
		"""
		# Determine which windows need to be loaded
		load_spans = get_load_spans(chrom, mnase_span, window_size=window_size)
		
		# Load and concatenate data from each window, using cache where possible
		loaded_data_arr = []
		
		for load_span in load_spans:
			# Create a cache key for this window
			cache_key = (chrom, load_span[0], load_span[1])
			
			# Check if this window is in the cache
			cached_data = WINDOW_CACHE.get(cache_key)
			
			if cached_data is not None:
				# Use cached data
				loaded_data_arr.append(cached_data)
			else:
				try:
					# Construct the file path for this window
					load_path = f'{self.outdir}/chr{chrom}/chr{chrom}_{load_span[0]}_{load_span[1]}_F.npy'
					
					# Load the data and reshape
					window_data = np.load(load_path)
					window_data = window_data.reshape((window_data.shape[0], 26, -1))
					
					# Add to cache
					WINDOW_CACHE.put(cache_key, window_data)
					
					loaded_data_arr.append(window_data)
				except (ValueError, FileNotFoundError):
					raise ValueError(f"File does not exist for {load_path}")
		
		if not loaded_data_arr:
			return None, None
			
		loaded_data = np.concatenate(loaded_data_arr, axis=2)
		
		# Calculate the full span that was loaded
		full_loaded_span = (load_spans[0][0], load_spans[-1][1])
		
		# Subset the loaded data to the desired span
		loaded_subset_data, loaded_subset_span = subset_data_to_span(
			loaded_data, 
			full_loaded_span, 
			mnase_span, 
			GlobalConstants.BIN_WIDTH, 
			chrom
		)
		
		# Store the results as instance variables
		self.loaded_subset_data = loaded_subset_data
		self.loaded_subset_span = loaded_subset_span
		
		return loaded_subset_data, loaded_subset_span
	
	def clear_cache(self):
		"""Clear the window cache."""
		WINDOW_CACHE.clear()


def get_load_spans(chrom, span, window_size=10000):
	"""
	Get the load spans that cover the given genomic span.
	
	Args:
		chrom: Chromosome name/number
		span: Tuple of (start, end) positions to load
		window_size: Size of the windows (default: 10000)
	
	Returns:
		List of (start, end) tuples for each window to load
	"""
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
