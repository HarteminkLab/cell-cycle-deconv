import numpy as np
from typing import Tuple

from src.chromatin_data_loader_abc import ChromatinDataLoader


class DeconvolvedChromatinDataLoader(ChromatinDataLoader):
	"""
	Loader for deconvolved chromatin data stored as numpy files.
	
	This class handles loading pre-processed deconvolved chromatin data
	that has been saved in 10kb window chunks.
	"""
	
	def __init__(self, output_dir: str, window_size: int = 10000, cache_size: int = 3):
		"""
		Initialize the deconvolved data loader.
		
		Parameters
		----------
		output_dir : str
			Base directory containing deconvolved data files
		window_size : int
			Size of windows for data loading (default: 10000 bp)
		cache_size : int
			Number of windows to keep in cache (default: 3)
		"""
		super().__init__(window_size=window_size, cache_size=cache_size)
		self.output_dir = output_dir
		
	def _load_window_data(self, chrom: int, window_span: Tuple[int, int]) -> np.ndarray:
		"""
		Load deconvolved data for a single window from numpy file.
		
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
			
		Raises
		------
		FileNotFoundError
			If the data file doesn't exist
		ValueError
			If the loaded data has unexpected shape
		"""
		# Construct file path for this window
		# Format: chr{chrom}/chr{chrom}_{start}_{end}_F.npy
		chromatin_data_directory = f"{self.output_dir}/chromatin_deconvolution/deconvolution_data"
		file_path = f'{chromatin_data_directory}/chr{chrom}/chr{chrom}_{window_span[0]}_{window_span[1]}_F.npy'
		
		try:
			# Load the numpy array
			window_data = np.load(file_path)
			
			# Reshape to expected format
			# Original shape is flattened, need to reshape to (timepoints, 26, positions)
			if len(window_data.shape) == 2:
				# Assume shape is (timepoints, flattened_data)
				# Need to reshape to (timepoints, fragment_lengths, positions)
				timepoints = window_data.shape[0]
				fragment_lengths = 26  # Standard fragment length bins
				positions = window_data.shape[1] // fragment_lengths
				
				if window_data.shape[1] % fragment_lengths != 0:
					raise ValueError(
						f"Data shape {window_data.shape} cannot be evenly divided "
						f"into {fragment_lengths} fragment length bins"
					)
				
				window_data = window_data.reshape(timepoints, fragment_lengths, positions)
				
			elif len(window_data.shape) == 3:
				# Already in correct shape
				pass
			else:
				raise ValueError(
					f"Unexpected data shape: {window_data.shape}. "
					f"Expected 2D or 3D array."
				)
				
			return window_data
			
		except FileNotFoundError:
			raise FileNotFoundError(
				f"Deconvolved data file not found: {file_path}\n"
				f"Please ensure deconvolution has been run for chr{chrom}"
			)
		except Exception as e:
			raise RuntimeError(
				f"Error loading deconvolved data from {file_path}: {str(e)}"
			)
	
	def get_data_info(self) -> dict:
		"""Get information about the data source."""
		info = super().get_cache_info()
		info.update({
			'data_type': 'deconvolved',
			'output_dir': self.output_dir,
			'file_pattern': 'chr{chrom}/chr{chrom}_{start}_{end}_F.npy'
		})
		return info