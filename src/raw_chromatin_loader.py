import numpy as np
from typing import Tuple

from src.chromatin_data_loader_abc import ChromatinDataLoader
from src.combined_chromatin_model import CombinedChromatinModel
from src.config import load_default_chrom_configs


class RawChromatinDataLoader(ChromatinDataLoader):
    """
    Loader for raw chromatin data from MNase-seq reads.
    
    This class handles loading raw MNase reads and processing them into
    binned histograms matching the format of deconvolved data.
    """
    
    def __init__(self, replicate: int = 1, window_size: int = 10000, cache_size: int = 3):
        """
        Initialize the raw data loader.
        
        Parameters
        ----------
        replicate : int
            Which replicate to load (1 or 2)
        window_size : int
            Size of windows for data loading (default: 10000 bp)
        cache_size : int
            Number of windows to keep in cache (default: 3)
        """
        super().__init__(window_size=window_size, cache_size=cache_size)
        
        self.replicate = replicate
        
        # Initialize the combined chromatin model
        config1, config2 = load_default_chrom_configs()
        self.combined_chromatin_model = CombinedChromatinModel(config1, config2)
        
        # Store which config/model to use based on replicate
        self.config = config1 if replicate == 1 else config2
        
    def _load_window_data(self, chrom: int, window_span: Tuple[int, int]) -> np.ndarray:
        """
        Load raw MNase data for a single window and process into histograms.
        
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
        # Load and process the raw data for this window
        self.combined_chromatin_model.load_mnase_span(chrom, window_span, verbose=False)
        
        # Retrieve the processed histogram data
        rep1_data, rep2_data = self._retrieve_G_imgs()
        
        # Select the appropriate replicate
        window_data = rep1_data if self.replicate == 1 else rep2_data
        
        # Validate shape
        if len(window_data.shape) != 3:
            raise ValueError(
                f"Unexpected data shape from ChromatinModel: {window_data.shape}. "
                f"Expected 3D array (timepoints, fragment_lengths, positions)."
            )
        
        return window_data
    
    def _retrieve_G_imgs(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve the downsampled raw histogram data for each replicate.
        
        Returns
        -------
        tuple of (np.ndarray, np.ndarray)
            Histogram data for replicate 1 and replicate 2
        """
        return (self.combined_chromatin_model.chrom1_model.G_imgs,
                self.combined_chromatin_model.chrom2_model.G_imgs)
    
    def get_data_info(self) -> dict:
        """Get information about the data source."""
        info = super().get_cache_info()
        info.update({
            'data_type': 'raw',
            'replicate': self.replicate,
            'config': self.config.name if hasattr(self.config, 'name') else 'unknown'
        })
        return info
    
    def switch_replicate(self, replicate: int):
        """
        Switch to loading data from a different replicate.
        
        Note: This will clear the cache since cached data is replicate-specific.
        
        Parameters
        ----------
        replicate : int
            Which replicate to load (1 or 2)
        """
        if replicate not in [1, 2]:
            raise ValueError("Replicate must be 1 or 2")
            
        if replicate != self.replicate:
            self.replicate = replicate
            config1, config2 = load_default_chrom_configs()