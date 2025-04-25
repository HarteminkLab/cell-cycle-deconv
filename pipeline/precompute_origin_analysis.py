import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.timer import Timer
from src.mnase_10kb_loader import MNase10kbLoader

class OriginFootprintAnalysis:
    """
    A simplified class to analyze origins of replication using MNase-seq data,
    focusing only on collecting 2D histograms of fragment lengths vs positions
    """
    
    def __init__(self, window_size=1000):
        """
        Initialize the OriginFootprintAnalysis
        
        Parameters:
        -----------
        window_size : int
            Size of the window around each origin (in bp)
        """
        self.timer = Timer()

        # Loaders for each replicate for caching
        self.mnase_loaders = {
            1: MNase10kbLoader(),
            2: MNase10kbLoader()
        }

        # Select the samples per replicate that we should poll
        # Timepoints when ORC binding signal is greatest (early S-phase)
        self.s_start_sample_timepoints = {
            1: 30, 
            2: 20
        }

        self.window_size = window_size
        self.origins = None
        
        # Store histograms by replicate and origin ID
        self.histograms_rep1 = {}  # For replicate 1
        self.histograms_rep2 = {}  # For replicate 2
        
        self.small_fragment_curve = None
        
    def load_origin_reference_dataset(self):
        """Load the origin reference dataset"""
        from src.origins import load_origins
        self.origins = load_origins(full=True)
        return self.origins

    def load_origin_data(self, origin, replicate):
        """
        Load MNase data for a single origin and create histograms for a specific replicate
        
        Parameters:
        -----------
        origin : pandas.Series
            Series containing origin information (chr, pos, etc.)
        replicate : int
            Replicate to use (1 or 2)
            
        Returns:
        --------
        hist_2d : numpy.ndarray
            2D histogram of fragment lengths vs positions
        """
        # Get chromosome and position
        chrom = origin['chr']
        mid = origin['pos']
        
        # Define window around origin
        half_window = self.window_size // 2
        window_start = mid - half_window
        window_end = mid + half_window
        
        # Initialize arrays to hold read positions and lengths
        positions = []
        lengths = []
        
        # Load MNase data for this chromosome
        mnase_data = self.mnase_loaders[replicate].load_mnase_data(replicate, chrom)
        
        # Filter reads in the window and by sample timepoint
        window_reads = mnase_data[
            (mnase_data['mid'] >= window_start) & 
            (mnase_data['mid'] <= window_end) &
            (mnase_data['sample'] == self.s_start_sample_timepoints[replicate])
        ]
        
        if len(window_reads) > 0:
            # Extract positions (relative to window start) and lengths
            read_positions = window_reads['mid'].values - window_start
            read_lengths = window_reads['length'].values
            
            # Append to our arrays
            positions.extend(read_positions)
            lengths.extend(read_lengths)

            del window_reads
        
        # Define histogram bins
        position_bins = np.arange(0, self.window_size + 2)
        max_length = 250  # Ensure we capture all relevant fragment lengths
        length_bins = np.arange(0, max_length + 1)

        # Create 2D histogram
        hist_2d, _, _ = np.histogram2d(
            lengths,
            positions, 
            bins=[length_bins, position_bins]
        )
        
        return hist_2d

    def collect_origin_histograms(self):
        """
        Collect 2D histograms for each origin, separated by replicates
        
        Returns:
        --------
        tuple
            (histograms_rep1, histograms_rep2) dictionaries of 2D histograms indexed by origin ID
        """
        if self.origins is None:
            self.load_origin_reference_dataset()
        
        # Process each origin
        total_origins = len(self.origins)
        self.timer.start()
        
        print(f"Collecting histograms for {total_origins} origins, separated by replicates...")
        
        histograms_rep1 = {}
        histograms_rep2 = {}
        
        for i, (idx, origin) in enumerate(self.origins.iterrows()):
            if i % 100 == 0:
                print(f"Processing origin {i+1}/{total_origins} - {self.timer.get_time()}")
            
            # Collect histogram for replicate 1
            hist_rep1 = self.load_origin_data(origin, 1)
            histograms_rep1[idx] = hist_rep1
            
            # Collect histogram for replicate 2
            hist_rep2 = self.load_origin_data(origin, 2)
            histograms_rep2[idx] = hist_rep2
        
        self.timer.print_time(f"Completed histogram collection for {len(histograms_rep1)} origins in both replicates")
        
        # Store the histograms in member variables
        self.histograms_rep1 = histograms_rep1
        self.histograms_rep2 = histograms_rep2
        
        return (self.histograms_rep1, self.histograms_rep2)

    def plot_origin(self, origin_id, replicate=None):
        """
        Plot the 2D histogram for a specific origin
        
        Parameters:
        -----------
        origin_id : int or str
            ID of the origin to plot
        replicate : int or None
            If 1 or 2, plots that specific replicate
            If None, plots the combined histogram
        """
        if replicate == 1:
            if origin_id not in self.histograms_rep1:
                hist = self.load_origin_data(self.origins.loc[origin_id], 1)
            else:
                hist = self.histograms_rep1[origin_id]
            title = f"Origin {origin_id} - Replicate 1"
        elif replicate == 2:
            if origin_id not in self.histograms_rep2:
                hist = self.load_origin_data(self.origins.loc[origin_id], 2)
            else:
                hist = self.histograms_rep2[origin_id]
            title = f"Origin {origin_id} - Replicate 2"
        else:
            # Combined view
            if origin_id not in self.histograms_rep1:
                hist1 = self.load_origin_data(self.origins.loc[origin_id], 1)
            else:
                hist1 = self.histograms_rep1[origin_id]
                
            if origin_id not in self.histograms_rep2:
                hist2 = self.load_origin_data(self.origins.loc[origin_id], 2)
            else:
                hist2 = self.histograms_rep2[origin_id]
            
            hist = hist1 + hist2
            title = f"Origin {origin_id} - Combined Replicates"
        
        plt.figure(figsize=(3, 1))
        plt.imshow(hist, vmax=1, origin='lower', aspect='auto', cmap='magma_r')
        plt.title(title)