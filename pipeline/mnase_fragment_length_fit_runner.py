import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.timer import Timer
from src.mnase_10kb_loader import MNase10kbLoader
from src.mnase_reference_data_analyzer import MNaseReferenceAnalyzer
from src.reference_data import read_brogaard_nucleosomes, read_macisaac_sites, read_rossi_sites

class FragmentLengthFitRunner:
    """
    A class to run fragment length distribution analysis for different reference datasets,
    fit Gaussian distributions to them, and generate visualization plots.
    """
    
    def __init__(self):
        """Initialize the FragmentLengthFitRunner"""
        self.timer = Timer()
        self.mnase_loader = MNase10kbLoader()
        self.analyzers = {}  # Store analyzers for different reference datasets
        
    def load_reference_datasets(self):
        """Load the reference datasets: Brogaard nucleosomes and MacIsaac ABF1 sites"""
        # Load Brogaard nucleosomes
        self.brogaard_nucleosomes = read_brogaard_nucleosomes()
        # Add missing keys needed for processing
        self.brogaard_nucleosomes['mid'] = self.brogaard_nucleosomes['position']
        self.brogaard_nucleosomes['chr'] = self.brogaard_nucleosomes['chromosome']
        
        # Load Rossi sites and filter for high scoring
        rossi_sites = read_rossi_sites()
        self.rossi_abf1_sites = rossi_sites[(rossi_sites.tf == 'Abf1') & (rossi_sites['?'] == 1000)]

        # Load MacIsaac sites and filter for ABF1
        # self.macisaac_sites = read_macisaac_sites()
        # self.macisaac_abf1_sites = self.macisaac_sites[self.macisaac_sites.TF == 'ABF1']
        
        print(f"Loaded {len(self.brogaard_nucleosomes)} Brogaard nucleosomes")
        print(f"Loaded {len(self.rossi_abf1_sites)} Rossi ABF1 sites")
        
        return self
        
    def create_analyzers(self, padding=80):
        """Create analyzers for each reference dataset"""
        # Create analyzer for Brogaard nucleosomes
        self.analyzers['brogaard'] = MNaseReferenceAnalyzer(
            self.brogaard_nucleosomes, 
            padding=padding
        )
        self.analyzers['brogaard'].title = "Top 2000 Brogaard nucleosomes"
        
        # Create analyzer for MacIsaac ABF1 sites
        self.analyzers['abf1'] = MNaseReferenceAnalyzer(
            self.rossi_abf1_sites, 
            padding=padding
        )
        self.analyzers['abf1'].title = f"{len(self.rossi_abf1_sites)} Rossi Abf1 sites"
        
        print("Created analyzers for Brogaard nucleosomes and MacIsaac ABF1 sites")
        return self
    
    def process_all_sites(self, chromosomes=range(1, 2)):
        """Process all sites for each analyzer"""
        for name, analyzer in self.analyzers.items():
            print(f"\nProcessing {name} sites...")
            analyzer.process_sites(chromosomes=chromosomes)
            
        return self
    
    def fit_gaussian_distributions(self):
        """Fit Gaussian distributions to the length distributions"""
        # Fit Gaussian for nucleosome fragments (Brogaard)
        print("\nFitting Gaussian for nucleosome fragments...")
        self.analyzers['brogaard'].fit_gaussian_to_length_distribution(
            fit_range=(140, 190),
            plot=False
        )
        
        # Fit Gaussian for small fragments (ABF1)
        print("\nFitting Gaussian for small fragments...")
        self.analyzers['abf1'].fit_gaussian_to_length_distribution(
            fit_range=(0, 90),
            plot=False
        )
        
        return self
    
    def plot_distributions(self, save_figures=False, output_dir=None):
        """
        Plot distributions for each analyzer
        
        Parameters:
        -----------
        save_figures : bool
            Whether to save the figures to files
        output_dir : str or None
            Directory to save figures to (if save_figures is True)
        """
        figures = {}
        
        for name, analyzer in self.analyzers.items():
            print(f"\nPlotting distribution for {name}...")
            fig, axs = analyzer.plot_distribution()
            figures[name] = (fig, axs)
            
            if save_figures:
                if output_dir is None:
                    output_dir = '.'
                    
                # Ensure the output directory exists
                import os
                os.makedirs(output_dir, exist_ok=True)
                
                # Save the figure
                filename = os.path.join(output_dir, f"{name}_distribution.png")
                fig.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"Saved figure to {filename}")
        
        return figures
    
    def run_all(self, padding=80, save_figures=False, output_dir=None):
        """
        Run the complete analysis pipeline
        
        Parameters:
        -----------
        padding : int
            Distance in bp to consider around each reference site
        save_figures : bool
            Whether to save the figures to files
        output_dir : str or None
            Directory to save figures to (if save_figures is True)
        """
        self.timer.start()
        
        print("Starting fragment length analysis pipeline...")
        
        self.load_reference_datasets()
        self.create_analyzers(padding=padding)
        self.process_all_sites()
        self.fit_gaussian_distributions()
        figures = self.plot_distributions(save_figures=save_figures, output_dir=output_dir)
        
        self.timer.print_time("Total analysis time")
        
        return figures
    
    def get_selection_curves(self):
        """
        Get the selection curves from the fitted distributions
        
        Returns:
        --------
        dict : Dictionary with selection curves for each reference dataset
        """
        selection_curves = {}
        
        for name, analyzer in self.analyzers.items():
            if hasattr(analyzer, 'fit_gaussian_loc') and hasattr(analyzer, 'fit_gaussian_scale'):
                x_data = analyzer.y_bins[:-1]  # Fragment lengths
                
                # Create selection curve as normalized Gaussian PDF
                from scipy.stats import norm
                ys = norm.pdf(x_data, loc=analyzer.fit_gaussian_loc, scale=analyzer.fit_gaussian_scale)
                selection_curve = pd.DataFrame(ys, index=x_data, columns=['weight'])
                
                # Zero out values outside the fitted range if available
                if hasattr(analyzer, 'fit_range'):
                    selection_curve.loc[selection_curve.index < analyzer.fit_range[0], 'weight'] = 0
                    selection_curve.loc[selection_curve.index > analyzer.fit_range[1], 'weight'] = 0
                
                # Normalize
                non_zero_mean = selection_curve.loc[selection_curve['weight'] > 0, 'weight'].mean()
                if non_zero_mean > 0:
                    selection_curve = selection_curve / non_zero_mean
                
                selection_curves[name] = selection_curve
                
        return selection_curves
