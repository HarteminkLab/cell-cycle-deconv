
import numpy as np
from matplotlib import pyplot as plt


class OriginsFootprintAnalysis:
    """
    Analyzes subnucleosomal footprint occupancy at replication origins
    throughout the cell cycle, reproducing and extending Li et al. 2021.
    
    Oriented by T-rich ACS strand (Belsky et al. 2015):
    - Plus-strand origins: position axis unchanged
    - Minus-strand origins: position axis flipped
    
    Footprint defined as <120bp fragments within ~50bp downstream of ACS 5' end.
    """

    # Li et al. parameters, converted to 10bp bins
    FRAGMENT_MAX_BP = 120
    FOOTPRINT_OFFSET_BINS = 0    # start of footprint relative to ACS center
    FOOTPRINT_WIDTH_BINS  = 5    # 50bp downstream

    def __init__(self, window_bp=2000):
        self.window_bp = window_bp          # total window around ACS
        self.window_bins = window_bp // 10  # in bins

    def setup(self, genome_deconv_analysis, origins):
        """Load genome analysis and origin data."""
        
        self.genome_deconv_analysis = genome_deconv_analysis
        self.origins = origins.copy()
        self.oriented_data = {}
        self.collect_all_oriented_data()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_oriented_origin_data(self, origin):
        """
        Load MNase window centered on origin.pos and orient by T-rich strand.
        Returns 3D array (time x fragment x position), strand-flipped if minus.
        """
        win_2 = self.window_bp // 2
        span = origin.pos - win_2, origin.pos + win_2
        loaded_data, _ = self.genome_deconv_analysis.load_mnase_span(origin.chr, span)

        if origin.strand == '-':
            loaded_data = np.flip(loaded_data, axis=2)

        return loaded_data


    def collect_all_oriented_data(self):
        """
        Collect oriented MNase data for all origins into a 4D numpy array
        (n_origins x time x fragment x position).
        
        Adds 'oriented_data_index' column to self.origins.
        Failed loads are skipped and marked with -1 in the index column.
        """
        self.origins['oriented_data_index'] = -1
        oriented_arrays = []
        current_index = 0

        for i, (oridb_id, origin) in enumerate(self.origins.iterrows()):
            try:
                oriented = self.load_oriented_origin_data(origin)
                oriented_arrays.append(oriented)
                self.origins.loc[oridb_id, 'oriented_data_index'] = current_index
                current_index += 1
            except Exception as e:
                print(f"Skipping {oridb_id}: {e}")
            if i % 20 == 0:
                print(f"{i+1}/{len(self.origins)}")

        self.oriented_data = np.stack(oriented_arrays, axis=0)

    # ------------------------------------------------------------------
    # Footprint signal extraction
    # ------------------------------------------------------------------

    def extract_footprint_timecourse(self, oriented_data):
        """
        Extract subnucleosomal footprint signal per time point.
        Applies fragment length and position slices per Li et al.
        Returns 1D array of length n_timepoints.
        """
        from src.config import load_default_chrom_configs
        config1, _ = load_default_chrom_configs()
        t_indices = config1.t_indices()

        center_bin = self.window_bins // 2
        frag_slice = slice(0, self.FRAGMENT_MAX_BP // 10)
        pos_slice  = slice(center_bin, center_bin + self.FOOTPRINT_WIDTH_BINS)

        trace = np.array([
            oriented_data[t, frag_slice, pos_slice].mean()
            for t in t_indices
        ])
        return trace

    def collect_footprint_timecourses(self, origins):
        """
        Run extract_footprint_timecourse for a set of origins.
        Uses self.oriented_data indexed via oriented_data_index.
        Returns 2D array (n_origins x n_timepoints).
        """
        indices = origins.oriented_data_index.values

        traces = np.array([
            self.extract_footprint_timecourse(self.oriented_data[idx])
            for idx in indices
        ])
        return traces


    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def compute_footprint_timecourses(self, origin_sets: dict):
        """
        Compute mean and std footprint timecourses for arbitrarily defined origin sets.
        Stores results as member variable.

        Parameters
        ----------
        origin_sets : dict
            Keys are legend labels, values are origin DataFrames.
            e.g. {'G1 & G2': g1_g2_origins, 'G1 only': g1_only_origins}
        """
        results = {}
        for label, origins in origin_sets.items():
            traces = self.collect_footprint_timecourses(origins)
            results[label] = {
                'traces': traces,
                'mean': traces.mean(axis=0),
                'sd': traces.std(axis=0),
                'n': len(traces)
            }
        self.footprint_timecourse_results = results

    def plot_footprint_timecourse(self, figsize=(8, 4)):
        """
        Plot mean footprint occupancy over time using results from
        compute_footprint_timecourses.
        """
        from src.config import load_default_chrom_configs, retrieve_phase_index_ticks
        import matplotlib.pyplot as plt

        config1, _ = load_default_chrom_configs()
        phase_ticks, edge_ticks, xtick_labels = retrieve_phase_index_ticks(
            't', config1, with_labels=True)

        colors = ['blue', 'green']

        fig, ax = plt.subplots(figsize=figsize)

        for (label, result), color in zip(self.footprint_timecourse_results.items(), colors):
            mean = result['mean']
            sd   = result['sd']
            n    = result['n']
            x    = np.arange(len(mean))

            ax.plot(x, mean, color=color, lw=1.5, label=f"{label} (n={n})")
            # ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.15)

        ax.set_xticks(phase_ticks)
        ax.set_xticklabels(xtick_labels, fontsize=9)
        ax.set_xticks(edge_ticks, minor=True)
        ax.tick_params(axis='x', which='major', length=0)
        ax.tick_params(axis='x', which='minor', length=10)

        for edge in edge_ticks[1::2]:
            ax.axvline(edge, color='#aaa', lw=0.5, ls='--', zorder=0)

        ax.set_xlim(edge_ticks[0], edge_ticks[-1])
        ax.set_ylabel('Mean subnucleosomal occupancy\n'
                      f'(0–{self.FRAGMENT_MAX_BP}bp fragments, '
                      f'{self.FOOTPRINT_WIDTH_BINS * 10}bp downstream of ACS)')
        ax.set_xlabel('')
        ax.legend()
        plt.tight_layout()

    def plot_composite_heatmap(self, origins, label):
        from src.plot_helpers import plot_composite_heatmap as plot_composite_heatmap_helper

        indices = origins['oriented_data_index']
        indices = indices[indices >= 0].values
        average_composite_data = self.oriented_data[indices].mean(axis=0)
        plot_composite_heatmap_helper(average_composite_data, label, len(indices), figsize=(5, 5),
            extent=(-1000, 1000, 0, 260), show_footprint_box=True)

    def compute_efficiency_correlation(self):
        """
        Compute Spearman correlation between per-origin footprint occupancy
        and origin efficiency at each time point.
        Stores results as member variables.
        """
        from scipy.stats import spearmanr

        footprint_origins = self.origins[
            self.origins.footprint_class.isin(['g1_and_g2_footprint', 'g1_only_footprint', 'g2_only_footprint'])
        ]

        traces = self.collect_footprint_timecourses(footprint_origins)
        efficiencies = footprint_origins\
            .derived_origin_efficiency_from_mcguffee_et_al_2013.values

        correlations = np.array([
            spearmanr(traces[:, t], efficiencies).correlation
            for t in range(traces.shape[1])
        ])

        self.footprint_origins = footprint_origins
        self.efficiencies = efficiencies
        self.correlations = correlations


    def plot_efficiency_correlation(self, figsize=(8, 4)):
        """
        Plot Spearman correlation timecourse computed by compute_efficiency_correlation.
        """
        from src.config import load_default_chrom_configs, retrieve_phase_index_ticks

        config1, _ = load_default_chrom_configs()
        phase_ticks, edge_ticks, xtick_labels = retrieve_phase_index_ticks(
            't', config1, with_labels=True)

        fig, ax = plt.subplots(figsize=figsize)
        x = np.arange(len(self.correlations))

        ax.plot(x, self.correlations, color='#333', lw=1.5)
        ax.axhline(0, color='#aaa', lw=0.8, ls='--', zorder=0)

        ax.set_xticks(phase_ticks)
        ax.set_xticklabels(xtick_labels, fontsize=9)
        ax.set_xticks(edge_ticks, minor=True)
        ax.tick_params(axis='x', which='major', length=0)
        ax.tick_params(axis='x', which='minor', length=10)

        for edge in edge_ticks[1::2]:
            ax.axvline(edge, color='#aaa', lw=0.5, ls='--', zorder=0)

        ax.set_xlim(edge_ticks[0], edge_ticks[-1])
        ax.set_ylabel('Spearman correlation\n(footprint occupancy vs. origin efficiency)')
        ax.set_xlabel('')
        ax.set_title(f'Correlation between small fragment footprint & efficiency\nacross time (deconvolved), '
                     f'n={len(self.footprint_origins)} origins',
                     fontweight='demi', fontsize=16)
        plt.tight_layout()
        plt.ylim(0.1, 0.25)