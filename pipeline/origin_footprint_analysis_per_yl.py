
import numpy as np
from matplotlib import pyplot as plt

# Li et al. parameters, converted to 10bp bins
FOOTPRINT_SMALL_MAX_BP = 120
FOOTPRINT_OFFSET = 0  # start of footprint relative to ACS center
FOOTPRINT_WIDTH  = 50 # 50bp downstream
FOOTPRINT_BOX = (FOOTPRINT_OFFSET, FOOTPRINT_OFFSET+FOOTPRINT_WIDTH, 0, FOOTPRINT_SMALL_MAX_BP)


class OriginsFootprintAnalysis:
    """
    Analyzes subnucleosomal footprint occupancy at replication origins
    throughout the cell cycle, reproducing and extending Li et al. 2021.
    
    Oriented by T-rich ACS strand (Belsky et al. 2015):
    - Plus-strand origins: position axis unchanged
    - Minus-strand origins: position axis flipped
    
    Footprint defined as <120bp fragments within ~50bp downstream of ACS 5' end.
    """

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
        from src.timer import Timer

        self.origins['oriented_data_index'] = -1
        oriented_arrays = []
        current_index = 0

        timer = Timer()

        print(f"Collecting oriented MNase data for all origins...")

        for i, (oridb_id, origin) in enumerate(self.origins.iterrows()):
            try:
                oriented = self.load_oriented_origin_data(origin)
                oriented_arrays.append(oriented)
                self.origins.loc[oridb_id, 'oriented_data_index'] = current_index
                current_index += 1
            except Exception as e:
                print(f"Skipping {oridb_id}: {e}")
            if i % 100 == 0:
                timer.print_time(f"{i+1}/{len(self.origins)}")

        self.oriented_data = np.stack(oriented_arrays, axis=0)

    # ------------------------------------------------------------------
    # Footprint signal extraction
    # ------------------------------------------------------------------

    def extract_footprint_timecourse(self, oriented_data, branch):
        """
        Extract subnucleosomal footprint signal per time point.
        Applies fragment length and position slices per Li et al.
        Returns 1D array of length n_timepoints.
        """
        from src.config import load_default_chrom_configs
        config1, _ = load_default_chrom_configs()

        if branch == 'i':
            indices = config1.i_indices()
        elif branch == 't':
            indices = config1.t_indices()
        elif branch == 'b':
            indices = config1.b_indices()

        center_bin = self.window_bins // 2
        frag_slice = slice(0, 120 // 10)
        pos_slice  = slice(center_bin, center_bin + 5)

        trace = np.array([
            oriented_data[t, frag_slice, pos_slice].mean()
            for t in indices
        ])
        return trace

    def collect_footprint_timecourses(self, origins, branch):
        """
        Run extract_footprint_timecourse for a set of origins.
        Uses self.oriented_data indexed via oriented_data_index.
        Returns 2D array (n_origins x n_timepoints).
        """
        indices = origins.oriented_data_index.values

        traces = np.array([
            self.extract_footprint_timecourse(self.oriented_data[idx], branch)
            for idx in indices
        ])
        return traces


    def compute_all_branch_timecourses(self, origin_sets: dict):
        """
        Compute footprint timecourses for all branches and all origin sets in one pass.
        Results stored in self.all_results[branch][label].
        
        Parameters
        ----------
        origin_sets : dict
            Keys are legend labels, values are origin DataFrames.
            e.g. {'Early G1 & G2': early_origins, 'Late G1 & G2': late_origins}
        """
        branches = {'recovery': 'i', 'mother': 't', 'daughter': 'b'}
        self.all_results = {}

        print(f"Computing footprint occupancies for all origins across all branches")

        for branch_name, branch_code in branches.items():
            self.all_results[branch_name] = {}
            for label, origins in origin_sets.items():
                traces = self.collect_footprint_timecourses(origins, branch_code)
                self.all_results[branch_name][label] = {
                    'traces': traces,
                    'mean': traces.mean(axis=0),
                    'median': np.median(traces, axis=0),
                    'sd': traces.std(axis=0),
                    'n': len(traces)
                }

    def compute_all_branch_g1_maxes(self):
        """
        Compute per-origin G1 peak footprint occupancy for all branches and labels.
        Adds 'g1_max' to each entry in self.all_results. Call after
        compute_all_branch_timecourses.
        """
        from src.config import load_default_chrom_configs
        config1, _ = load_default_chrom_configs()
        n_g1 = len(config1.get_Hpositions_for_phase('CG1'))

        for branch_name, branch_results in self.all_results.items():
            for label, result in branch_results.items():
                result['g1_max'] = result['traces'][:, :n_g1].max(axis=1)

    def compute_footprint_timecourses(self, origin_sets: dict, branch):
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
            traces = self.collect_footprint_timecourses(origins, branch)
            results[label] = {
                'traces': traces,
                'mean': traces.mean(axis=0),
                'median': np.median(traces, axis=0),
                'sd': traces.std(axis=0),
                'n': len(traces)
            }
        self.footprint_timecourse_results = results

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_all_conditions_boxplot(self, figsize=(6, 4)):
        """
        Plot G1 peak footprint occupancy for all branches and origin sets.
        Requires compute_all_branch_timecourses and compute_g1_max to have been called.
        
        Groups: Recovery, Mother, Daughter (left to right)
        Within each group: Early vs Late
        """

        early_color = plt.cm.Reds(0.5)
        late_color = plt.cm.Blues(0.5)

        branch_display_names = {
            'recovery': 'Recovery from α-factor G1',
            'mother': 'Mother',
            'daughter': 'Daughter',
        }
        colors = {
            'Early G1 & G2': early_color,
            'Late G1 & G2': late_color
        }

        fig, ax = plt.subplots(figsize=figsize)

        # Build ordered list of (group_label, origin_label, g1_max_values)
        box_data = []
        tick_positions = []
        tick_labels = []
        group_centers = {}

        pos = 1
        for branch_name, display_name in branch_display_names.items():
            group_positions = []
            for label, result in self.all_results[branch_name].items():
                box_data.append(result['g1_max'])
                k = len(result['g1_max']) # Each group should have the same k
                tick_positions.append(pos)
                tick_labels.append(label.replace(' G1 & G2', ''))  # shorten label
                group_positions.append(pos)
                pos += 1
            group_centers[display_name] = np.mean(group_positions)
            pos += 1  # gap between groups

        # Draw boxplots
        bp = ax.boxplot(
            box_data,
            positions=tick_positions,
            showmeans=True,
            showfliers=False,
            patch_artist=True,
            widths=0.6,
        )

        # Color boxes by Early/Late
        all_labels = [
            label
            for branch_name in branch_display_names
            for label in self.all_results[branch_name]
        ]
        for patch, label in zip(bp['boxes'], all_labels):
            color = colors[label]
            patch.set_facecolor(color)
            patch.set_alpha(0.3)

        for median in bp['medians']:
            median.set_color('orange')

        # X axis
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=9)

        # Group labels below x axis
        for display_name, center in group_centers.items():
            ax.text(center, 1.0125, display_name, ha='center', va='bottom',
                    fontsize=10, fontweight='regular',
                    transform=ax.get_xaxis_transform())

        # Group separator lines
        for branch_name, display_name in list(branch_display_names.items())[:-1]:
            last_pos = group_centers[display_name] + 1
            ax.axvline(last_pos + 0.5, color='#ccc', lw=0.8, ls='--')

        ax.set_ylim(-0.5, 6.25)
        ax.set_ylabel('Occupancy')
        ax.set_title('Maximal footprint occupancy in G1 per branch',
                     fontweight='demi', fontsize=16, pad=26)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=early_color, alpha=0.3, label=f'Early, n={k}'),
            Patch(facecolor=late_color, alpha=0.3, label=f'Late, n={k}'),
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        plt.tight_layout()

        return fig, ax

    def plot_footprint_timecourse(self, branch_name, normalize=False, figsize=(6, 3), 
        colors=[plt.cm.Reds(0.5), plt.cm.Blues(0.5)],
        plot_max_g1=False, agg_method='mean'):
        """
        Plot mean footprint occupancy over time using results from
        compute_footprint_timecourses.
        """
        from src.config import load_default_chrom_configs, retrieve_phase_index_ticks
        import matplotlib.pyplot as plt

        branch_codes = {'recovery': 'i', 'mother': 't', 'daughter': 'b'}
        branch_code = branch_codes[branch_name]

        config1, _ = load_default_chrom_configs()
        phase_ticks, edge_ticks, xtick_labels = retrieve_phase_index_ticks(
            branch_code, config1, with_labels=True)

        fig, ax = plt.subplots(figsize=figsize)

        for (label, result), color in zip(self.all_results[branch_name].items(), colors):
            y_values = result[agg_method]
            sd   = result['sd']
            n    = result['n']
            x    = np.arange(len(y_values))

            if normalize:
                y_values = (y_values-y_values.min())/(y_values.max()-y_values.min())

            label = label.replace(' G1 & G2', '')

            ax.plot(x, y_values, color=color, lw=2.5, label=f"{label} (n={n})")
            # ax.fill_between(x, y_values - sd, y_values + sd, color=color, alpha=0.15)

            if plot_max_g1:
                len_g1 = len(config1.get_Hpositions_for_phase('CG1'))
                g1_values =  y_values[:len_g1]
                max_g1_index = g1_values.argmax()
                ax.scatter(max_g1_index, y_values[max_g1_index], color='green', marker='^',
                    label='G1 max', s=40, zorder=10)

        ax.set_xticks(phase_ticks)
        ax.set_xticklabels(xtick_labels, fontsize=9)
        ax.set_xticks(edge_ticks, minor=True)
        ax.tick_params(axis='x', which='major', length=0)
        ax.tick_params(axis='x', which='minor', length=10)

        for edge in edge_ticks[1::2]:
            ax.axvline(edge, color='#aaa', lw=0.5, ls='--', zorder=0)

        ax.set_xlim(edge_ticks[0], edge_ticks[-1])
        ax.set_ylabel('Occupancy')
        ax.set_xlabel('')

        if agg_method == 'mean':
            ax.set_ylim(0.75, 2.1)
        else:
            ax.set_ylim(0.0, 1.6)

        ax.legend()
        plt.tight_layout()

        if agg_method == 'mean':
            title = f"Average small fragment footprint, {branch_name}"
        else:
            title = f"Median small fragment footprint, {branch_name}"

        plt.title(title, pad=9, fontweight='demi', fontsize=16)


    def plot_footprint_timecourse_all_branches(self, normalize=False, figsize=(6, 3.5),
            colors=[plt.cm.Reds(0.5), plt.cm.Blues(0.5)],
            plot_max_g1=False, agg_method='mean'):
        """
        Plot mean footprint occupancy for all three branches overlaid on a single shared
        real-time axis. Each branch's G1 occupies its own time span, converging at S-phase
        entry. S and G2/M are shared and overlap exactly.
        Branch styles: mother=dashed, daughter=solid, recovery=dotted.
        """
        from src.config import load_default_chrom_configs, retrieve_phase_ticks
        import matplotlib.patches as mpatches
        import matplotlib.lines as mlines

        branch_styles = {
            'recovery': {'ls': 'dotted',  'code': 'i'},
            'mother':   {'ls': 'solid', 'code': 't'},
            'daughter': {'ls': 'dashed',  'code': 'b'},
        }

        config1, config2 = load_default_chrom_configs()

        from src.config import get_average_timepoints_for_branch

        fig, ax = plt.subplots(figsize=figsize)

        for branch_name, style in branch_styles.items():
            branch_code = style['code']
            timepoints = get_average_timepoints_for_branch(config1, config2, branch_code)

            for (label, result), color in zip(self.all_results[branch_name].items(), colors):
                y_values = result[agg_method].copy()
                n = result['n']


                if normalize:
                    y_values = (y_values - y_values.min()) / (y_values.max() - y_values.min())

                clean_label = label.replace(' G1 & G2', '')

                ax.plot(timepoints, y_values, color=color, lw=2.5, ls=style['ls'],
                        label=f"{clean_label} — {branch_name} (n={n})")

                if plot_max_g1 and branch_name == 'mother':
                    len_g1 = len(config1.get_Hpositions_for_phase('CG1'))
                    g1_values = y_values[:len_g1]
                    max_g1_index = g1_values.argmax()
                    ax.scatter(timepoints[max_g1_index], y_values[max_g1_index],
                               color='green', marker='^', label='G1 max', s=40, zorder=10)

        # Ticks from mother branch — S and G2/M positions are shared across all branches
        phase_ticks, edge_ticks, xtick_labels = retrieve_phase_ticks(
            'b', config1, config2, with_labels=True)

        # Skip G1 mid tick; keep only S and G2/M labels
        ax.set_xticks(phase_ticks)
        ax.set_xticklabels(xtick_labels, fontsize=9)
        ax.set_xticks(edge_ticks, minor=True)
        ax.tick_params(axis='x', which='major', length=0)
        ax.tick_params(axis='x', which='minor', length=10)

        for edge in edge_ticks[1::2]:
            ax.axvline(edge, color='#aaa', lw=0.5, ls='--', zorder=0)

        # xlim: leftmost G1 start across all branches, rightmost shared G2/M end
        xlim_left = edge_ticks[0]
        ax.set_xlim(xlim_left, edge_ticks[-1])
        ax.set_ylabel('Occupancy')
        ax.set_xlabel('Time, min')

        if agg_method == 'mean':
            ax.set_ylim(0.67, 2.25)
        else:
            ax.set_ylim(0.0, 1.6)

        # Condition handles (color, no specific line style)
        color_handles = [
            mpatches.Patch(color=colors[0], label=f'Early, n={n}'),
            mpatches.Patch(color=colors[1], label=f'Late, n={n}'),
        ]

        # Branch handles (gray, distinct line styles)
        gray = '#555555'
        branch_handles = [
            mlines.Line2D([], [], color=gray, lw=2, ls='dotted', label='Recovery from α-factor G1'),
            mlines.Line2D([], [], color=gray, lw=2, ls='solid',  label='Mother'),
            mlines.Line2D([], [], color=gray, lw=2, ls='dashed', label='Daughter'),
        ]

        ax.legend(handles=color_handles + branch_handles, fontsize=8)
        # ax.legend(fontsize=8)

        plt.tight_layout()

        if agg_method == 'mean':
            title = "Average small fragment footprint"
        else:
            title = "Median small fragment footprint"

        plt.title(title, pad=9, fontweight='demi', fontsize=16)


    def plot_composite_heatmap(self, origins, label):
        from src.plot_helpers import plot_composite_heatmap as plot_composite_heatmap_helper

        indices = origins['oriented_data_index']
        indices = indices[indices >= 0].values
        average_composite_data = self.oriented_data[indices].mean(axis=0)
        plot_composite_heatmap_helper(average_composite_data, label, len(indices), figsize=(5, 5),
            extent=(-1000, 1000, 0, 260), show_footprint_box=True,
            footprint_box=FOOTPRINT_BOX)

    def compute_efficiency_correlation(self, selected_origins=None):
        """
        Compute Spearman correlation between per-origin footprint occupancy
        and origin efficiency at each time point.
        Stores results as member variables.
        """
        from scipy.stats import spearmanr

        if selected_origins is None:
            footprint_origins = self.origins[
                self.origins.footprint_class.isin(['g1_and_g2_footprint', 'g1_only_footprint', 'g2_only_footprint'])
            ]
        else:
            footprint_origins = selected_origins

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
        # plt.ylim(0.1, 0.25)
