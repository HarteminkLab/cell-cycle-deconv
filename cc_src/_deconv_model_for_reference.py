
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class DeconvolvedAnalysis:

    def __init__(self, model):

        self.load_model(model.config.model_wt1_file)
        # self.load_deconvolved_data(deconvolved_output_path)

        # gene_names_d = pd.read_csv(f'{parentdir}/all_genes.csv').columns.values

        # deconv_f = pd.read_csv(f'{parentdir}/all_genes_f.csv', header=None)
        # deconv_f['gene_name'] = gene_names_d
        # deconv_f = deconv_f.set_index('gene_name')

        # deconv_g = pd.read_csv(f'{parentdir}/all_genes_g.csv', header=None)
        # deconv_g['gene_name'] = gene_names_d
        # deconv_g = deconv_g.set_index('gene_name')


        # # TODO: we are copying replicate 2 twice for this deconvolution, so we can just ignore half of the columns
        # # Eventually we need to update the deconvolution code to handle 1 or 2 replicates
        # deconv_g = deconv_g[deconv_g.columns[range(len(timepoints))]]
        # deconv_g.columns = timepoints

        # self.timepoints = timepoints
        # self.deconv_g = deconv_g
        # self.deconv_f = deconv_f
        # self.gene_names_d = gene_names_d

        # self.deconv_f_df = pd.DataFrame(self.deconv_f, index=self.gene_names_d)
        # self.deconv_g_df = pd.DataFrame(self.deconv_g, index=self.gene_names_d)


    def compute_predicted_g(self):
        f = self.deconv_f
        g = self.deconv_g

        H = self.H

        # Just take half of H, as we are only using one replicate
        # TODO: Revisit this
        half_H = H[0:g.shape[1]]

        f.shape, half_H.T.shape, 
        predicted_g = np.matmul(f.values, half_H.T.values)

        self.predicted_g = pd.DataFrame(predicted_g, index=g.index.values, columns=g.columns)


    def load_timepoints_from_config(self, deconv_config_path):

        # Read the deconvolution config file to get the timepoints
        with open(deconv_config_path, 'r') as t:
            lines = t.readlines()
        
        # Find where the wild type 2  timepoints are defined
        for l in lines:
            line = l.strip()
            if line.startswith("WT2_TP"):
                timepoints_str = line.split('=')[1].replace('[', '').replace('];', '')
                timepoints = timepoints_str.split('\t')

                # Parse as matlab range string (start:step:end)
                if ':' in timepoints_str:
                    first, step, end = tuple([int(val) for val in "38:16:262".split(':')])
                    timepoints = np.arange(first, end, step)
                # Parse as actual integer values (0, 10, 20, etc..)
                else:
                    timepoints = [int(t) for t in timepoints]

        return timepoints

    def plot_H(self):
        plt.imshow(self.H, aspect='auto')
        plt.yticks([])
        plt.xticks([])

    def load_model(self, model_path):

        self.model_path = model_path
        with open(model_path, 'r') as f:
            lines = f.readlines()

        line_splits = [[t.strip() for t in line.split(' ')] for line in lines]

        # Parse the model
        state = None
        description = {}
        intervals = {}
        lengths = {}
        for l in line_splits:

            # Parse state
            if l[0] == '#' and l[1] == 'lengths':
                state = "Lengths"
            elif l[0] == '#' and l[1] == 'description':
                state = "Description"
            elif l[0] == '#' and (l[1] == 'i' or l[1] == 't' or l[1] == 'b' or l[1] == 'h'):
                state = l[1]

            # Parse state contents
            if state == 'Lengths':
                if l[0] == "#":
                    pass
                else:
                    lengths[l[0]] = float(l[1])
            elif state == "Description":
                if l[0] == "#":
                    pass
                else:
                    description[l[0]] = l[1:]
                    desc_indices = {}
                    for i in range(1, len(l)-1, 2):
                        desc_indices[l[i]] = int(l[i+1])
                    description[l[0]] = desc_indices
            elif state == "i" or state == 't' or state == 'h' or state == 'b':
                if l[0] == "#":
                    intervals[state] = []
                else:
                    cur = intervals[state]
                    cur.append([float(t) for t in l])
            
        self.description = description
        self.lengths = lengths
        self.intervals = intervals

        interval_counts = {}
        for interval, vals in self.intervals.items():
            interval_counts[interval] = np.array([len(v) for v in vals]).sum()
        self.interval_counts = interval_counts
        state_sums = {}
        for cc_label, v in description.items():
            cur_sum = 0

            for i_name, index in v.items():
                x = intervals[i_name][index]
                cur_sum += len(x)

            label = f"{cc_label}, N={cur_sum}"
            state_sums[cc_label] = cur_sum
        self.state_sums = state_sums
        self.compute_phase_indices()

    def compute_phase_indices(self, log=False):
        """
        We have the description which defines how the H matrix is constructed
        read through the description in order and get the interval lengths 
        from the intervals object to construct the indexing definitions of the H matrix
        """

        model_desc = self.description
        keys = list(self.description.keys())

        phase_indices = {}

        current_length = 0
        for phase in keys:
            if log: print(f"The {phase} ", end="")
            phase_branch_dic = model_desc[phase]
            for branch_name, interval_index in phase_branch_dic.items():
                
                phase_len = len(self.intervals[branch_name][interval_index])
                start = current_length
                end = current_length+phase_len-1
                
                if log:
                    print(f"defined by branch {branch_name} on interval {interval_index} is {phase_len} long")
                    print(f"  So, the indices in H are from {start} to {end}\n")

                current_length = end

                phase_indices[phase] = np.arange(start, end)
                
                # For the construction of H, we only need one of the branch's counts
                break
        self.phase_indices = phase_indices


    # need to reverse lookup the phases from the branch name
    def get_phases_for_branch(self, search_branch):
        collect = []

        for phase, branch_i_kv in self.description.items():
            for branch, index in branch_i_kv.items():
                if branch == search_branch: collect.append((phase, index))
        return collect

    def get_phase_timepoints_indices_for_branch(self, search_branch):
        phases_i_kv = self.get_phases_for_branch(search_branch)
        collect = []
        for phase, interval in phases_i_kv:
            collect.append((phase, self.intervals[search_branch][interval], self.phase_indices[phase]))
            
        return collect

    def plot_deconvolved_gene(self, gene_name, deconv_f, deconv_g, title=None):

        # If I want to plot the initial branch,
        # I need the branch name: i
        # the phases:   R, CG1, postG1
        # and their associated timepoints and indices:
            # Indices is done
            # timepoints are looked up in the intervals object

        from src.sgd import get_orfname

        if gene_name in deconv_f.index.values:
            orfname = gene_name
        else:
            orfname = get_orfname(gene_name)

        f_data = deconv_f.loc[orfname]
        g_data = deconv_g.loc[orfname]
        pred_g_data = self.predicted_g.loc[orfname]

        fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(16, 3))

        # Plot the raw data
        ax0.plot(g_data, c=self.color_for_key('raw'), lw=3)
        ax0.plot(pred_g_data, c=self.color_for_key('fit'), lw=3)

        # Initial branch
        phases_tp_i = self.get_phase_timepoints_indices_for_branch('i')
        for phase, tp, indices in phases_tp_i:
            ax1.plot(tp[:-1], f_data[indices], lw=3, c=self.color_for_key(phase))
            ax1.set_title('Initial')

        # Top branch
        phases_tp_i = self.get_phase_timepoints_indices_for_branch('t')
        for phase, tp, indices in phases_tp_i:
            ax2.plot(tp[:-1], f_data[indices], lw=3, c=self.color_for_key(phase))
            ax2.set_title('Top')

        # Bottom branch
        phases_tp_i = self.get_phase_timepoints_indices_for_branch('b')
        for phase, tp, indices in phases_tp_i:
            ax3.plot(tp[:-1], f_data[indices], lw=3, c=self.color_for_key(phase))
            ax3.set_title('Bottom')

        plt.subplots_adjust(top=0.8)

        if title is None:
            if gene_name == orfname:
                plt.suptitle(gene_name)
            else:
                plt.suptitle(f"{gene_name} / {orfname}")
        else:
            plt.suptitle(title)

        return fig


    def calculate_ptrs(self):
        from src.peak_to_trough import calculate_ptr, calculate_combined_mother_daughter_ptr

        mother_indices = self.indices_for_branch('t')
        daughter_indices = self.indices_for_branch('b')

        raw_ptrs = self.deconv_g_df.apply(calculate_ptr, axis=1)
        mother_ptrs = self.deconv_f_df[mother_indices].apply(calculate_ptr, axis=1)
        daughter_ptrs = self.deconv_f_df[mother_indices].apply(calculate_ptr, axis=1)
        combined_ptr = calculate_combined_mother_daughter_ptr(mother_ptrs, daughter_ptrs)

        ptrs = self.deconv_f_df[[]].copy()

        ptrs['raw_ptr'] = raw_ptrs
        ptrs['mother_ptr'] = mother_ptrs
        ptrs['daughter_ptr'] = daughter_ptrs
        ptrs['combined_ptr'] = combined_ptr

        self.ptrs = ptrs
        return ptrs

    def plot_ptrs(self, genes_to_plot=[], title="", xlim=(0, 500), ylim=(0, 500)):

        data = self.ptrs.copy()

        from src.sgd import get_orfname
        from src.DensityPTRPlotter import DensityPTRPlotter

        x = data['raw_ptr']
        y = data['combined_ptr']

        ptr_plotter = DensityPTRPlotter(x, y,
                             title=f'{title}',
                             xlabel='Original PTR',
                             ylabel='Deconvolved PTR',
                             xlim=xlim,
                             ylim=ylim,
                             genes_to_plot=plotting_gene_set())
        ptr_plotter.plot()

    def plot_entropy_ptrs(self, genes_to_plot=[], title="", xlim=(0, 3), ylim=(0, 3), scale='log',
        bw=(0.001, 0.001)):

        data = self.ptrs.copy()

        from src.sgd import get_orfname
        from src.DensityPTRPlotter import DensityPTRPlotter

        x = data['raw_ptr']
        y = data['combined_ptr']

        ptr_plotter = DensityPTRPlotter(x, y,
                             title=f'{title}',
                             xlabel='Original PTR',
                             ylabel='Deconvolved PTR',
                             xlim=xlim,
                             ylim=ylim,
                             genes_to_plot=plotting_gene_set())

        sctr_pltr = ptr_plotter.density_scatter_pltr
        sctr_pltr.scale = scale
        sctr_pltr.cmap = 'viridis'
        sctr_pltr.vmax = 10
        sctr_pltr.bw = bw
        sctr_pltr.alpha = 0.25

        # ptr_plotter.segmentation_lines = 1.3
        ptr_plotter.plot()


    def indices_for_branch(self, branch):
        """Return the subset of deconvolved f data for a branch. Looks up the appropriate indices
        assigned to the branch"""
        phases_tp_i = self.get_phase_timepoints_indices_for_branch(branch)
        branch_indices = np.array([])
        for phase, tp, indices in phases_tp_i:
            branch_indices = np.concatenate([branch_indices, indices])
        return branch_indices 


    def color_for_key(self, key):

        color_map = {
             "raw": np.array([158, 50, 50])/255.,
             "fit": np.array([145, 180, 98])/255.,
             "R": np.array([199, 148, 144])/255.,
             "RG1": np.array([199, 148, 144])/255.,
             "CG1": np.array([147, 168, 198])/255.,
             "DG1": np.array([165, 197, 204])/255.,
             "postG1": np.array([223, 192, 158])/255.,
             "H": np.array([100, 100, 100])/255.
        }

        return color_map[key]


    def plot_model(self):
        intervals = self.intervals
        description = self.description

        def _plot(x, y, label):
            color = self.color_for_key(label)
            y = [y] * len(x)
            return plt.scatter(x, y, s=20, marker='s', color=color, label=label)

        y_intervals = {'i': 0, 't': 1, 'b': -1, 'h': -2}
        plt.figure(figsize=(12, 6))
        legend_items = []
        legend_labels = []

        # Define the right most boundary of the branches, they should all be the same
        # So just take i
        def _op_list_arrs(num_list, op, outer_op=None): 
            if outer_op is None: outer_op = op
            return outer_op([op(m) for m in num_list])

        max_right = _op_list_arrs(intervals['i'], max)
        min_top = _op_list_arrs(intervals['t'], min)
        min_bottom = _op_list_arrs(intervals['b'], min)
        offset_top = max_right-min_top
        offset_bottom = max_right-min_bottom

        for cc_label, v in description.items():
            cur_sum = 0

            for interval_name, index in v.items():
                x = intervals[interval_name][index]
                cur_sum += len(x)

            for interval_name, index in v.items():
                y = y_intervals[interval_name]
                x = np.array(intervals[interval_name][index])

                # The right end are all aligned

                # Take that maximum value, subtract by the start of the I and B branches
                # and offset by the difference

                if interval_name == 't':
                    x = x+offset_top
                elif interval_name == 'b':
                    x = x+offset_bottom

                ret = _plot(x, y, cc_label)
                plt.text(x[len(x)//2], y-0.05, f"{cc_label}\n{len(x)}", fontsize=16, 
                    va='top',
                    ha='center')
                
                label = f"{cc_label}, N={cur_sum}"
                if label not in legend_labels:
                    legend_items.append(ret)
                    legend_labels.append(label)

        plt.ylim(-1.2, 1.2)
        plt.yticks([])
        plt.xticks([])

        plt.gca().spines[['bottom', 'left', 'right', 'top']].set_visible(False)

        interval_counts = self.interval_counts

        lambd = self.lengths['delta']
        plt.plot([max_right, max_right], [y_intervals['t'], y_intervals['b']], 
            color='gray', 
            linestyle='dotted', lw=1, zorder=0)
        
        def _plot_branch_text(x, y, name):
            plt.text(x-3, y, name, ha='right', fontsize=16, va='center')

            _plot_branch_text(_op_list_arrs(intervals['i'], min), y_intervals['i'], 
                f"Initial\nN={_op_list_arrs(intervals['i'], len, sum)}")
            _plot_branch_text(max_right, y_intervals['t'], 
                f"Top\nN={_op_list_arrs(intervals['t'], len, sum)}")
            _plot_branch_text(max_right, y_intervals['b'], 
                f"Bottom\nN={_op_list_arrs(intervals['b'], len, sum)}")
            plt.xticks([])
            plt.yticks([])

            ax = plt.gca()
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.patch.set_alpha(0.0)


        def plot_g_f_hm(self):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2, 8))

            ax1.imshow(np.log2(deconv_g+1), aspect='auto', cmap='plasma_r', vmax=13.31)
            ax1.set_yticks([])

            ax2.imshow(np.log2(deconv_f+1), aspect='auto', cmap='viridis_r', vmax=13.31)
            ax2.set_yticks([])


def plotting_gene_set():
    """The set of genes to plot from Xin, 2011"""
    return ['MF(ALPHA)1', 'CLN2', 'PCL1', 'CDC20', 'SIC1', 'SSK22']
