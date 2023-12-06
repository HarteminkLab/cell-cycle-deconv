
import matplotlib.pyplot as plt
import numpy as np


class ModelFile:

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
            elif l[0] == '#' and (l[1] == 'i' or l[1] == 't' or l[1] == 'b'):
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
            elif state == "i" or state == 't' or state == 'b':
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

    def plot_model(self):
        intervals = self.intervals
        description = self.description

        color_mapping = {}
        color_names = ["raw", "fit", "R", "RG1", "CG1", "DG1", "postG1"];
        colors = [np.array([158, 50, 50])/255.,
             np.array([145, 180, 98])/255.,
             np.array([199, 148, 144])/255.,
             np.array([199, 148, 144])/255.,
             np.array([147, 168, 198])/255.,
             np.array([165, 197, 204])/255.,
             np.array([223, 192, 158])/255.]
        for i in range(len(color_names)):
            color_mapping[color_names[i]] = colors[i]

        def _plot(x, y, label):
            y = [y] * len(x)
            return plt.scatter(x, y, s=20, marker='s', color=color_mapping[label], label=label)

        y_intervals = {'i': 0, 't': 1, 'b': -1}
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
                plt.text(x[len(x)//2], y-0.05, f"{cc_label}\n{len(x)}", fontsize=16, va='top',
                    ha='center')
                
                label = f"{cc_label}, N={cur_sum}"
                if label not in legend_labels:
                    legend_items.append(ret)
                    legend_labels.append(label)

        plt.ylim(-1.2, 1.2)
        interval_counts = self.interval_counts

        lambd = self.lengths['delta']
        plt.plot([max_right, max_right], [y_intervals['t'], y_intervals['b']], color='gray', 
            linestyle='dotted', lw=1, zorder=0)
        
        def _plot_branch_text(x, y, name):
            plt.text(x-3, y, name, ha='right', fontsize=16, va='center')

        _plot_branch_text(_op_list_arrs(intervals['i'], min), y_intervals['i'], 
            f"Initial\nN={_op_list_arrs(intervals['i'], len, sum)}")
        _plot_branch_text(max_right, y_intervals['t'], f"Top\nN={_op_list_arrs(intervals['t'], len, sum)}")
        _plot_branch_text(max_right, y_intervals['b'], f"Bottom\nN={_op_list_arrs(intervals['b'], len, sum)}")
        plt.xticks([])
        plt.yticks([])

        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.patch.set_alpha(0.0)
