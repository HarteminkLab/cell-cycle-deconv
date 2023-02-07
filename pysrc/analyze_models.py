
import matplotlib.pyplot as plt
import numpy as np


class Model:

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

        intervals['b'] = [np.array(b) + lengths['delta'] + lengths['lambda']
            for b in intervals['b']]

        intervals['t'] = [np.array(t) + lengths['lambda']
                    for t in intervals['t']]

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

        cmap = plt.get_cmap('tab10')
        colors = {}
        keys = list(description.keys())
        for i in range(len(description.keys())):
            colors[keys[i]] = cmap(i)

        def _plot(x, y, label):
            y = [y] * len(x)
            return plt.scatter(x, y, s=2, color=colors[label], label=label)

        y_intervals = {'i': 0, 't': 1, 'b': -1}
        plt.figure(figsize=(12, 6))
        legend_items = []
        legend_labels = []

        for cc_label, v in description.items():
            cur_sum = 0

            for i_name, index in v.items():
                x = intervals[i_name][index]
                cur_sum += len(x)

            for i_name, index in v.items():
                y = y_intervals[i_name]
                x = intervals[i_name][index]
                ret = _plot(x, y, cc_label)
                plt.text(x[len(x)//2], y-0.05, f"{len(x)}", va='top')
                
                label = f"{cc_label}, N={cur_sum}"
                if label not in legend_labels:
                    legend_items.append(ret)
                    legend_labels.append(label)

        plt.ylim(-1.2, 1.2)
        interval_counts = self.interval_counts

        lambd = self.lengths['lambda']
        plt.legend(legend_items, legend_labels)
        plt.text(lambd-1, 0+0.05, f"Initial N={interval_counts['i']}", ha='right')
        plt.text(lambd-1, 1+0.05, f"Top N={interval_counts['t']}, $\\lambda$", ha='right')
        plt.text(lambd-1, -1+0.05, f"Bottom N={interval_counts['b']}, $\\lambda + \\delta$",
            ha='right')
        plt.title(self.model_path)
        plt.axvline(lambd, color='gray', linestyle='dotted', lw=1)
