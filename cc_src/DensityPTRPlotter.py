


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.sgd import get_orfname
import matplotlib.patheffects as patheffects
from src.DensityScatterPlotter import DensityScatterPlotter


class DensityPTRPlotter:

    def __init__(self, x, y, xlabel, ylabel, 
                 xlim, ylim, 
                 title, genes_to_plot=[]):

        self.x_data = x
        self.y_data = y
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xlim = xlim
        self.ylim = ylim
        self.title = title
        self.genes_to_plot = genes_to_plot
        self.segmentation_lines = 2
        self.density_scatter_pltr = DensityScatterPlotter()
        self.density_scatter_pltr.cmap = 'magma_r'
        self.density_scatter_pltr.alpha = 1.
        self.scale = 'log'

    def plot(self):

        # Segments the lower left quadrant
        lower_segment_val = self.segmentation_lines

        title = self.title
        xlabel = self.xlabel
        ylabel = self.ylabel
        genes_to_plot = self.genes_to_plot
        x_data = self.x_data.copy()
        y_data = self.y_data.copy()

        ylim, xlim = self.ylim, self.xlim

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        plt.subplots_adjust(top=0.85)

        sel_plot_ylim = ylim[1]
        y_data[y_data > sel_plot_ylim] = sel_plot_ylim

        sel_plot_xlim = xlim[1]
        x_data[x_data > sel_plot_xlim] = sel_plot_xlim

        plotter = self.density_scatter_pltr
        plotter.set_data(x_data, y_data)
        plotter.plot_ax(ax, plot_colorbar=False)

        # --------- Plot selected genes ----------

        for gene_name in genes_to_plot:

            orfname = get_orfname(gene_name)
            x, y = x_data.loc[orfname], y_data.loc[orfname]

            # Offscreen so adjust the text alignment
            va = 'bottom'
            ha = 'left'

            # if y >= sel_plot_ylim:
            #     va = 'top'

            # if x >= sel_plot_xlim:
            #     ha = 'right'

            zorder = 3+int(np.log2(y)*10) # Plot higher values above, a little hacky
            _plot_ann_text(x, y, gene_name, zorder=zorder, ha=ha, va=va, fontsize=8,
                bordercolor='gray')
            
            ax.scatter(x, y, s=30, edgecolor='blue', facecolors='none', 
                zorder=3,
                marker='D')

        # ----------------------------------------

        # Segmentation lines
        max_lim = max(xlim[1], ylim[1])
        ax.plot([0, max_lim], [0, max_lim], color='red', linestyle='dashed', 
            alpha=0.75, lw=1, zorder=100)

        # x_line = [0, xlim]
        # y_line = [lower_segment_val,lower_segment_val]

        # # Draw the vertical and horizontal lines segmenting the lower threshold
        # ax.plot(x_line, y_line, c='red', linestyle='dashed', lw=1)

        # x_line = [lower_segment_val,lower_segment_val]
        # y_line = [0, ylim]
        # ax.plot(x_line, y_line, c='red', linestyle='dashed', lw=1)

        # ------- Let's add some counts for each segment -------

        data_combined = pd.DataFrame({'x': x_data, 'y': y_data})
        n = len(data_combined)

        # def _plot_count(x, y, m, n):
        #     text = f"{m}\n({m/n*100:.1f}%)"
        #     _plot_ann_text(x, y, text, zorder=100, fontsize=13, ha='right',
        #         va='top', textcolor='red', bordercolor='white')

        # lower_quadrant_pos = lower_segment_val*0.99
        # mid_quadrant_pos = xlim*0.7
        # upper_quadrant_pos = xlim*0.99

        # # --- Lower left quadrant -------

        # sel_data = data_combined[(data_combined['x'] < lower_segment_val) & 
        # (data_combined['y'] < lower_segment_val)]
        # m = len(sel_data)
        # _plot_count(lower_quadrant_pos, 
        #     lower_quadrant_pos, m, n)

        # # ----- Upper left quadrant ------

        # sel_data = data_combined[(data_combined['x'] < lower_segment_val) & 
        #                          (data_combined['y'] >= lower_segment_val)]
        # m = len(sel_data)
        # _plot_count(lower_quadrant_pos, 
        #     mid_quadrant_pos, m, n)

        # # ----- Lower right quadrant ------

        # sel_data = data_combined[(data_combined['x'] >= lower_segment_val) & 
        #                          (data_combined['y'] < lower_segment_val)]
        # m = len(sel_data)
        # _plot_count(mid_quadrant_pos, lower_quadrant_pos, m, n)

        # # ----- Upper left main triangle ------

        # sel_data = data_combined[(data_combined['x'] >= lower_segment_val) & 
        #                          (data_combined['y'] >= lower_segment_val) & 
        #                          (data_combined['y'] >= data_combined['x'])]
        # m = len(sel_data)
        # _plot_count(mid_quadrant_pos, upper_quadrant_pos, m, n)

        # # ----- Lower right main triangle ------

        # sel_data = data_combined[(data_combined['x'] >= lower_segment_val) & 
        #                          (data_combined['y'] >= lower_segment_val) & 
        #                          (data_combined['y'] < data_combined['x'])]
        # m = len(sel_data)
        # _plot_count(upper_quadrant_pos, mid_quadrant_pos, m, n)

        # ----------------------------------------

        # if self.scale == 'log':
        #     ax.set_yscale('log', base=10)
        #     ax.set_xscale('log', base=10)

        ax.set_ylim(*ylim)
        ax.set_xlim(*xlim)

        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)

        plt.suptitle(f'{title}, n={n}', fontsize=18)

        return ax

def _plot_ann_text(x, y, text, fontsize=16, 
    ha='left', va='bottom', zorder=1, 
    textcolor='white', bordercolor='black'):
    """
    Can't plot this to ax for some reason, revisit this later, check cd 
    paper code for plotting arbitrary text
    """
    plt.text(x, y, text, fontsize=fontsize, weight='normal', style='italic', 
        path_effects=[patheffects.withStroke(linewidth=3, foreground=bordercolor)],
        color=textcolor, va=va, ha=ha, zorder=zorder)
