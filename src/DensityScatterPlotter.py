
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt


class DensityScatterPlotter:
    """
    Class to make density scatter plots easier. Can configure as an object and 
    feed x and y data.
    """

    def __init__(self):

        self.bw = [0.01, 0.01]
        self.cmap = 'viridis'
        self.s = 10
        self.alpha = 1.
        self.logz = False

    def set_data(self, x, y):
        self.x = x
        self.y = y

    def plot_ax(self, ax, plot_colorbar=False, vmax=None):

        x, y = self.x, self.y
        s = self.s
        cmap = self.cmap
        bw = self.bw
        zorder = 1

        try:
            kde = sm.nonparametric.KDEMultivariate(data=[x, y], var_type='cc', bw=bw)
            z = kde.pdf([x, y])
        except ValueError:
            z = np.array([0] * len(x))

        sorted_idx = np.argsort(z)
        x, y, z = x[sorted_idx], y[sorted_idx], z[sorted_idx]

        if self.logz:
            z = np.log2(z+1.)

        scatter = ax.scatter(x, y, lw=1, facecolor='None', edgecolor='#ddd', s=s+3,
            alpha=1., rasterized=True, zorder=0)

        scatter = ax.scatter(x, y, c=z, lw=0, edgecolor=None, s=s, cmap=cmap,
           alpha=self.alpha, rasterized=True, zorder=zorder+1, vmax=vmax)

        if plot_colorbar: 
            plt.colorbar(scatter)

        return ax