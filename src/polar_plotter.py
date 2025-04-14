
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.peak_to_trough import compute_quantile_ptr_2d
from scipy import stats

class PolarPlot:
	def __init__(self):
		"""Initialize the PolarPlot class with empty data."""
		self.timepoints = None
		self.amplitudes = None
		self.fig = None
		self.ax = None
		self.ylims = -.5, 4
		
	def set_data(self, data, tp_key='peak_idx', ampl_key='ptr', bin_key='bin',
		min_time=0, suffix='', max_time=64):
		"""
		Set the data for the polar plot.
		
		Parameters:
		-----------
		timepoints : array-like
			Time points to be converted to angles in polar coordinates
		amplitudes : array-like
			Amplitude values for each time point (radial distance)
		"""
			
		self.data = data.copy()
		self.min_time = min_time
		self.max_time = max_time

		# Convert timepoints to polar angles
		peak_angles = self._convert_to_polar_angles(self.data['peak_idx'+suffix])
		trough_angles = self._convert_to_polar_angles(self.data['trough_idx'+suffix])
		self.data['peak_angle'] = peak_angles
		self.data['trough_angle'] = trough_angles
		self.tp_key = tp_key+suffix
		self.ampl_key = ampl_key+suffix
		self.bin_key = bin_key+suffix
		
	def _convert_to_polar_angles(self, data):
		"""
		Convert timepoints to polar angles (0 to 2π).
		
		Parameters:
		-----------
		min_time : float, optional
			Minimum time value. If None, uses minimum of timepoints.
		max_time : float, optional
			Maximum time value. If None, uses maximum of timepoints.
			
		Returns:
		--------
		angles : ndarray
			Time points converted to angles in radians
		"""
			
		# Convert time to angles (0 to 2π)
		angles = 2 * np.pi * (np.array(data) - self.min_time) / (self.max_time - self.min_time)
		return angles
	
	def plot(self, g1_key, ax=None, threshold=None, figsize=(4, 4),
		plot_skew_cat=False):
		"""
		Plot the polar plot data
		"""
		if self.data is None:
			raise ValueError("Data not set. Use set_data() method first.")
		
		# Create the polar plot
		if ax is None:
			self.fig, self.ax = plt.subplots(figsize=figsize, subplot_kw={'projection': 'polar'})
		else:
			self.ax = ax

		self.ax.xaxis.grid(False)
		bin_key = self.bin_key
		
		# Plot the genes that are above the threshold
		bins_to_plot = [g1_key, 'postG1']
		thresholded_data = self.data[self.data[bin_key].isin(bins_to_plot)]
		below_thresholded_data = self.data[self.data[bin_key] == 'Not cell cycle']

		# The default color for cell cycle genes
		default_color = '#777'

		self.ax.scatter(thresholded_data.peak_angle, thresholded_data[self.ampl_key], s=2, c=default_color, 
			zorder=2)

		# For skewed data, as in gene expression, we have the option to plot
		# the genes that peak and trough differently
		if plot_skew_cat:
			peak_color = default_color #plt.cm.Reds(0.65)
			trough_color = plt.cm.Blues(0.65)

			sel = thresholded_data.cell_cycle_skew_t == 'peak'
			skew_data = thresholded_data.loc[sel]
			self.ax.scatter(skew_data.peak_angle, skew_data[self.ampl_key], s=2, color=peak_color, zorder=2)

			sel = thresholded_data.cell_cycle_skew_t == 'trough'
			skew_data = thresholded_data.loc[sel]
			self.ax.scatter(skew_data.trough_angle, skew_data[self.ampl_key], s=2, color=trough_color, zorder=2)

		self.ax.scatter(below_thresholded_data.peak_angle, below_thresholded_data[self.ampl_key], s=2, c='#bbb', zorder=2)
		self.ax.set_ylim(*self.ylims)

		add_radius_circle(ax, threshold, lw=0.75, ls='solid', color='black', zorder=2, alpha=0.5)

		xticks = np.arange(0, 64, 8)
		xtick_angles = self._convert_to_polar_angles(xticks)
		self.ax.set_yticks([])

		self.ax.set_xticks(xtick_angles)
		self.ax.set_xticklabels(xticks)

		from src.plot_helpers import color_for_key

		num_g1 = len(thresholded_data[thresholded_data[bin_key] == g1_key])
		num_postg1 = len(thresholded_data) - num_g1

		self.plot_annotation(0, 42, color_for_key(g1_key), f"{g1_key}, {num_g1}", 
			threshold, self.ylims[1])
		self.plot_annotation(42, 64, color_for_key('postG1'), f'S/G2/M, {num_postg1}', 
			threshold, self.ylims[1])

		num_non_cc = len(below_thresholded_data)
		self.ax.text(np.pi/2., self.ylims[0], f"{num_non_cc}", color='#999', ha='center',
			va='center')


	def plot_annotation(self, time_min, time_max, color, text, min_radius, max_radius):
		"""
		Add a colored sector annotation to the plot.
		"""
		if self.ax is None:
			raise ValueError("Plot not created yet. Call plot() method first.")
			
		# Convert time to angles
		angle_min = self._convert_to_polar_angles([time_min])[0]
		angle_max = self._convert_to_polar_angles([time_max])[0]
		
		# Create a set of angles for the sector
		angles = np.linspace(angle_min, angle_max, 100)
		
		# Fill the sector
		self.ax.fill_between(angles, min_radius, max_radius, color=color, alpha=0.15, zorder=0)

		# Add text
		radius = 0.75 * max_radius
		angle_text = (angle_min + angle_max) / 2
		self.ax.text(angle_text, radius, text, color=color,
			fontfamily='Open Sans', horizontalalignment='center', 
			verticalalignment='center')


def plot_polar_branches_data(expression_combined_polar_data_df,
		tx_ptr_threshold, tx_qval, title, ylims, tp_key='peak', plot_skew_cat=False):
	fig, axs = plt.subplots(1, 2, figsize=(7.5, 4), 
		subplot_kw={'projection': 'polar'})
	from src.polar_plotter import add_filled_circle, add_radius_circle

	# Mother branch
	polar_plotter = PolarPlot()
	polar_plotter.ylims = ylims
	polar_plotter.set_data(expression_combined_polar_data_df,
		suffix='_t')
	polar_plotter.plot('MG1', axs[0], threshold=tx_ptr_threshold, plot_skew_cat=plot_skew_cat)

	axs[0].set_title("Mother branch")

	# Daughter branch

	polar_plotter = PolarPlot()
	polar_plotter.ylims = ylims
	polar_plotter.set_data(expression_combined_polar_data_df, suffix='_b')
	polar_plotter.plot('DG1', axs[1], threshold=tx_ptr_threshold, plot_skew_cat=plot_skew_cat)
	plt.subplots_adjust(wspace=0.125, top=0.7)
	axs[1].set_title("Daughter branch")

	plt.suptitle(f"{title} PTRs,\nthreshold: {tx_ptr_threshold:.2f} (perc. {tx_qval*100:.0f})",
				fontsize=16)


def add_radius_circle(ax, radius, **kwargs):
	"""
	Add a circle of constant radius to a polar plot.
	
	Parameters:
	-----------
	ax : matplotlib.axes.Axes
		The polar axes to draw on
	radius : float
		The radius of the circle
	**kwargs : dict
		Additional keyword arguments to pass to ax.plot
	"""
	theta = np.linspace(0, 2*np.pi, 100)
	ax.plot(theta, np.ones_like(theta) * radius, **kwargs)

def add_filled_circle(ax, radius, **kwargs):
	theta = np.linspace(0, 2*np.pi, 100)
	ax.fill_between(theta, np.ones_like(theta) * radius, 0, **kwargs)

def prep_polar_plot_data(branch_expressions, g1_phase_key, eps):

	lo = 0.1
	hi = 0.9
	branch_expressions = branch_expressions.copy()
	branch_expressions.columns = np.arange(len(branch_expressions.columns))
	branch_ptrs = compute_quantile_ptr_2d(branch_expressions, eps=eps, lo=lo, hi=hi)
	branch_peak_idx = branch_expressions.idxmax(1)
	branch_trough_idx = branch_expressions.idxmin(1)
	branch_polar_data_df = pd.DataFrame({
		'peak_idx': branch_peak_idx, 
		'trough_idx': branch_trough_idx,
		'skew': stats.skew(branch_expressions, axis=1),
		'mean': branch_expressions.mean(1),
		'ptr': branch_ptrs},
		index=branch_expressions.index)

	# Assign peak phase based on location of peak index
	branch_polar_data_df['peak_phase'] = g1_phase_key
	branch_polar_data_df.loc[branch_polar_data_df.peak_idx >= 44, 'peak_phase'] = 'postG1'

	# Assign peak phase based on location of trough index
	branch_polar_data_df['trough_phase'] = g1_phase_key
	branch_polar_data_df.loc[branch_polar_data_df.trough_idx >= 44, 'trough_phase'] = 'postG1'

	return branch_polar_data_df


def create_combined_ptr_data_set(gene_data, config1, q_threshold, eps, assign_p_or_t=False,
	skew_threshold=None):
	# Sort through the polar data and assign the genes into MG1, DG1, or postG1 based
	# on highest PTR
	b_polar_data_df = prep_polar_plot_data(gene_data[config1.b_indices()],
		'DG1', eps)
	t_polar_data_df = prep_polar_plot_data(gene_data[config1.t_indices()],
		'MG1', eps)

	ptr_values = np.concatenate([b_polar_data_df.ptr.values, t_polar_data_df.ptr.values])
	ptr_threshold = np.quantile(ptr_values, q=q_threshold)

	# For each branch, assign to the non-cell cycle bin if below the ptr threshold
	def assign_bin(polar_data_df, threshold):

		bin_assignments = polar_data_df['peak_phase'].copy()

		# For trough genes, assign based on trough location
		if assign_p_or_t:
			trough_genes = polar_data_df[polar_data_df.cell_cycle_skew == 'trough'].index
			bin_assignments.loc[trough_genes] = polar_data_df.loc[trough_genes, 'trough_phase']

		bin_assignments.loc[polar_data_df.ptr < threshold] = 'Not cell cycle'

		return bin_assignments.values

	def classify_peak_or_trough_genes(polar_data_df, skew_threshold=None, ptr_threshold=None):
		"""Classifies cell cycle genes as peak or trough based on skew, default to 'even' 
		if the gene is balanced, or "Not cell Cycle" if the gene is not a cell cycle gene"""
		
		polar_data_df = polar_data_df.copy()

		polar_data_df['peak_or_trough_gene'] = None
		
		# Default to even, equally balanced skew
		polar_data_df['peak_or_trough_gene'] = 'even'
		
		# If positive skew, skews positive and is a peaky gene
		polar_data_df.loc[(polar_data_df['skew'] > skew_threshold), 'peak_or_trough_gene'] = 'peak'
		
		# If negative skew, skews positive and is a troughy gene
		polar_data_df.loc[(polar_data_df['skew'] < -skew_threshold), 'peak_or_trough_gene'] = 'trough'
		
		# Set non-cell cycle genes to not cell cycling category
		# Both the peak phase and trough phase will be not cell cycle so set on checking either
		polar_data_df.loc[(polar_data_df.ptr < ptr_threshold), 'peak_or_trough_gene'] = 'Not cell cycle'

		return polar_data_df['peak_or_trough_gene']

	if assign_p_or_t:
		b_polar_data_df['cell_cycle_skew'] = classify_peak_or_trough_genes(b_polar_data_df, 
			skew_threshold, ptr_threshold)
		t_polar_data_df['cell_cycle_skew'] = classify_peak_or_trough_genes(t_polar_data_df, 
			skew_threshold, ptr_threshold)

	b_polar_data_df['bin'] = assign_bin(b_polar_data_df, ptr_threshold)
	t_polar_data_df['bin'] = assign_bin(t_polar_data_df, ptr_threshold)

	# Then assign genes based on the highest ptr
	combined_polar_data_df = t_polar_data_df.join(b_polar_data_df, lsuffix='_t', rsuffix='_b')

	return combined_polar_data_df, ptr_threshold


def plot_skew(polar_data_df, skew_threshold=0.25):
    
    plt.figure(figsize=(8, 2))
    plt.subplot(1, 2, 1)
    plt.hist(polar_data_df.skew_t, bins=12)
    plt.title("Mother")
    plt.axvline(skew_threshold, c='red', lw=1, ls='dotted')
    plt.axvline(-skew_threshold, c='red', lw=1, ls='dotted')

    plt.subplot(1, 2, 2)
    plt.hist(polar_data_df.skew_b, bins=12)
    plt.axvline(skew_threshold, c='red', lw=1, ls='dotted')
    plt.axvline(-skew_threshold, c='red', lw=1, ls='dotted')
    plt.title("Daughter")    
    plt.suptitle("Gene expression skew")

