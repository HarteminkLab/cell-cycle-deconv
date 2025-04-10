import numpy as np
import matplotlib.pyplot as plt

class PolarPlot:
	def __init__(self):
		"""Initialize the PolarPlot class with empty data."""
		self.timepoints = None
		self.amplitudes = None
		self.fig = None
		self.ax = None
		self.ylims = -.5, 4
		
	def set_data(self, data, tp_key='peak_idx', ampl_key='ptr', min_time=0, max_time=64):
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
		angles = self._convert_to_polar_angles(self.data[tp_key])
		self.data['angle'] = angles

		self.tp_key = tp_key
		self.ampl_key = ampl_key
		
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
	
	def plot(self, g1_key, ax=None, threshold=None, figsize=(4, 4)):
		"""
		"""
		if self.data is None:
			raise ValueError("Data not set. Use set_data() method first.")
		
		# Create the polar plot
		if ax is None:
			self.fig, self.ax = plt.subplots(figsize=figsize, subplot_kw={'projection': 'polar'})
		else:
			self.ax = ax

		self.ax.xaxis.grid(False)
		
		# Plot the genes that are above the threshold
		bins_to_plot = [g1_key, 'postG1']
		thresholded_data = self.data[self.data.bin.isin(bins_to_plot)]
		below_thresholded_data = self.data[self.data.bin == 'Not cell cycle']

		self.ax.scatter(thresholded_data.angle, thresholded_data.ptr, s=2, c='#777', zorder=2)
		self.ax.scatter(below_thresholded_data.angle, below_thresholded_data.ptr, s=2, c='#bbb', zorder=2)
		self.ax.set_ylim(*self.ylims)

		add_radius_circle(ax, threshold, lw=1, ls='solid', color='red', zorder=2, alpha=0.5)

		xticks = np.arange(0, 64, 8)
		xtick_angles = self._convert_to_polar_angles(xticks)
		self.ax.set_yticks([])

		self.ax.set_xticks(xtick_angles)
		self.ax.set_xticklabels(xticks)

		from src.plot_helpers import color_for_key

		num_g1 = len(thresholded_data[thresholded_data.bin == g1_key])
		num_postg1 = len(thresholded_data) - num_g1

		self.plot_annotation(0, 42, color_for_key(g1_key), f"{g1_key}, {num_g1}", 
			threshold, self.ylims[1])
		self.plot_annotation(42, 64, color_for_key('postG1'), f'S/G2/M, {num_postg1}', 
			threshold, self.ylims[1])


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
		tx_ptr_threshold, tx_qval, title, ylims):
	fig, axs = plt.subplots(1, 3, figsize=(12, 4), 
	    subplot_kw={'projection': 'polar'})

	polar_plotter = PolarPlot()
	polar_plotter.ylims = ylims
	polar_plotter.set_data(expression_combined_polar_data_df)
	polar_plotter.plot('MG1', axs[0], threshold=tx_ptr_threshold)

	axs[0].set_title("Mother branch")

	polar_plotter = PolarPlot()
	polar_plotter.ylims = ylims
	polar_plotter.set_data(expression_combined_polar_data_df)
	polar_plotter.plot('DG1', axs[1], threshold=tx_ptr_threshold)
	plt.subplots_adjust(wspace=0.25, top=0.75)
	axs[1].set_title("Daughter branch")

	from src.polar_plotter import add_filled_circle, add_radius_circle

	non_cell_cycle_ax = axs[2]
	non_cell_cycle_ax.set_xticks([])
	non_cell_cycle_ax.set_yticks([])
	non_cell_cycle_ax.spines['polar'].set_visible(False)
	add_filled_circle(non_cell_cycle_ax, tx_ptr_threshold, color='#ddd')
	add_radius_circle(non_cell_cycle_ax, tx_ptr_threshold, color='red', lw=0.75)
	non_cell_cycle_ax.text(np.pi/2., tx_ptr_threshold*1.25, "Non cell-cycle genes", ha='center')
	non_cc_genes = expression_combined_polar_data_df[
	    expression_combined_polar_data_df.bin == 'Not cell cycle']
	non_cell_cycle_ax.text(np.pi/2., 0, f"{len(non_cc_genes)}", ha='center',
	                      color='#555')
	non_cell_cycle_ax.set_ylim(*ylims)

	plt.suptitle(f"{title} PTRs, threshold: {tx_ptr_threshold:.2f} (perc. {tx_qval*100:.0f})",
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

