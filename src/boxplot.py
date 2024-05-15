import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from src.plot_helpers import plot_rect2
import pandas as pd


class BoxPlotPlotter():

	def __init__(self):
		pass

	def set_data(self, dfs_to_plot, data_key, group_key, group_name_key, category_names):

		self.quantile_dfs_to_plot = dfs_to_plot
		self.data_key = data_key
		self.group_key = group_key
		self.group_name_key = group_name_key
		self.category_names = category_names

	def create_boxplot_data(self, arr_dat):
		q1, median, q3 = np.percentile(arr_dat, [25, 50, 75])
		iqr = q3 - q1
		lower_whisker = np.min(arr_dat[arr_dat >= q1 - 1.5 * iqr])
		upper_whisker = np.max(arr_dat[arr_dat <= q3 + 1.5 * iqr])
		outliers = arr_dat[(arr_dat < lower_whisker) | (arr_dat > upper_whisker)]
		box_plot_data = q1, q3, median, lower_whisker, upper_whisker, outliers
		return box_plot_data

	def plot_box(self, box_plot_dat, x_location, group_index, ax):

		q1, q3, median, lower_whisker, upper_whisker, outliers = box_plot_dat

		color = plt.get_cmap('plasma_r')(group_index/self.num_categories*0.6+0.2)

		# box plot width
		width = 0.075
		padding = 0.025 # between grouped plots

		x_center = x_location # Center of the group of box plots
		# Total width
		total_width = (self.num_categories * width) + (self.num_categories-1)*padding

		# start offset by half the width, step by group index and box width
		# Add width/2 because line plots are centered on half the width
		step = width+padding
		x_location = (x_center - total_width/2. + width/2) + step*group_index
		
		# Plot a box for the IQR
		x1 = x_location - width / 2
		y1 = q1
		x2 = x_location + width / 2
		y2 = q3

		plot_rect2(ax, x1, y1, x2, y2, zorder=10,
			color=color)

		plt.plot([x_location, x_location], [lower_whisker, q1], color=color, zorder=2, lw=1)
		plt.plot([x_location, x_location], [q3, upper_whisker], color=color, zorder=2, lw=1)
		plt.scatter([x_location] * len(outliers), outliers, color='#777', s=3)  # Outliers

		plt.plot([x_location - width / 2, x_location + width / 2], [median, median], 
			color='black', linestyle='-', linewidth=1, solid_capstyle='butt', zorder=11)

		return color


	def plot_box_plot(self, ax):
		"""Plot a box plot of a dataframe with a quantile column to separate
		the data into distinct columns"""

		self.num_categories = len(self.quantile_dfs_to_plot)

		xticks = []
		xticklabels = []
		colors = []

		for category_index in range(self.num_categories):
			dataframe = self.quantile_dfs_to_plot[category_index]
			group_keys = sorted(dataframe[self.group_key].unique())
		
			# For each quantile plot a box and whisker plot
			for group_index in range(len(group_keys)):
				quantile_name = group_keys[group_index]

				group_data = dataframe[dataframe[self.group_key] == quantile_name]
				current_data_values = group_data[self.data_key]
				current_data_values = current_data_values.dropna()
				box_plot_dat = self.create_boxplot_data(current_data_values)
				color = self.plot_box(box_plot_dat, group_index, category_index, ax)

				group_name = group_data.iloc[0][self.group_name_key]

				xticks.append(group_index)
				xticklabels.append(group_name)

			colors.append(color)

		# Create placeholder artists for the legend.
		# These artists do not actually paint anything on the ax they are not added to ax.
		import matplotlib.lines as mlines

		legend_items = []
		for i in range(len(self.category_names)):
			color = colors[i]
			legend_line = mlines.Line2D([], [], color=color, linewidth=4, label=self.category_names[i])
			legend_items.append(legend_line)

		# Add these lines to the legend.
		ax.legend(handles=legend_items, loc='center right', title="Replication")

		ax.set_xticks(xticks)
		ax.set_xticklabels(xticklabels)
		ax.set_xlim(xticks[0]-.5, xticks[-1]+1.5)
		ax.set_ylim(1.8, 4.2)

		ax.set_xlabel("Gene expression cutoffs, VST")
		ax.set_ylabel("Nucleosome entropy")
		ax.set_title("Nucleosome Entropy and expression during replication")
