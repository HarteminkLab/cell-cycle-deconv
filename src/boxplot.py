import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from src.plot_helpers import plot_rect2
import pandas as pd


class BoxPlotPlotter():

	def __init__(self):
		pass

	def set_data(self, dfs_to_plot, data_key, group_key, group_name_key, category_names):

		self.dfs_to_plot = dfs_to_plot
		self.data_key = data_key
		self.group_key = group_key
		self.group_name_key = group_name_key
		self.category_names = category_names
		self.plot_outliers = True
		self.plot_whiskers = True
		self.legend = True
		self.auto_xticks = True
		self.color_prop_override = False
		self.color = None
		self.group_colors = None

		# todo: default ylims for first box plot example
		# set this for future boxplots
		self.ylims = 1.8, 4.2

		# Box plot width
		self.width = 0.075
		self.padding = 0.025 # between grouped plots

	def create_boxplot_data(self, arr_dat):
		q1, median, q3 = np.percentile(arr_dat, [25, 50, 75])
		iqr = q3 - q1
		mean = np.mean(arr_dat)
		lower_whisker = np.min(arr_dat[arr_dat >= q1 - 1.5 * iqr])
		upper_whisker = np.max(arr_dat[arr_dat <= q3 + 1.5 * iqr])
		outliers = arr_dat[(arr_dat < lower_whisker) | (arr_dat > upper_whisker)]
		box_plot_data = q1, q3, median, mean, lower_whisker, upper_whisker, outliers
		return box_plot_data

	def plot_box(self, box_plot_dat, x_location, group_index, ax):

		q1, q3, median, mean, lower_whisker, upper_whisker, outliers = box_plot_dat

		# If overriding color index for manually coloring
		if self.color_prop_override:
			color_prop = self.color_prop_override
		else:
			color_prop = group_index/self.num_categories

		if self.color is not None:
			color = self.color
		else:
			color = plt.get_cmap('plasma_r')(color_prop*0.6+0.2)

		# For no categories, let's color the groups by different colors
		if self.group_colors is not None:
			color = self.group_colors[x_location]

		# box plot width
		width = self.width
		padding = self.padding # between grouped plots

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

		if self.plot_whiskers:
			plt.plot([x_location, x_location], [lower_whisker, q1], color=color, zorder=2, lw=1)
			plt.plot([x_location, x_location], [q3, upper_whisker], color=color, zorder=2, lw=1)

		if self.plot_outliers:
			plt.scatter([x_location] * len(outliers), outliers, color='#777', s=3)  # Outliers

		plt.scatter([x_location], [mean], marker='D', s=9, zorder=12, facecolors=color,
			edgecolor='black', lw=1)  # Outliers

		plt.plot([x_location - width / 2, x_location + width / 2], [median, median], 
			color='black', linestyle='-', linewidth=1, solid_capstyle='butt', zorder=11)

		return color


	def plot_box_plot(self, ax, title):
		"""Plot a box plot of a dataframe with a quantile column to separate
		the data into distinct columns"""

		self.num_categories = len(self.dfs_to_plot)

		xticks = []
		xticklabels = []
		colors = []
		
		for category_index in range(self.num_categories):
			dataframe = self.dfs_to_plot[category_index]
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

		# Add these lines to the legend
		if self.legend:
			ax.legend(handles=legend_items, loc='center right', title="Replication")

		# Auto xticks
		if self.auto_xticks:
			ax.set_xticks(xticks)
			ax.set_xticklabels(xticklabels)
			ax.set_xlim(xticks[0]-.5, xticks[-1]+.5)

		if self.ylims is not None:
			ax.set_ylim(*self.ylims)

		ax.set_title(title)


# Reduce the columns to plot fewer box plots
def get_boxplot_data_for_metric_df(dat, subselect_index, data_key, binstep=1, index_range_key='H_range'):
	"""Convert a dataframe of genes x h-indexes to a narrow format """

	from src.helpers import combine_columns_with_bins

	# Select the columns to combine
	selected_dat = dat.loc[subselect_index]
	bins = np.arange(0, dat.columns[-1], binstep)
	combined_cols_df = combine_columns_with_bins(selected_dat, 
	   bins=bins)

	# Narrow format for the box plotter
	narrow_format_df = combined_cols_df.unstack().reset_index()
	narrow_format_df.columns = [index_range_key, "orf_name", data_key]
	narrow_format_df[index_range_key] = narrow_format_df[index_range_key].astype(int)

	return narrow_format_df