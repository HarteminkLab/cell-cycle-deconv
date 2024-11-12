
from matplotlib import pyplot as plt
from src.chromatin_model import draw_phase_label_annotations
from src.figure_configs import FiguresConfig
import numpy as np


class TimecoursePlotter:


	def __init__(self, config):

		self.config = config
		self.color_1 = plt.get_cmap('Reds')(0.75)
		self.color_2 = plt.get_cmap('Blues')(0.75)

	def plot_timecourse(self, data, index_1, index_2,
								  title, label_1, label_2, ylim=None,
								  ylabel=None, fig=None):
		config = self.config
		t_indices = config.get_Hpositions_for_branch('t')
		t_timepoints = config.get_timepoints_for_branch('t')

		if fig is None:
			fig = plt.figure(figsize=(6, 4))

		data_1 = data.loc[index_1][t_indices]\
			.quantile(q=[0.25, 0.5, 0.75], axis=0).dropna().astype(float)

		data_2 = data.loc[index_2][t_indices]\
			.quantile(q=[0.25, 0.5, 0.75], axis=0).dropna().astype(float)

		plt.fill_between(t_timepoints, data_1.loc[0.25].values, 
						 data_1.loc[0.75].values, alpha=0.35, color=self.color_1,
						 label=f"{label_1}, n={len(index_1)}")
		plt.plot(t_timepoints, data_1.loc[0.5], color=self.color_1)

		plt.fill_between(t_timepoints, data_2.loc[0.25].values, 
						 data_2.loc[0.75].values, alpha=0.35,
						 label=f"{label_2}, n={len(index_2)}",
						 color=self.color_2)
		plt.plot(t_timepoints, data_2.loc[0.5], color=self.color_2)
		plt.title(title,
				 fontsize=FiguresConfig.FIG_TITLE_FONTSIZE, pad=9)
		plt.legend()

		ax = plt.gca()

		if ylim is None:
			ylim = ax.get_ylim()

		ax.set_ylabel(ylabel)

		annotations_offset = (ylim[1]-ylim[0]) * 0.05

		draw_phase_label_annotations(ax, config, flip=True, annotations_x=ylim[0]+annotations_offset)
		plt.xticks([])
		plt.xlim(t_timepoints[0], t_timepoints[-1])
		plt.ylim(*ylim)
		plt.subplots_adjust(top=0.82)
