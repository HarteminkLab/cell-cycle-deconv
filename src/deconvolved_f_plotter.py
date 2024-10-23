	
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.global_config import GlobalConstants
from src.figure_configs import FiguresConfig
from src.config import load_configs_by_config_type
from src.orf_plotter import ORFAnnotationPlotter
from src.geneset import get_deconvolved_geneset


class DeconvolvedFPlotter:
	"""docstring for DeconvolvedFPlotter"""
	def __init__(self):

		self.smooth = False
		self.normalize = False
		self.vmax = 1
		self.figsize = (13, 11)
		self.xlims = None
		self.plot_intervals = ['CG1 - Early',
		 'CG1 - Late',
		 'S - Early',
		 'S - Late',
		 'G2/M - Early',
		 'G2/M - Late']
		self.plot_orfs = False
		self.ax_func = None
		self.orf_ax_func = None

	def set_f_imgs(self, f_imgs, mnase_span):
		self.f_imgs = f_imgs
		self.mnase_span = mnase_span
		

	def plot(self, vmax=None):

		f_imgs = self.f_imgs

		if vmax is None:
			vmax = self.vmax
		else:
			self.vmax = vmax

		figsize = self.figsize
		normalize = self.normalize
		smooth = self.smooth
		mnase_span = self.mnase_span
		xlims = self.xlims
		plot_intervals = self.plot_intervals

		config, _  = load_configs_by_config_type('shared')
		indices, label_names = config.get_full_phase_indices()
		label_indices_dic = {}
		for i in range(len(indices)):
			label_indices_dic[label_names[i]] = indices[i]

		plot_orfs_subplot_offset = (1 if self.plot_orfs else 0)
		fig, axs = plt.subplots(len(plot_intervals)+plot_orfs_subplot_offset,
			1, figsize=figsize)

		if self.plot_orfs:
			# Orf annotation ax
			ax = axs[0]
			ax.set_xticks([])
			ax.set_yticks([])

			if self.orf_ax_func is not None:
				self.orf_ax_func(ax)

		from src.helpers import smooth_data
		extents = [mnase_span[0], mnase_span[-1], 0, GlobalConstants.MAX_Y_LEN]

		for i in range(len(plot_intervals)):
			ax = axs[i+plot_orfs_subplot_offset]
			label_name = plot_intervals[i]
			index = label_indices_dic[label_name]
			
			current_f_img = f_imgs[index]

			if smooth:
				current_f_img = smooth_data(current_f_img, size=5, sigma=0.5)

			if normalize:
				current_f_img = current_f_img / current_f_img.sum() * 200.

			ax.imshow(current_f_img, origin='lower', cmap='magma_r', vmin=1, vmax=vmax, aspect='auto',
					 extent=extents)
			ax.set_ylabel(label_name, rotation=0, ha='right', labelpad=9)
			ax.set_yticks([])
			
			if i == (len(indices)-1):
				ax.set_xticks(np.arange(mnase_span[0], mnase_span[1], 2000), minor=False)
				ax.set_xticks(np.arange(mnase_span[0], mnase_span[1], 200), minor=True)
			else:
				ax.set_xticks([])
			
			if xlims is not None:
				ax.set_xlim(*xlims)

			if self.ax_func:
				self.ax_func(ax, index, i, len(plot_intervals), label_name)

		plt.subplots_adjust(left=0.2, top=0.923)

		return fig


def plot_pseudo_gene(ax, gene_start=-40, gene_len=1080, xlim=None):
	from src.orf_plotter import plot_gene_annotation
	from src.global_config import GlobalConstants

	TSS = gene_start
	PAS = gene_start+gene_len

	plot_gene_annotation(ax, gene_start, gene_start+gene_len, 0, 36, '#ccc', (0, 0), 1,
		40, TSS, PAS, True)

	ax.set_ylim(-50, 100)

	if xlim is None:
		ax.set_xlim(-GlobalConstants.PROM_LEN, GlobalConstants.GB_LEN*3)

	ax.set_yticks([])
	ax.set_xticks([])

