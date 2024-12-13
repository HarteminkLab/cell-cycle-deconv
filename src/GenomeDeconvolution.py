
import matplotlib.pyplot as plt
import numpy as np
from src.global_config import GlobalConstants
from src.figure_configs import FiguresConfig
from src.orf_plotter import ORFAnnotationPlotter
from src.geneset import get_deconvolved_geneset
import pandas as pd


class GenomeDeconvolution(object):
	"""Load and plot figures for the first result figure"""

	def __init__(self, save_dir, configs=None):

		self.save_directory = save_dir

		# Create the H for the updated model config to include the H config
		if configs is None:
			from src.config import load_configs_by_config_type
			config1, config2 = load_configs_by_config_type('shared')
		else:
			config1, config2 = tuple(configs)

		H, Hpositions = config1.calcH_function(config1.intervals_wt1, config1.WT1_TIMEPOINTS)
		self.H = H
		self.config1 = config1
		self.config2 = config2

		from src.combined_chromatin_model import CombinedChromatinModel
		combined_model = CombinedChromatinModel(self.config1, self.config2)
		combined_model.gamma = 0.007
		self.combined_model = combined_model


	def load_chrom_span(self, chrom, span, log=True, downsample=True):
		self.combined_model.load_mnase_span(chrom, span, log=log, downsample=downsample)

		if downsample:
			self.combined_model.setup_deconv_model()


	def deconvolve(self, G1=None, G2=None):
		self.combined_model.deconvolve(G1=G1, G2=G2)
	

	def plot_deconvolved_result(self, smooth=False, 
			normalize=False, vmin=1, vmax=20, figwidth=23):
		from src.global_config import GlobalConstants
		from src.figure_configs import FiguresConfig

		f_imgs = self.combined_model.chrom1_model.get_f_images()
		chrom = self.combined_model.chrom1_model.chr
		span = self.combined_model.chrom1_model.mnase_span

		#normalized_f_imgs = f_imgs/f_imgs.mean(axis=1).mean(axis=1).reshape((-1, 1, 1))
		normalized_f_imgs = f_imgs

		indices, label_names = self.combined_model\
			.chrom1_model.config.get_full_phase_indices()

		fig, axs = plt.subplots(len(indices)+1, 1, figsize=(figwidth, 13))

		ax = axs[0]
		genes = get_deconvolved_geneset()
		orf_plotter = ORFAnnotationPlotter(genes)
		orf_plotter.set_span_chrom(self.combined_model.chrom1_model.mnase_span,
								  self.combined_model.chrom1_model.chr)
		orf_plotter.plot_orf_annotations(ax)

		from src.helpers import smooth_data

		for i in range(len(indices)):
			ax = axs[i+1]
			label_name = label_names[i]
			
			current_f_img = normalized_f_imgs[indices[i]]

			if smooth:
				current_f_img = smooth_data(current_f_img, size=5, sigma=0.5)

			if normalize:
				current_f_img = current_f_img / current_f_img.sum() * 2000

			ax.imshow(current_f_img, origin='lower', cmap='magma_r', vmax=vmax, aspect='auto',
					 extent=self.combined_model.chrom1_model.bin_extents, 
					 vmin=vmin)
			ax.set_ylabel(label_name, rotation=0, ha='right', labelpad=9)
			ax.set_yticks([])
			
			if i == (len(indices)-1):
				ax.set_xticks(np.arange(span[0], span[1], 2000), minor=False)
				ax.set_xticks(np.arange(span[0], span[1], 200), minor=True)
			else:
				ax.set_xticks([])

		plt.suptitle(f"Chr{chrom}: {span[0]}-{span[1]}", fontsize=FiguresConfig.FIG_SUPTITLE_FONTSIZE)
		plt.subplots_adjust(top=0.923)
		return fig


	def plot_raw_data(self, replicate):

		# Compare to the raw data to ensure that the deconvolution bins are being created properly
		if replicate == 1:
			raw_imgs_rep = self.combined_model.chrom1_model.deconv_hist_unflattened
			timepoints = GlobalConstants.CHROM_WT1_TIMEPOINTS
		else:
			raw_imgs_rep = self.combined_model.chrom2_model.deconv_hist_unflattened
			timepoints = GlobalConstants.CHROM_WT2_TIMEPOINTS

		fig, axs = plt.subplots(len(timepoints)+1, 1, figsize=(23, 16))

		genes = get_deconvolved_geneset()
		span = self.combined_model.chrom1_model.mnase_span
		chrom = self.combined_model.chrom1_model.chr
		orf_plotter = ORFAnnotationPlotter(genes)
		orf_plotter.set_span_chrom(self.combined_model.chrom1_model.mnase_span,
								  self.combined_model.chrom1_model.chr)
		orf_plotter.plot_orf_annotations(axs[0])

		for i in range(len(timepoints)):
			ax = axs[i+1]
			label_name = f"{timepoints[i]} min"
			
			current_f_img = raw_imgs_rep[i]

			ax.imshow(current_f_img, origin='lower', cmap='magma_r', vmax=20, aspect='auto',
					 extent=self.combined_model.chrom1_model.bin_extents)
			ax.set_ylabel(label_name, rotation=0, ha='right', labelpad=9)
			ax.set_yticks([])
			
			if i == (len(timepoints)-1):
				ax.set_xticks(np.arange(span[0], span[1], 2000), minor=False)
				ax.set_xticks(np.arange(span[0], span[1], 200), minor=True)
			else:
				ax.set_xticks([])

		plt.suptitle(f"Chr{chrom}: {span[0]}-{span[1]}\nReplicate {replicate}", fontsize=FiguresConfig.FIG_SUPTITLE_FONTSIZE)
		plt.subplots_adjust(top=0.923)

		return fig

	def save_gamma_to_disk(self):
		"""Save optimal gamma and rn/sn values to disk"""
		from src.utils import mkdirs_safe

		span = self.combined_model.chrom1_model.mnase_span
		chrom = self.combined_model.chrom1_model.chr

		gamma_save_dir = f"{self.save_directory}/data/gamma"
		gamma_save_path = f"{gamma_save_dir}/chr{chrom}_{span[0]}_{span[1]}.csv"
		mkdirs_safe([gamma_save_dir])

		save_dic = {'gamma': self.combined_model.gamma,
			'rn': self.combined_model.rn,
			'sn': self.combined_model.sn}
		pd.DataFrame(save_dic, index=[(chrom, span[0], span[1])]).to_csv(gamma_save_path)


	def save_to_disk(self):

		from src.utils import mkdirs_safe
		
		outdir = self.save_directory
		
		chrom, span = self.combined_model.chrom1_model.chr,\
		    self.combined_model.chrom1_model.mnase_span

		# Save paths
		raw_rep1_plot_dir = f"{outdir}/plots/raw/replicate1/chr{chrom}"
		raw_rep2_plot_dir = f"{outdir}/plots/raw/replicate2/chr{chrom}"
		deconvolved_plot_dir = f"{outdir}/plots/deconvolved/chr{chrom}"
		data_dir = f"{outdir}/data/chr{chrom}"

		# Make directories if needed
		mkdirs_safe([raw_rep1_plot_dir, raw_rep2_plot_dir, deconvolved_plot_dir, data_dir])

		# File save paths
		data_save_path = f"{data_dir}/chr{chrom}_{span[0]}_{span[1]}"
		deconvolved_plot_save_path = f"{deconvolved_plot_dir}/chr{chrom}_{span[0]}_{span[1]}.png"
		raw_rep1_plot_save_path = f"{raw_rep1_plot_dir}/rep1_chr{chrom}_{span[0]}_{span[1]}.png"
		raw_rep2_plot_save_path = f"{raw_rep2_plot_dir}/rep2_chr{chrom}_{span[0]}_{span[1]}.png"

		# Save to disk

		# Deconvolved data files
		f_imgs = self.combined_model.chrom1_model.get_f_images()
		np.save(data_save_path, f_imgs)

		# Deconvolved plot
		fig = self.plot_deconvolved_result(smooth=False)
		plt.savefig(deconvolved_plot_save_path, dpi=150)
		plt.close(fig)
		
		# Replicate 1 raw data plot
		fig = self.plot_raw_data(1)
		plt.savefig(raw_rep1_plot_save_path, dpi=150)
		plt.close(fig)
		
		# Replicate 2 raw data plot
		fig = self.plot_raw_data(2)
		plt.savefig(raw_rep2_plot_save_path, dpi=150)
		plt.close(fig)


def write_10k_windows_dataset_to_disk():
	import pandas as pd
	import numpy as np

	from src.sgd import get_chromosome_length
	def get_chrom_10k_windows(chrom):
		chrom_len = get_chromosome_length(chrom)
		starts = np.arange(0, chrom_len, 10000)
		ends = np.concatenate([starts[1:], np.array([chrom_len])])
		return starts, ends

	all_windows = pd.DataFrame()
	for chrom in range(1, 17):
		starts, ends = get_chrom_10k_windows(chrom)
		chrom_windows = pd.DataFrame({'chr': np.repeat(chrom, len(starts)),
									  'start': starts,
									  'end': ends})
		all_windows = pd.concat([all_windows, chrom_windows])

	data_10k_windows_filepath = 'data/reference_data/sacCer3_genome_10k_windows.csv'
	all_windows.to_csv(data_10k_windows_filepath, index=False)
	print(f"Wrote {data_10k_windows_filepath} to disk.")

