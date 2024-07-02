
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.figure_configs import FiguresConfig
from src.figure_configs import save_figure_for_paper

class Figure4DeconvolvedOrigins(object):
	"""Fourth figure: deconvolved origins analysis. Show how the deconvolution
	algorithm is able to track known biology around origin of replication Mcm2-7
	loading and unloading at replication time and difference between early and
	late replicating origins"""

	def __init__(self):

		import glob

		origin_deconv_directory = "output/deconvolve_origins_g0066_2024_06_28"

		# Load the tracking information for NFR and origin occupancy
		pattern = f'{origin_deconv_directory}/chromatin/*tracking*'
		tracking_paths = glob.glob(pattern)

		all_nfrs_df = pd.DataFrame()
		all_origins_occupancy_df = pd.DataFrame()

		for tracking_path in tracking_paths:

			path_split = tracking_path.split('/')[-1].split('_')
			oridb, ars_name = path_split[3], path_split[4].replace('.csv', '')
			oridb, ars_name

			tracking_df = pd.read_csv(tracking_path)
			nfr = np.abs(tracking_df['+1']-tracking_df['-1'])
			origin_occupancy = tracking_df['origin_occupancy']
			index = f"oridb_{oridb}"
			gene_nfr = pd.Series(nfr, name=index)
			gene_occupancy = pd.Series(origin_occupancy, name=index)

			nfr_df = pd.DataFrame(gene_nfr).T
			origin_occupancy_df = pd.DataFrame(gene_occupancy).T
			all_nfrs_df = pd.concat([all_nfrs_df, nfr_df])
			all_origins_occupancy_df = pd.concat([all_origins_occupancy_df, origin_occupancy_df])

		self.all_origins_occupancy_df = all_origins_occupancy_df
		self.all_nfrs_df = all_nfrs_df

		from src.config import load_configs_by_config_type
		from src.origins import load_origins_w_replication

		origins_w_replication = load_origins_w_replication(full=True)
		config1, config2 = load_configs_by_config_type('shared')

		t_indices = config1.get_Hpositions_for_branch('t')
		origins_w_replication = origins_w_replication.sort_values('replication_time')

		self.origins_w_replication = origins_w_replication
		self.t_indices = t_indices

	def plot_heatmap_of_tracking(self, full=True):

		normalized_nfrs_df = normalize_z_score_by_row(self.all_nfrs_df[self.t_indices])
		normalized_origins_df = normalize_z_score_by_row(self.all_origins_occupancy_df[self.t_indices])

		from src.figure_configs import FiguresConfig

		# Plot only origins with g1 and g2 footprint
		if not full:
			origins_w_replication = self.origins_w_replication[self.origins_w_replication.footprint_class == 'g1_and_g2_footprint']
			origins_w_replication_sorted = \
				origins_w_replication.sort_values('replication_time')

		# Plot all 798 origins
		else:
			origins_w_replication_sorted = \
				self.origins_w_replication.sort_values('replication_time')

		index = origins_w_replication_sorted.index

		plt.figure(figsize=(6, 4))
		plt.subplot(1, 2, 1)
		plt.imshow(normalized_nfrs_df.loc[index], 
			aspect='auto', cmap='RdBu_r', vmin=-15, vmax=15, interpolation='none')
		plt.yticks([])
		plt.title("Normalized NFR size")
		plt.xticks([])
		plt.colorbar()

		plt.subplot(1, 2, 2)
		plt.imshow(self.all_origins_occupancy_df.loc[index][self.t_indices].tail(200),
				   aspect='auto',
				  vmin=0, vmax=120, cmap='Purples', interpolation='none')
		plt.yticks([])
		plt.title(f"Origin occupancy")
		plt.colorbar()
		plt.xticks([])

		plt.suptitle(f"Origin chromatin dynamics sorted\nby replication time,"+
					 f" n={len(index)}", 
					 fontsize=FiguresConfig.FIG_SUPTITLE_FONTSIZE)
		plt.subplots_adjust(top=0.8)
		# todo: label x axis with cell cycle phases

def normalize_z_score_by_row(df, norm_sd=False):
	"""Normalize such that each origin is centered on the mean and scaled to std 1"""
	mean, sd = df.mean(axis=1), df.std(axis=1)
	z_norm = (df - mean.values.reshape((-1, 1)))
	
	if norm_sd: z_norm /= (sd.values.reshape(-1, 1))
	return z_norm
