

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.figure_configs import FiguresConfig
from src.geneset import get_deconvolved_geneset


class Figure3ChromatinMetrics(object):
	"""Create figures for chromatin metrics vs gene expression"""

	def __init__(self):

		from src.deconv_data import load_f_files
		chromatin_dir = 'output/deconvolve_sharedg1_0066_cc_2024_06_13/chromatin/'
		self.all_gene_f_df = load_f_files(chromatin_dir)
		self.genes = get_deconvolved_geneset()

		from src.config import load_configs_by_config_type
		self.config1, self.config2 = load_configs_by_config_type('shared')


	def track_img(self, gene_name):
		from src.global_config import GlobalConstants
		from src.sgd import get_gene_name_orf_name
		from src.chromatin_metric_tracking import ChromatinMetricTracking

		orf_name, gene_name = get_gene_name_orf_name(gene_name)
		gene = self.genes.loc[orf_name]

		gene_f = self.all_gene_f_df.loc[gene.name]
		gene_f_values = gene_f.values

		# Note max y length is updated to 264, but this run
		# of the deconvolution goes up to 240
		image_shape = GlobalConstants.IMAGE_SHAPE
		image_shape = image_shape[0], image_shape[1]
		# print("Image shape is: ", image_shape)

		img = gene_f_values.reshape((-1, *image_shape))

		# Flip if crick
		if gene.strand == '-':
			img = np.flip(img, axis=2)

		self.img = img

		# Add half a bin to define the bins as centered
		# on each position, these positions are relative to a computed plus one
		# at 0 position, left to right, flip the f images if necessary
		genomic_x_positions = np.arange(GlobalConstants.PROM_REGION[0], 
		                                GlobalConstants.GB_REGION[1],
		                                GlobalConstants.BIN_WIDTH)+GlobalConstants.BIN_WIDTH//2.

		from src.chromatin_metrics import fragment_lengths_definitions

		# Fragment length and genomic spans
		sm, med, nuc = fragment_lengths_definitions()

		self.promoter_span = np.array(GlobalConstants.PROM_REGION)+GlobalConstants.BIN_WIDTH//2.
		self.gb_span = np.array(GlobalConstants.GB_REGION)-GlobalConstants.BIN_WIDTH//2.

		# Promoter occupancy tracker
		prom_occ_tracker = ChromatinMetricTracking(img_data=img, 
		    x_genomic_positions=genomic_x_positions)
		prom_occ_tracker.select_range(self.promoter_span, sm)
		prom_occ_tracker.track_occupancy()

		# Gene body entropy
		gb_tracker = ChromatinMetricTracking(img_data=img, 
		    x_genomic_positions=genomic_x_positions)
		gb_tracker.select_range(self.gb_span, nuc)
		gb_tracker.track_entropy()
		gb_tracker.track_occupancy()

		# +1
		p1_tracker = ChromatinMetricTracking(img_data=img, 
		    x_genomic_positions=genomic_x_positions)
		p1_tracker.select_range((-GlobalConstants.BIN_WIDTH*2, GlobalConstants.BIN_WIDTH*3), 
		    nuc)
		p1_tracker.track_genomic_movement()

		self.tracking_df = pd.DataFrame({'promoter_occupancy': prom_occ_tracker.total_occupancy,
			'+1': p1_tracker.called_peak_weighted_mean,
			'gene_body_entropy': gb_tracker.entropy_values,
			'gene_body_occupancy': gb_tracker.total_occupancy})

		self.gene = gene

	def plot_tracking(self):

		from src.config import load_configs_by_config_type
		t_indices = self.config1.get_Hpositions_for_branch('t')
		ys = np.arange(len(t_indices))
		plt.figure(figsize=(4, 3))
		plt.scatter(self.tracking_df['+1'][t_indices], ys, color='#666', s=10)
		plt.xlim(self.promoter_span[0], self.gb_span[1])
		plt.title(f"{self.gene.gene}")

		prom_occ = self.tracking_df['promoter_occupancy'][t_indices].values
		prom_occ = prom_occ.reshape((-1, 1))
		plt.imshow(prom_occ, extent=[-200, -100, 0, len(t_indices)], aspect='auto', 
		    origin='lower', cmap='Oranges', vmin=0, vmax=120)

		gb_entropy = self.tracking_df['gene_body_entropy'][t_indices].values
		gb_entropy = gb_entropy.reshape((-1, 1))

		plt.imshow(gb_entropy, extent=[150, 400, 0, len(t_indices)], aspect='auto', 
		    origin='lower', cmap='Purples', vmin=3.2, vmax=3.5)

	def deconvolve_pry1(self):
		self.deconvolve_gene("PRY1")

	def deconvolve_hxt4(self):
		self.deconvolve_gene("HXT4")

	def deconvolve_gene(self, gene_name):
		from src.config import load_configs_by_config_type, load_combined_gene_expression_by_config_type
		combined_ge_config = load_combined_gene_expression_by_config_type('shared')

		from src.model import Model
		combined_ge_model = Model(combined_ge_config, gene_name)
		combined_ge_model.deconvolve_find_optimal_gamma()
		from src.combined_chromatin_model import CombinedChromatinModel
		combined_model = CombinedChromatinModel(self.config1, self.config2)
		combined_model.load_combined_mnase_gene(gene_name)
		combined_model.gamma = 0.007
		combined_model.deconvolve()

		self.combined_model = combined_model
		self.combined_ge_model = combined_ge_model

	def plot_deconvolved_gene(self):

		fig = self.combined_model.create_deconvolution_plots_abbreviated_flipped(
			ge_model=self.combined_ge_model, vmax=50, num_rows=3, figsize=(11, 7))