
import matplotlib.pyplot as plt
import numpy as np
from src.global_config import GlobalConstants
import pandas as pd


class GenomeDeconvolutionAnalysis(object):
	"""Class to perform analysis on genome-wide deconvolution
	results. 


	Common tasks:
	- Retrieve the MNase data surrounding all genes
		- Flip crick stranded genes
		- And various subsets of genes

	- Retrieve the MNase data surrounding all origins
	- Retrieve nucleosome reads

	Notes:
	- Note that stacking genes by +1 nucleosome will not align bins perfectly, 
	  so the stacked histogram will need to be in base pairs (or rounded to 
	  the nearest bin.)

	"""

	def __init__(self, outdir):
		self.outdir = outdir

	def load_gene_expression(self, outdir):
		"""Load the deconvolved gene expression data"""
		from src.geneset import get_deconvolved_geneset
		from src.gene_clustering import GeneClustering

		genes = get_deconvolved_geneset()	
		gene_clustering = GeneClustering(outdir)
		genes_replication = pd.read_csv('data/replication_timing/yl_2019/genes_replication_timing_shared.csv').set_index('orf_name')
		genes_replication = genes_replication.sort_values('replication_time')
		t_indices = gene_clustering.config.get_Hpositions_for_branch('t')
		deconvolved_gene_expression_t = gene_clustering.gene_expression[t_indices]

		self.config = gene_clustering.config
		self.genes_replication = genes_replication
		self.t_indices = t_indices
		self.deconvolved_gene_expression_t = deconvolved_gene_expression_t

	def select_cg1_g2m_genes(self, prop_thresh):

		config = self.config
		deconvolved_gene_expression_t = self.deconvolved_gene_expression_t

		t_timepoints = config.get_timepoints_for_branch('t')
		cg1_indices = config.get_Hpositions_for_phase('CG1')
		g2m_indices = config.get_Hpositions_for_phase('G2M')

		# Split genes by cg1 expressed genes and G2M expressed genes
		mean_cg1 = deconvolved_gene_expression_t[cg1_indices].mean(axis=1)
		mean_g2m = deconvolved_gene_expression_t[g2m_indices].mean(axis=1)

		cg1_genes = deconvolved_gene_expression_t.loc[mean_cg1 > (mean_g2m)*(1+prop_thresh)].index
		g2m_genes = deconvolved_gene_expression_t.loc[mean_g2m > (mean_cg1)*(1+prop_thresh)].index

		self.cg1_genes = cg1_genes
		self.g2m_genes = g2m_genes


	def load_mnase_span(self, chrom, mnase_span):
		"""Load the MNase data for a given span"""

		load_spans = self.get_load_spans_10k(chrom, mnase_span)
		loaded_f_dat = None

		for load_span in load_spans:

			try:
				load_path = f'{self.outdir}/data/chr{chrom}/chr{chrom}_{load_span[0]}_{load_span[1]}.npy'
				loaded_f = np.load(load_path)
			except ValueError:
				continue

			if loaded_f_dat is None:
				loaded_f_dat = loaded_f
			else:
				loaded_f_dat = np.concatenate([loaded_f_dat, loaded_f], axis=2)

		def subset_10k_to_desired_span(gene_10k_data, load_span, desired_span):
			# Subset the loaded 10k window to the desired span, update
			# the span if the span is not a multiple of the bin width

			from src.sgd import get_chromosome_length
			max_bp = get_chromosome_length(chrom)

			bin_width = GlobalConstants.BIN_WIDTH
			first_bp = load_span[0]
			load_indices = (desired_span[0]-first_bp)//bin_width, \
				(desired_span[1]-first_bp)//bin_width
			load_bps = load_indices[0]*bin_width+first_bp, \
				load_indices[1]*bin_width+first_bp

			# If loading is prior to or at the end of the chromosome add appropriate padding
			start_padding = 0
			end_padding = 0

			if load_indices[0] < 0:
				start_padding = -load_indices[0]
				load_indices = 0, load_indices[1]

			if load_bps[1] > max_bp:
				end_padding = (load_bps[1]-max_bp)//bin_width

			selected_loaded_data = gene_10k_data[:, :, load_indices[0]:load_indices[1]]
			selected_loaded_span = load_bps

			if start_padding > 0:
				shape = selected_loaded_data.shape
				selected_loaded_data = np.concatenate([np.zeros((shape[0], shape[1], start_padding)),
					selected_loaded_data], axis=2)

			if end_padding > 0:
				shape = selected_loaded_data.shape
				selected_loaded_data = np.concatenate([selected_loaded_data, 
					np.zeros((shape[0], shape[1], end_padding))], axis=2)

			return selected_loaded_data, selected_loaded_span

		if loaded_f_dat is None:
			return None, None

		loaded_span = (load_spans[0][0], load_spans[-1][1])
		loaded_subset_data, loaded_subset_span = subset_10k_to_desired_span(loaded_f_dat, 
			loaded_span, mnase_span)

		return loaded_subset_data, loaded_subset_span


	def get_load_spans_10k(self, chrom, span):
		"""Get the 10k load spans that span the given span. Assumes the desired span is not larger than 10k."""
		import math
		from src.sgd import get_chromosome_length

		max_bp = get_chromosome_length(chrom)

		start, end = span
		
		scale = 10000
		
		start_int, end_int = int(math.floor(start / scale)), int(math.ceil(end / scale))    

		start_int = max(start_int, 0)

		start_bp = start_int*scale
		end_bp = end_int*scale
		end_bp = min(end_bp, max_bp)

		# If the given span is between two 10k windows, return two spans
		if end_int-start_int > 1 and end_bp - start_bp >= 10000:

			spans = [
				(start_bp, (start_int+1)*scale),
				((start_int+1)*scale, end_bp),
			]
		else:
			spans = [(start_bp, end_bp)]
		
		return spans


	def plot_deconvolved_result(self, f_imgs, mnase_span, smooth=False, 
		normalize=False, vmax=1, figsize=(13, 11), xlims=None):
		from src.global_config import GlobalConstants
		from src.figure_configs import FiguresConfig
		from src.config import load_configs_by_config_type
		from src.orf_plotter import ORFAnnotationPlotter
		from src.geneset import get_deconvolved_geneset

		config, _  = load_configs_by_config_type('shared')
		indices, label_names = config.get_full_phase_indices()

		fig, axs = plt.subplots(len(indices)+1, 1, figsize=figsize)

		ax = axs[0]
		ax.set_xticks([])
		ax.set_yticks([])

		from src.helpers import smooth_data
		extents = [mnase_span[0], mnase_span[-1], 0, GlobalConstants.MAX_Y_LEN]

		for i in range(len(indices)):
			ax = axs[i+1]
			label_name = label_names[i]
			
			current_f_img = f_imgs[indices[i]]

			if smooth:
				current_f_img = smooth_data(current_f_img, size=5, sigma=0.5)

			if normalize:
				current_f_img = current_f_img / current_f_img.sum() * 200.

			ax.imshow(current_f_img, origin='lower', cmap='magma_r', vmax=vmax, aspect='auto',
					 extent=extents, vmin=1)
			ax.set_ylabel(label_name, rotation=0, ha='right', labelpad=9)
			ax.set_yticks([])
			
			if i == (len(indices)-1):
				ax.set_xticks(np.arange(mnase_span[0], mnase_span[1], 2000), minor=False)
				ax.set_xticks(np.arange(mnase_span[0], mnase_span[1], 200), minor=True)
			else:
				ax.set_xticks([])

			ax.axvline(0, c='black', lw=1, alpha=0.5, ls='dotted')

			if xlims is not None:
				ax.set_xlim(*xlims)

		plt.subplots_adjust(top=0.923)

		return fig

	def load_stacked_mnase_data_for_origins(self, origin, padding=800):

		center = origin.pos
		chrom = origin.chr

		span = int(center-padding), int(center+padding)
		self.origin_mnase_data, self.loaded_span = self.load_mnase_span(chrom, span)


	def load_stacked_mnase_data_for_rossi_sites(self, sites, padding=400):

		from src.chromatin_metric_tracking import ChromatinMetricTracking

		self.padding = padding

		self.all_tfs_mnase = None
		self.all_tf_occupancies = None

		n = len(sites)
		tf_frag_lens = (20, 120)
		tf_span = (-40, 40)

		i = 0
		for _, site in sites.iterrows():
			self.load_stacked_mnase_data_for_rossi_site(site, padding)
			tracker = ChromatinMetricTracking(self.tf_mnase, tf_span)
			tracker.select_region(tf_span, tf_frag_lens)
			occ = tracker.track_occupancy()

			if self.all_tfs_mnase is None:
				self.all_tfs_mnase = np.zeros(self.tf_mnase.shape)
				self.all_tf_occupancies = pd.DataFrame(np.zeros((n, len(occ))), index=sites.index)

			self.all_tfs_mnase = self.all_tfs_mnase + self.tf_mnase/n
			self.all_tf_occupancies.loc[i] = occ

			i += 1


	def load_stacked_mnase_data_for_rossi_site(self, site, padding=400):

		self.site = site
		center = (site.start + site.stop)//2
		span = center-padding, center+padding
		self.padding = padding

		self.tf_mnase, self.loaded_span = self.load_mnase_span(site.chr, span)


	def load_stacked_mnase_data_for_genes(self, genes, chroms=range(1, 17), 
		center_mode='+1', padding=2000, normalize=False, 
		compute_gb_occ=False, compute_custom_func=None):
		"""Load set of genes and flip crick strand genes"""
		from src.geneset import get_deconvolved_geneset
		from src.timer import Timer

		timer = Timer()

		if 'chr' not in genes.columns:
			genes_reference = get_deconvolved_geneset()
			genes = genes.join(genes_reference[['chr', 'strand', 'TSS', 'PAS']], how='inner')
		
		# Combine gene data with +1
		if center_mode == "+1":
			rep1_plus1s = pd.read_csv('datasets/computed_mnase/rep1_plus_ones.csv')\
				.set_index('orf_name')
			rep2_plus1s = pd.read_csv('datasets/computed_mnase/rep2_plus_ones.csv')\
				.set_index('orf_name')
			combined_plus1s = (rep1_plus1s + rep2_plus1s)/2
			genes = combined_plus1s.join(genes, how='inner')
			center_key = '+1'
			backup_center_key = 'TSS'
		elif center_mode == 'PAS_nuc':
			PAS_nuc = pd.read_csv('datasets/computed_mnase/computed_PAS_nucs.csv')\
				.set_index('orf_name')
			PAS_nuc['mean_PAS_nuc'] = PAS_nuc.mean(axis=1)
			genes = PAS_nuc.join(genes, how='inner')
			center_key = 'mean_PAS_nuc'
			backup_center_key = 'PAS'
		else:
			raise ValueError(f"Invalid center_mode: {center_mode}")

		loaded_gene_dat = None
		i = 0
		failed_file_not_found_count = 0

		# todo: Parameterize selecting for gene body nucleosomes for occupancy:

		if center_mode == '+1':
			gb_nuc_indices = np.arange(90, 161)
		elif center_mode == 'PAS_nuc':
			 # Select all but the last nucleosome, image is 200 bins wide (halfway is 100)
			gb_nuc_indices = np.arange(40, 90)
		else:
			raise ValueError(f"Invalid center_mode: {center_mode}")

		custom_metric_df = pd.DataFrame(columns=range(178), index=genes.index)

		for chrom in chroms:
			chr_genes = genes[(genes.chr == chrom)]
			for orf_name, gene in chr_genes.iterrows():

				if np.isnan(gene[center_key]):
					center = gene[backup_center_key]
				else: 
					center = gene[center_key]

				span = int(center-padding), int(center+padding)
				
				try:
					gene_data, loaded_span = self.load_mnase_span(chrom, span)
					if gene_data is None:
						print("Failed to find center key for: ", gene.name, chrom, span)
						continue

				except FileNotFoundError:
					print("Failed to find center key for: ", gene.name, chrom, span)
					failed_file_not_found_count += 1
					continue

				if gene.strand == '-':
					gene_data = np.flip(gene_data, axis=2)

				if normalize:
					TOTAL = 1000
					gene_data_sum = gene_data.sum(axis=1).sum(axis=1).reshape((-1, 1, 1))
					gene_data = gene_data / gene_data_sum * TOTAL

				if loaded_gene_dat is None:
					loaded_gene_dat = np.zeros(gene_data.shape)

				if compute_custom_func is None and compute_gb_occ:

					def compute_gb_nuc_occ_func(gene_data, gb_nuc_indices):
						gb_nuc_occ = gene_data[:, :, gb_nuc_indices].sum(axis=1).sum(axis=1)
						return gb_nuc_occ

					compute_custom_func = compute_gb_nuc_occ_func

				if compute_custom_func is not None:
					values = compute_custom_func(gene_data, gb_nuc_indices)
					custom_metric_df.loc[orf_name, :] = values

				loaded_gene_dat += gene_data

				i += 1

				if i % 200 == 0:
					timer.print_time(f"{i}/{len(genes)}")
				
		if failed_file_not_found_count > 0:
			print(f"Failed to load {failed_file_not_found_count} files")

		mean_gene_dat = loaded_gene_dat / len(genes)

		if compute_custom_func is not None:
			return mean_gene_dat, custom_metric_df

		return mean_gene_dat


	def plot_tss_pas_chrom(self, selected_genes, title=None):
		
		from src.figure_configs import FiguresConfig

		# Padding of the window surrounding the TSS and PAS to select
		padding = 800
		inset = 300
		xlims = (-padding*2+inset, padding*2-inset)

		self.selected_genes = selected_genes
		
		tss_f = self.load_stacked_mnase_data_for_genes(selected_genes, padding=padding,
													  center_mode='+1')
		pas_f = self.load_stacked_mnase_data_for_genes(selected_genes, padding=padding,
													  center_mode='PAS_nuc')

		self.tss_f = tss_f
		self.pas_f = pas_f

		tss_pas_concat_f = np.concatenate([tss_f, pas_f], axis=2)

		from src.deconvolved_f_plotter import DeconvolvedFPlotter
		from src.deconvolved_f_plotter import plot_pseudo_gene

		def plot_pseudo_orf(orf_ax):
			plot_pseudo_gene(orf_ax, gene_start=-padding-50, gene_len=padding*2+100)
			orf_ax.set_xlim(*xlims) 
			orf_ax.axvline(0, ls='solid', c='black', alpha=1, zorder=99, lw=1)

		def format_ax(ax, index, i, num_rows, label_name):
			ax.axvline(0, c='black', lw=0.75, alpha=1, ls='solid')    
			ax.axvline(-padding, c='black', lw=0.75, alpha=0.5, ls='dashed')
			ax.axvline(padding, c='black', lw=0.75, alpha=0.5, ls='dashed')

			if index == num_rows-1:
				ax.set_xticks([-padding, -padding+500, padding])
				ax.set_xticklabels(['+1 nuc.', '+500', 'last nuc.'])
				
				ax.set_xticks(np.arange(xlims[0], xlims[1], 100), minor=True)

		plotter = DeconvolvedFPlotter()
		plotter.set_f_imgs(tss_pas_concat_f, (-padding*2, padding*2))
		plotter.figsize = (5, 6)
		plotter.plot_orfs = True
		plotter.xlims = xlims
		plotter.ax_func = format_ax
		plotter.orf_ax_func = plot_pseudo_orf
		self.plotter = plotter

		fig = plotter.plot(vmax=5)
		plt.subplots_adjust(top=0.89)

		if title is not None:
			plt.suptitle(title, fontsize=FiguresConfig.FIG_SUPTITLE_FONTSIZE)

		return fig


def compute_entropy_func(gene_data, gb_nuc_indices):
	"""Compute the entropy of the gene body nucleosome occupancy data"""
	from src.helpers import calc_entropy
	gb_nucs = gene_data[:, :, gb_nuc_indices].reshape((gene_data.shape[0], -1))
	gb_entropy = np.apply_along_axis(lambda row: calc_entropy(row+0.001), axis=1, arr=gb_nucs)
	return gb_entropy

