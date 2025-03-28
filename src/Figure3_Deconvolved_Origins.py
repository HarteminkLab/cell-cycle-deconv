
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.figure_configs import FiguresConfig
from src.chromatin_metric_tracking import ChromatinMetricTracking
from src.figure_configs import save_figure_for_paper
from src.origins import get_origin_title_name
from src.plot_helpers import plot_heatmap_cell_cycle_tps
from src.chromatin_model import draw_phase_label_annotations


ORIGIN_FRAG_SPAN = (0, 100)

from src.config import load_configs_by_config_type

config, _ = load_configs_by_config_type('shared')
t_indices = config.get_Hpositions_for_branch('t')


class Figure4DeconvolvedOrigins(object):
	"""Fourth figure: deconvolved origins analysis. Show how the deconvolution
	algorithm is able to track known biology around origin of replication Mcm2-7
	loading and unloading at replication time and difference between early and
	late replicating origins"""


	def __init__(self):

		from src.GenomeDeconvolutionAnalysis import GenomeDeconvolutionAnalysis

		outdir = "output/deconvolved_genome_g0066_offset1_10k_10x10_2024_10_11/"
		analysis = GenomeDeconvolutionAnalysis(outdir)
		from src.origins import load_origins_w_replication

		origins = load_origins_w_replication()

		self.origins = origins
		self.analysis = analysis

	def load_origin(self, origin=None):

		if origin is None:
			self.origin = self.origins.loc['oridb_809']
		else:
			self.origin = origin

		self.padding = 600
		self.analysis.load_stacked_mnase_data_for_origins(self.origin, padding=self.padding)

		# Flip if +, upstream
		# or      -, downstream
		mnase_data = self.analysis.origin_mnase_data
		if (self.origin.strand == '+' and
			self.origin.mcm_loading_class == 'upstream') or \
		   (self.origin.strand == '-' and
			self.origin.mcm_loading_class == 'downstream'):
			self.analysis.origin_mnase_data = np.flip(mnase_data, axis=2)

	def plot_loaded_origin(self):
		from src.deconvolved_f_plotter import DeconvolvedFPlotter

		plotter = DeconvolvedFPlotter()
		plotter.set_f_imgs(self.analysis.origin_mnase_data, (-self.padding, self.padding))
		plotter.figsize = (5, 6.5)
		plotter.plot_orfs = False

		from src.config import load_configs_by_config_type
		config1, config2 = load_configs_by_config_type('shared')

		cg1_index = config1.get_Hpositions_for_branch('t')[0]

		def ax_func(ax, index, i, n, label_name):

			ax.axvline(self.m1_track[index], lw=1, c='red', alpha=0.5, ls='solid')
			ax.axvline(self.p1_track[index], lw=1, c='red', alpha=0.5, ls='solid')

			ax.axvline(self.m1_track[cg1_index], lw=1, c='gray', alpha=1, ls='dotted')
			ax.axvline(self.p1_track[cg1_index], lw=1, c='gray', alpha=1, ls='dotted')

		plotter.ax_func = ax_func

		from src.origins import get_origin_title_name

		fig = plotter.plot(vmax=20)
		plt.suptitle(f"{get_origin_title_name(self.origin)}, {self.origin.activation_time} replicating,\n"
			f"efficiency={self.origin.derived_origin_efficiency_from_mcguffee_et_al_2013:.3f}", 
			fontsize=FiguresConfig.FIG_SUPTITLE_FONTSIZE)
		plt.subplots_adjust(top=0.87)

	def plot_tracking_summary_hm(self):

		# Load the small fragments for the entire window, create a summary depiction
		# of the small fragments occupancy over time
		from src.Figure3_Deconvolved_Origins import ORIGIN_FRAG_SPAN

		f_mnase_span = -self.padding, self.padding

		mnase_data = self.analysis.origin_mnase_data

		origin_tracker = ChromatinMetricTracking(mnase_data, f_mnase_span)
		origin_tracker.select_region(f_mnase_span, ORIGIN_FRAG_SPAN) 

		# Idea will be to merge with the nucleosome reads over time to show both nucleosome
		# positioning and origin occupancy throughout the time course
		from src.global_config import fragment_lengths_definitions

		_, _, nuc_lens = fragment_lengths_definitions()
		f_mnase_span = -self.padding, self.padding
		nuc_tracker = ChromatinMetricTracking(mnase_data, f_mnase_span)
		nuc_tracker.select_region(f_mnase_span, nuc_lens)
				
		nuc_dat = nuc_tracker.selected_img_data.sum(axis=1)
		sm_dat = origin_tracker.selected_img_data.sum(axis=1)

		# Normalize so lowest value is 0 (to adjust for deconvolution pseudocount)
		nuc_med, sm_med = np.median(nuc_dat.flatten()), np.median(sm_dat.flatten())
		nuc_dat -= nuc_med
		sm_dat -= sm_med

		# Plot the difference to show both origin occupancy and nucleosome
		# occupancy/positioning
		plt_data = nuc_dat - sm_dat

		# Segment by g1 and s/g2m
		cg1_indices = config.get_Hpositions_for_phase('CG1')
		postG1_indices = config.get_Hpositions_for_phase('postG1')

		# define extents
		cg1_tps = config.get_phase_timepoints_for_phase('CG1')
		postG1_tps = config.get_phase_timepoints_for_phase('postG1')
		tps = config.get_timepoints_for_branch('t')

		n = len(t_indices)

		g1_extent = [f_mnase_span[0], f_mnase_span[1], cg1_tps[0], cg1_tps[-1]]
		postG1_extent = [f_mnase_span[0], f_mnase_span[1], cg1_tps[-1], postG1_tps[-1]]
 
		fig = plt.figure(figsize=(6, 4))
		ax = plt.gca()
		ax.imshow(plt_data[cg1_indices], extent=g1_extent, aspect='auto',
			cmap='BrBG', vmin=-50, vmax=50, origin='lower')

		ax.imshow(plt_data[postG1_indices], extent=postG1_extent, aspect='auto',
			cmap='BrBG', vmin=-50, vmax=50, origin='lower')
		ax.set_ylim(g1_extent[2], postG1_extent[3])

		ys = tps
		ax.plot(self.p1_track[t_indices], ys, c='red', ls='solid', lw=1.5, alpha=0.5)
		ax.plot(self.m1_track[t_indices], ys, c='red', ls='solid', lw=1.5, alpha=0.5)

		plt.suptitle(f"{get_origin_title_name(self.origin)}, {self.origin.activation_time} replicating,\n"
			f"efficiency={self.origin.derived_origin_efficiency_from_mcguffee_et_al_2013:.3f}", 
			fontsize=FiguresConfig.FIG_SUPTITLE_FONTSIZE)

		cg1_len = config.get_g1_lens('CG1')
		plt.axhline(self.origin.replication_time-cg1_len, c='red', ls='dotted')

		draw_phase_label_annotations(ax, config,
			annotations_x=g1_extent[0]-35)
		ax.set_xlim(g1_extent[0]-70, g1_extent[1])
		ax.set_ylabel("Cell cycle time, min")
		ax.set_xlabel("Genomic position, bp")
		plt.subplots_adjust(bottom=0.15, top=0.83)

		ax.set_ylim(postG1_extent[3], g1_extent[2])


	def track_nfr(self):
		from src.global_config import GlobalConstants
		from src.global_config import fragment_lengths_definitions

		sm_span, med, nuc_span = fragment_lengths_definitions()

		f_img_data = self.analysis.origin_mnase_data
		f_mnase_span = -self.padding, self.padding	

		# Find the origin and set the -1 and +1 positions from that
		tracker = ChromatinMetricTracking(f_img_data, f_mnase_span)
		tracker.select_region((-100, 100), ORIGIN_FRAG_SPAN)
		tracker.track_peak_position()

		origin_pos = tracker.weighted_mean_tracking.mean()
		self.origin_tracker = tracker
		self.origin_occ = tracker.track_occupancy()

		# Find the origin and set the -1 and +1 positions from that
		tracker = ChromatinMetricTracking(f_img_data, f_mnase_span)
		span = int(origin_pos), int(origin_pos+250)
		tracker.select_region(span, nuc_span)
		tracker.track_peak_position()
		self.p1_track = tracker.weighted_mean_tracking
		self.p1_tracker = tracker

		tracker = ChromatinMetricTracking(f_img_data, f_mnase_span)
		span = int(origin_pos)-250, int(origin_pos)
		tracker.select_region(span, nuc_span)
		tracker.track_peak_position()
		self.m1_track = tracker.weighted_mean_tracking
		self.m1_tracker = tracker


	def plot_heatmap_nfr(self, all_p1s_df, all_m1s_df, annotate_oridbs=[]):

		sorted_origins = self.origins.sort_values('replication_time')
		nfr_df = all_p1s_df - all_m1s_df

		normalized = (nfr_df - nfr_df.mean(axis=1).values.reshape((-1, 1))) #/ \
		   #nfr_df.std(axis=1).values.reshape((-1, 1))

		self.nfr_widths = normalized
		self.normalized_nfr_width = normalized

		plt_data = normalized[t_indices].loc[sorted_origins.index]
		plt.figure(figsize=(6, 8))

		ax = plt.gca()
		im = plot_heatmap_cell_cycle_tps(ax, config, plt_data, -20, 20, 'RdBu_r',
			annotations_x_offset=5, ylim_offset=10)
		ax.set_yticks([])
		ax.set_ylabel("Origins, sorted by replication time")

		plt.title(f"Origin NFR dynamics, n={len(plt_data)}", pad=10, 
			fontsize=FiguresConfig.FIG_SUPTITLE_FONTSIZE)
		cbar = plt.colorbar(im, ax=ax, pad=0.1)
		cbar.ax.set_ylabel("Normalized NFR width", rotation=270, va='bottom')
		ax.set_xlabel("Single cell time, min")

		cg1_len = config.get_g1_lens('CG1')
		ys = np.arange(len(plt_data))
		ax.scatter(sorted_origins.replication_time-cg1_len, ys, marker='D', s=1, 
			color='red')

		if len(annotate_oridbs) > 0:
			oridb_indices = self.origins.loc[sorted_origins.index].copy()
			oridb_indices['origin_index'] = ys

			tick_positions = []
			for oridb_name in annotate_oridbs:
				
				oridb_row = oridb_indices.loc[oridb_name]
				tick_positions.append(oridb_row.origin_index)

			ax2 = ax.twinx()
			ax2.set_ylim(ax.get_ylim())
			ax2.set_yticks(tick_positions)
			ax2.set_yticklabels(oridb_indices.loc[annotate_oridbs].ars_name.values,
				rotation=270, ha='center', va='center')
			ax2.tick_params(axis='y', pad=5) 


	def plot_heatmap_occupancy(self, all_occs_df, annotate_oridbs=[]):

		sorted_origins = self.origins.sort_values(
			'replication_time', ascending=True)

		# sorted_origins = self.origins.sort_values(
		#     'derived_origin_efficiency_from_mcguffee_et_al_2013', ascending=False)

		normalized_occs = (all_occs_df - all_occs_df.mean(axis=1).values.reshape((-1, 1))) /\
			all_occs_df.std(axis=1).values.reshape((-1, 1))

		plt_data = normalized_occs[t_indices].loc[sorted_origins.index]
		plt.figure(figsize=(6, 4))
		ax = plt.gca()
		im = plot_heatmap_cell_cycle_tps(ax, config, plt_data, -2, 2, 'RdBu_r')
		ax.set_yticks([])
		ax.set_ylabel("Origins, sorted by replication time")
		plt.title(f"Origin binding dynamics, n={len(plt_data)}", pad=10, fontsize=FiguresConfig.FIG_SUPTITLE_FONTSIZE)

		cbar = plt.colorbar(im)
		cbar.ax.set_ylabel("Normalized origin occupancy", rotation=270, va='bottom')
		ax.set_xlabel("Single cell time, min")

		cg1_len = config.get_g1_lens('CG1')
		ys = np.arange(len(plt_data))
		ax.scatter(sorted_origins.replication_time-cg1_len, ys, marker='D', s=1, 
			color='red')

		if len(annotate_oridbs) > 0:
			oridb_indices = self.origins.loc[sorted_origins.index].copy()
			oridb_indices['origin_index'] = ys

			tick_positions = []
			for oridb_name in annotate_oridbs:
				
				oridb_row = oridb_indices.loc[oridb_name]
				tick_positions.append(oridb_row.origin_index)

			ax2 = ax.twinx()
			ax2.set_ylim(ax.get_ylim())
			ax2.set_yticks(tick_positions)
			ax2.set_yticklabels(oridb_indices.loc[annotate_oridbs].ars_name.values,
				rotation=270, ha='center', va='center')
			ax2.tick_params(axis='y', pad=5) 


	def plot_occupancy_partition_hm(self, normalized_occs, sorted_origins):
		"""Plot heatmap partitioning out origins with peak origin occupancy in G1, see if there
		is a connection with when the NFR is at its peak size."""

		sel_g1_origin = (normalized_occs[t_indices[20]] > normalized_occs[t_indices[20]]*0.1)

		fig = plt.figure(figsize=(4, 4))
		grid = plt.GridSpec(3, 2, hspace=0.15, wspace=0.1)

		# Create the first two 1x1 axes (top left and top right)
		ax1 = fig.add_subplot(grid[0, 0])
		ax2 = fig.add_subplot(grid[0, 1])

		# Create the remaining two 2x1 axes (spanning the bottom two rows)
		ax3 = fig.add_subplot(grid[1:, 0])
		ax4 = fig.add_subplot(grid[1:, 1])

		plt_data = normalized_occs[t_indices].loc[sorted_origins.index].loc[sel_g1_origin]
		plot_heatmap_cell_cycle_tps(ax1, config, plt_data, -50, 50, 'RdBu_r', plot_phase_labels=False)

		ax1.set_xticks([])
		ax1.set_ylabel(f"G1 peak origins,\nn={sel_g1_origin.sum()}")
		ax1.set_yticks([])
		ax1.set_title("Origin occupancy")

		plt_data = normalized_occs[t_indices].loc[sorted_origins.index].loc[~sel_g1_origin]
		plot_heatmap_cell_cycle_tps(ax3, config, plt_data, -50, 50, 'RdBu_r')
		ax3.set_ylabel(f"non-G1 peak origins,\nn={(sel_g1_origin==0).sum()}")
		ax3.set_yticks([])
		ax3.set_xticks([])

		plt_data = self.normalized_nfr_width[t_indices]\
					   .loc[sorted_origins.index].loc[sel_g1_origin]
		plot_heatmap_cell_cycle_tps(ax2, config, plt_data, -100, 100, 'RdBu_r', plot_phase_labels=False)
		ax2.set_yticks([])
		ax2.set_xticks([])
		ax2.set_title("NFR width")

		plt_data = self.normalized_nfr_width[t_indices]\
					   .loc[sorted_origins.index].loc[~sel_g1_origin]
		plot_heatmap_cell_cycle_tps(ax4, config, plt_data, -100, 100, 'RdBu_r')
		ax4.set_yticks([])
		ax4.set_xticks([])


	def compute_m1_p1_tracks(self):
		from src.timer import Timer

		timer = Timer()

		all_occs_df = pd.DataFrame()
		all_m1s_df = pd.DataFrame()
		all_p1s_df = pd.DataFrame()

		for oridb, origin in self.origins.iterrows():

			self.load_origin(origin)
			self.track_nfr()
			m1_track = self.m1_track
			p1_track = self.p1_track
			occ_track = self.origin_occ
			m1_df = pd.DataFrame(m1_track, columns=[oridb]).T
			p1_df = pd.DataFrame(p1_track, columns=[oridb]).T
			occ_df = pd.DataFrame(occ_track, columns=[oridb]).T

			all_occs_df = pd.concat([all_occs_df, occ_df])
			all_m1s_df = pd.concat([all_m1s_df, m1_df])
			all_p1s_df = pd.concat([all_p1s_df, p1_df])

		timer.print_time()
		self.all_occs_df = all_occs_df
		self.all_m1s_df = all_m1s_df
		self.all_p1s_df = all_p1s_df


	def filter_p1_m1_origins(self):

		all_p1s_df = self.all_p1s_df
		all_m1s_df = self.all_m1s_df

		from src.config import load_configs_by_config_type
		config, _ = load_configs_by_config_type('shared')
		t_indices = config.get_Hpositions_for_branch('t')

		shift_cutoff = 4

		# todo: Better convey the nucleosome shift, Belsky shows this shift as +20 nt 
		# and that the upstream nucleosome may also shift up...
		p1s_t = all_p1s_df[t_indices]
		p1_mean = p1s_t[t_indices].mean(axis=1)
		p1_shift = p1s_t - p1_mean.values.reshape((-1, 1))

		m1s_t = all_m1s_df[t_indices]
		m1_mean = m1s_t[t_indices].mean(axis=1)
		m1_shift = m1s_t - m1_mean.values.reshape((-1, 1))

		m1_q10_shift = m1_shift.quantile(0.1, axis=1)
		p1_q90_shift = p1_shift.quantile(0.9, axis=1)

		m1_shifted = m1_q10_shift[m1_q10_shift < -shift_cutoff]
		p1_shifted = p1_q90_shift[p1_q90_shift > shift_cutoff]

		self.m1_shifted = m1_shifted
		self.p1_shifted = p1_shifted

		plt.figure(figsize=(9, 2))
		plt.subplot(1, 2, 1)
		plt.hist(m1_q10_shift, bins=50)
		plt.title("-1 nucleosome shift")

		plt.subplot(1, 2, 2)
		plt.hist(p1_q90_shift, bins=50)
		plt.title("+1 nucleosome shift")

		all_p1s_normalized = all_p1s_df[t_indices] - \
					all_p1s_df[t_indices].mean(axis=1).values.reshape((-1, 1))
		all_m1s_normalized = all_m1s_df[t_indices] - \
			all_m1s_df[t_indices].mean(axis=1).values.reshape((-1, 1))

		self.select_both = all_p1s_normalized.index.isin(p1_shifted.index) & \
			all_p1s_normalized.index.isin(m1_shifted.index)
		self.select_only_p1 = all_p1s_normalized.index.isin(p1_shifted.index) & \
			~all_p1s_normalized.index.isin(m1_shifted.index)
		self.select_only_m1 = ~all_p1s_normalized.index.isin(p1_shifted.index) & \
			all_p1s_normalized.index.isin(m1_shifted.index)
		self.select_neither = ~all_p1s_normalized.index.isin(p1_shifted.index) & \
			~all_p1s_normalized.index.isin(m1_shifted.index)

		self.orfs_m1_p1_both = all_p1s_normalized.loc[self.select_both].index
		self.orfs_m1_p1_only_p1 = all_p1s_normalized.loc[self.select_only_p1].index
		self.orfs_m1_p1_only_m1 = all_p1s_normalized.loc[self.select_only_m1].index
		self.orfs_m1_p1_neither = all_p1s_normalized.loc[self.select_neither].index
		

	def plot_m1_p1_heatmap(self):

		select_both = self.select_both  
		select_only_p1 = self.select_only_p1  
		select_only_m1 = self.select_only_m1  
		select_neither = self.select_neither  

		m1_shifted = self.m1_shifted
		p1_shifted = self.p1_shifted
		all_p1s_df = self.all_p1s_df
		all_m1s_df = self.all_m1s_df

		all_p1s_normalized = all_p1s_df[t_indices] - \
			all_p1s_df[t_indices].mean(axis=1).values.reshape((-1, 1))
		all_m1s_normalized = all_m1s_df[t_indices] - \
			all_m1s_df[t_indices].mean(axis=1).values.reshape((-1, 1))

		vmax = 20
		plt.figure(figsize=(8, 11))

		def plot_m1p1_heatmaps_axs(plt_idx, selection):
			# Filtered for both +1 and -1 shifted origins
			filtered_m1_shifts_normalized = all_m1s_normalized.loc[selection][t_indices]
			filtered_p1_shifts_normalized = all_p1s_normalized.loc[selection][t_indices]

			plt.subplot(4, 2, plt_idx)
			plt.imshow(filtered_m1_shifts_normalized, cmap='RdBu_r', 
			           vmin=-vmax, vmax=vmax, aspect='auto')
			plt.yticks([])
			plt.xticks([])

			plt.subplot(4, 2, plt_idx+1)
			plt.imshow(filtered_p1_shifts_normalized, cmap='RdBu_r',
			           vmin=-vmax, vmax=vmax, aspect='auto')
			plt.yticks([])
			plt.xticks([])


		# Filtered for both +1 and -1 shifted origins
		plot_m1p1_heatmaps_axs(1, self.orfs_m1_p1_both)

		# Filtered for +1
		plot_m1p1_heatmaps_axs(3, self.orfs_m1_p1_only_m1)

		# Filtered for -1
		plot_m1p1_heatmaps_axs(5, self.orfs_m1_p1_only_p1)

		# Filtered for neither
		plot_m1p1_heatmaps_axs(7, self.orfs_m1_p1_neither)
