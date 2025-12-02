import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from src.utils import mkdir_safe
from src.figure_configs import save_figure_for_paper
from src.peak_to_trough import compute_quantile_ptr_2d


class FigureOrigins:
	"""
	Analyze origins of replications, examining global footprint and nucleosome
dynamics. Then sharing some clear examples of these dynamics.

	"""
	
	def __init__(self, output_dir="output/draft4_run/", window_size=160, 
				 subset_qval=0.95, random_seed=123):
		"""
		Initialize the analyzer with configuration parameters.
		"""
		# Configuration parameters
		self.output_dir = output_dir
		self.save_dir = f"{self.output_dir}/figure_origins"
		self.figures_dir = f'{self.output_dir}/Figures'
		mkdir_safe(self.save_dir)

	def setup_data(self):
		from src.origins import load_origins
		from src.replication_timing import ReplicationTiming
		self.origins = load_origins(full=True)
		self.replication_timings = ReplicationTiming(self.output_dir)

		origin_timings = self.origins.copy()
		replication_timings = self.replication_timings
		self.distance_for_entropy_analysis = 32000

		from src.config import load_mean_dg1_mg1_length
		g1_length = load_mean_dg1_mg1_length()
		origin_timings['inferred_firing'] = False

		for oridb, origin in origin_timings.iterrows():
			
			replication_time = replication_timings\
				.load_replication_entry_for(origin.chr, origin.pos).replication_time
			
			# Check neighbors, if the timing for this origin is earlier than
			# its neighbors, we can infer this origin as firing from the replication
			# data
			replication_time_left = replication_timings\
				.load_replication_entry_for(origin.chr, origin.pos-2000).replication_time
			replication_time_right = replication_timings\
				.load_replication_entry_for(origin.chr, origin.pos+2000).replication_time
			
			inferred_firing = replication_time < replication_time_left and\
				replication_time < replication_time_right

			if replication_time < replication_timings.replications_df.replication_time.max() and \
				origin.derived_origin_efficiency_from_mcguffee_et_al_2013 > 0:
				origin_timings.loc[oridb, 'replication_time'] = replication_time+g1_length
				origin_timings.loc[oridb, 'inferred_firing'] = inferred_firing

		# Omit edge cases, negative efficiency and uncalled replication timing
		origin_timings = origin_timings[~origin_timings.replication_time.isna()]
		self.origin_timings = origin_timings
		self.compute_set_of_nearest_and_termination_sites()

		# Compute set of inferred firing and separated/isolated origins
		min_distance_cutoff_from_neighbor = 5000
		origin_nearest = self.origin_nearest
		origins_filtered_by_nearest = origin_nearest[origin_nearest.min_distance_to_neighbor >\
													 min_distance_cutoff_from_neighbor]
		self.inferred_isolated_origins = origins_filtered_by_nearest[origins_filtered_by_nearest.inferred_firing]

	def setup_processors(self, force_recompute=False):

		from src.replication_fork_processor import OriginReplicationForkProcessor
		fork_processor = OriginReplicationForkProcessor(self.output_dir)
		fork_processor.process_all_origins_fork_progression(self.origin_timings,
														 force_recompute=force_recompute)
		self.fork_processor = fork_processor

		from src.replication_fork_processor import OriginReplicationForkProcessor
		fork_processor_no_copy = OriginReplicationForkProcessor(self.output_dir, copy_correct=False)
		fork_processor_no_copy.process_all_origins_fork_progression(self.origin_timings,
														 force_recompute=force_recompute)
		self.fork_processor_no_copy = fork_processor_no_copy

		from src.GenomeDeconvolutionAnalysis import GenomeDeconvolutionAnalysis
		from pipeline.transcription_processor import ExpressionAnalysisProcessor
		from src.deconvolved_tpm_plotter import DeconvolvedTPMPlotter

		# For locus plotting
		self.expression_processor = ExpressionAnalysisProcessor(self.output_dir)
		self.expression_processor.setup_data_loaders()
		self.expression_processor.load_deconvolved_expression()
		self.tpm_plotter = DeconvolvedTPMPlotter(self.expression_processor.expression_data,
								   self.expression_processor.all_transcripts_set)	

		self.genome_deconv_analysis = GenomeDeconvolutionAnalysis(
			'output/draft4_run/')

		# Compute entropies for origins and termination
		self.compute_origin_entropies()
		self.compute_termination_site_entropies()

		# Collect composite data from genome deconv analysis
		# and origin efficiencies
		self.collect_composite_data()

	def plot_replication_fork_heatmaps(self):

		_ = self.fork_processor.plot_all_origin_percentiles('occupancy', normalize=True,
						   branch='t', num_percentiles=5, 
						   inferred=True,
						   percentiles_to_plot=[0, 4],
						   figsize=(7.5, 4.75))
		save_figure_for_paper(f"{self.save_dir}/fork_quintiles_1_5_occupancy.png")

		_ = self.fork_processor.plot_all_origin_percentiles('occupancy', normalize=True,
						inferred=True, branch='t', num_percentiles=5, figsize=(7, 7),
											  plot_phase_labels=False)
		save_figure_for_paper(f"{self.save_dir}/fork_quintiles_inferred_occupancy.png")

		_ = self.fork_processor.plot_all_origin_percentiles('occupancy', normalize=False,
			inferred=True, branch='t', num_percentiles=5, figsize=(7, 7),
													  plot_phase_labels=False)
		plt.subplots_adjust(hspace=0.125)
		save_figure_for_paper(f"{self.save_dir}/fork_quintiles_inferred_not_normalized.png")

		_ = self.fork_processor_no_copy.plot_all_origin_percentiles('occupancy', normalize=True,
						inferred=True, branch='t', num_percentiles=5, figsize=(7, 7),
											  plot_phase_labels=False, plotting_xlim=(-20000, 20000))
		save_figure_for_paper(f"{self.save_dir}/fork_quintiles_no_copy.png")


	def plot_origin_timing(self):

		def _create_attribute(selection, false_value, true_value):
			attribute_values = np.zeros(len(selection)).astype('object')
			attribute_values[~selection] = false_value
			attribute_values[selection] = true_value
			return attribute_values

		origin_timings = self.origin_timings.copy()
		inferred_and_isolated = self.inferred_isolated_origins
		origin_timings['inferred_and_isolated'] = False
		origin_timings.loc[inferred_and_isolated.index, 'inferred_and_isolated'] = True
		early_origins = origin_timings[origin_timings.activation_time == 'early']
		late_origins = origin_timings[origin_timings.activation_time == 'late']

		early_color = '#d14c43'
		late_color = '#3f78d4'

		early_facecolors = _create_attribute(early_origins.inferred_and_isolated, 'none', 
			early_color)
		late_facecolors = _create_attribute(late_origins.inferred_and_isolated, 'none', 
			late_color)

		early_edgecolors = _create_attribute(early_origins.inferred_and_isolated, early_color, early_color)
		late_edgecolors = _create_attribute(late_origins.inferred_and_isolated, late_color, late_color)

		plt.figure(figsize=(6, 4))
		plt.scatter(early_origins.replication_time, 
				  early_origins.derived_origin_efficiency_from_mcguffee_et_al_2013,
				  s=12, edgecolor=early_edgecolors, facecolor=early_facecolors,
				  label=f"Early firing, n={len(early_origins)}, {early_origins.inferred_and_isolated.sum()}", 
					lw=0.3)

		plt.scatter(late_origins.replication_time, 
				  late_origins.derived_origin_efficiency_from_mcguffee_et_al_2013,
				  s=11, edgecolor=late_edgecolors,
				  facecolor=late_facecolors, 
					marker='D',
				  label=f"Late firing, n={len(late_origins)}, {late_origins.inferred_and_isolated.sum()}", 
					lw=0.3)


		num_local_max = len(inferred_and_isolated)

		plt.xlabel("Replication time, min")
		plt.ylabel("Origin efficiency, derived McGuffee et al (2013)")
		plt.title(f"Origin replication timing\nn={len(origin_timings)}, "
		  f"{num_local_max} inferred firing and separated",
		 fontweight='demi', fontsize=16, pad=12)

		from matplotlib.lines import Line2D

		legend_elements_2 = [
			Line2D([0], [0], marker='o', markeredgecolor=early_color, linewidth=0, 
				markerfacecolor='none', markeredgewidth=0.3,
				   markersize=4, label=f'Early firing, n={len(early_origins)}'),
			Line2D([0], [0], marker='D', markeredgecolor=late_color, linewidth=0, 
				markerfacecolor='none', markeredgewidth=0.3,
				   markersize=3, label=f'Late firing, n={len(late_origins)}'),

			Line2D([0], [0], marker='o', markeredgecolor=early_color, linewidth=0, 
				markerfacecolor=early_color, markeredgewidth=0.3,
				   markersize=4, label=f'Early inferred and separated, n={early_origins.inferred_and_isolated.sum()}'),
			Line2D([0], [0], marker='D', markeredgecolor=late_color, linewidth=0, 
				markerfacecolor=late_color, markeredgewidth=0.3,
				   markersize=3, label=f'Late inferred and separated, n={late_origins.inferred_and_isolated.sum()}'),
		]
		plt.legend(handles=legend_elements_2, ncols=2)

		plt.ylim(-0.05, 1.05)
		save_figure_for_paper(f"{self.save_dir}/origin_replication_times.png")

	def plot_all_and_inferred_enrichments(self):
		fig = plt.figure(figsize=(7, 4))
		ax1 = plt.subplot(1, 2, 1)
		self.analyze_enrichment(inferred_firing=False, ax=ax1)

		ax2 = plt.subplot(1, 2, 2)
		self.analyze_enrichment(inferred_firing=True, ax=ax2)
		ax2.set_ylabel("")

		plt.suptitle("Early origin enrichment using replication time", fontweight='demi',
					fontsize=16)
		plt.tight_layout()
		save_figure_for_paper(f"{self.save_dir}/early_origin_enrichments.png")

	def analyze_enrichment(self, inferred_firing=True, ax=None):

		if inferred_firing:
			# origins = origins.loc[origins.inferred_firing]
			origins = self.inferred_isolated_origins
			color = plt.cm.Oranges(0.35)
			title = "Inferred firing and separated origins"
		else:
			color = '#555'
			origins = self.origin_timings
			title = "All origins with replication timing"

		# Sort by replication time (earliest first)
		origins_ranked = origins.sort_values('replication_time').reset_index(drop=True)
		origins_ranked['rank'] = range(1, len(origins_ranked) + 1)

		# Separate the groups
		early_group = origins[origins['activation_time'] == 'early']['replication_time']
		late_group = origins[origins['activation_time'] == 'late']['replication_time']
		
		# Plot replication times by group
		if ax is None:
			fig, ax = plt.subplots(1, 1, figsize=(6, 4))

		# Plot GSEA-like enrichment plot
		es_early = calculate_enrichment_score(origins_ranked, 'early')

		max_possible_enrichment = len(early_group)  # If all early origins were at top
		normalized_score = (es_early / max_possible_enrichment) * 100

		observed_es, p_value, permuted_scores = permutation_test(origins_ranked, 'early',
			n_permutations=20000)

		ax.plot(range(1, len(es_early)+1), normalized_score, lw=2,
			c=color)
		ax.set_xlabel('Rank (earliest to latest replication)')
		ax.set_ylabel('Enrichment score, % of all early origins')
		ax.set_title(f'{title},\nn={len(origins_ranked)}, '
				  f"p-value={p_value:.3g}")
		ax.set_ylim(0, 105)
		ax.set_xlim(0, len(origins))

	def plot_early_fraction_cdf(self, 
								time_col='replication_time',
								label_col='activation_time',
								positive_label='early'):
		"""
		Plot cumulative fraction of early origins vs replication time
		for a full dataset and for a subset, as single lines per panel.

		Parameters
		----------
		df_full, df_subset : pandas.DataFrame
			DataFrames containing replication time and activation label.
		time_col : str
			Column holding replication time values.
		label_col : str
			Column holding 'early' / 'late' labels.
		positive_label : str
			Label that counts as 'early'.

		Returns
		-------
		fig, axes : matplotlib Figure and Axes
		"""
		
		df_full = self.origin_timings.sort_values('replication_time')
		df_subset = self.inferred_isolated_origins.sort_values('replication_time')

		def cumulative_early_fraction(df):
			"""Return replication_time sorted and cumulative fraction of early labels."""
			data = df[[time_col, label_col]].dropna().sort_values(time_col)
			is_early = (data[label_col] == positive_label).astype(int)
			cum_early = is_early.cumsum() / len(is_early[is_early == 1])
			return data[time_col].values, np.clip(cum_early.values, 0, 1)

		fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))

		_, full_p_value, permuted_scores = permutation_test(df_full, 'early',
			n_permutations=20000)
		_, inferred_p_value, permuted_scores = permutation_test(df_subset, 'early',
			n_permutations=20000)

		for data, color, title in zip(
			[df_full, df_subset],
			['#333', 'darkorange'],
			[f"All (n={len(df_full)}),\np-value={full_p_value:.3g}",
			 f"Inferred, separated (n={len(df_subset)}),\np-value={inferred_p_value:.3g}"]
		):
			x, y = cumulative_early_fraction(data)
			ax.step(x, y, lw=1.5, label=title, color=color)
			ax.set_xlim(df_full[time_col].min(), df_full[time_col].max())
			ax.set_ylim(0, 1.005)
			ax.set_xlabel("Replication time (min)")
		
		ax.set_ylabel("Cumulative fraction of early origins")

		plt.title("Cumulative fraction of early-origins\nacross replication timing", 
			fontweight='demi', fontsize=13, pad=10)
		plt.legend()
		save_figure_for_paper(f"{self.save_dir}/cumulative_early_origin_enrichments.png")


	def compute_set_of_nearest_and_termination_sites(self):

		from pipeline.origin_metrics_processor import compute_nearest_neighbor_distance,\
			create_termination_sites

		# Create  a set origin termination sites
		origins = self.origin_timings.copy()

		# For each origin, pompute nearest distances to nearet neighboring origin
		origin_nearest = compute_nearest_neighbor_distance(origins)
		origin_nearest = origin_nearest.sort_values('min_distance_to_neighbor', 
			ascending=False)
		self.origin_nearest = origin_nearest

		self.termination_sites = create_termination_sites(origin_nearest[
			~origin_nearest.nearest_neighbor_id.isna()], N=50)

	def compute_termination_site_entropies(self):
		from pipeline.origin_metrics_processor import compute_nearest_neighbor_distance,\
			create_termination_sites

		from src.origin_entropy_processor import generate_origin_entropy_matrix

		# Now that we have termination sites, let's collect the entropy for these sites and plot
		# the result
		furthest_dist = self.distance_for_entropy_analysis
		
		self.termination_entropies = generate_origin_entropy_matrix(
			self.termination_sites, 
			self.fork_processor.deconvolved_loader,
			furthest_dist=furthest_dist, position_key='termination_pos')


	def plot_termination_entropies(self):
		from src.config import load_default_chrom_configs
		from src.origin_entropy_processor import plot_entropy_by_distance
		furthest_dist = self.distance_for_entropy_analysis

		config1, _ = load_default_chrom_configs()
		plot_entropy_by_distance(self.termination_entropies, config1, furthest_dist=furthest_dist,
		title=f"Nucleosome entropy\nat termination sites, n={len(self.termination_entropies)}",
		xlabel="Distance from termination")
		save_figure_for_paper(f"{self.save_dir}/termination_site_entropy.png")


	def compute_origin_entropies(self):
		from src.origin_entropy_processor import select_region_away, fold_halves_together, _compute_entropy_3d
		from src.origin_entropy_processor import generate_origin_entropy_matrix

		deconvolved_loader = self.fork_processor.deconvolved_loader
		furthest_dist = self.distance_for_entropy_analysis
		origins = self.origin_timings.sort_values('replication_time')
		
		# Inferred
		# inferred_firing_origins = origins[origins.inferred_firing]
		# self.inferred_entropies = generate_origin_entropy_matrix(
		# 	inferred_firing_origins, deconvolved_loader, furthest_dist=furthest_dist)

		# Inferred and 10kb away from neighboring origins
		self.inferred_isolated_origin_entropies = generate_origin_entropy_matrix(
			self.inferred_isolated_origins, deconvolved_loader, furthest_dist=furthest_dist)

		# Efficient and 10kb away from neighboring origins
		# efficiency_cutoff = np.quantile(origin_nearest.derived_origin_efficiency_from_mcguffee_et_al_2013, q=0.9)
		# eff_and_isolated_origins = origin_nearest[(origin_nearest.min_distance_to_neighbor >\
		# 											 min_distance_cutoff_from_neighbor) &
		# 											 (origin_nearest.derived_origin_efficiency_from_mcguffee_et_al_2013 >
		# 											  efficiency_cutoff)]
		# self.efficient_isolated_origins = eff_and_isolated_origins
		# self.efficient_isolated_origin_entropies = generate_origin_entropy_matrix(
		# 	self.efficient_isolated_origins, deconvolved_loader, furthest_dist=furthest_dist)


	def plot_origin_entropies(self):
		from src.origin_entropy_processor import plot_entropy_by_distance
		from src.config import load_default_chrom_configs

		furthest_dist = self.distance_for_entropy_analysis

		config1, _ = load_default_chrom_configs()
		plot_entropy_by_distance(self.inferred_isolated_origin_entropies, config1,
								furthest_dist=furthest_dist)
		save_figure_for_paper(f"{self.save_dir}/inferred_firing_entropies.png")


	def plot_origin_locus(self, origin, label):

		from src.utils import round_nearest

		ars_center = (origin.ars_start+origin.ars_end)//2
		window_size = 4000
		span = ars_center-window_size//2, ars_center+window_size//2
		span = round_nearest(span[0], 100), round_nearest(span[1], 100)

		loaded_data = self.genome_deconv_analysis.load_mnase_span(origin.chr, span)

		self.genome_deconv_analysis.plot_loaded_data(figsize=(13, 11), 
			title=f"{label}, {origin.ars_name}",
			plot_index_labels=False,
			tpm_plotter=self.tpm_plotter)

	def collect_composite_data(self):
		"""Collect composite chromatin signal at early and late origins"""

		from src.GenomeDeconvolutionAnalysis import GenomeDeconvolutionAnalysis
		genome_analysis = self.genome_deconv_analysis
		#origins_sorted = self.origin_timings.sort_values('replication_time')
		#origins_sorted = origins_sorted#[origins_sorted.inferred_firing]

		origins_sorted = self.inferred_isolated_origins.sort_values('replication_time')

		k = 20
		early_origins = origins_sorted.head(k)
		late_origins = origins_sorted.tail(k)

		def collect_composite_origin_data(genome_analysis, origins, window=4000):
			win_2 = window//2
			origin_data = []
			for i in range(k):
				origin = origins.iloc[i]
				span = origin.pos-win_2, origin.pos+win_2
				loaded_data, loaded_span = genome_analysis.load_mnase_span(origin.chr, span)
				origin_data.append(loaded_data)
			average_composite = np.mean(origin_data, axis=0)
			return average_composite

		self.average_earliest = collect_composite_origin_data(genome_analysis, early_origins)
		self.average_latest = collect_composite_origin_data(genome_analysis, late_origins)
		self.num_composite = k

	def plot_composite_heatmap(self, which):
		from src.config import load_default_chrom_configs

		if which == 'early':
			average_composite_data = self.average_earliest
		else:
			average_composite_data = self.average_latest
		
		config1, _ = load_default_chrom_configs()
		t_indices = config1.t_indices()
		b_indices = config1.b_indices()

		num_rows = 8
		fig, axs = plt.subplots(num_rows, 1, figsize=(5, 5))
		step = len(t_indices)//num_rows
		extent = [-2000, 2000, 0, 260]

		for i in range(0, num_rows):
			ax = axs[i]

			plot_index_t = t_indices[i*step]
			plot_index_b = b_indices[i*step]

			composite_data = (average_composite_data[plot_index_t]+
							  average_composite_data[plot_index_b])/2.

			ax.imshow(composite_data, cmap='magma_r', 
					 origin='lower', aspect='auto', vmin=0, vmax=15, 
					 extent=extent)

			# Major ticks
			if i == 1: # G1 Tick
				ax.set_yticks([0])
				ax.set_yticklabels(['Mean G1'], rotation=90, ha='right', va='center')
			elif i == 4:
				ax.set_yticks([0])
				ax.set_yticklabels(['S'], rotation=90, ha='right', va='center')
			elif i == 6:
				ax.set_yticks([0])
				ax.set_yticklabels(['G2/M'], rotation=90, ha='right', va='center')
			else:
				ax.set_yticks([])

			# Minor ticks separating phases
			if i == 0:
				ax.set_yticks([260], minor=True)
			elif i in [3, 5, 7]:
				ax.set_yticks([0], minor=True)

			if i == num_rows-1:
				ax.set_xticks(np.arange(extent[0], extent[1], 500))
				ax.set_xlabel("Position from origin center, bp")
			else:
				ax.set_xticks([])

			ax.tick_params(axis='y', which='major', length=0, pad=2)
			ax.tick_params(axis='y', which='minor', length=13)

			ax.set_xlim(-1000, 1000)

		plt.subplots_adjust(hspace=0)

		if which == 'early':
			plt.suptitle(f"Early inferred, separated origins, n={self.num_composite}", 
				fontweight='demi', fontsize=16)
		elif which == 'late':
			plt.suptitle(f"Late inferred, separated origins, n={self.num_composite}", 
				fontweight='demi', fontsize=16)
		

	def plot_all(self):

		self.plot_origin_timing()
		self.plot_all_and_inferred_enrichments()
		self.plot_early_fraction_cdf()

		# 'oridb_526', # ARS1213, early firing with downstream shift
		# 'oridb_189', # Late firing origin
		self.plot_origin_locus(self.origin_timings.loc['oridb_526'], "Early firing")
		save_figure_for_paper(f"{self.save_dir}/early_origin_locus.png")

		self.plot_origin_locus(self.origin_timings.loc['oridb_189'], "Late firing")
		save_figure_for_paper(f"{self.save_dir}/late_origin_locus.png")

		# Plot all inferred origin entropy heatmaps
		self.plot_origin_entropies()

		# Plot termination
		self.plot_termination_entropies()

		# Plot composite heatmaps
		self.plot_composite_heatmap(which='early')
		save_figure_for_paper(f"{self.save_dir}/early_origins_composite.png")

		self.plot_composite_heatmap(which='late')
		save_figure_for_paper(f"{self.save_dir}/late_origins_composite.png")


	def layout_panel(self):

		from pipeline.figure_composer import FigureCompositor
		from pipeline.figure_composer_helpers import layout_images_vertically, \
			add_panel_labels_to_images, layout_images_horizontally

		# Create compositor with wider dimensions for horizontal layout
		compositor = FigureCompositor(1024, 410, debug_mode=True)

		# ---------- Origins ------------------

		image_paths = [
			f'{self.save_dir}/early_origin_locus.png',
			f'{self.save_dir}/late_origin_locus.png',
			f'{self.save_dir}/inferred_firing_entropies.png',
			f'{self.save_dir}/termination_site_entropy.png',
		]

		placed_images = layout_images_horizontally(
			compositor,
			image_paths[:3],
			between_padding=32,
			offsets=[(0, 0), (0, 0), (0, 0)],
			width_proportions=[0.44, 0.44, 0.16],
			margin=(33, 30),
			image_keys=['early_locus', 'late_locus', 'inferred_entropies']  # Custom keys
		)

		inferred_ent_img = placed_images['inferred_entropies']

		e_img = compositor.place_image(
			image_paths[3], inferred_ent_img['logical_position'][0],
			inferred_ent_img['logical_position'][1]+inferred_ent_img['logical_size'][1]+30,
			width=inferred_ent_img['logical_size'][0], name='termination_entropies'
		)

		# ------- Labels ----------

		offsets = [(-20, 20)]*len(compositor.placed_images)

		# Add panel labels
		add_panel_labels_to_images(
			compositor, 
			compositor.placed_images,
			"abcd",
			font_size=30,
			offsets=offsets
		)

		# Save the composite figure
		compositor.save(f'{self.figures_dir}/Figure4_Origins.png')

	def layout_supplemental_panel(self):

		from pipeline.figure_composer import FigureCompositor
		from pipeline.figure_composer_helpers import layout_images_vertically, \
			add_panel_labels_to_images, layout_images_horizontally

		# Create compositor with wider dimensions for horizontal layout
		compositor = FigureCompositor(1024, 560, debug_mode=True)

		image_paths = [
			f'{self.save_dir}/early_origins_composite.png',
			f'{self.save_dir}/late_origins_composite.png',
		]

		placed_images = layout_images_horizontally(
			compositor,
			image_paths,
			between_padding=30,
			offsets=[(0, 0), (0, 0)],
			width_proportions=[1, 1],
			margin=(30, 30),
			image_keys=['early_composite', 'late_composite']
		)

		# Add panel labels
		add_panel_labels_to_images(
			compositor, 
			compositor.placed_images,
			font_size=36,
			offset=[-16, 29]
		)

		# Save the composite figure
		compositor.save(f'{self.figures_dir}/Supplemental9_Origins.png')


	def layout_supplemental_replication_origins_validation(self):

		from pipeline.figure_composer import FigureCompositor
		from pipeline.figure_composer_helpers import layout_images_vertically, \
			add_panel_labels_to_images, layout_images_horizontally

		# Create compositor with wider dimensions for horizontal layout
		compositor = FigureCompositor(1024, 420, debug_mode=True)

		nucleosomes_fig_dir = self.save_dir
		origins_fig_dir = f"{self.output_dir}/figure_origins"

		# ---------- Origins ------------------

		image_paths = [
			f'{origins_fig_dir}/origin_replication_times.png',
			f'{origins_fig_dir}/cumulative_early_origin_enrichments.png',
		]

		placed_images = layout_images_horizontally(
			compositor,
			image_paths,
			between_padding=32,
			offsets=[(0, 0), (0, 0)],
			width_proportions=[0.44, 0.44],
			margin=(30, 30),
			image_keys=['replication_times', 'cumualtive_early_origins']
		)

		# ------- Labels ----------

		# Add panel labels
		add_panel_labels_to_images(
			compositor, 
			compositor.placed_images,
			font_size=30,
			offset=(-15, 22)
		)

		# Save the composite figure
		compositor.save(f'{self.figures_dir}/Supplemental2.7_Replication_Origins_Validation.png')


def calculate_enrichment_score(ranked_df, group_label):
	"""Calculate running enrichment score for a group"""
	n_total = len(ranked_df)
	is_in_set = (ranked_df['activation_time'] == group_label).astype(int)
	n_in_set = is_in_set.sum()
	
	# Running sum: +1 for hits, -1/(n_total-n_in_set) for misses
	hit_increment = 1
	miss_increment = -1 / (n_total - n_in_set) if n_in_set < n_total else 0
	
	running_score = []
	current_score = 0
	
	for hit in is_in_set:
		if hit:
			current_score += hit_increment
		else:
			current_score += miss_increment

		running_score.append(current_score)
	
	return np.array(running_score)


def permutation_test(ranked_df, group_label, n_permutations=1000):
	"""Test significance of enrichment score via permutation"""
	
	np.random.seed(123)

	observed_es = calculate_enrichment_score(ranked_df, group_label).max()
	
	permuted_scores = []
	for _ in range(n_permutations):
		# Shuffle group labels
		shuffled_df = ranked_df.copy()
		shuffled_df['activation_time'] = np.random.permutation(shuffled_df['activation_time'])
		permuted_es = calculate_enrichment_score(shuffled_df, group_label).max()
		permuted_scores.append(permuted_es)
	
	# P-value: fraction of permutations with score >= observed
	# Add pseudocount to avoid exact p=0
	n_greater_equal = np.sum(np.array(permuted_scores) >= observed_es)
	p_value = (n_greater_equal + 1) / (n_permutations + 1)

	return observed_es, p_value, permuted_scores
