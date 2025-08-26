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

	def setup_processors(self, force_recompute=False):
		from pipeline.origin_metrics_processor import OriginFootprintProcessor
		self.origin_processor = OriginFootprintProcessor(self.output_dir)
		self.origin_processor.process_all_origins(self.origin_timings, force_recompute=force_recompute)

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


	def plot_origin_timing(self):

		def _create_attribute(selection, false_value, true_value):
			attribute_values = np.zeros(len(selection)).astype('object')
			attribute_values[~selection] = false_value
			attribute_values[selection] = true_value
			return attribute_values

		origin_timings = self.origin_timings
		early_origins = origin_timings[origin_timings.activation_time == 'early']
		late_origins = origin_timings[origin_timings.activation_time == 'late']

		early_color = '#d14c43'
		late_color = '#3f78d4'

		early_facecolors = _create_attribute(early_origins.inferred_firing, 'none', 
			early_color)
		late_facecolors = _create_attribute(late_origins.inferred_firing, 'none', 
			late_color)

		early_edgecolors = _create_attribute(early_origins.inferred_firing, early_color, early_color)
		late_edgecolors = _create_attribute(late_origins.inferred_firing, late_color, late_color)

		plt.figure(figsize=(6, 4))
		plt.scatter(early_origins.replication_time, 
				  early_origins.derived_origin_efficiency_from_mcguffee_et_al_2013,
				  s=12, edgecolor=early_edgecolors, facecolor=early_facecolors,
				  label=f"Early firing, n={len(early_origins)}, {early_origins.inferred_firing.sum()}", 
					lw=0.3)

		plt.scatter(late_origins.replication_time, 
				  late_origins.derived_origin_efficiency_from_mcguffee_et_al_2013,
				  s=11, edgecolor=late_edgecolors,
				  facecolor=late_facecolors, 
					marker='D',
				  label=f"Late firing, n={len(late_origins)}, {late_origins.inferred_firing.sum()}", 
					lw=0.3)

		num_local_max = origin_timings.inferred_firing.sum()

		plt.xlabel("Replication time, min")
		plt.ylabel("Origin efficiency, derived McGuffee et al (2013)")
		plt.title(f"Origin replication timing\nn={len(origin_timings)}, "
		  f"{num_local_max} inferred firing",
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
				   markersize=4, label=f'Early inferred, n={early_origins.inferred_firing.sum()}'),
			Line2D([0], [0], marker='D', markeredgecolor=late_color, linewidth=0, 
				markerfacecolor=late_color, markeredgewidth=0.3,
				   markersize=3, label=f'Late inferred, n={late_origins.inferred_firing.sum()}'),
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
		origins = self.origin_timings
		origins = origins.sort_values('replication_time')

		if inferred_firing:
			origins = origins.loc[origins.inferred_firing]
			color = plt.cm.Oranges(0.35)
			title = "Inferred firing"
		else:
			color = '#555'
			title = "All origins with replication timing"

		# Sort by replication time (earliest first)
		origins_ranked = origins.sort_values('replication_time').reset_index(drop=True)
		origins_ranked['rank'] = range(1, len(origins_ranked) + 1)

		# Separate the groups
		early_group = origins[origins['activation_time'] == 'early']['replication_time']
		late_group = origins[origins['activation_time'] == 'late']['replication_time']
		
		# Plot replication times by group
		if ax is None:
			fig = plt.figure(figsize=(6, 4))

		# Plot GSEA-like enrichment plot
		es_early = calculate_enrichment_score(origins_ranked, 'early')

		max_possible_enrichment = len(early_group)  # If all early origins were at top
		normalized_score = (es_early / max_possible_enrichment) * 100

		observed_es, p_value, permuted_scores = permutation_test(origins_ranked, 'early',
			n_permutations=1000)
		observed_es, p_value

		ax.plot(range(1, len(es_early)+1), normalized_score, lw=2,
			c=color)
		ax.set_xlabel('Rank (earliest to latest replication)')
		ax.set_ylabel('Enrichment score, % of all early origins')
		ax.set_title(f'{title},\nn={len(origins_ranked)}, '
				  f"p_value={p_value:.4f}")
		ax.set_ylim(0, 105)
		ax.set_xlim(0, len(origins))


	def plot_origin_locus(self, origin, label):

		from src.utils import round_nearest

		ars_center = (origin.ars_start+origin.ars_end)//2
		span = ars_center-1000, ars_center+1000
		span = round_nearest(span[0], 100), round_nearest(span[1], 100)

		loaded_data = self.genome_deconv_analysis.load_mnase_span(origin.chr, span)

		self.genome_deconv_analysis.plot_loaded_data(figsize=(7, 11), 
			title=f"{label}, {origin.ars_name}",
			plot_index_labels=False,
			tpm_plotter=self.tpm_plotter)


	def plot_all(self):

		self.plot_origin_timing()
		self.plot_all_and_inferred_enrichments()

		self.origin_processor.plot_origin_metrics_heatmaps()
		save_figure_for_paper(f"{self.save_dir}/all_origin_metrics_heatmap.png")

		inferred_firing_origin = self.origin_timings
		inferred_firing_origin = inferred_firing_origin[inferred_firing_origin.inferred_firing].sort_values('replication_time')
		self.origin_processor.plot_origin_metrics_heatmaps(inferred_firing_origin.index)
		save_figure_for_paper(f"{self.save_dir}/inferred_firig_origin_metrics_heatmap.png")

		# 'oridb_526', # ARS1213, early firing with downstream shift
		# 'oridb_189', # Late firing origin
		self.plot_origin_locus(self.origin_timings.loc['oridb_526'], "Early firing")
		save_figure_for_paper(f"{self.save_dir}/early_origin_locus.png")

		self.plot_origin_locus(self.origin_timings.loc['oridb_189'], "Late firing")
		save_figure_for_paper(f"{self.save_dir}/late_origin_locus.png")

	def layout_panel(self):

		from pipeline.figure_composer import FigureCompositor
		from pipeline.figure_composer_helpers import layout_images_vertically, \
			add_panel_labels_to_images, layout_images_horizontally

		# Create compositor with wider dimensions for horizontal layout
		compositor = FigureCompositor(1024, 610, debug_mode=True)

		image_paths = [
			f'{self.save_dir}/origin_replication_times.png',
			f'{self.save_dir}/early_origin_enrichments.png',
			f'{self.save_dir}/inferred_firig_origin_metrics_heatmap.png',
			f'{self.save_dir}/early_origin_locus.png',
			f'{self.save_dir}/late_origin_locus.png',
		]

		placed_images = layout_images_vertically(
			compositor,
			image_paths[:3],
			between_padding=30,
			offsets=[(0, 0), (0, 0), (0, 0)],
			widths=[260, 260, 260],
			margin=(30, 30),
			image_keys=['origins_repl', 'enrichments', 'chromatin']  # Custom keys
		)

		e_img = compositor.place_image(
			image_paths[3], placed_images['chromatin']['logical_position'][0]+
			placed_images['chromatin']['logical_size'][0]+30, 
			30, height=560, name='early'
		)

		compositor.place_image(
			image_paths[4], e_img['logical_position'][0]+e_img['logical_size'][0]+30, 
			30, height=560, name='late'
		)

		# Add panel labels
		add_panel_labels_to_images(
			compositor, 
			compositor.placed_images,
			font_size=26,
			offset=[-10, 0]
			#offsets=[(-65, 0), (-15, 0), (-15, 0)]  # Adjust offset as needed
		)

		# Save the composite figure
		compositor.save(f'{self.figures_dir}/Supplemental5.7_Origins.png')


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
	
	np.random.choice(123)

	observed_es = calculate_enrichment_score(ranked_df, group_label).max()
	
	permuted_scores = []
	for _ in range(n_permutations):
		# Shuffle group labels
		shuffled_df = ranked_df.copy()
		shuffled_df['activation_time'] = np.random.permutation(shuffled_df['activation_time'])
		permuted_es = calculate_enrichment_score(shuffled_df, group_label).max()
		permuted_scores.append(permuted_es)
	
	# P-value: fraction of permutations with score >= observed
	p_value = np.mean(np.array(permuted_scores) >= observed_es)
	return observed_es, p_value, permuted_scores
