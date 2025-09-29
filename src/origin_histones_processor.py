import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from src.plot_helpers import create_proportional_subplots_3rows, get_truncated_RdBu_r
from src.histones import HistoneModificationOrganizer, histones_ordering


class OriginHistonesProcessor:
	"""
	A class for plotting histone modifications for origins. Here we'll plot the modifications
	grouped by modification type. This is a simplifed version of the histone_group_plotter 
	used in the nuclesomes analysis.

	"""
	
	def __init__(self, origins, termination_sites):
		"""
		Initialize the plotter with enrichment results data.
		
		Parameters:
		-----------
		enrichment_results : dict
			Dictionary containing 'occupancy', 'entropy', and 'positioning' metrics
			with statistical results for different groups
		subset_n : int
			Number of nucleosomes in the subset for labeling purposes
		"""

		from src.nucleosome_histone_dataset import HistonesNucleosomesDataset
		histones_nucleosomes_dataset = HistonesNucleosomesDataset()
		histones_nucleosomes_dataset.load_all()

		self.organizer = HistoneModificationOrganizer()
		self.histones_nucleosomes_dataset = histones_nucleosomes_dataset
		self.origins = origins
		self.termination_sites = termination_sites

	def retrieve_histone_modifications_per_origin(self, origins):
		#self.diff_df = self.retrieve_histone_modifications_per_site(self.origins[\
		#		self.origins.inferred_firing], 'pos')

		self.diff_df = self.retrieve_histone_modifications_per_site(
			origins, 
			'pos')


	def retrieve_histone_modifications_per_site(self, origins, position_key):

		histones_nucleosomes_dataset = self.histones_nucleosomes_dataset
		weiner_nucs = histones_nucleosomes_dataset.weiner_nucleosomes
		average_histone_modifications = histones_nucleosomes_dataset.weiner_histones.mean(0)
		histone_mods = histones_nucleosomes_dataset.weiner_histones

		def _select_weiner_nucs(weiner_nucs, origin, offset=0, window=2000):
			# Select weiner nuclesosomes for an origin

			win_2 = window//2
			selected_nucs = weiner_nucs[(weiner_nucs.chr == origin.chr) & 
										(weiner_nucs.center > origin[position_key]-win_2+offset) &
										(weiner_nucs.center < origin[position_key]+win_2+offset)]
			return selected_nucs

		def retrieve_origin_mods_per_dist(origin):
			furthest = 32000
			distances_away = np.arange(-furthest, furthest+1000, 1000)

			current_origin_mods_per_dist = []
			for offset in distances_away:
				selected_nucs = _select_weiner_nucs(weiner_nucs, origin, offset=offset)
				selected_nuc_histone_mods = histone_mods.loc[selected_nucs.index].mean(0)
				current_origin_mods_per_dist.append(selected_nuc_histone_mods)

			origin_histone_mods_per_dist = pd.DataFrame(current_origin_mods_per_dist, 
				index=distances_away)
			return origin_histone_mods_per_dist

		# For each of the inferred origin, let's collect modification values
		# at the origin and progressively further from the origin
		from src.timer import Timer
		timer = Timer()
		all_origin_mods = []
		for i, (_, origin) in enumerate(origins.iterrows()):
			origin_mods_per_dist = retrieve_origin_mods_per_dist(origin)
			all_origin_mods.append(origin_mods_per_dist)
		if i % 50 == 0: timer.print_time(f"{i}/{len(origins)}")

		all_mod_values = [all_origin_mods[i].fillna(0).values for i in range(len(all_origin_mods))]
		mean_mods_all_origins = np.mean(all_mod_values, axis=(0))

		half_index = len(mean_mods_all_origins-1)//2

		from src.origin_entropy_processor import fold_halves_together
		folded_mods = fold_halves_together(mean_mods_all_origins, axis=0)

		diff = folded_mods - average_histone_modifications.values[None, :]

		average_histone_mods_df = pd.DataFrame(average_histone_modifications, 
			columns=['average_value'])
		average_histone_mods_df['original_index'] = np.arange(len(average_histone_mods_df))
		diff_df = pd.DataFrame(diff, columns=average_histone_mods_df.index)

		return diff_df

	def plot_histone_modifications(self):
		from src.plot_helpers import create_proportional_subplots_vertical
		from src.VerticalHistoneModificationGroupPlotter import VerticalHistoneModificationGroupedPlotter

		# Define modification type groupings
		group_keys = ['Acetylation', 'Methylation', 'Phosphorylation', 'Histone Variant']
		
		# Calculate group counts for proportional sizing
		group_counts_mapping = self.organizer.df.groupby('modification_type').count().rename(
			columns={'modification_name': 'count'})['count']
		group_counts = list(group_counts_mapping.loc[group_keys].values)

		fig, axes = create_proportional_subplots_vertical(group_counts,
											 figsize=(3, 7), vertical_padding=0.3)

		organizer = self.organizer
		histone_groupings = organizer.get_grouped_modifications()
		diff_df = self.diff_df

		for i, group_name in enumerate(histone_groupings):

			modifications_by_group = organizer.get_modifications_flattened_by_group_name(group_name)
			current_heatmap_data = diff_df[modifications_by_group].T

			ax = axes[i]
			im = ax.imshow(current_heatmap_data, cmap='RdBu_r', vmin=-0.5, vmax=0.5, aspect='auto', 
					interpolation='none', extent=[-500, 32500, len(current_heatmap_data)-0.5, -0.5],
						  origin='upper')
			ax.set_yticks(range(0, len(current_heatmap_data)),
						 modifications_by_group)
			ax.set_xticks([])
			ax.set_title(group_name)

			if i == len(axes)-1:
				xticks = np.arange(0, 35000, 10000)
				xtick_labels = [f"{x/1000:.0f}kb" for x in xticks]
				ax.set_xticks(xticks)
				ax.set_xticklabels(xtick_labels)
				ax.set_xlabel("Distance from origin")

			ax.set_xlim(-500, 30500)

		plt.suptitle(f"Histone modifications at\nfiring origins, n={len(self.origins[self.origins.inferred_firing])}", 
			fontweight='demi', fontsize=16, y=1.0)