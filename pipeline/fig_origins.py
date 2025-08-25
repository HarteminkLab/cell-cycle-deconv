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

	def plot_origin_timing(self):

		def _create_attribute(selection, false_value, true_value):
		    attribute_values = np.array([false_value] * len(selection)).astype('object')
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

		plt.figure(figsize=(6, 4))
		plt.scatter(early_origins.replication_time, 
		          early_origins.derived_origin_efficiency_from_mcguffee_et_al_2013,
		          s=12, edgecolor=early_color, facecolor=early_facecolors,
		          label=f"Early firing, n={len(early_origins)}, {early_origins.inferred_firing.sum()}", 
		            lw=0.3)

		plt.scatter(late_origins.replication_time, 
		          late_origins.derived_origin_efficiency_from_mcguffee_et_al_2013,
		          s=11, edgecolor=late_color,
		          facecolor=late_facecolors, 
		            marker='D',
		          label=f"Late firing, n={len(late_origins)}, {late_origins.inferred_firing.sum()}", 
		            lw=0.3)

		num_local_max = origin_timings.inferred_firing.sum()

		plt.xlabel("Replication time, min")
		plt.ylabel("Origin efficiency")
		plt.title(f"Origin replication timing\nn={len(origin_timings)}, "
          f"{num_local_max} inferred firing",
         fontweight='demi', fontsize=16, pad=12)

		from matplotlib.lines import Line2D

		legend_elements_2 = [
		    Line2D([0], [0], marker='o', color=early_color, linewidth=0,
		           markersize=2, label=f'Early firing, n={len(early_origins)}, '
		           f'{early_origins.inferred_firing.sum()}'),
		    Line2D([0], [0], marker='D', color=late_color, linewidth=0,
		           markersize=2, label=f'Late firing, n={len(late_origins)}, '
		           f'{late_origins.inferred_firing.sum()}'),
		]
		plt.legend(handles=legend_elements_2)

		plt.ylim(-0.05, 0.925)
