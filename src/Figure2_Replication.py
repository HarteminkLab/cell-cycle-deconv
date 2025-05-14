
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.utils import mkdir_safe

from src.figure_configs import FiguresConfig
from src.figure_configs import save_figure_for_paper
from src.plot_helpers import adjust_lightness_saturation


class Figure2ReplicationDeconvolution():
	"""Create figures for the replication deconvolution"""

	def __init__(self, output_directory):
		self.output_directory = output_directory
		self.save_dir = f'{self.output_directory}/replication_figures'

		# Set the math text parameters to computer modern
		plt.rcParams['mathtext.fontset'] = 'cm'

		mkdir_safe(self.save_dir)

		# Examine the replication data, does replicate 2's cell cycle length need to be longer?
		from src.CombinedReplicateDeconvolutionRunner import CombinedReplicateDeconvolutionRunner
		from src.config import load_default_chrom_configs

		config1, conself = load_default_chrom_configs()
		runner = CombinedReplicateDeconvolutionRunner(chrom=4, save_dir=None, 
			config1=config1, conself=conself)
		self.runner = runner
		self.config1 = config1
		self.conself = conself


	def setup_data(self):
		self.runner.start_runs(num_epochs=1)


	def plot_diagram_replication(self):
		import numpy as np
		import matplotlib.pyplot as plt
		from matplotlib.collections import LineCollection
		from scipy import interpolate
		from src.plot_helpers import adjust_lightness_saturation

		def plot_strand(loc_x, loc_y, color):
			# Create data for the helix
			t = np.linspace(0, 2*np.pi, 100)
			amplitude = 0.15

			strand1_x = t
			strand1_y = amplitude * np.sin(t)
			plt.plot(strand1_x*0.15+loc_x, strand1_y+loc_y, linewidth=2,
					c=color)

		def plot_strand_series(start_x, start_y, xs, ys, repl_index,
							  color):
			for i, x in enumerate(xs):
				y = ys[i]
				plot_strand(start_x+x,  start_y+y, color)
				
				if i >= repl_index:
					plot_strand(start_x+x,  start_y+y+0.5, color)            

		fig = plt.figure(figsize=(19, 2))

		early_repl = 3
		late_repl = 5

		xlims = 0, 18

		plt.subplot(1, 3, 1)
		xs = np.arange(0, 18, 2)
		ys = np.repeat(0, len(xs))

		early_color = plt.cm.Reds(0.65)
		late_color = plt.cm.Blues(0.65)

		early_color = adjust_lightness_saturation(early_color, 1.0, 0.75)

		num_tps = len(xs)
		plot_strand_series(0.5, -1.25, xs, ys, early_repl, early_color)
		plot_strand_series(0.5,  1.25, xs, ys, late_repl, late_color)

		plt.ylim(-3, 3)
		plt.xlim(*xlims)
		plt.yticks([])
		plt.title("Genomic DNA", fontsize=18, pad=10)
		plt.axvspan(6, 12, 0, 1, color='black', alpha=0.05, lw=0)
		plt.xticks([3, 9, 15], ['G1', 'S', 'G2/M'], fontsize=14)
		plt.tick_params(axis='x', length=0, pad=10)
		ax1 = plt.gca()

		# ------------------------

		plt.subplot(1, 3, 2)
		plt.yticks([])

		num_tps = 19
		xs = np.arange(0, num_tps).astype(float)

		early_repl = 6
		late_repl = 12

		xs[early_repl-1] = xs[early_repl]
		xs[late_repl-1] = xs[late_repl]

		# Inset the replication timings a bit
		# visual clarity
		xs[early_repl-1:early_repl+1] = xs[early_repl]+0.5
		xs[late_repl-1:late_repl+1] = xs[late_repl]-0.75

		early_repl_ys = np.ones(num_tps)
		early_repl_ys[early_repl:] = 2

		late_repl_ys = np.ones(num_tps)
		late_repl_ys[late_repl:] = 2

		line1 = plt.plot(xs, early_repl_ys+0.01, label="Early replicating", c=early_color, lw=4)[0]
		line2 = plt.plot(xs, late_repl_ys-0.01, label="Late replicating", c=late_color, lw=4)[0]

		plt.title("Replication profiles", fontsize=18, pad=10)
		plt.xlim(*xlims)
		plt.ylim(0.75, 2.25)
		plt.axvspan(6, 12, 0, 1, color='black', alpha=0.05, lw=0)
		plt.xticks([3, 9, 15], ['G1', 'S', 'G2/M'], fontsize=14)
		plt.tick_params(axis='x', length=0, pad=10)

		fig.legend(
		    [line1, line2],
		    ['Early replicating', 'Late replicating'],
			loc='upper center',
    		bbox_to_anchor=(0.25, -0.0), # Place below the first subplot
		    ncol=2,
		    fontsize=14,
		    frameon=False
		)

		# ------------------------

		plt.subplot(1, 3, 3)

		plt.plot(xs, early_repl_ys)
		plt.plot(xs, late_repl_ys)

		total_repls = early_repl_ys + late_repl_ys
		early_repl_ys = early_repl_ys / total_repls
		late_repl_ys = late_repl_ys / total_repls

		plt.fill_between(xs, early_repl_ys, 0, color=early_color)
		plt.fill_between(xs, early_repl_ys+late_repl_ys, early_repl_ys, color=late_color)

		plt.ylim(0, 1)
		plt.xlim(*xlims)
		plt.yticks([])
		plt.xticks([3, 9, 15], ['G1', 'S', 'G2/M'], fontsize=14)
		plt.tick_params(axis='x', length=0, pad=10)
		plt.axvspan(6, 12, 0, 1, color='white', alpha=0.05, lw=0)
		plt.title("Relative proportion of total DNA", fontsize=18, pad=10)
		plt.plot(xs, early_repl_ys, lw=2, c='black', solid_joinstyle='miter')

		plt.subplots_adjust(hspace=0.4, top=0.85, bottom=0.2)
		plt.suptitle("Effect of replication on normalized DNA counts",
			fontweight='demi', fontsize=29, y=1.37)
		save_figure_for_paper(f"{self.save_dir}/DNA_replication_diagram.png")

	def plot_N_G_Fr_B_components(self):
		fig = self.runner.deconvolution.plot_N_G_Fr_B_diagram()
		save_figure_for_paper(f"{self.save_dir}/Replication_components.png")


	def plot_GNHFrB_diagram(self):
		fig = self.runner.deconvolution.plot_G_N_H_Fr_B_diagram()
		save_figure_for_paper(f"{self.save_dir}/Replication_diagram.png")

	def layout_panel(self):
		from pipeline.figure_composer import FigureCompositor
		from pipeline.figure_composer_helpers import layout_images_vertically,\
		    add_panel_labels_to_images


		compositor = FigureCompositor(1024, 960, debug_mode=True)

		image_paths = [
		    f'{self.save_dir}/DNA_replication_diagram.png',
		    f'{self.save_dir}/Replication_diagram.png',
		    f'{self.save_dir}/Replication_components.png',
		]

		placed_images = layout_images_vertically(
		    compositor,
		    image_paths,
		    height_proportions=[0.35, 0.4, 0.6],
		    between_padding=40,
		    margin=(30, 30),
		    image_keys=['DNA', 'Replication', 'Components']  # Custom keys for the images
		)

		add_panel_labels_to_images(
		    compositor, 
		    compositor.placed_images,
		    font_size=36,
		    offset=(-10, -12)
		)

		compositor.save(f'{self.save_dir}/Figure2_Replication.png')
