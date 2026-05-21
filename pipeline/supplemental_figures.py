import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests
from src.utils import mkdir_safe
from src.figure_configs import save_figure_for_paper

from src.nucleosome_histone_dataset import HistonesNucleosomesDataset
from src.nucleosome_metrics_processor import NucleosomeDataLoader
from src.peak_to_trough import compute_quantile_ptr_2d


class FigureSupplemental:
	"""
	A class to layout supplemental panels that are not tied to any existing 
	figure panel/story.
	"""
	
	def __init__(self, output_dir="output/draft4_run/"):
		"""
		Initialize the analyzer with configuration parameters.
		"""
		# Configuration parameters
		self.output_dir = output_dir
		self.save_dir = f"{self.output_dir}/supplemental_various"
		self.figures_dir = f'{self.output_dir}/Figures'
		mkdir_safe(self.save_dir)

	def plot_promoter_intergenic_distribution(self):
		# Create a histogram of the distance between consecutive genes to determine promoter 
		# lengths. We're checking to make sure we aren't overlapping too much with our promoter
		# size selection

		from src.transcripts_dataset import load_transcripts_sets

		genes, _ = load_transcripts_sets('output/draft4_run/')

		import pandas as pd
		import matplotlib.pyplot as plt
		import numpy as np

		def compute_upstream_distances(df):
			"""
			Calculate intergenic distance upstream of each gene's TSS.
			
			Parameters:
			-----------
			df : pandas.DataFrame
				DataFrame with columns: chr, TSS, strand, full_transcript_start, full_transcript_end
			
			Returns:
			--------
			pandas.DataFrame
				Original DataFrame with added 'upstream_distance' column
			"""
			
			# Initialize output column
			df = df.copy()
			df['upstream_distance'] = np.nan
			
			# Process each chromosome separately
			for chromosome in df['chr'].unique():
				
				# Get all genes on this chromosome
				chr_genes = df[df['chr'] == chromosome].copy()
				
				# Sort by genomic position (use full_transcript_start for sorting)
				chr_genes = chr_genes.sort_values('full_transcript_start')
				
				# For each gene on this chromosome
				for idx, gene in chr_genes.iterrows():
					
					if gene['strand'] == '+':
						# Plus strand: find nearest gene that ENDS before this TSS
						upstream_genes = chr_genes[chr_genes['full_transcript_end'] < gene['TSS']]
						
						if len(upstream_genes) > 0:
							nearest_end = upstream_genes['full_transcript_end'].max()
							upstream_distance = gene['TSS'] - nearest_end
						else:
							upstream_distance = np.nan  # No upstream gene (chromosome start)
							
					elif gene['strand'] == '-':
						# Minus strand: find nearest gene that STARTS after this TSS
						upstream_genes = chr_genes[chr_genes['full_transcript_start'] > gene['TSS']]
						
						if len(upstream_genes) > 0:
							nearest_start = upstream_genes['full_transcript_start'].min()
							upstream_distance = nearest_start - gene['TSS']
						else:
							upstream_distance = np.nan  # No upstream gene (chromosome end)
					
					else:
						# Handle any unexpected strand values
						upstream_distance = np.nan
					
					# Store result in original dataframe
					df.at[idx, 'upstream_distance'] = upstream_distance
			
			return df

		upstream_distances_df = compute_upstream_distances(genes)
		plt.figure(figsize=(8, 4))
		plt.hist(upstream_distances_df.upstream_distance, bins=np.linspace(0, 2000, 100),
				color=plt.cm.Blues(0.5))
		plt.axvline(200, c='red', lw=1, label="200 bp promoter length selection")
		plt.title("Length distribution of intergenic regions upstream of TSSs", 
			fontweight='demi', fontsize=16, pad=10)
		plt.xlabel("Genomic length, bp")
		plt.ylabel("Number of genes")

		for dist_cutoff in [200, 250, 300]:
			num_less_num = len(upstream_distances_df.upstream_distance[
				upstream_distances_df.upstream_distance < dist_cutoff])
			print(f"There are {num_less_num} genes with promoters less than {dist_cutoff} bp long")
			if dist_cutoff == 200:
				num_less_200 = num_less_num

		plt.xlim(0, 1500)
		plt.ylim(0, 350)

		n = len(upstream_distances_df)
		plt.text(105, 250, f"{num_less_200} genes\n{num_less_200/n*100:.0f}%", c='red',
				ha='center')
		plt.text(700, 250, f"{n-num_less_200} genes\n{(n-num_less_200)/n*100:.0f}%", c='black',
				ha='center')
		plt.text(210, 340, f"200 bp promoter length selection", c='red',
				ha='left', va='top')

		save_figure_for_paper(f'{self.save_dir}/intergenic_spacing_distribution.png')


	def plot_facs_path(self, tsv_path, ax, replicate):
		import matplotlib.colors as mcolors
		
		# ── Load data ──────────────────────────────────────────────────────────────────
		df = pd.read_csv(tsv_path, sep="\t")

		# First column = Y-axis bin edges; remaining columns = X-axis log10 bin labels
		y_bins = df.iloc[:, 0].values.astype(float)          # e.g. 0, 10, 20, … 150
		x_bins = df.columns[1:].astype(float)                # log10 values 0 → 10
		Z      = df.iloc[:, 1:].values.astype(float)         # shape (n_y_bins, n_x_bins)
		Z = Z+1e-9

		# ── Convert log10 column labels → linear scale for the x-axis ─────────────────
		x_linear = x_bins                              # e.g. 1 → 10^10

		# pcolormesh needs (n_y+1, n_x+1) edge arrays for proper cell boundaries.
		# Build X edges from the linear values; Y edges from the bin values.
		x_edges = np.concatenate([[x_linear[0] * 0.9],
								   np.sqrt(x_linear[:-1] * x_linear[1:]),   # geometric midpoints
								   [x_linear[-1] * 1.1]])
		y_step  = y_bins[1] - y_bins[0] if len(y_bins) > 1 else 10
		y_edges = np.append(y_bins - y_step / 2, y_bins[-1] + y_step / 2)

		mesh = ax.pcolormesh(
			x_edges, y_edges, Z,
			cmap="viridis",
			norm=mcolors.Normalize(vmin=Z.min()+1e-9, vmax=Z.max()*0.75),
			shading="flat",
		)

		ax.set_title(f"Heatmap of the flow cytometry data, replicate {replicate}", fontsize=14,
		 fontweight="demi", pad=9)
		ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
		ax.set_xlim(6.5, 10)
		ax.set_yticks(y_bins)
		ax.set_xlabel("log-fluorescence")
		ax.set_ylabel("Experimental time, min")

	def plot_facs(self):
		# ── Figure ─────────────────────────────────────────────────────────────────────
		fig, ax = plt.subplots(1, 1, figsize=(6, 4))

		tsv1_path = 'data/facs/cell_cycle/replicate_1_cleaned_for_cloccs.tsv'
		tsv2_path = 'data/facs/cell_cycle/replicate_1_cleaned_for_cloccs.tsv'

		self.plot_facs_path(tsv1_path, ax, 1)
		save_figure_for_paper(f"{self.save_dir}/facs_replicate1.png")

		fig, ax = plt.subplots(1, 1, figsize=(6, 4))
		self.plot_facs_path(tsv2_path, ax, 2)
		save_figure_for_paper(f"{self.save_dir}/facs_replicate2.png")


	def layout_supplemental_flow_cytometry(self):

		from pipeline.figure_composer import FigureCompositor
		from pipeline.figure_composer_helpers import layout_images_vertically, \
			add_panel_labels_to_images

		# Create compositor with wider dimensions for horizontal layout
		compositor = FigureCompositor(1024, 1460, debug_mode=True)

		image_paths = [
			f"{self.save_dir}/facs_replicate1.png",
			f"{self.save_dir}/facs_replicate2.png"
		]

		placed_images = layout_images_vertically(
			compositor,
			image_paths,
			between_padding=30,
			margin=(30, 30),
			image_keys=['hm1', 'hm2']  # Custom keys
		)

		# Add panel labels
		add_panel_labels_to_images(
			compositor, 
			compositor.placed_images,
			offset=(-10, 46),
			font_size=48,
		)

		# Save the composite figure
		compositor.save(f'{self.figures_dir}/Supplemental2_flow_cytometry.png')


	def layout_supplemental_promoters(self):

		from pipeline.figure_composer import FigureCompositor
		from pipeline.figure_composer_helpers import layout_images_horizontally, \
			add_panel_labels_to_images

		# Create compositor with wider dimensions for horizontal layout
		compositor = FigureCompositor(1024, 600, debug_mode=True)

		image_paths = [
			f'{self.save_dir}/intergenic_spacing_distribution.png',
		]

		placed_images = layout_images_horizontally(
			compositor,
			image_paths,
			between_padding=30,
			margin=(30, 30),
			image_keys=['intergenic']  # Custom keys
		)

		# Save the composite figure
		compositor.save(f'{self.figures_dir}/Supplemental4.5_Intergenic_distribution.png')

	def layout_supplemental_cloccs_fits(self):

		from pipeline.figure_composer import FigureCompositor
		from pipeline.figure_composer_helpers import layout_images_vertically, \
			add_panel_labels_to_images

		# Create compositor with wider dimensions for horizontal layout
		compositor = FigureCompositor(1024, 1250, debug_mode=True)

		image_paths = [
			f'./data/2019_cloccs_fits/yl_2019_replicate1/fit_curves_rep1.png',
			f'./data/2019_cloccs_fits/yl_2019_replicate2/fit_curves_rep2.png',
		]

		placed_images = layout_images_vertically(
			compositor,
			image_paths,
			between_padding=90,
			margin=(30, 60),
			image_keys=['fit1', 'fit2']  # Custom keys
		)

		compositor.add_panel_label_to_image('fit1', 'Replicate 1 CLOCCS fit', offset=(300, -10), 
			font_size=32, font_type='semi_bold')

		compositor.add_panel_label_to_image('fit2', 'Replicate 2 CLOCCS fit', offset=(300, -10), 
			font_size=32, font_type='semi_bold')

		# Add panel labels
		add_panel_labels_to_images(
			compositor, 
			compositor.placed_images,
			offset=(-20, -20),
			font_size=36,
		)

		# Save the composite figure
		compositor.save(f'{self.figures_dir}/Supplemental1_fit_curves.png')

