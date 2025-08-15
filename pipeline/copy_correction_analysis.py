
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.RealDataReplication import read_n_fr_b, read_no_copy_correction_n_fr_b


from src.config import load_default_expression_configs
config1, config2 = load_default_expression_configs()


class CopyCorrectionAnalysis():
	"""Analysis of copy correction"""

	def __init__(self, output_directory):
		self.output_directory = output_directory
		self.copy_correction_data_directory = f'{output_directory}/chromatin_deconvolution/'
		self.no_copy_correction_data_directory = f'{output_directory}/chromatin_deconvolution_no_copy/'
		windows = pd.read_csv('data/reference_data/sacCer3_genome_10k_windows.csv')
		self.windows = windows


	def initialize_replication_time_colormaps(self):

		import matplotlib as mpl
		norm = mpl.colors.Normalize(vmin=20, vmax=70)
		cmap = plt.cm.inferno_r  # The _r suffix reverses the colormap

		# Create a ScalarMappable object with the colormap
		self.repl_sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
		self.repl_cmap = cmap
		self.repl_norm = norm

	
	def load_means_all_chromosomes(self):

		from src.timer import Timer

		timer = Timer()
		all_cc_b_means = []
		all_no_cc_means = []
		all_repl_times = []
		
		for chrom in range(1, 17):

			print(f"Chromosome {chrom}", end='...')
			self.load_chromosome(chrom=chrom)
			self.load_replication_timing()

			def _add_chrom_to_index(dat, chrom):
				dat.index.name = 'start'
				new_dat = dat.reset_index()
				new_dat['chr'] = chrom
				new_dat = new_dat.set_index(['chr', 'start'])
				return new_dat

			chrom_cc_w_b_means = _add_chrom_to_index(self.chrom_b_copy_corrected_means_df, chrom)
			chrom_no_cc_means = _add_chrom_to_index(self.chrom_no_cc_means_df, chrom)
			chrom_repl_times = _add_chrom_to_index(self.chrom_replication_times, chrom)

			all_cc_b_means.append(chrom_cc_w_b_means)
			all_no_cc_means.append(chrom_no_cc_means)
			all_repl_times.append(chrom_repl_times)
			
			timer.print_time()

		(self.all_cell_cycle_b_means, self.all_no_cell_cycle_means,
		 self.all_replication_times) = (pd.concat(all_cc_b_means), 
			pd.concat(all_no_cc_means), pd.concat(all_repl_times))

	def load_chromosome(self, chrom):

		self.chrom = chrom
		chr1_cc_data_directory = f'{self.copy_correction_data_directory}/deconvolution_data/chr{chrom}/'
		chr1_no_cc_data_directory = f'{self.no_copy_correction_data_directory}/deconvolution_data/chr{chrom}/'

		windows = self.windows
		windows = windows[windows.chr == chrom]
		n = len(windows)

		F_data_cc = np.zeros((n, 257, 26000))
		F_data_no_cc = np.zeros((n, 257, 26000))

		i = 0
		num_skip = 0

		# todo: skipping last index which is not a whole 10kb
		for idx, row in windows[:-1].iterrows():

			mnase_span = row.start, row.end+1

			no_cc_path = f"{chr1_no_cc_data_directory}/chr{row.chr}_{mnase_span[0]}_{mnase_span[1]}_F.npy"
			cc_path = f"{chr1_cc_data_directory}/chr{row.chr}_{mnase_span[0]}_{mnase_span[1]}_F.npy"
			
			try:
				data_no_cc = np.load(no_cc_path)
				data_cc = np.load(cc_path)
			except ValueError:
				num_skip += 1
				i += 1
				continue
			except FileNotFoundError as e:
				num_skip += 1
				i += 1
				continue
			
			# Skip the last row, for ease of computation
			if i == len(windows)-1: break
				
			F_data_cc[i] = data_cc
			F_data_no_cc[i] = data_no_cc
			i += 1

		self.chrom_windows = windows
		self.chrom_cc_means = F_data_cc.mean(axis=2)
		self.chrom_no_cc_means = F_data_no_cc.mean(axis=2)

		starts = self.chrom_windows['start']
		self.chrom_starts = starts

		self.load_replication_timing()
		self.chrom_no_cc_means_df = pd.DataFrame(self.chrom_no_cc_means, index=starts)
		self.chrom_cc_means_df = pd.DataFrame(self.chrom_cc_means, index=starts)

		# Correct with the b term from the replication deconvolution
		# (baseline occupancy correction scalar)
		self.chrom_b_copy_corrected_means = self.chrom_cc_means_df.values * \
			self.chrom_bs_df.values[:, None]

		self.chrom_b_copy_corrected_means_df = pd.DataFrame(self.chrom_b_copy_corrected_means, index=starts)

		# To avoid memory issues, we'll delete the data after it has been loaded
		del F_data_cc
		del F_data_no_cc

		print(f"Skipped {num_skip} entries")


	def plot_heatmap_comparison(self, plot_chrom):
		from src.config import load_default_expression_configs

		config1, config2 = load_default_expression_configs()

		branch_indices = config1.t_indices()
			
		fig = plt.figure(figsize=(13, 9))

		nrows = 4

		plot_chrom_cc_means_df = self.all_cell_cycle_b_means.loc[plot_chrom].values
		plot_chrom_nocc_means_df = self.all_no_cell_cycle_means.loc[plot_chrom].values

		replication_timing = self.all_replication_times.loc[plot_chrom]

		# Load the full replication timing for reference
		from src.RealDataReplication import load_replication_Fr_df
		Fr_df, replication_indices = load_replication_Fr_df(self.output_directory, plot_chrom)
		rep1_timing = config1.timepoints_df.set_index('Hpos').loc[replication_indices]
		rep2_timing = config2.timepoints_df.set_index('Hpos').loc[replication_indices]
		mean_replication_timing = ((rep1_timing + rep2_timing)/2).mean(1)

		plt.subplot(nrows, 1, 1)
		plt.imshow(plot_chrom_cc_means_df.T[\
			branch_indices], 
			vmin=0, vmax=2, cmap='RdBu_r', aspect='auto', interpolation='none')
		plt.xticks([])
		plt.title("No copy correction", fontsize=12)

		plt.subplot(nrows, 1, 2)
		plt.imshow(plot_chrom_nocc_means_df.T[\
			branch_indices], 
			vmin=0, vmax=2, cmap='RdBu_r', aspect='auto', interpolation='none')
		plt.xticks([])
		plt.title("With copy correction", fontsize=12)

		plt.subplot(nrows, 1, 3)
		plt.imshow((plot_chrom_cc_means_df-\
						plot_chrom_nocc_means_df).T[\
			branch_indices], 
			vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto', interpolation='none')
		plt.xticks([])
		plt.title("Copy correction - No copy correction", fontsize=12)

		plt.subplot(nrows, 1, 4)
		plt.plot(replication_indices.index, mean_replication_timing)
		plt.ylim(68, 26)
		plt.xlim(0, replication_indices.index[-1])
		plt.title("Replication time", fontsize=12)

		plt.suptitle(f"Copy correction affect on chromatin deconvolution, chrom{plot_chrom} 10kb means",
			fontsize=16)

		plt.tight_layout()

	def select_example_windows(self):
		copy_correction_ptr_comparison_t = self.ptrs_df[\
			['cc_ptr_t', 'no_cc_ptr_t']]
		repl_ptrs_df = copy_correction_ptr_comparison_t.join(
			self.all_replication_times)
		sorted_repl_ptrs = repl_ptrs_df.sort_values('replication_time')

		# Example indices on chromosome 4
		self.sample_plot_windows = [

			# Early and efficient example ARS1635, from locus figure, decreases in PTR
			(16, 770000),
			
			# Median replication timing, Unchanging PTR:
			(8, 150000),

			# Late replication region, decreases in PTR
			(16, 730000), # Distal to the ARS1635 locus

			# Example where PTR increases
			# (13, 610000),
		]

		# Name the windows for lookup on the PTR Plot
		self.window_names = [
		"(1) Early, decreased PTR",
		"(2) Mid, unchanged PTR",
		"(3) Late, decreased PTR"]#,
		# "(4) Increased PTR"]

	def plot_sample_curves(self, figsize=(11, 2.5),
		override_color=None):
		copy_correction_ptr_comparison_t = self.ptrs_df[\
			['cc_ptr_t', 'no_cc_ptr_t']]
		repl_ptrs_df = copy_correction_ptr_comparison_t.join(
			self.all_replication_times)
		sorted_repl_ptrs = repl_ptrs_df.sort_values('replication_time')

		i_indices = config1.i_indices()
		t_indices = config1.t_indices()
		b_indices = config1.b_indices()

		from src.config import get_average_timepoints_for_branch
		from src.plot_helpers import create_subplot_pairs, add_pair_title

		i_tps = get_average_timepoints_for_branch(config1, config2, 'i')
		t_tps = get_average_timepoints_for_branch(config1, config2, 't')
		b_tps = get_average_timepoints_for_branch(config1, config2, 'b')
		tb_tps = (t_tps + b_tps)/2.

		def plot_window(axs_pair, window_idx, show_xticks, show_yticks, color,
			plot_ylabel):
			if override_color is not None:
				color = override_color

			def retrieve_branch_vals(dat, window_idx, indices):
				retrieved_data = dat.loc[window_idx][indices].values
				return retrieved_data

			def plot_comparison_branch(ax, branch, tps, no_copy_values, copy_values):

				from src.config import retrieve_phase_ticks

				import matplotlib.patheffects as path_effects

				ax.plot(tps, no_copy_values,
									color='#aaaaaa',
									lw=2,# ls=(0, (1, 1)),
									label="Uncorrected")
				line = ax.plot(tps, copy_values,
						color=color,
						lw=3, label="Copy corrected")[0]
				line.set_path_effects([
					path_effects.Stroke(linewidth=4, foreground='black'),  # Border
					path_effects.Normal()  # Original line on top
				])

				ax.set_ylim(0, 1.5)
				ax.set_xlim(tps[0], tps[-1])

				xticks = retrieve_phase_ticks(branch, config1, config2, with_labels=True)
				phase_ticks, edge_ticks, xtick_labels = xticks

				ax.set_xticks(phase_ticks)
				ax.set_xticklabels(xtick_labels, fontsize=8)
				ax.set_xticks(edge_ticks, minor=True)

				ax.tick_params(axis='x', which='major', length=0)
				ax.tick_params(axis='x', which='minor', length=10) 

			# No copy correction values for recovery
			i_no_copy_correction_values = retrieve_branch_vals(self.all_no_cell_cycle_means, 
				window_idx, i_indices)

			# Copy corrected values (average of the mother and daughter branches)
			i_copy_correction_values = retrieve_branch_vals(self.all_cell_cycle_b_means, 
				window_idx, i_indices)

			# No copy correction values for the window (average the top and bottom branches)
			tb_no_copy_correction_values = (retrieve_branch_vals(self.all_no_cell_cycle_means, 
				window_idx, t_indices) + retrieve_branch_vals(self.all_no_cell_cycle_means, 
				window_idx, b_indices))/2.

			# Copy corrected values (average of the mother and daughter branches)
			tb_copy_correction_values = (retrieve_branch_vals(self.all_cell_cycle_b_means, 
				window_idx, t_indices) + retrieve_branch_vals(self.all_cell_cycle_b_means, 
				window_idx, b_indices))/2.

			plot_comparison_branch(axs_pair[0], 'i', i_tps, 
				i_no_copy_correction_values, i_copy_correction_values)

			plot_comparison_branch(axs_pair[1], 'tb', tb_tps, 
				tb_no_copy_correction_values, tb_copy_correction_values)
			
			for ax in axs_pair:
				if not show_xticks: 
					ax.set_xticks([])
				else:
					ax.set_xlabel("")

			# No yticks for second pair ever
			axs_pair[1].set_yticks([])

			# No yticks unless first column
			if not show_yticks:
				axs_pair[0].set_yticks([])

			if plot_ylabel:
				axs_pair[0].set_ylabel("Window occupancy")
		
		def plot_windows_row(fig, axs_pairs, windows, show_xticks,
			window_names):

			from src.config import load_mean_dg1_mg1_length
			g1_len = load_mean_dg1_mg1_length()

			"""Plot a row of windows (either all early or all late)"""
			for col, window in enumerate(windows):
				repl_time = self.all_replication_times.loc[window].replication_time+g1_len

				window_name = window_names[col]
				color = self.lookup_color_for_repl_time(repl_time)
				
				show_yticks = (col % 2 == 0)  # Only show y-ticks on leftmost column
				
				axs_pair = axs_pairs[col]

				plot_window(axs_pair, window, show_xticks=show_xticks, 
						   show_yticks=(col == 0), color=color, 
						   plot_ylabel=(col == 0))
				
				start = window[1]
				end = start + 10000

				title = f"{window_name}\n(chr{window[0]}: {start//1000}k-{end//1000}k)"
				add_pair_title(fig, axs_pair, title, fontweight='demi', fontsize=12)
				
				# Add legend to first column
				axs_pairs[col][1].legend(loc='lower right', ncols=1)
		
		sample_windows = self.sample_plot_windows

		fig, axs_pairs = create_subplot_pairs(pair_rows=1, pair_cols=3, 
			pair_spacing=0.15, hspacing=0.5, figsize=figsize)
		
		# Plot late windows row
		window_names = self.window_names
		plot_windows_row(fig, axs_pairs, sample_windows, show_xticks=True, 
			window_names=window_names)

		# plt.suptitle("Selected window occupancy changes", fontsize=24, fontweight='demi')


	def load_replication_timing(self):

		from src.RealDataReplication import load_replication_Fr_df, load_B_df

		def _retrieve_closest_replication_bin(source_starts, target_starts):
			"""
			Convert the target 10kb start positions to the replication start positions.
			
			Source starts are the full replication window start values that the profile 
			estimations were computed over. These are 10kb windows 2kb apart
			
			The target starts are the 10kb window delineations from the chromatin deconvolution.
			These are 10kb windows 10kb apart.
			"""
			from src.mnase_10kb_loader import get_bin_for_position

			# Get the midpoint of the target start positions
			mid_10ks = target_starts + 5000 # fixed 10kb windows/2
			closest_source_starts = [get_bin_for_position(row, source_starts)[1] for row in mid_10ks]
			return closest_source_starts

		# Retrieve the 10kb/2kb replication timings and b values
		Fr_df, replication_indices = load_replication_Fr_df(self.output_directory, self.chrom)
		B, b_df = load_B_df(self.output_directory, self.chrom, Fr_df.columns)

		# The copy correction is performed on the 10kb windows, thus convert to 
		# a lower resolution set of start positions

		# Retrieve the closest bin starts to the start indices from the deconvolution
		closest_starts = _retrieve_closest_replication_bin(b_df.index.values, 
			self.chrom_starts)

		selected_replication_indices = replication_indices.loc[closest_starts].values

		# todo: Switch to: retrieve_replication_timing
		rep1_timing = config1.timepoints_df.set_index('Hpos').loc[selected_replication_indices]
		rep2_timing = config2.timepoints_df.set_index('Hpos').loc[selected_replication_indices]
		mean_replication_timing = ((rep1_timing + rep2_timing)/2).mean(1) # mean of two replicates and the start and end

		self.chrom_replication_times = pd.DataFrame({
			'replication_index': selected_replication_indices,
			'replication_time': mean_replication_timing.values
		}, index=self.chrom_starts.values)


		self.chrom_bs_df = b_df.loc[closest_starts]
		self.full_replication_indices = replication_indices


	def compute_ptrs(self):

		from src.peak_to_trough import compute_quantile_ptr_2d

		cc_means = self.all_cell_cycle_b_means.values
		no_cc_means = self.all_no_cell_cycle_means.values

		cc_ptrs = compute_quantile_ptr_2d(cc_means)
		no_cc_ptrs = compute_quantile_ptr_2d(no_cc_means)

		i_indices = config1.i_indices()
		t_indices = config1.t_indices()
		b_indices = config1.b_indices()

		t_cc_ptrs = compute_quantile_ptr_2d(cc_means[:, t_indices])
		t_no_cc_ptrs = compute_quantile_ptr_2d(no_cc_means[:, t_indices])

		b_cc_ptrs = compute_quantile_ptr_2d(cc_means[:, b_indices])
		b_no_cc_ptrs = compute_quantile_ptr_2d(no_cc_means[:, b_indices])

		i_cc_ptrs = compute_quantile_ptr_2d(cc_means[:, i_indices])
		i_no_cc_ptrs = compute_quantile_ptr_2d(no_cc_means[:, i_indices])

		self.ptrs_df = pd.DataFrame({
			'cc_ptr_all': cc_ptrs,
			'no_cc_ptr_all': no_cc_ptrs,

			'cc_ptr_t': t_cc_ptrs,
			'no_cc_ptr_t': t_no_cc_ptrs,

			'cc_ptr_b': b_cc_ptrs,
			'no_cc_ptr_b': b_no_cc_ptrs,

			'cc_ptr_i': i_cc_ptrs,
			'no_cc_ptr_i': i_no_cc_ptrs,
		}, index=self.all_cell_cycle_b_means.index)

	def plot_genomic_location_ptr_changes(self):
		# Next let's examine the set of windows for which the PTRs increased. 
		# Is there anything distinctive about these windows compared to the rest of the genome?


		def _identify_ptr_clusters(positions, min_cluster_size=3, max_gap=10000):
			"""
			Identify clusters of increased PTR windows along a chromosome.
			
			Parameters:
			- positions: sorted array of genomic positions
			- min_cluster_size: minimum number of windows to form a cluster
			- max_gap: maximum distance between adjacent windows in a cluster
			
			Returns:
			- list of tuples: [(cluster_start, cluster_end), ...]
			"""
			if len(positions) < min_cluster_size:
				return []
			
			clusters = []
			current_cluster_start = positions[0]
			current_cluster_positions = [positions[0]]
			
			for i in range(1, len(positions)):
				gap = positions[i] - positions[i-1]
				
				if gap <= max_gap:
					# Continue current cluster
					current_cluster_positions.append(positions[i])
				else:
					# End current cluster if it meets size criteria
					if len(current_cluster_positions) >= min_cluster_size:
						cluster_end = current_cluster_positions[-1]
						clusters.append((current_cluster_start, cluster_end))
					
					# Start new cluster
					current_cluster_start = positions[i]
					current_cluster_positions = [positions[i]]
			
			# Handle final cluster
			if len(current_cluster_positions) >= min_cluster_size:
				cluster_end = current_cluster_positions[-1]
				clusters.append((current_cluster_start, cluster_end))
			
			return clusters

		all_ptrs_df = self.ptrs_df[['no_cc_ptr_all', 'cc_ptr_all']].copy()
		all_ptrs_df['difference'] = all_ptrs_df['cc_ptr_all'] / all_ptrs_df['no_cc_ptr_all']

		ptr_increase_rows = all_ptrs_df[all_ptrs_df.cc_ptr_all > all_ptrs_df.no_cc_ptr_all].copy()
		ptr_decrease_rows = all_ptrs_df[all_ptrs_df.cc_ptr_all <= all_ptrs_df.no_cc_ptr_all].copy()

		print(f"{len(ptr_increase_rows)} of {len(all_ptrs_df)} windows increased in PTR")
		print(f"{len(ptr_decrease_rows)} of {len(all_ptrs_df)} windows decreased in PTR")

		# Top 200
		#ptr_increase_rows = ptr_increase_rows.sort_values('difference', ascending=False).head(200)

		# Bottom 200
		#ptr_decrease_rows = ptr_decrease_rows.sort_values('difference', ascending=False).tail(200)

		from src.sgd import load_aux_annotations

		aux = load_aux_annotations()
		centromeres = aux[aux.cat == 'centromere'].set_index('chr')

		from src.origins import load_origins
		all_origins = load_origins()

		plt.figure(figsize=(9, 6))
		for chrom in range(1, 17):
			from src.sgd import get_chromosome_length
			
			chrom_len = get_chromosome_length(chrom)
			plt.plot([0, chrom_len], [chrom, chrom], c='black', zorder=0)

			centromere = centromeres.loc[chrom]
			centromere_mid = (centromere.start+centromere.stop)/2.
			plt.scatter(centromere_mid, chrom, 
				lw=4, c='black', marker='|', label=("Centromere" if chrom == 1 else None))
			
			# label = "Decreased PTR" if chrom == 1 else None
			# try:
			# 	selected_windows = ptr_decrease_rows.loc[chrom].index
			# 	plt.scatter(selected_windows, np.repeat(chrom, len(selected_windows)),
			# 			   color=plt.cm.Greys(0.5), alpha=0.25, label=label, edgecolor='none')

			# except KeyError:
			# 	continue

			label = "Increased PTR" if chrom == 1 else None
			try:
				selected_windows = ptr_increase_rows.loc[chrom].index

				# Identify and highlight clusters
				clusters = _identify_ptr_clusters(
					sorted(selected_windows)
				)

				plt.scatter(selected_windows, np.repeat(chrom, len(selected_windows)),
					   color=plt.cm.Greens(0.65), alpha=0.5, label=label, lw=0)

				# Draw rectangles around clusters
				from src.plot_helpers import plot_rect2
				ax = plt.gca()
				for cluster_start, cluster_end in clusters:
					plot_rect2(
						ax, 
						x1=cluster_start-6000, y1=chrom-0.3,  # Slight vertical offset
						x2=cluster_end+6000, y2=chrom+0.3,
						facecolor='none', alpha=0.75,
						edgecolor=plt.cm.Greens(1.0), lw=0.5,
						zorder=30)
					
			except KeyError:
				continue

			chrom_origins = all_origins[(all_origins.chr == chrom) & 
				(all_origins.activation_time == 'early')].copy()
			alpha_value = 0.75
			plt.scatter(chrom_origins.pos, np.repeat(chrom, len(chrom_origins)), alpha=alpha_value,
				color='red', s=10, marker='D', label='Early origin' if chrom == 1 else None,
				lw=0)

		plt.yticks(range(1, 17))
		plt.suptitle("Regions with increased PTR following copy correction", 
			fontweight='demi', fontsize=21)
		plt.legend()
		plt.xlabel("Genomic position, bp")
		plt.ylabel("Chromosome")
		plt.ylim(17, 0)


	def plot_ptrs(self, key_1='no_cc_ptr_all', key_2='cc_ptr_all'):
		ptrs1 = self.ptrs_df[key_1]
		ptrs2 = self.ptrs_df[key_2]

		plot_replication_times_df = self.all_replication_times

		fig = plt.figure(figsize=(5.25, 4.5))

		# Offset for start of S phase
		from src.config import load_mean_dg1_mg1_length
		mean_g1_len = load_mean_dg1_mg1_length()
		replication_times = plot_replication_times_df.replication_time+mean_g1_len

		plot_df = pd.DataFrame({
			key_1: ptrs1,
			key_2: ptrs2,
			'replication_time': replication_times
		}, index=self.ptrs_df.index)

		np.random.seed(123)
		random_index = np.random.permutation(plot_df.index)
		plot_df = plot_df.loc[random_index]

		plt.plot([0, 2], [0, 2], c='black', lw=0.75, ls='dotted')
		plt.scatter(plot_df[key_1], plot_df[key_2], s=10, alpha=1, 
					edgecolor='gray', facecolor='none')
		plt.scatter(plot_df[key_1], plot_df[key_2], s=8, alpha=1, 
					c=plot_df.replication_time,
					cmap=self.repl_cmap, norm=self.repl_norm)
		plt.xlim(1, 1.4)
		plt.ylim(1, 1.4)
		plt.xlabel("No correction, PTRs")
		plt.ylabel("Copy corrected, PTRs")

		# Create the colorbar
		cbar = plt.colorbar()
		cbar.set_label('Replication time, min', rotation=270, va='bottom')

		import matplotlib.patheffects as patheffects

		offset_amount = 0.003
		default_text_x_offset = -offset_amount
		default_text_y_offset = offset_amount

		default_ha = 'right'
		default_va = 'bottom'

		# Map the window names to any custom formatting
		custom_formatting = {
			1: {
				'ha': 'left',
				'va': 'top',
				'offset': (offset_amount, -offset_amount)
			},
			2: {
				'ha': 'left',
				'va': 'top',
				'offset': (offset_amount, -offset_amount)
			},
		}

		for i, window in enumerate(self.sample_plot_windows):

			name = self.window_names[i]

			# Retrieve numeric name
			numeric_name = int(name.split(" ")[0].replace('(', '').replace(')', ''))

			plot_row = plot_df.loc[window]
			replication_time = plot_replication_times_df.loc[window].replication_time
			#color = #self.lookup_color_for_repl_time(replication_time)

			#from src.plot_helpers import adjust_lightness_saturation
			#adjusted_color = adjust_lightness_saturation(color, 0.4, 1.0)
			color = 'black'

			plt.scatter(plot_row[key_1], plot_row[key_2], s=100, edgecolor=color,
				facecolor='none', lw=2, marker='D')

			if numeric_name in custom_formatting:
				row_formatting = custom_formatting[numeric_name]
				text_x_offset, text_y_offset = row_formatting['offset']
				ha, va = row_formatting['ha'], row_formatting['va']
			else:
				ha, va, text_x_offset, text_y_offset = default_ha, default_va, \
					default_text_x_offset, default_text_y_offset

			x, y = plot_row[key_1]+text_x_offset, plot_row[key_2]+text_y_offset

			plt.text(x, y, numeric_name,
				fontsize=14,
				color=color,
				ha=ha, va=va,
				path_effects=[patheffects.withStroke(linewidth=3, foreground='white')])

		key_1_mean = np.mean(plot_df[key_1])
		key_2_mean = np.mean(plot_df[key_2])

		plt.suptitle("PTR change following copy correction", fontsize=18, fontweight='demi')
		plt.tight_layout()

	def plot_branch_ptrs(self):

		plt.figure(figsize=(9, 8))
		plt.subplot(2, 2, 1)
		self.plot_ptrs('no_cc_ptr_all', 'cc_ptr_all')
		plt.title("All branches")

		plt.subplot(2, 2, 2)
		self.plot_ptrs('no_cc_ptr_i', 'cc_ptr_i')
		plt.title("Recovery")

		plt.subplot(2, 2, 3)
		self.plot_ptrs('no_cc_ptr_t', 'cc_ptr_t')
		plt.title("Top")

		plt.subplot(2, 2, 4)
		self.plot_ptrs('no_cc_ptr_b', 'cc_ptr_b')
		plt.title("Bottom")

		plt.suptitle(f"PTR change from copy correction", fontsize=16, y=0.97)
		plt.tight_layout()


	def lookup_color_for_repl_time(self, value, alpha=1.0):
		normalized_value = self.repl_norm(value)
		rgba_color = self.repl_cmap(normalized_value)
		rgba_color = rgba_color[0], rgba_color[1], rgba_color[2], alpha
		return rgba_color
