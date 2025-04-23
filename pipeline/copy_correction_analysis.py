
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
		norm = mpl.colors.Normalize(vmin=30, vmax=60)
		cmap = plt.cm.RdBu  # The _r suffix reverses the colormap

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

		F_data_cc = np.zeros((n, 149, 26000))
		F_data_no_cc = np.zeros((n, 149, 26000))

		i = 0

		# todo: skipping last index which is not a whole 10kb
		for idx, row in windows[:-1].iterrows():

			mnase_span = row.start, row.end+1
			
			try:
				data_no_cc = np.load(f"{chr1_no_cc_data_directory}/chr{row.chr}_{mnase_span[0]}_{mnase_span[1]}_F.npy")
				data_cc = np.load(f"{chr1_cc_data_directory}/chr{row.chr}_{mnase_span[0]}_{mnase_span[1]}_F.npy")
			except FileNotFoundError:
				print(f"Skipping index {i}")
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
		self.chrom_b_copy_corrected_means = self.chrom_cc_means_df.values * \
			self.chrom_bs_df.values[:, None]
		self.chrom_b_copy_corrected_means_df = pd.DataFrame(self.chrom_b_copy_corrected_means, index=starts)

		# To avoid memory issues, we'll delete the data after it has been loaded
		del F_data_cc
		del F_data_no_cc


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
		self.early_windows = [sorted_repl_ptrs.iloc[i].name for i in [21, 33, 67]]
		self.late_windows = [sorted_repl_ptrs.iloc[i].name for i in [-28, -16, -30]]

	def plot_sample_curves(self, early_windows=None, late_windows=None, figsize=(6, 6),
		override_color=None):
		copy_correction_ptr_comparison_t = self.ptrs_df[\
			['cc_ptr_t', 'no_cc_ptr_t']]
		repl_ptrs_df = copy_correction_ptr_comparison_t.join(
			self.all_replication_times)
		sorted_repl_ptrs = repl_ptrs_df.sort_values('replication_time')

		if early_windows is None:
			early_windows = self.early_windows

		if late_windows is None:
			late_windows = self.late_windows

		indices = config1.t_indices()

		no_cc_mean = self.chrom_no_cc_means_df.mean().mean()
		cc_mean = self.chrom_b_copy_corrected_means_df.mean().mean()

		from src.config import get_average_timepoints_for_branch

		tps = get_average_timepoints_for_branch(config1, config2, 't')

		def _plot_window(ax, window_idx, show_xticks, show_yticks, color):

			if override_color is not None:
				color = override_color

			ax.plot(tps, self.all_no_cell_cycle_means\
						 .loc[window_idx][indices] / no_cc_mean,
					color='#555555',
					lw=2,
					ls=(0, (1, 1)),
					label="No copy correction")
			ax.plot(tps, self.all_cell_cycle_b_means\
						 .loc[window_idx][indices] / cc_mean,
					color=color,
					lw=2,
					label="With copy correction")
			ax.set_ylim(-0.5, 2)
			ax.set_xlim(tps[0], tps[-1])
			ax.axhline(1, c='black', lw=0.25)
			if not show_xticks: 
				ax.set_xticks([])
			else:
				ax.set_xlabel("Average single cell time, minutes")
			if not show_yticks: 
				ax.set_yticks([])
			else:
				ax.set_ylabel("Average occupancy")

		nrows = len(early_windows)

		fig, axs = plt.subplots(nrows, 2, figsize=figsize)

		for row in range(nrows):
			early_window, late_window = early_windows[row], late_windows[row]

			early_repl_time = self.all_replication_times.loc[early_window].replication_time
			early_color = self.lookup_color_for_repl_time(early_repl_time)

			late_repl_time = self.all_replication_times.loc[late_window].replication_time
			late_color = self.lookup_color_for_repl_time(late_repl_time)

			row_ax = axs[row]
			mid = early_window[1]
			_plot_window(row_ax[0], early_window, show_yticks=row==nrows-1, show_xticks=(row==nrows-1),
				color=early_color)

			if nrows < 5:
				row_ax[0].set_title(f"Early {row+1}, chr{early_window[0]}: {mid-5000}-{mid+5000}")

			if row == 0:
				row_ax[0].legend(loc='lower right')

			mid = late_window[1]
			_plot_window(row_ax[1], late_window, show_yticks=False, show_xticks=(row==nrows-1),
				color=late_color)

			if nrows < 5:
				row_ax[1].set_title(f"Late {row+1}: chr{late_window[0]}: {mid-5000}-{mid+5000}")

			if row == 0:
				row_ax[1].legend(loc='lower right')

		plt.suptitle("Example copy correction occupancy", fontsize=16)
		plt.tight_layout()


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

	def plot_ptrs(self, key_1, key_2):
		ptrs1 = self.ptrs_df[key_1]
		ptrs2 = self.ptrs_df[key_2]
		plot_replication_times_df = self.all_replication_times
		plt.plot([0, 2], [0, 2], c='black', lw=0.75, ls='dotted')
		plt.scatter(ptrs1, ptrs2, s=10, alpha=1, 
					edgecolor='gray', facecolor='none')
		plt.scatter(ptrs1, ptrs2, s=8, alpha=1, 
					c=plot_replication_times_df.replication_time,
					cmap=self.repl_cmap, norm=self.repl_norm)
		plt.xlim(1, 1.4)
		plt.ylim(1, 1.4)
		plt.xlabel("No correction, PTRs")
		plt.ylabel("Copy corrected, PTRs")
		plt.colorbar()

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
