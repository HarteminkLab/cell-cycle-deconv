
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
		self.chrom_b_copy_corrected_means_df = pd.DataFrame(self.chrom_cc_means_df, index=starts)

		# To avoid memory issues, we'll delete the data after it has been loaded
		del F_data_cc
		del F_data_no_cc


	def plot_heatmap_comparison(self):
		from src.config import load_default_expression_configs

		config1, config2 = load_default_expression_configs()

		branch_indices = config1.t_indices()
			
		fig = plt.figure(figsize=(13, 9))

		nrows = 4

		plt.subplot(nrows, 1, 1)
		plt.imshow(self.chrom_no_cc_means.T[\
			branch_indices], 
			vmin=0, vmax=2, cmap='RdBu_r', aspect='auto', interpolation='none')
		plt.xticks([])
		plt.title("No copy correction", fontsize=12)

		plt.subplot(nrows, 1, 2)
		plt.imshow(self.chrom_b_copy_corrected_means.T[\
			branch_indices], 
			vmin=0, vmax=2, cmap='RdBu_r', aspect='auto', interpolation='none')
		plt.xticks([])
		plt.title("With copy correction", fontsize=12)

		plt.subplot(nrows, 1, 3)
		plt.imshow((self.chrom_b_copy_corrected_means-\
						self.chrom_no_cc_means).T[\
			branch_indices], 
			vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto', interpolation='none')
		plt.xticks([])
		plt.title("No copy correction - Copy correction", fontsize=12)

		plt.subplot(nrows, 1, 4)
		plt.plot(self.chrom_replication_times.index,
				 self.chrom_replication_times.replication_time)
		plt.ylim(60, 30)
		plt.xlim(0, self.chrom_windows.start.values[-1])
		plt.title("Replication time", fontsize=12)

		plt.suptitle(f"Copy correction affect on chromatin deconvolution, chrom{self.chrom} 10kb means",
			fontsize=16)

		plt.tight_layout()


	def plot_sample_curves(self):
		copy_correction_ptr_comparison_t = self.ptrs_df[\
			['cc_ptr_t', 'no_cc_ptr_t']]
		copy_correction_ptr_comparison_t

		repl_ptrs_df = copy_correction_ptr_comparison_t.join(
			self.chrom_replication_times)

		sorted_repl_ptrs = repl_ptrs_df.sort_values('replication_time')

		early_window = sorted_repl_ptrs.iloc[10].name
		late_window = sorted_repl_ptrs.iloc[-6].name

		window_idx = early_window
		indices = config1.t_indices()

		no_cc_mean = self.chrom_no_cc_means_df.mean().mean()
		cc_mean = self.chrom_b_copy_corrected_means_df.mean().mean()

		def _plot_window(window_idx):
			tps = config1.get_timepoints_for_branch('t')
			plt.plot(tps, self.chrom_no_cc_means_df\
						 .loc[window_idx][indices] / no_cc_mean,
					color='black',
					lw=1,
					ls=(5, (1, 1)),
					label="No copy correction")
			plt.plot(tps, self.chrom_b_copy_corrected_means_df\
						 .loc[window_idx][indices] / cc_mean,
					color=plt.cm.Reds(0.6),
					lw=2,
					label="With copy correction")
			plt.ylim(0, 2)

		plt.figure(figsize=(9, 4))
		plt.subplot(1, 2, 1)
		_plot_window(early_window)
		plt.title("Early replicating window")

		plt.subplot(1, 2, 2)
		_plot_window(late_window)
		plt.title("Late replicating window")
		plt.legend()

		plt.suptitle("Mother branch correction examples")
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

		cc_means = self.chrom_b_copy_corrected_means
		no_cc_means = self.chrom_no_cc_means_df.values

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
		}, index=self.chrom_starts)


	def plot_branch_ptrs(self):

		window_replication_times = self.chrom_replication_times

		def plot_ptrs(no_cc_ptrs, cc_ptrs):
			plt.plot([0, 2], [0, 2], c='black', lw=0.75)
			plt.scatter(no_cc_ptrs, cc_ptrs, s=12, alpha=1, 
						edgecolor='gray', facecolor='none')
			plt.scatter(no_cc_ptrs, cc_ptrs, s=10, alpha=1, 
						c=window_replication_times.replication_time[:len(no_cc_ptrs)],
					   cmap='RdBu', vmin=30, vmax=60)
			plt.xlim(1, 1.3)
			plt.ylim(1, 1.3)
			plt.xlabel("No correction, PTRs")
			plt.ylabel("Copy corrected, PTRs")
			plt.colorbar()

		no_cc_ptrs = self.ptrs_df['no_cc_ptr_all']
		cc_ptrs = self.ptrs_df['cc_ptr_all']

		i_no_cc_ptrs = self.ptrs_df['no_cc_ptr_i']
		i_cc_ptrs = self.ptrs_df['cc_ptr_i']

		t_no_cc_ptrs = self.ptrs_df['no_cc_ptr_t']
		t_cc_ptrs = self.ptrs_df['cc_ptr_t']

		b_no_cc_ptrs = self.ptrs_df['no_cc_ptr_b']
		b_cc_ptrs = self.ptrs_df['cc_ptr_b']

		plt.figure(figsize=(9, 8))
		plt.subplot(2, 2, 1)
		plot_ptrs(no_cc_ptrs, cc_ptrs)
		plt.title("All branches")

		plt.subplot(2, 2, 2)
		plot_ptrs(i_no_cc_ptrs, i_cc_ptrs)
		plt.title("Recovery")

		plt.subplot(2, 2, 3)
		plot_ptrs(t_no_cc_ptrs, t_cc_ptrs)
		plt.title("Top")

		plt.subplot(2, 2, 4)
		plot_ptrs(b_no_cc_ptrs, b_cc_ptrs)
		plt.title("Bottom")

		plt.suptitle(f"PTR comparison, chr{self.chrom} 10kb means", fontsize=16, y=0.97)
		plt.tight_layout()