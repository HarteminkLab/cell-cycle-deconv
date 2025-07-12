
import pandas as pd

class ReplicationTiming:
	"""Class to handle loading and retrieving replication timing 
	for genomic features"""

	def __init__(self, output_dir):
		"""Load with the appropriate deconvolution run.

		We may cache this, if this takes to long to load every time
		"""
		self.output_dir = output_dir

		from src.RealDataReplication import load_replication_Fr_df

		repl_dfs = []
		for chrom in range(1, 17):
			_, chrom_replication_df = load_replication_Fr_df(output_dir, chrom, 
				with_replication_timing=True)
			chrom_replication_df['chr'] = chrom
			repl_dfs.append(chrom_replication_df)

		self.replications_df = pd.concat(repl_dfs).reset_index().set_index(['chr', 'start'])
		self.load_joined_muller_data()

	def compute_peak_annotations(self):
		"""Compute whether a replication timing occurs at a peak, that is it is the
		earlier than its neighbors. Peaks will be annotated as shared or absolute. For
		shared peaks, it will be equal to either neighbor."""
		
		chrom_peaks_dfs = []
		for chrom in range(1, 17):
			chrom_peaks = self.compute_peak_annotations_chrom(chrom)
			chrom_peaks['chr'] = chrom
			chrom_peaks_dfs.append(chrom_peaks)
		peaks_df = pd.concat(chrom_peaks_dfs).reset_index().set_index(['chr', 'start'])
		self.replications_df = self.replications_df.join(peaks_df)

	# Now let's identify indices with peaks
	def compute_peak_annotations_chrom(self, chrom):
		"""Compute whether a replication timing occurs at a peak, that is it is the
		earlier than its neighbors. Peaks will be annotated as shared or absolute. For
		shared peaks, it will be equal to either neighbor."""
		chrom_repl_timings = self.replications_df.loc[chrom]
		peak_annotations = chrom_repl_timings[[]].copy()
		peak_annotations['peak_type'] = False
		start_indices = chrom_repl_timings.index
		peaks = []
		for index, start in enumerate(start_indices):
			if index > 0 and index < len(start_indices)-1:
				prev_entry = chrom_repl_timings.iloc[index-1]
				cur_entry = chrom_repl_timings.iloc[index]
				next_entry = chrom_repl_timings.iloc[index+1]

				if (cur_entry.replication_time < prev_entry.replication_time and
					cur_entry.replication_time < next_entry.replication_time):
					peak_annotations.loc[start, 'peak_type'] = 'absolute'

				elif (cur_entry.replication_time <= prev_entry.replication_time and
					cur_entry.replication_time <= next_entry.replication_time):
					peak_annotations.loc[start, 'peak_type'] = 'shared'
		return peak_annotations

	def load_replication_entry_for(self, chrom, position=None, span=None):

		from src.mnase_10kb_loader import get_bin_for_position
		start_indices = self.replications_df.loc[chrom].index
		if span is not None:
			position = (span[0]+span[1])/2

		bin_idx, start = get_bin_for_position(position, start_indices)
		return self.replications_df.loc[chrom].loc[start]

	def load_joined_muller_data(self):
		from src.reference_data import load_muller_replication_timing_copy_number_ratio
		from src.mnase_10kb_loader import get_bin_for_position
		from src.read_bam import _fromRoman


		# Load the muller replication timing data
		muller_timing = load_muller_replication_timing_copy_number_ratio()
		muller_timing.chromosome = muller_timing.chromosome.str.replace('chr', '').astype(int)
		muller_timing = muller_timing.set_index(['chromosome', 'position'])

		chrom_joined_dfs_arr = []

		for chrom in range(1, 17):
			chrom_muller_timing = muller_timing.loc[chrom]
			chrom_repl_timing = self.replications_df.loc[chrom]

			# Convert the muller positions into the bin positions that correspond
			# with our replication timing
			muller_bp_positions = [get_bin_for_position(b, chrom_repl_timing.index.values,
			bp_only=True) for b in chrom_muller_timing.index]
			chrom_muller_timing['bin'] = muller_bp_positions
			chrom_muller_timing = chrom_muller_timing.groupby('bin').mean()

			chrom_joined_df = chrom_muller_timing.join(chrom_repl_timing[['replication_time']])
			chrom_joined_df['chrom'] = chrom
			chrom_joined_df = chrom_joined_df.reset_index().set_index(['chrom', 'bin'])
			
			chrom_joined_dfs_arr.append(chrom_joined_df)

		joined_replication_df = pd.concat(chrom_joined_dfs_arr)
		from scipy.stats import pearsonr

		self.joined_muller_replication_df = joined_replication_df
		self.muller_pearsonr = pearsonr(-joined_replication_df.copy_number_ratio, joined_replication_df.replication_time)

	def plot_muller_correlation(self):
		from src.DensityScatterPlotter import DensityScatterPlotter
		import matplotlib.pyplot as plt

		fig = plt.figure(figsize=(5.25, 5))
		plotter = DensityScatterPlotter()
		plotter.set_data(self.joined_muller_replication_df['copy_number_ratio'].values,
				   self.joined_muller_replication_df['replication_time'].values)
		plotter.bw = [0.02, 0.5]
		plotter.cmap = 'Purples'
		plotter.plot_ax(plt.gca())
		plt.ylim(35, -15)
		plt.title(f"Deep sequencing vs MNase-seq\n"
				  f"n={len(self.joined_muller_replication_df)}, Pearson r = {self.muller_pearsonr[0]:.2g}",
				 fontweight='demi', fontsize=18, y=1.02)
		plt.ylabel("Replication time, min (Deconvolved MNase-seq)")
		plt.xlabel("Copy # ratio (Deep Sequencing, Müller, 2014)")

	def plot_chrom_timing(self, chrom):
		
		import matplotlib.pyplot as plt

		fig = plt.figure(figsize=(7, 4))

		chr_replication = self.replications_df.loc[chrom]

		plt.subplot(2, 1, 1)
		plt.scatter(self.joined_muller_replication_df.loc[chrom].index,
			self.joined_muller_replication_df.loc[chrom].copy_number_ratio, s=1,
			c='#777')
		plt.xlim(chr_replication.index[0], chr_replication.index[-1])
		plt.ylim(0.75, 2.25)
		plt.title("Deep sequencing, (Müller, 2014)")
		plt.xticks([])
		plt.ylabel("Copy # ratio")

		plt.subplot(2, 1, 2)
		plt.scatter(chr_replication.index,
			chr_replication['replication_time'], s=1, color=plt.cm.Oranges(0.75))
		plt.xlim(chr_replication.index[0], chr_replication.index[-1])
		plt.ylim(38, -25)
		plt.title("Deconvolved MNase-seq")
		plt.ylabel("Replication time, min")
		plt.xlabel("Genomic position, bp")

		plot_origins = False
		if plot_origins:
			from src.origins import load_origins
			from src.mnase_10kb_loader import get_bin_for_position

			origins = load_origins(full=True)
			chr_origins = origins[origins.chr == chrom].copy()

			closest_origin_bp = [get_bin_for_position(pos, chr_replication.index.values,
				bp_only=True) for pos in chr_origins.pos.values]
			chr_origins['start_bin'] = closest_origin_bp

			chr_origins = chr_origins.reset_index()[['start_bin', 'derived_origin_efficiency_from_mcguffee_et_al_2013']].\
				groupby('start_bin').mean()
			chr_origins = chr_origins.join(chr_replication[['replication_time']])

			alpha_values = chr_origins['derived_origin_efficiency_from_mcguffee_et_al_2013'].values
			alpha_values[alpha_values < 0] = 0
			plt.scatter(chr_origins.index, chr_origins.replication_time, 
				alpha=alpha_values, s=10, c='blue')

		plt.suptitle(f"Replication profile for chr{chrom}", 
			fontsize=20, fontweight='demi')
		plt.tight_layout()
		plt.subplots_adjust(hspace=0.5)
