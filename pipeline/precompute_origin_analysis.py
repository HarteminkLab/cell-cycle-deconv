import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.timer import Timer
from src.mnase_10kb_loader import MNase10kbLoader
from src.global_config import GlobalConstants

class OriginFootprintAnalysis:
	"""
	A simplified class to analyze origins of replication using MNase-seq data,
	storing small fragment summaries and composite data for all timepoints
	"""
	
	def __init__(self, window_size=1000, chromosomes=np.arange(1, 17)):
		"""
		Initialize the OriginFootprintAnalysis
		
		Parameters:
		-----------
		window_size : int
			Size of the window around each origin (in bp)
		chromosomes : array-like
			List of chromosomes to process
		"""
		self.timer = Timer()
		self.chromosomes = chromosomes

		# Loaders for each replicate for caching
		self.mnase_loaders = {
			1: MNase10kbLoader(),
			2: MNase10kbLoader()
		}

		# Get all timepoints for each replicate
		self.wt1_timepoints = GlobalConstants.CHROM_WT1_TIMEPOINTS
		self.wt2_timepoints = GlobalConstants.CHROM_WT2_TIMEPOINTS

		# Default timepoints for early S-phase
		self.default_s_phase_timepoints = {
			1: 30,
			2: 20
		}

		self.window_size = window_size
		self.origins = None
		
		# Member variables to store data
		# Small fragment data: indexed by replicate -> timepoint -> orf -> fragment counts
		self.small_fragments_data = {
			1: {},  # For replicate 1
			2: {}   # For replicate 2
		}
		
		# Composite data: indexed by replicate -> timepoint -> 2D histogram
		self.composite_data = {
			1: {},  # For replicate 1
			2: {}   # For replicate 2
		}
		
		# Separate storage for composite counts
		self.composite_counts = {
			1: {},  # For replicate 1
			2: {}   # For replicate 2
		}
		
	def load_origin_reference_dataset(self):
		"""Load the origin reference dataset"""
		from src.origins import load_origins
		self.origins = load_origins(full=True)
		return self.origins

	def generate_small_fragments_for_chromosome(self, chrom, replicate, frag_sel=(0, 100), strand_correct=True):
		"""
		Generate small fragment summaries for all origins on a chromosome, for all timepoints
		
		Parameters:
		-----------
		chrom : str
			Chromosome name
		replicate : int
			Replicate (1 or 2)
		frag_sel : tuple
			Fragment size selection range (min, max)
		strand_correct : bool
			Whether to flip the data for origins on the Crick strand
			
		Returns:
		--------
		None (updates member variables in place)
		"""
		if self.origins is None:
			self.load_origin_reference_dataset()
			
		# Get all origins on this chromosome
		origins_on_chrom = self.origins[self.origins['chr'] == chrom]
		
		if len(origins_on_chrom) == 0:
			return
			
		print(f"  Processing chromosome {chrom} ({len(origins_on_chrom)} origins) - {self.timer.get_time()}")
		
		# Load MNase data for this chromosome (all timepoints included)
		mnase_data = self.mnase_loaders[replicate].load_mnase_data(replicate, chrom)
		
		# Get all timepoints for this replicate
		timepoints = self.wt1_timepoints if replicate == 1 else self.wt2_timepoints
		
		# Process each origin on this chromosome
		for idx, origin in origins_on_chrom.iterrows():
			# Get strand for strand correction
			strand = origin['strand']
			
			# Define window around origin
			mid = origin['pos']
			half_window = self.window_size // 2
			window_start = mid - half_window
			window_end = mid + half_window
			
			# For each timepoint
			for timepoint in timepoints:
				# Skip if already processed
				if (timepoint in self.small_fragments_data[replicate] and 
					idx in self.small_fragments_data[replicate][timepoint]):
					continue
					
				# Initialize timepoint dictionary if needed
				if timepoint not in self.small_fragments_data[replicate]:
					self.small_fragments_data[replicate][timepoint] = {}
				
				# Initialize composite data and counts if needed
				if timepoint not in self.composite_data[replicate]:
					self.composite_data[replicate][timepoint] = None
					self.composite_counts[replicate][timepoint] = 0
				
				# Filter reads for this timepoint
				timepoint_data = mnase_data[mnase_data['sample'] == timepoint]
				
				# Filter reads in the window
				window_reads = timepoint_data[
					(timepoint_data['mid'] >= window_start) & 
					(timepoint_data['mid'] <= window_end)
				]
				
				# Initialize arrays for positions and lengths
				positions = []
				lengths = []
				
				if len(window_reads) > 0:
					# Extract positions (relative to window start) and lengths
					read_positions = window_reads['mid'].values - window_start
					read_lengths = window_reads['length'].values
					
					# Append to our arrays
					positions.extend(read_positions)
					lengths.extend(read_lengths)
				
				# Define histogram bins
				position_bins = np.arange(0, self.window_size + 2)
				max_length = 250
				length_bins = np.arange(0, max_length + 1)

				# Create 2D histogram
				hist_2d, _, _ = np.histogram2d(
					lengths,
					positions, 
					bins=[length_bins, position_bins]
				)
				
				# Calculate small fragment mean
				small_fragment_mean = hist_2d[frag_sel[0]:frag_sel[1], :].mean(axis=0)
				
				# Apply strand correction if needed
				if strand_correct and strand == '-':
					small_fragment_mean = np.flip(small_fragment_mean)
				
				# Store in member variable
				self.small_fragments_data[replicate][timepoint][idx] = small_fragment_mean
				
				# For composite data, flip the histogram if on Crick strand
				hist_for_composite = hist_2d.copy()
				if strand_correct and strand == '-':
					hist_for_composite = np.flip(hist_for_composite, axis=1)
				
				# Add to composite (will normalize later)
				if self.composite_data[replicate][timepoint] is None:
					self.composite_data[replicate][timepoint] = hist_for_composite
				else:
					self.composite_data[replicate][timepoint] += hist_for_composite
				
				self.composite_counts[replicate][timepoint] += 1

	def finalize_composite_data(self):
		"""
		Normalize all composite histograms by the number of origins
		"""
		for replicate in [1, 2]:
			for timepoint in self.composite_data[replicate].keys():
				# Skip if no data for this timepoint
				if self.composite_data[replicate][timepoint] is None:
					continue
				
				# Normalize by count
				count = self.composite_counts[replicate][timepoint]
				if count > 0:
					self.composite_data[replicate][timepoint] /= count

	def generate_summary_histogram_data(self, frag_sel=(0, 100), strand_correct=True):
		"""
		Generate small fragment summaries and composite data for all origins, timepoints, and replicates
		
		Parameters:
		-----------
		frag_sel : tuple
			Fragment size selection range (min, max)
		strand_correct : bool
			Whether to flip the data for origins on the Crick strand
			
		Returns:
		--------
		tuple
			(small_fragments_data, composite_data) - references to the member variables
		"""
		if self.origins is None:
			self.load_origin_reference_dataset()
			
		self.timer.start()
		
		print(f"Generating summary histogram data for all timepoints and replicates...")
		
		# Process replicate 1
		print(f"Processing replicate 1...")
		for chrom in self.chromosomes:
			self.generate_small_fragments_for_chromosome(chrom, 1, frag_sel, strand_correct)
		
		# Process replicate 2
		print(f"Processing replicate 2...")
		for chrom in self.chromosomes:
			self.generate_small_fragments_for_chromosome(chrom, 2, frag_sel, strand_correct)
		
		# Normalize composite data
		self.finalize_composite_data()
		
		self.timer.print_time(f"Completed summary histogram data generation for all timepoints and replicates")
		
		return (self.small_fragments_data, self.composite_data)
	
	def get_small_fragments_df(self, replicate, timepoint):
		"""
		Convert the small fragments data for a specific replicate and timepoint to a DataFrame
		
		Parameters:
		-----------
		replicate : int
			Replicate (1 or 2)
		timepoint : int
			Timepoint
			
		Returns:
		--------
		pandas.DataFrame
			DataFrame with origins as rows and positions as columns
		"""
		if replicate not in self.small_fragments_data or timepoint not in self.small_fragments_data[replicate]:
			raise ValueError(f"No data available for replicate {replicate}, timepoint {timepoint}")
		
		# Get the data dictionary
		data_dict = self.small_fragments_data[replicate][timepoint]
		
		# Create a list of arrays
		data_arrays = []
		indices = []
		
		for idx, array in data_dict.items():
			data_arrays.append(array)
			indices.append(idx)
		
		# Create positions for columns
		win_2 = self.window_size // 2
		positions = np.arange(-win_2, win_2+1)
		
		# Create DataFrame
		df = pd.DataFrame(
			data_arrays,
			index=indices,
			columns=positions
		)
		
		return df
	
	def get_composite_histogram(self, replicate, timepoint):
		"""
		Get the composite histogram for a specific replicate and timepoint
		
		Parameters:
		-----------
		replicate : int
			Replicate (1 or 2)
		timepoint : int
			Timepoint
			
		Returns:
		--------
		numpy.ndarray
			2D composite histogram
		"""
		if replicate not in self.composite_data or timepoint not in self.composite_data[replicate]:
			raise ValueError(f"No composite data available for replicate {replicate}, timepoint {timepoint}")
		
		return self.composite_data[replicate][timepoint]
	
	#--------------------------------
	# Plotting Methods
	#--------------------------------
	
	def plot_small_fragment_heatmap(self, replicate, timepoint, origins_sorted=None, max_value=0.02):
		"""
		Plot heatmap of small fragment means for a specific replicate and timepoint
		
		Parameters:
		-----------
		replicate : int
			Replicate (1 or 2)
		timepoint : int
			Timepoint
		origins_sorted : pandas.DataFrame or None
			Pre-sorted origins to plot
			If None, uses the index order from the DataFrame
		max_value : float
			Maximum value for color scale
			
		Returns:
		--------
		matplotlib.figure.Figure
			The created figure
		"""
		# Get the data
		df = self.get_small_fragments_df(replicate, timepoint)
		
		# Use provided sorting if available
		if origins_sorted is not None:
			df = df.reindex(origins_sorted.index)
		
		# Create figure
		fig, ax = plt.subplots(figsize=(3, 4))
		
		# Window half-size for plot extents
		win_2 = self.window_size // 2
		
		# Plot heatmap
		n = len(df)
		ax.imshow(df, vmax=max_value, aspect='auto',
				 extent=[-win_2, win_2, 0, n], origin='lower', interpolation='none',
				 cmap='Blues')
		ax.set_ylim(n, 0)
		
		# Add title
		ax.set_title(f"WT{replicate} @ {timepoint}")
		
		# Add dividing line if there's a classification in the origins
		if origins_sorted is not None and 'footprint_class' in origins_sorted.columns:
			num_footprint = sum(origins_sorted.footprint_class == 'g1_and_g2_footprint')
			if num_footprint > 0:
				ax.axhline(num_footprint, c='black', lw=1, ls='dashed')
		
		plt.tight_layout()
		return fig
	
	def plot_small_fragment_heatmaps(self, replicates, timepoints, origins_sorted=None, max_value=0.02):
		"""
		Plot multiple heatmaps of small fragment means
		
		Parameters:
		-----------
		replicates : list
			List of replicates to plot
		timepoints : list or dict
			List of timepoints to plot (one per replicate) or dict mapping replicate to timepoint
		origins_sorted : pandas.DataFrame or None
			Pre-sorted origins to plot
			If None, uses the index order from the DataFrame
		max_value : float
			Maximum value for color scale
			
		Returns:
		--------
		matplotlib.figure.Figure
			The created figure
		"""
		# Create figure
		n_plots = len(replicates)
		fig, axs = plt.subplots(1, n_plots, figsize=(3 * n_plots, 4))
		if n_plots == 1:
			axs = [axs]
		
		# Window half-size for plot extents
		win_2 = self.window_size // 2
		
		# Plot each heatmap
		for i, rep in enumerate(replicates):
			# Determine timepoint
			if isinstance(timepoints, dict):
				tp = timepoints[rep]
			else:
				tp = timepoints[i]
			
			# Get the data
			df = self.get_small_fragments_df(rep, tp)
			
			# Use provided sorting if available
			if origins_sorted is not None:
				df = df.reindex(origins_sorted.index)
			
			# Plot heatmap
			n = len(df)
			axs[i].imshow(df, vmax=max_value, aspect='auto',
						 extent=[-win_2, win_2, 0, n], origin='lower', interpolation='none',
						 cmap='Blues')
			axs[i].set_ylim(n, 0)
			
			# Add title
			axs[i].set_title(f"WT{rep} @ {tp}")
			
			# Add dividing line if there's a classification in the origins
			if origins_sorted is not None and 'footprint_class' in origins_sorted.columns:
				num_footprint = sum(origins_sorted.footprint_class == 'g1_and_g2_footprint')
				if num_footprint > 0:
					axs[i].axhline(num_footprint, c='black', lw=1, ls='dashed')
			
			# Remove y-ticks for all but the first plot
			if i > 0:
				axs[i].set_yticks([])
		
		plt.tight_layout()
		return fig
	
	def plot_composite_histogram(self, replicate, timepoint, max_value=0.1, cmap='magma_r'):
		"""
		Plot composite histogram for a specific replicate and timepoint
		
		Parameters:
		-----------
		replicate : int
			Replicate (1 or 2)
		timepoint : int
			Timepoint
		max_value : float
			Maximum value for color scale
		cmap : str
			Colormap to use
			
		Returns:
		--------
		matplotlib.figure.Figure
			The created figure
		"""
		# Get the data
		hist = self.get_composite_histogram(replicate, timepoint)
		
		# Create figure
		fig, ax = plt.subplots(figsize=(3, 1.5))
		
		# Window half-size for plot extents
		win_2 = self.window_size // 2
		
		# Plot composite histogram
		ax.imshow(hist, cmap=cmap, origin='lower', vmax=max_value,
				 extent=[-win_2, win_2, 0, 250], aspect='auto', interpolation='none')
		
		# Add title
		ax.set_title(f"WT{replicate} @ {timepoint}")
		
		plt.tight_layout()
		return fig
	
	def plot_composite_histograms(self, replicates, timepoints, max_value=0.1, cmap='magma_r'):
		"""
		Plot multiple composite histograms
		
		Parameters:
		-----------
		replicates : list
			List of replicates to plot
		timepoints : list or dict
			List of timepoints to plot (one per replicate) or dict mapping replicate to timepoint
		max_value : float
			Maximum value for color scale
		cmap : str
			Colormap to use
			
		Returns:
		--------
		matplotlib.figure.Figure
			The created figure
		"""
		# Create figure
		n_plots = len(replicates)
		fig, axs = plt.subplots(1, n_plots, figsize=(5.5, 1.5))
		if n_plots == 1:
			axs = [axs]
		
		# Window half-size for plot extents
		win_2 = self.window_size // 2
		
		# Plot each composite histogram
		for i, rep in enumerate(replicates):
			# Determine timepoint
			if isinstance(timepoints, dict):
				tp = timepoints[rep]
			else:
				tp = timepoints[i]
			
			# Get the data
			hist = self.get_composite_histogram(rep, tp)
			
			# Plot composite histogram
			axs[i].imshow(hist, cmap=cmap, origin='lower', vmax=max_value,
						 extent=[-win_2, win_2, 0, 250], aspect='auto', interpolation='none')
			
			# Add title
			axs[i].set_title(f"WT{rep} @ {tp}")
			
			# Remove y-ticks for all but the first plot
			if i > 0:
				axs[i].set_yticks([])
		
		plt.tight_layout()
		plt.subplots_adjust(wspace=0.1)
		return fig
	
	def plot_timecourse_heatmaps(self, replicate, timepoints, origins_sorted=None, max_value=0.02):
		"""
		Plot small fragment heatmaps for a timecourse
		
		Parameters:
		-----------
		replicate : int
			Replicate to plot
		timepoints : list
			List of timepoints to plot
		origins_sorted : pandas.DataFrame or None
			Pre-sorted origins to plot
			If None, uses the index order from the DataFrame
		max_value : float
			Maximum value for color scale
			
		Returns:
		--------
		matplotlib.figure.Figure
			The created figure
		"""
		# Create figure
		n_plots = len(timepoints)
		fig, axs = plt.subplots(1, n_plots, figsize=(3 * n_plots, 4))
		if n_plots == 1:
			axs = [axs]
		
		# Window half-size for plot extents
		win_2 = self.window_size // 2
		
		# Plot each heatmap
		for i, tp in enumerate(timepoints):
			# Get the data
			df = self.get_small_fragments_df(replicate, tp)
			
			# Use provided sorting if available
			if origins_sorted is not None:
				df = df.reindex(origins_sorted.index)
			
			# Plot heatmap
			n = len(df)
			axs[i].imshow(df, vmax=max_value, aspect='auto',
						 extent=[-win_2, win_2, 0, n], origin='lower', interpolation='none',
						 cmap='Blues')
			axs[i].set_ylim(n, 0)
			
			# Add title
			axs[i].set_title(f"WT{replicate} @ {tp}")
			
			# Add dividing line if there's a classification in the origins
			if origins_sorted is not None and 'footprint_class' in origins_sorted.columns:
				num_footprint = sum(origins_sorted.footprint_class == 'g1_and_g2_footprint')
				if num_footprint > 0:
					axs[i].axhline(num_footprint, c='black', lw=1, ls='dashed')
			
			# Remove y-ticks for all but the first plot
			if i > 0:
				axs[i].set_yticks([])
		
		plt.tight_layout()
		return fig
	
	def plot_timecourse_composites(self, replicate, timepoints, max_value=0.1, cmap='magma_r'):
		"""
		Plot composite histograms for a timecourse vertically
		
		Parameters:
		-----------
		replicate : int
			Replicate to plot
		timepoints : list
			List of timepoints to plot
		max_value : float
			Maximum value for color scale
		cmap : str
			Colormap to use
			
		Returns:
		--------
		matplotlib.figure.Figure
			The created figure
		"""
		# Create figure with vertical layout
		n_plots = len(timepoints)
		fig, axs = plt.subplots(n_plots, 1, figsize=(1.5, 0.75 * n_plots))
		if n_plots == 1:
			axs = [axs]
		
		# Window half-size for plot extents
		win_2 = self.window_size // 2
		
		# Plot each composite histogram
		for i, tp in enumerate(timepoints):
			# Get the data
			hist = self.get_composite_histogram(replicate, tp)
			
			# Plot composite histogram
			axs[i].imshow(hist, cmap=cmap, origin='lower', vmax=max_value,
						 extent=[-win_2, win_2, 0, 250], aspect='auto', interpolation='none')
			
			axs[i].set_ylabel(f"{tp}'")
			axs[i].set_yticks([])
			
			# Remove x-ticks for all but the last plot
			if i < n_plots - 1:
				axs[i].set_xticks([])
		
		plt.suptitle(f"WT{replicate}", y=0.98)
		plt.tight_layout()
		plt.subplots_adjust(hspace=0)  # Remove vertical spacing between plots
		return fig