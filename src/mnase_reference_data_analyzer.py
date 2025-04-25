import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.timer import Timer
from src.mnase_10kb_loader import MNase10kbLoader
from scipy.stats import norm

class MNaseReferenceAnalyzer:
	def __init__(self, reference_sites, padding=100, length_bins=250):
		"""
		Initialize the MNase Reference Analyzer
		
		Parameters:
		-----------
		reference_sites : pandas.DataFrame
			DataFrame containing reference sites with columns 'chr', 'mid', and 'strand'
		padding : int
			Distance in bp to consider around each reference site
		length_bins : int
			Number of bins for fragment length dimension
		"""
		self.reference_sites = reference_sites
		self.padding = padding
		self.mnase_loader = MNase10kbLoader()
		self.timer = Timer()
		
		# Initialize histogram bins
		self.x_bins = np.arange(-padding, padding+1, 1)
		self.y_bins = np.arange(0, length_bins+1)
		
		# Initialize the histogram matrix
		self.hist = np.zeros((len(self.x_bins)-1, len(self.y_bins)-1))
		
		# Keep track of total reads processed
		self.total_reads = 0
		
	def filter_reads(self, reads, span):
		"""Filter reads within the given span"""
		return reads[(reads.mid > span[0]) & (reads.mid < span[1])]
	
	def process_sites(self, chromosomes=range(1, 17)):
		"""
		Process all reference sites on the given chromosomes
		
		Parameters:
		-----------
		chromosomes : iterable
			Chromosomes to process
		"""
		for replicate in [1, 2]:

			print(f"Replicate {replicate}...Chromosomes: ")

			for chrom in chromosomes:
				print(f"{chrom}", end=",")
				chrom_sites = self.reference_sites[self.reference_sites.chr == chrom]
				chrom_reads = self.mnase_loader.load_mnase_data(replicate, chrom)
				
				for _, chrom_site in chrom_sites.iterrows():
					span = chrom_site.mid - self.padding, chrom_site.mid + self.padding
					site_reads = self.filter_reads(chrom_reads, span)
					
					# Calculate translated midpoints relative to reference site
					translated_mids = site_reads.mid - chrom_site.mid
					
					# Adjust for strand orientation
					if 'strand' in chrom_site.index and chrom_site.strand == '-':
						translated_mids = translated_mids * -1
					
					# Update histogram incrementally for this site
					if len(translated_mids) > 0:
						H, _, _ = np.histogram2d(
							translated_mids, 
							site_reads['length'], 
							bins=[self.x_bins, self.y_bins]
						)
						self.hist += H
						self.total_reads += len(translated_mids)
			print()
			self.timer.print_time(f"Number of total reads {self.total_reads}")
	
	def plot_heatmap(self, vmax=1e3, cmap='magma_r'):
		"""Plot the 2D histogram as a heatmap"""
		plt.figure(figsize=(10, 6))
		plt.imshow(self.hist.T, vmax=vmax, cmap=cmap, origin='lower', 
				   extent=[self.x_bins[0], self.x_bins[-1], self.y_bins[0], self.y_bins[-1]])
		plt.colorbar(label='Read count')
		plt.xlabel('Distance from reference site (bp)')
		plt.ylabel('Fragment length (bp)')
		plt.title(f'MNase reads around reference sites (n={self.total_reads})')
		return plt.gca()
	
	def plot_length_distribution(self):
		"""Plot the length distribution of fragments"""
		plt.figure(figsize=(10, 4))
		length_dist = self.hist.mean(0)/self.hist.mean()  # Average across all positions
		plt.plot(self.y_bins[:-1], length_dist)
		plt.xlabel('Fragment length (bp)')
		plt.ylabel('Average read count')
		plt.title('Fragment length distribution')
		return plt.gca()

	def plot_distribution(self, vmax=None, cmap='magma_r', show_fit=True):
		"""
		Plot both the 2D histogram heatmap and length distribution side by side
		with optional fitted Gaussian curve on the length distribution
		
		Parameters:
		-----------
		vmax : float
			Maximum value for color scaling in heatmap
		cmap : str
			Colormap to use for the heatmap
		show_fit : bool
			Whether to show the fitted Gaussian curve (if available)
			
		Returns:
		--------
		fig : matplotlib.figure.Figure
			The figure containing both plots
		axs : list of matplotlib.axes.Axes
			The two axes objects for further customization
		"""
		# Create figure with two subplots side by side
		fig, axs = plt.subplots(1, 2, figsize=(6, 4), gridspec_kw={'width_ratios': [1, 1]})
		
		# Plot heatmap on the left subplot
		im = axs[0].imshow(
			self.hist.T, 
			vmax=vmax, 
			cmap=cmap, 
			origin='lower',
			extent=[self.x_bins[0], self.x_bins[-1], self.y_bins[0], self.y_bins[-1]],
			aspect='auto'
		)
		axs[0].set_xlabel('Position, bp')
		axs[0].set_ylabel('Fragment length, bp')
		axs[0].set_title(f'Read dist. (n={self.total_reads})')
		
		# Add colorbar to the heatmap
		cbar = fig.colorbar(im, ax=axs[0])
		cbar.set_label('Read count')
		
		# Plot length distribution on the right subplot
		length_dist = self.hist.mean(0)/self.hist.mean()  # Average across all positions
		axs[1].plot(length_dist, self.y_bins[:-1], 'b-', label='Raw')
		axs[1].set_xlabel('Average read count')
		axs[1].set_title('Fragment length distribution')
		
		# Add fitted Gaussian curve if available and requested
		if show_fit and hasattr(self, 'fit_gaussian_loc') and hasattr(self, 'fit_gaussian_scale'):
			# Generate x values for plotting
			x_data = self.y_bins[:-1]
			
			# Calculate the fitted curve
			fitted_curve = gaussian(x_data, 
									self.fit_gaussian_loc, 
									self.fit_gaussian_scale, 
									self.fit_gaussian_amplitude)
			
			# Plot the fitted curve
			axs[1].plot(fitted_curve, x_data, 'r-', 
					   label=f'Fit (μ={self.fit_gaussian_loc:.1f}, σ={self.fit_gaussian_scale:.1f})')
			
			# If we have a normalized selection curve, plot it too
			if hasattr(self, 'fit_range'):
				# Create selection curve (using the same approach as in fit_gaussian_to_length_distribution)
				ys = norm.pdf(x_data, loc=self.fit_gaussian_loc, scale=self.fit_gaussian_scale)
				selection_curve = pd.DataFrame(ys, index=x_data, columns=['weight'])
				
				# Zero out values outside our desired range
				selection_curve.loc[selection_curve.index < self.fit_range[0], 'weight'] = 0
				selection_curve.loc[selection_curve.index > self.fit_range[1], 'weight'] = 0
				
				# Normalize and plot
				non_zero_mean = selection_curve.loc[selection_curve['weight'] > 0, 'weight'].mean()
				if non_zero_mean > 0:
					selection_curve = selection_curve / non_zero_mean
					
				# Scale for visualization and plot
				max_val = length_dist.max()
				selection_scaled = selection_curve['weight'] * (max_val / selection_curve['weight'].max()) \
								  if selection_curve['weight'].max() > 0 else selection_curve['weight']
				axs[1].plot(selection_scaled, x_data, 'g--', 
						   label='Normalized selection curve')
			
			# Add a legend
			axs[1].legend(loc='lower right')
		
		# Make the y-axis on the right align with the heatmap
		axs[1].set_ylim(self.y_bins[0], self.y_bins[-1])
		
		# Adjust layout to prevent overlap
		plt.tight_layout()
		plt.suptitle(self.title, fontsize=16, fontweight='demi', y=1.05)
		
		return fig, axs

	def fit_gaussian_to_length_distribution(self, initial_guess=None, plot=True, 
									   fit_range=(140, 190)):
		"""
		Fit a Gaussian distribution to the fragment length distribution
		within a restricted range
		"""
		from scipy.optimize import curve_fit
		import pandas as pd
		import numpy as np
		
		# Get the length distribution
		length_dist = self.hist.mean(0)
		
		# Perform the fit on restricted range
		x_data = self.y_bins[:-1]  # Use bin centers
		y_data = length_dist / length_dist.mean()
		
		# Create mask for the range we want to fit
		mask = (x_data >= fit_range[0]) & (x_data <= fit_range[1])
		x_fit = x_data[mask]
		y_fit = y_data[mask]
		
		# Make initial guess if not provided
		if initial_guess is None:
			peak_idx = np.argmax(y_fit)
			loc_guess = x_fit[peak_idx]
			scale_guess = 20  # Fallback default
			amplitude_guess = y_fit.max()
			initial_guess = (loc_guess, scale_guess, amplitude_guess)
		else:
			# If only loc and scale are provided, add amplitude
			if len(initial_guess) == 2:
				amplitude_guess = y_fit.max()
				initial_guess = (initial_guess[0], initial_guess[1], amplitude_guess)
		
		# Fit only to the restricted range
		params, _ = curve_fit(gaussian, x_fit, y_fit, p0=initial_guess)
		loc, scale, amplitude = params
		
		# Create the selection curve for all data points
		xs = x_data
		ys = norm.pdf(xs, loc=loc, scale=scale)
		selection_curve = pd.DataFrame(ys, index=xs, columns=['weight'])
		
		# Zero out values outside our desired range
		selection_curve.loc[selection_curve.index < fit_range[0], 'weight'] = 0
		selection_curve.loc[selection_curve.index > fit_range[1], 'weight'] = 0
		
		# Normalize the curve for weighting (mean of 1.0 for non-zero values)
		non_zero_mean = selection_curve.loc[selection_curve['weight'] > 0, 'weight'].mean()
		if non_zero_mean > 0:
			selection_curve = selection_curve / non_zero_mean
		
		# Plot the results if requested
		if plot:
			plt.figure(figsize=(10, 5))
			
			# Plot the raw data
			plt.plot(x_data, y_data, 'b-', label='Raw length distribution')
			
			# Plot the fitted curve (scaled for visualization)
			fitted_curve = gaussian(x_data, loc, scale, amplitude)
			plt.plot(x_data, fitted_curve, 'r-', 
					 label=f'Gaussian fit (μ={loc:.1f}, σ={scale:.1f})')
			
			plt.xlabel('Fragment length (bp)')
			plt.ylabel('Density')
			plt.legend()
			plt.title('Gaussian fit to fragment length distribution',
				fontsize=13)

		self.fit_gaussian_loc = loc
		self.fit_gaussian_scale = scale
		self.fit_gaussian_amplitude = amplitude

		return params

	def get_histogram(self):
		"""Return the computed histogram and bin edges"""
		return self.hist, self.x_bins, self.y_bins

# Define our model - a Gaussian/normal PDF
def gaussian(x, loc, scale, amplitude):
	return amplitude * norm.pdf(x, loc=loc, scale=scale)
