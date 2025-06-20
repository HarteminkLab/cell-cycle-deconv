import numpy as np
import matplotlib.pyplot as plt
from src.rna_seq_intermediates import RNASeqIntermediateManager


class RNASeqPileupPlotter:
	"""
	A class for plotting RNA-seq pileup data with Watson and Crick strand visualization.
	"""
	
	def __init__(self, output_directory):
		"""
		Initialize the plotter with the output directory containing the data.
		
		Parameters:
		-----------
		output_directory : str
			Path to the directory containing the RNA-seq data
		"""
		self.output_directory = output_directory
		self.watson_data = None
		self.crick_data = None
		self.span = None
		self.chromosome = None
		self.replicate = None
		
	def set_chrom_span(self, chrom, span, replicate):
		"""
		Set the chromosome, genomic span, and replicate, and load the corresponding data.
		
		Parameters:
		-----------
		chrom : int
			Chromosome number
		span : tuple
			Genomic coordinates (start, end)
		replicate : int
			Replicate number (1 or 2)
		"""
		self.chromosome = chrom
		self.span = span
		self.replicate = replicate
		
		# Initialize manager and load data
		manager = RNASeqIntermediateManager(
			output_directory=self.output_directory, 
			chromosome=chrom
		)
		watson_r1, crick_r1, watson_r2, crick_r2 = manager.load_both_replicates_pileups()
		
		# Select the appropriate replicate
		if replicate == 1:
			watson_full = watson_r1
			crick_full = crick_r1
		elif replicate == 2:
			watson_full = watson_r2
			crick_full = crick_r2
		else:
			raise ValueError("Replicate must be 1 or 2")
		
		# Select the genomic span
		self.watson_data = watson_full[range(span[0], span[1])]
		self.crick_data = crick_full[range(span[0], span[1])]
		
	def plot_pileup(self, ax=None, mode='timepoints', 
		smooth=True):
		"""
		Plot the Watson and Crick pileup data.
		
		Parameters:
		-----------
		ax : matplotlib.axes.Axes, optional
			Axes to plot on. If None, current axes will be used.
		mode : str
			Plotting mode: 'timepoints' or 'minmax'
			- 'timepoints': Plot each timepoint as separate lines
			- 'minmax': Plot min/max range with median line
			
		Returns:
		--------
		ax : matplotlib.axes.Axes
			The axes object containing the plot
		"""
		if self.watson_data is None or self.crick_data is None:
			raise ValueError("Data not loaded. Call set_chrom_span() first.")
		
		if ax is None:
			plt.figure(figsize=(13, 1))
			ax = plt.gca()
		
		from src.pileup_helpers import smooth_rna_curve

		timepoints = self.watson_data.index
		xs = self.watson_data.columns
		
		if mode == 'timepoints':
			# Plot each timepoint with color gradients
			reds = [plt.cm.Reds(i/len(timepoints)) for i in range(len(timepoints))]
			blues = [plt.cm.Blues(i/len(timepoints)) for i in range(len(timepoints))]
			
			for i, time in enumerate(timepoints):

				watson_values = self.watson_data.loc[time]
				crick_values = self.crick_data.loc[time]

				if smooth:
					watson_values = smooth_rna_curve(watson_values)
					crick_values = smooth_rna_curve(crick_values)

				watson_values = np.log2(watson_values+1)
				crick_values = np.log2(crick_values+1)

				ax.plot(xs, watson_values, c=blues[i])
				ax.plot(xs, -crick_values, c=reds[i])
				
		elif mode == 'minmax':
			# Plot min/max ranges with mean lines
			watson_lower = np.quantile(self.watson_data, q=0., axis=0)
			watson_upper = np.quantile(self.watson_data, q=1., axis=0)
			crick_lower = np.quantile(self.crick_data, q=0., axis=0)
			crick_upper = np.quantile(self.crick_data, q=1., axis=0)

			watson_med = self.watson_data.mean(0)
			crick_med = self.crick_data.mean(0)

			if smooth:
				watson_lower = smooth_rna_curve(watson_lower)
				watson_upper = smooth_rna_curve(watson_upper)
				crick_lower = smooth_rna_curve(crick_lower)
				crick_upper = smooth_rna_curve(crick_upper)

				watson_med = smooth_rna_curve(watson_med)
				crick_med = smooth_rna_curve(crick_med)

			# log transform
			watson_upper = np.log2(watson_upper+1)
			watson_lower = np.log2(watson_lower+1)
			watson_med = np.log2(watson_med+1)

			crick_upper = np.log2(crick_upper+1)
			crick_lower = np.log2(crick_lower+1)
			crick_med = np.log2(crick_med+1)

			ax.fill_between(xs, watson_upper, 
						   watson_lower, color=plt.cm.Blues(0.2), lw=0)
			ax.plot(xs, watson_med, color=plt.cm.Blues(0.75))
			
			ax.fill_between(xs, -crick_upper, 
						   -crick_lower, color=plt.cm.Reds(0.2), lw=0)
			ax.plot(xs, -crick_med, color=plt.cm.Reds(0.75))
			
		else:
			raise ValueError("Mode must be 'timepoints' or 'minmax'")
		
		# Set axis properties
		ax.set_ylim(-20, 20)
		ax.set_xlim(*self.span)
		
		return ax
	