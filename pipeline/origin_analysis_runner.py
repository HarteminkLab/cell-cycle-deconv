import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pipeline.deconvolved_origin_analysis import DeconvolvedOriginAnalysis
from pipeline.origin_consensus_correlation import OriginConsensusCorrelationAnalysis
from src.config import load_default_chrom_configs

config1, config2 = load_default_chrom_configs()

class DeconvolvedOriginAnalysisRunner:
	"""
	Runner class to execute the deconvolved origins of replication analysis workflow
	and save results to disk.
	
	This class automates the workflow shown in the notebook for analyzing 
	origins of replication, creating consensus origins, and performing correlation
	analyses against this consensus to characterize early and late firing origins.
	"""
	
	def __init__(self, chromatin_data_directory, save_directory):
		"""
		Initialize the runner with paths for data and results.
		
		Args:
			chromatin_data_directory (str): Path to the chromatin deconvolution data
			save_directory (str): Path where results and figures will be saved
		"""
		self.chromatin_data_directory = chromatin_data_directory
		self.save_directory = save_directory
		self.deconv_analysis = None
		self.consensus_analysis = None
		self.results = None
		self.histogram_data = None
		
		# Create save directory if it doesn't exist
		os.makedirs(save_directory, exist_ok=True)
		
		# Initialize timestamp for logging
		self.start_time = datetime.now()
		
	def initialize_analyses(self):
		"""Initialize the analysis objects and load data"""
		print(f"[{self._get_elapsed_time()}] Initializing analysis objects...")
		
		# Initialize the deconvolved origin analysis
		self.deconv_analysis = DeconvolvedOriginAnalysis(
			chromatin_data_path=self.chromatin_data_directory
		)
		
		# Initialize the consensus correlation analysis
		self.consensus_analysis = OriginConsensusCorrelationAnalysis(
			self.deconv_analysis
		)
		
		print(f"[{self._get_elapsed_time()}] Analysis objects initialized.")
		
	def generate_histogram_data(self):
		"""Generate the summary histogram data for all timepoints"""
		print(f"[{self._get_elapsed_time()}] Generating summary histogram data...")
		
		self.histogram_data = self.deconv_analysis.generate_summary_histogram_data()
		self.consensus_analysis.initialize_data()
		
		print(f"[{self._get_elapsed_time()}] Summary histogram data generated.")
		return self.histogram_data
	
	def compute_correlations(self, shift_range=(-10, 10), time_indices=None):
		"""
		Compute correlations between origins and consensus with optimal shifts
		
		Args:
			shift_range (tuple): Range of shifts to consider (min, max)
			time_indices (list): List of time indices to use for correlation
		
		Returns:
			dict: Results of correlation analysis
		"""
		print(f"[{self._get_elapsed_time()}] Computing correlations...")
		
		self.results = self.consensus_analysis.compute_correlations(
			shift_range=shift_range,
			find_optimal_shift=True,
			time_indices=time_indices
		)
		
		# Save the correlation results
		results_file = os.path.join(self.save_directory, 'correlation_results.pkl')
		pd.to_pickle(self.results, results_file)
		print(f"[{self._get_elapsed_time()}] Correlation results saved to {results_file}")
		
		return self.results
	
	def generate_and_save_figures(self, time_indices=None):
		"""
		Generate all figures and save them to the save directory
		
		Args:
			time_indices (list): List of time indices to use for plots
		"""
		print(f"[{self._get_elapsed_time()}] Generating and saving figures...")
		
		# 1. Small fragment occupancy comparison
		self.deconv_analysis.plot_small_fragment_occupancy_comparison()
		self._save_figure('small_fragment_occupancy_comparison')
		
		# 2. Composite histograms
		self.deconv_analysis.plot_composite_histograms(
			't', max_value=2000, highlight_center=True
		)
		self._save_figure('composite_histograms')
		
		# 3. Distribution plots
		self.consensus_analysis.plot_distributions()
		self._save_figure('correlation_distributions')
		
		# 4. Correlation heatmap
		self.consensus_analysis.plot_origin_correlation_heatmap(
			time_indices=time_indices
		)
		self._save_figure('origin_correlation_heatmap')
		
		# 5. Early origin example
		early_origin_id = 'oridb_10'  # Can be parameterized if needed
		fig1 = self.consensus_analysis.plot_consensus_vs_origin(
			origin_id=early_origin_id,
			time_indices=time_indices, 
			title="early firing"
		)
		self._save_figure(f'early_origin_{early_origin_id}_consensus_comparison')
		
		fig2 = self.deconv_analysis.plot_origin_heatmap(early_origin_id, 't')
		self._save_figure(f'early_origin_{early_origin_id}_heatmap')
		
		# 6. Late origin example
		late_origin_id = 'oridb_498'  # Can be parameterized if needed
		fig1 = self.consensus_analysis.plot_consensus_vs_origin(
			origin_id=late_origin_id, 
			time_indices=time_indices, 
			title="late firing"
		)
		self._save_figure(f'late_origin_{late_origin_id}_consensus_comparison')
		
		fig2 = self.deconv_analysis.plot_origin_heatmap(late_origin_id, 't')
		self._save_figure(f'late_origin_{late_origin_id}_heatmap')
		
		print(f"[{self._get_elapsed_time()}] All figures saved to {self.save_directory}")
	
	def analyze_all_origins(self, generate_data=True, time_indices=None):
		"""
		Run the full analysis pipeline and save all results
		
		Args:
			time_indices (list): Optional list of time indices to use for analyses
		"""
		print(f"[{self._get_elapsed_time()}] Starting full analysis pipeline...")
		
		if generate_data:
			# 1. Initialize analysis objects
			self.initialize_analyses()
			
			# 2. Generate histogram data
			self.generate_histogram_data()
			
			# 3. Compute correlations
			self.compute_correlations(time_indices=time_indices)
		
		# 4. Generate and save all figures
		self.generate_and_save_figures(time_indices=time_indices)
		
		print(f"[{self._get_elapsed_time()}] Full analysis pipeline completed.")
	
	def get_early_and_late_origins(self, n=10):
		"""
		Extract the top N early and late firing origins based on correlation shift
		
		Args:
			n (int): Number of early/late origins to return
			
		Returns:
			tuple: (early_origins, late_origins) DataFrames
		"""
		if self.results is None:
			raise ValueError("Correlations have not been computed yet.")
		
		shifts_df = pd.DataFrame(self.results['optimal_shifts'], 
								index=['shift']).T.reset_index()
		shifts_df.columns = ['origin_id', 'shift']
		
		# Sort by shift (negative = early, positive = late)
		early_origins = shifts_df.sort_values('shift').head(n)
		late_origins = shifts_df.sort_values('shift', ascending=False).head(n)
		
		# Save results to CSV
		early_origins.to_csv(os.path.join(self.save_directory, f'top_{n}_early_origins.csv'), index=False)
		late_origins.to_csv(os.path.join(self.save_directory, f'top_{n}_late_origins.csv'), index=False)
		
		print(f"[{self._get_elapsed_time()}] Top {n} early and late origins saved to CSV.")
		return early_origins, late_origins
	
	def _save_figure(self, name):
		"""Save a figure to the save directory with the given name"""
		from src.figure_configs import save_figure_for_paper
		filepath = os.path.join(self.save_directory, f"{name}.png")
		save_figure_for_paper(filepath)
		return filepath
	
	def _get_elapsed_time(self):
		"""Get elapsed time since initialization as a string"""
		elapsed = datetime.now() - self.start_time
		return str(elapsed).split('.')[0]  # Remove microseconds
