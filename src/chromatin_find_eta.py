import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

from src.config import load_default_chrom_configs
from src.combined_chromatin_model import CombinedChromatinModel
from src.timer import Timer


class EtaOptimizer:
	"""
	Class for optimizing eta values in chromatin deconvolution analysis.
	
	Performs linear eta search to find optimal regularization parameter
	that ensures halted cells remain similar to recovery G1 (RG1) cells.
	"""
	
	# Target similarity parameters
	TARGET_MAX_DIFFERENCE = 0.15  # 15% max relative difference (as fraction, not percentage)
	DEFAULT_SEARCH_POINTS = 20    # Number of points in linear search
	
	def __init__(self, output_dir, save_dir, gamma, kappa):
		"""
		Initialize the eta optimizer.
		
		Parameters:
		-----------
		output_dir : str
			Directory containing the chromatin model output data
		save_dir : str
			Directory to save results (dataframes and plots)
		gamma : float
			Fixed gamma value to use during eta optimization
		kappa : float
			Fixed kappa value to use during eta optimization
		"""
		self.output_dir = Path(output_dir)
		self.save_dir = Path(save_dir)
		self.gamma = gamma
		self.kappa = kappa
		
		# Create save directory if it doesn't exist
		self.save_dir.mkdir(parents=True, exist_ok=True)
		
		# Initialize chromatin model
		config1, config2 = load_default_chrom_configs()
		self.combined_model = CombinedChromatinModel(config1=config1, config2=config2)
		self.config1 = config1
		self.config2 = config2
		
		# Current analysis parameters
		self.chrom = None
		self.mnase_span = None
		self.current_results = None
		
		# Search parameters
		self.eta_upper_bound = None
		self.optimal_eta = None
		
	def set_chromosome_span(self, chrom, mnase_span):
		"""
		Set the chromosome and genomic span for analysis.
		
		Parameters:
		-----------
		chrom : int
			Chromosome number
		mnase_span : tuple
			(start, end) genomic coordinates
		"""
		self.chrom = chrom
		self.mnase_span = mnase_span
		
		# Load the data for this region
		self.combined_model.load_mnase_span(chrom, mnase_span)
		self.combined_model.setup_deconv_model(copy_correct=False)
		
	def compute_solution(self, eta):
		"""
		Compute solution for given eta and return (solution, sn, rn, rg1_halted_difference).
		
		Parameters:
		-----------
		eta : float
			Eta value for regularization
			
		Returns:
		--------
		tuple
			(solution, sn, rn, rg1_halted_difference)
		"""
		# Run deconvolution with current eta (gamma and kappa are fixed)
		self.combined_model.deconvolve(gamma=self.gamma, kappa=self.kappa, eta=eta)
		F = self.combined_model.solver.F
		
		# Get indices for RG1 and halted phases
		f_rg1 = self.config1.get_Hpositions_for_phase('RG1')
		f_halted = self.config1.get_Hpositions_for_phase('Halted')
		
		# Calculate similarity metric (matches the regularization term exactly)
		rg1_mean = np.mean(F[f_rg1])
		halted_value = F[f_halted[0]].mean() # Should be single value
		rg1_halted_difference = abs(rg1_mean - halted_value)
		
		# Extract other metrics
		sn = self.combined_model.solver.sn
		rn = self.combined_model.solver.rn
		
		return F, sn, rn, rg1_halted_difference
		
	def find_eta_upper_bound(self, eta_test_values=[1.0, 5.0, 10.0, 50.0], tolerance=1e-4, verbose=True):
		"""
		Find eta upper bound where RG1-halted difference becomes very small.
		
		Parameters:
		-----------
		eta_test_values : list, default=[1.0, 5.0, 10.0, 50.0]
			Test values to find where difference becomes small
		tolerance : float, default=1e-4
			Threshold for "very small" difference
		verbose : bool, default=True
			Print progress information
			
		Returns:
		--------
		float
			Upper bound eta value
		"""
		if verbose:
			print(f"\tFinding eta upper bound (target difference < {tolerance:.1e})...")
			
		for eta_test in eta_test_values:
			_, _, _, difference = self.compute_solution(eta_test)
			
			if verbose:
				print(f"\t\tη={eta_test}: RG1-halted difference = {difference:.6f}")
				
			if difference < tolerance:
				self.eta_upper_bound = eta_test
				if verbose:
					print(f"\t\tUpper bound found: η={eta_test}")
				return eta_test
		
		# If no test value achieved tolerance, use the highest test value
		self.eta_upper_bound = eta_test_values[-1]
		if verbose:
			print(f"\t\tUsing maximum test value as upper bound: η={self.eta_upper_bound}")
		return self.eta_upper_bound
		
	def find_optimal_eta_linear(self, 
								eta_min=0.0,
								eta_max=None,
								num_points=None,
								verbose=True,
								save_results=True):
		"""
		Find optimal eta using linear search approach.
		
		Parameters:
		-----------
		eta_min : float, default=0.0
			Minimum eta value for search range
		eta_max : float, optional
			Maximum eta value for search range (if None, will find upper bound)
		num_points : int, optional
			Number of points in linear search (default uses class constant)
		verbose : bool, default=True
			Print progress information
		save_results : bool, default=True
			Save dataframe and plot to disk
			
		Returns:
		--------
		tuple
			(optimal_eta, results_df)
		"""
		if self.chrom is None or self.mnase_span is None:
			raise ValueError("Must set chromosome and span before running eta optimization")
			
		if num_points is None:
			num_points = self.DEFAULT_SEARCH_POINTS
			
		if verbose:
			print(f"Running linear eta optimization on chr{self.chrom}:{self.mnase_span[0]}-{self.mnase_span[1]}")
			print(f"Using gamma={self.gamma}, kappa={self.kappa}")
		
		timer = Timer() if verbose else None
		
		# Step 1: Find upper bound if not provided
		if eta_max is None:
			if verbose and timer:
				timer.print_time("Step 1: Finding eta upper bound")
			eta_max = self.find_eta_upper_bound(verbose=verbose)
		else:
			self.eta_upper_bound = eta_max
		
		# Step 2: Linear search
		if verbose and timer:
			timer.print_time("Step 2: Linear search for optimal eta")
		self.optimal_eta, results_df = self._log_search(eta_min, eta_max, num_points, verbose, 
			timer=timer)
		
		# Step 3: Save results
		if save_results:
			self._save_linear_results(results_df)
			
		if verbose:
			print(f"\nLinear eta optimization completed.")
			print(f"Optimal eta: {self.optimal_eta:.6g}")
			print(f"Results shape: {results_df.shape}")
			
		return self.optimal_eta, results_df
		
	def _log_search(self, eta_min, eta_max, num_points, verbose=True, timer=None):
		"""
		Perform log2 search between eta_min and eta_max.
		
		Parameters:
		-----------
		eta_min : float
			Minimum eta value
		eta_max : float
			Maximum eta value
		num_points : int
			Number of points to test
		verbose : bool, default=True
			Print progress information
			
		Returns:
		--------
		tuple
			(optimal_eta, results_df)
		"""
		# Create linearly spaced eta array

		log_eta_start, log_eta_end = np.log2(eta_min+1e-5), np.log2(eta_max)
		log_eta_vals = np.linspace(log_eta_start, log_eta_end, num_points)
		eta_array = 2**log_eta_vals
		
		if verbose:
			print(f"\t\tLog search using {num_points} points between η={eta_min:.3f} and η={eta_max:.3f}")
		
		# Collect results
		rg1_halted_differences = []
		sns = []
		rns = []
		solutions = []
		
		for i, eta in enumerate(eta_array):
			solution, sn, rn, rg1_halted_difference = self.compute_solution(eta)
			
			rg1_halted_differences.append(rg1_halted_difference)
			sns.append(sn)
			rns.append(rn)
			solutions.append(solution)
			
			if verbose:
				if timer is not None:
					time_str = " - " + timer.get_time()
				else:
					time_str = ""
				print(f"\t\tη: {eta:.6g}\tdifference: {rg1_halted_difference:.6f}\trn: {rn:.4f}\tsn: {sn:.4f}" +
					time_str)
		
		# Create results dataframe
		results_df = pd.DataFrame({
			'eta': eta_array,
			'rg1_halted_diff': rg1_halted_differences,
			'rn': rns,
			'sn': sns,
		})
		results_df.set_index('eta', inplace=True)

		from kneed import KneeLocator
		kn = KneeLocator(eta_array, rg1_halted_differences, curve='convex', direction='decreasing')
		optimal_eta = eta_array[kn.maxima_indices[0]]
		
		if verbose:
			print(f"\t\tOptimal eta found: η={optimal_eta:.6g} "
				f"(difference={results_df.loc[optimal_eta, 'rg1_halted_diff']:.6f})")
		
		self.optimal_eta = optimal_eta
		self.current_results = results_df
		
		return optimal_eta, results_df
		
	def _save_linear_results(self, results_df):
		"""Save linear search results dataframe and plot to disk."""
		# Generate filename based on current analysis
		base_filename = f"eta_linear_chr{self.chrom}_{self.mnase_span[0]}_{self.mnase_span[1]}_gamma{self.gamma}_kappa{self.kappa}"
		
		# Save dataframe
		csv_path = self.save_dir / f"{base_filename}.csv"
		results_df.to_csv(csv_path)
		print(f"Saved results to: {csv_path}")
		
		# Create and save plot
		fig_path = self.save_dir / f"{base_filename}_plot.png"
		self._create_linear_plot(results_df, save_path=fig_path)
		print(f"Saved plot to: {fig_path}")
		
		# Save summary statistics
		summary_path = self.save_dir / f"{base_filename}_summary.txt"
		self._save_summary_stats(summary_path)
		print(f"Saved summary to: {summary_path}")
		
	def _create_linear_plot(self, results_df, save_path=None, figsize=(12, 4)):
		"""
		Create linear eta search visualization plot.
		
		Parameters:
		-----------
		results_df : pd.DataFrame
			Results from linear eta search
		save_path : str or Path, optional
			Path to save the plot
		figsize : tuple, default=(12, 4)
			Figure size
		"""
		fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
		
		# Plot 1: RG1-Halted difference vs eta
		ax1.plot(results_df.index, results_df['rg1_halted_diff'], 'o-b', linewidth=2, markersize=4)
		ax1.set_xlabel('RG1-Halted regularization, η')
		ax1.set_ylabel('RG1-Halted relative\ndifference')
		ax1.grid(True, alpha=0.3)
		ax1.set_title('Similarity vs η')
		
		# Highlight optimal eta
		if self.optimal_eta is not None:
			optimal_diff = results_df.loc[self.optimal_eta, 'rg1_halted_diff']
			ax1.scatter(self.optimal_eta, optimal_diff, c='red', s=50, zorder=5)
			ax1.annotate(f'Optimal η={self.optimal_eta:.3f}\ndiff={optimal_diff:.4f}', 
						(self.optimal_eta, optimal_diff),
						xytext=(10, 10), textcoords='offset points',
						bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.3))
		
		# Plot 2: Smoothing norm vs Residual norm
		ax2.plot(results_df['sn'], results_df['rn'], 'o-r', linewidth=2, markersize=4)
		ax2.set_xlabel('Smoothing norm (sn)')
		ax2.set_ylabel('Residual norm (rn)')
		ax2.grid(True, alpha=0.3)
		ax2.set_title('SN vs RN Trade-off')
		
		# Highlight optimal point
		if self.optimal_eta is not None:
			optimal_sn = results_df.loc[self.optimal_eta, 'sn']
			optimal_rn = results_df.loc[self.optimal_eta, 'rn']
			ax2.scatter(optimal_sn, optimal_rn, c='red', s=50, zorder=5)
		
		# Plot 3: Eta vs both SN and RN
		ax3_twin = ax3.twinx()
		
		line1 = ax3.plot(results_df.index, results_df['sn'], 'g-o', linewidth=2, markersize=3, label='SN')
		line2 = ax3_twin.plot(results_df.index, results_df['rn'], 'r-s', linewidth=2, markersize=3, label='RN')
		
		ax3.set_xlabel('Eta (η)')
		ax3.set_ylabel('Smoothing Norm', color='green')
		ax3_twin.set_ylabel('Residual Norm', color='red')
		ax3.grid(True, alpha=0.3)
		ax3.set_title('Model Quality vs η')
		
		# Highlight optimal eta
		if self.optimal_eta is not None:
			ax3.axvline(x=self.optimal_eta, color='blue', linestyle='--', alpha=0.7, 
					   label=f'Optimal η')
		
		# Combine legends
		lines1, labels1 = ax3.get_legend_handles_labels()
		lines2, labels2 = ax3_twin.get_legend_handles_labels()
		ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
		
		plt.suptitle(f'Linear Eta Search: Chr{self.chrom}:{self.mnase_span[0]}-{self.mnase_span[1]} (γ={self.gamma}, κ={self.kappa})')
		plt.tight_layout()
		
		if save_path:
			plt.savefig(save_path, dpi=150, bbox_inches='tight')
			
	def _save_summary_stats(self, summary_path):
		"""Save summary statistics to text file."""
		with open(summary_path, 'w') as f:
			f.write(f"Linear Eta Optimization Summary\n")
			f.write(f"===============================\n\n")
			f.write(f"Region: Chr{self.chrom}:{self.mnase_span[0]}-{self.mnase_span[1]}\n")
			f.write(f"Fixed Parameters:\n")
			f.write(f"  Gamma: {self.gamma}\n")
			f.write(f"  Kappa: {self.kappa}\n\n")
			
			f.write(f"Search Parameters:\n")
			f.write(f"  Upper bound: {self.eta_upper_bound}\n")
			f.write(f"  Target max difference: {self.TARGET_MAX_DIFFERENCE:.3f}\n")
			if self.current_results is not None:
				f.write(f"  Points tested: {len(self.current_results)}\n\n")
			
			f.write(f"Optimal Selection:\n")
			f.write(f"  Optimal eta: {self.optimal_eta:.6g}\n")
			if self.current_results is not None and self.optimal_eta is not None:
				optimal_diff = self.current_results.loc[self.optimal_eta, 'rg1_halted_diff']
				optimal_sn = self.current_results.loc[self.optimal_eta, 'sn']
				optimal_rn = self.current_results.loc[self.optimal_eta, 'rn']
				f.write(f"  RG1-Halted difference: {optimal_diff:.6f}\n")
				f.write(f"  Smoothing norm: {optimal_sn:.6f}\n")
				f.write(f"  Residual norm: {optimal_rn:.6f}\n")
			
			f.write(f"\nSelection Criterion: Lowest eta yielding ≤{self.TARGET_MAX_DIFFERENCE:.3f} difference\n")
			f.write(f"Biological Goal: Halted cells similar to RG1 cells with minimal constraint\n")
			
	def plot_current_results(self, figsize=(12, 4)):
		"""Plot results from the most recent linear eta search."""
		if self.current_results is None:
			raise ValueError("No results to plot. Run linear eta search first.")
		self._create_linear_plot(self.current_results, figsize=figsize)
		
	def get_summary_stats(self):
		"""
		Get summary statistics from the current linear search results.
		
		Returns:
		--------
		dict
			Summary statistics including optimal eta, difference ranges, etc.
		"""
		if self.current_results is None:
			raise ValueError("No results available. Run linear eta search first.")
			
		df = self.current_results
		
		summary = {
			'optimal_eta': self.optimal_eta,
			'eta_upper_bound': self.eta_upper_bound,
			'target_max_difference': self.TARGET_MAX_DIFFERENCE,
			'total_points_tested': len(df),
			'difference_range': (df['rg1_halted_diff'].min(), df['rg1_halted_diff'].max()),
			'sn_range': (df['sn'].min(), df['sn'].max()),
			'rn_range': (df['rn'].min(), df['rn'].max()),
			'fixed_gamma': self.gamma,
			'fixed_kappa': self.kappa
		}
		
		if self.optimal_eta is not None:
			summary['optimal_difference'] = df.loc[self.optimal_eta, 'rg1_halted_diff']
			summary['optimal_sn'] = df.loc[self.optimal_eta, 'sn']
			summary['optimal_rn'] = df.loc[self.optimal_eta, 'rn']
			
			# Count how many etas meet the target
			meeting_target = df[df['rg1_halted_diff'] <= self.TARGET_MAX_DIFFERENCE]
			summary['etas_meeting_target'] = len(meeting_target)
			summary['percentage_meeting_target'] = len(meeting_target) / len(df) * 100
		
		return summary
		
	def load_previous_results(self, csv_path):
		"""
		Load previously saved results from CSV.
		
		Parameters:
		-----------
		csv_path : str or Path
			Path to the CSV file with previous results
		"""
		self.current_results = pd.read_csv(csv_path, index_col='eta')
		print(f"Loaded previous results from: {csv_path}")
		return self.current_results
	
	def plot_rg1_halted_comparison(self, eta_values=None, figsize=(10, 6)):
		"""
		Plot comparison of RG1 and halted cell profiles at different eta values.
		
		Parameters:
		-----------
		eta_values : list, optional
			List of eta values to compare (default: [0.0, optimal_eta, upper_bound])
		figsize : tuple, default=(10, 6)
			Figure size
		"""
		if eta_values is None:
			eta_values = [0.0]
			if self.optimal_eta is not None:
				eta_values.append(self.optimal_eta)
			if self.eta_upper_bound is not None and self.eta_upper_bound != self.optimal_eta:
				eta_values.append(self.eta_upper_bound)
		
		fig, axes = plt.subplots(1, len(eta_values), figsize=figsize)
		if len(eta_values) == 1:
			axes = [axes]
		
		for i, eta in enumerate(eta_values):
			solution, _, _, difference = self.compute_solution(eta)
			
			# Get RG1 and halted indices and timepoints
			f_rg1 = self.config1.get_Hpositions_for_phase('RG1')
			f_halted = self.config1.get_Hpositions_for_phase('Halted')
			rg1_tps = self.config1.get_timepoints_for_phase('RG1')
			halted_tps = self.config1.get_timepoints_for_phase('Halted')
			
			# Plot RG1 profile
			axes[i].plot(rg1_tps, solution[f_rg1], 'b-o', label='RG1 cells', linewidth=2, markersize=4)
			
			# Plot halted cell (single point)
			axes[i].plot(halted_tps, solution[f_halted], 'r-s', label='Halted cells', 
						linewidth=2, markersize=6)
			
			# Add mean line for RG1
			rg1_mean = np.mean(solution[f_rg1])
			axes[i].axhline(y=rg1_mean, color='blue', linestyle='--', alpha=0.5, 
						   label=f'RG1 mean ({rg1_mean:.3f})')
			
			axes[i].set_title(f'η = {eta:.3f}\nDifference = {difference:.4f}')
			axes[i].set_xlabel('Time')
			axes[i].set_ylabel('Chromatin Occupancy')
			axes[i].legend(fontsize=8)
			axes[i].grid(True, alpha=0.3)
		
		plt.suptitle(f'RG1 vs Halted Cell Comparison\nChr{self.chrom}:{self.mnase_span[0]}-{self.mnase_span[1]}')
		plt.tight_layout()
		plt.show()