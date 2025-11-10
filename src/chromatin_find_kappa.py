import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

from src.config import load_default_chrom_configs
from src.combined_chromatin_model import CombinedChromatinModel
from src.dg1_mg1_island_analysis import ChromatinOccupancyIslandDetector
from src.timer import Timer


class ChromatinKappaOptimizer:
	"""
	Class for optimizing kappa values in chromatin deconvolution analysis.
	
	Performs focused kappa search analysis to find optimal regularization parameter
	by measuring dynamic chromatin percentage (mother/daughter-specific regions).
	"""
	
	# Boundary calculation parameters (similar to GammaOptimizer)
	RIGHT_DYNAMICS_THRESHOLD = 0.001  # 0.001% dynamics for right boundary
	FOCUSED_SEARCH_BINS = 20  # Number of points in focused search
	TARGET_DYNAMICS_MIN = 2.0  # Minimum target dynamics percentage
	TARGET_DYNAMICS_MAX = 5.0  # Maximum target dynamics percentage
	
	def __init__(self, output_dir, save_dir, gamma=0.07):
		"""
		Initialize the kappa optimizer.
		
		Parameters:
		-----------
		output_dir : str
			Directory containing the chromatin model output data
		save_dir : str
			Directory to save results (dataframes and plots)
		gamma : float, default=0.07
			Fixed gamma value to use during kappa optimization
		"""
		self.output_dir = Path(output_dir)
		self.save_dir = Path(save_dir)
		self.gamma = gamma
		
		# Create save directory if it doesn't exist
		self.save_dir.mkdir(parents=True, exist_ok=True)
		
		# Initialize chromatin model
		config1, config2 = load_default_chrom_configs()
		self.combined_model = CombinedChromatinModel(config1=config1, config2=config2,
			output_dir=self.output_dir)
		self.config1 = config1
		self.config2 = config2
		
		# Current analysis parameters
		self.chrom = None
		self.mnase_span = None
		self.current_results = None
		
		# Island detection parameters (set as defaults)
		self.island_params = {
			'min_occupancy': 1e-3,
			'std_multiplier': 1,
			'max_distance': 4,
			'min_size': 3,
			'occupancy_threshold': 5
		}
		
		# Focused search results storage
		self.max_dynamics = None
		self.min_dynamics = None
		self.left_threshold = None
		self.right_threshold = None
		self.kappa_left = None
		self.kappa_right = None
		self.optimal_kappa = None
		
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
		
	def compute_solution(self, kappa, eta=0):
		"""
		Compute solution for given kappa and return (solution, sn, rn, dynamic_percentage).
		This follows the same pattern as GammaOptimizer's compute_solution function.
		
		Parameters:
		-----------
		kappa : float
			Kappa value for regularization
		eta : float, default=0
			Eta parameter for deconvolution
			
		Returns:
		--------
		tuple
			(solution, sn, rn, dynamic_percentage)
		"""
		# Run deconvolution with current kappa
		self.combined_model.deconvolve(gamma=self.gamma, kappa=kappa, eta=eta)
		F = self.combined_model.solver.F
		
		# Run island detection analysis
		detector = ChromatinOccupancyIslandDetector(self.config1, self.config2, F)
		detector.step1_preprocess_data(min_occupancy=self.island_params['min_occupancy'])
		detector.step2_analyze_distribution(std_multiplier=self.island_params['std_multiplier'], plot=False)
		detector.detect_all_island_types(
			max_distance=self.island_params['max_distance'],
			min_size=self.island_params['min_size'],
			occupancy_threshold=self.island_params['occupancy_threshold']
		)
		
		# Extract metrics
		stats = detector.calculate_chromatin_composition()
		dynamic_percentage = stats['dynamics']['dynamic_percent']
		sn = self.combined_model.solver.sn
		rn = self.combined_model.solver.rn
		
		return F, sn, rn, dynamic_percentage
		
	def calculate_dynamic_range(self, kappa_min=1e-5, kappa_max=1.0, verbose=True):
		"""
		Calculate the maximum and minimum achievable dynamic percentages.
		Similar to calculate_base_error() in GammaOptimizer.
		
		Parameters:
		-----------
		kappa_min : float, default=1e-5
			Minimum kappa value to test
		kappa_max : float, default=1.0
			Maximum kappa value to test
		verbose : bool, default=True
			Print progress information
			
		Returns:
		--------
		tuple
			(max_dynamics, min_dynamics)
		"""
		if verbose:
			print(f"\tCalculating dynamic range...")
			
		# Test minimum kappa (should give maximum dynamics)
		_, _, _, max_dynamics = self.compute_solution(kappa_min)
		
		# Test maximum kappa (should give minimum dynamics)
		_, _, _, min_dynamics = self.compute_solution(kappa_max)
		
		self.max_dynamics = max_dynamics
		self.min_dynamics = min_dynamics
		
		if verbose:
			print(f"\tDynamic range:")
			print(f"\t\tMax dynamics (κ={kappa_min:.1e}): {max_dynamics:.3f}%")
			print(f"\t\tMin dynamics (κ={kappa_max:.1e}): {min_dynamics:.3f}%")
			
		return max_dynamics, min_dynamics
		
	def calculate_boundary_thresholds(self, verbose=True):
		"""
		Calculate left and right boundary thresholds for focused search.
		Similar to calculate_error_boundaries() in GammaOptimizer.
		
		Parameters:
		-----------
		verbose : bool, default=True
			Print progress information
			
		Returns:
		--------
		tuple
			(left_threshold, right_threshold)
		"""
		if self.max_dynamics is None:
			raise ValueError("Must calculate dynamic range first")
			
		# Target the threshold that reaches the target max threshold (5%)
		# And is below/near zero on the right.
		self.left_threshold = self.TARGET_DYNAMICS_MAX
		self.right_threshold = self.RIGHT_DYNAMICS_THRESHOLD
		
		if verbose:
			print(f"\tBoundary thresholds:")
			print(f"\t\tLeft (5%): {self.left_threshold:.3f}%")
			print(f"\t\tRight (over-smoothing): {self.right_threshold:.3f}%")
			
		return self.left_threshold, self.right_threshold
		
	def find_kappa_for_dynamics(self, target_dynamics, kappa_low, kappa_high, 
							   tolerance=1e-6, verbose=True):
		"""
		Binary search to find kappa that achieves target dynamics percentage.
		Similar to find_gamma_for_error() in GammaOptimizer.
		
		Parameters:
		-----------
		target_dynamics : float
			Target dynamics percentage to achieve
		kappa_low : float
			Lower bound for kappa search
		kappa_high : float
			Upper bound for kappa search
		tolerance : float, default=1e-6
			Convergence tolerance for binary search
		verbose : bool, default=True
			Print progress information
			
		Returns:
		--------
		float
			Kappa value achieving target dynamics
		"""
		previous_dynamics = None
		iteration_count = 0
		
		while (kappa_high - kappa_low) > tolerance:
			kappa_mid = (kappa_low + kappa_high) / 2
			_, current_sn, current_rn, current_dynamics = self.compute_solution(kappa_mid)
			iteration_count += 1
			
			if verbose:
				print(f"\t\tκ: {kappa_mid:.6g}\tdynamics: {current_dynamics:.3f}%\trn: {current_rn:.4f}\tsn: {current_sn:.4f}")
			
			# Check if we've reached the target dynamics within tolerance
			if abs(current_dynamics - target_dynamics) < tolerance:
				return kappa_mid
			elif current_dynamics > target_dynamics:
				kappa_low = kappa_mid  # Need higher kappa to reduce dynamics
			else:
				kappa_high = kappa_mid  # Need lower kappa to increase dynamics
			
			previous_dynamics = current_dynamics
				
		return (kappa_low + kappa_high) / 2
		
	def find_boundary_kappas(self, kappa_min=1e-5, kappa_max=1.0, verbose=True):
		"""
		Find kappa values achieving left and right boundary thresholds.
		Similar to find_boundary_gammas() in GammaOptimizer.
		
		Parameters:
		-----------
		kappa_min : float, default=1e-5
			Minimum kappa value for search range
		kappa_max : float, default=1.0
			Maximum kappa value for search range
		verbose : bool, default=True
			Print progress information
			
		Returns:
		--------
		tuple
			(kappa_left, kappa_right)
		"""
		if self.left_threshold is None or self.right_threshold is None:
			raise ValueError("Must calculate boundary thresholds first")
			
		if verbose:
			print(f"\tSearching for left boundary kappa ({self.left_threshold:.3f}% dynamics)...")
				
		self.kappa_left = self.find_kappa_for_dynamics(
			self.left_threshold, kappa_min, kappa_max, verbose=verbose
		)
		
		if verbose:
			print(f"\tLeft boundary kappa found: {self.kappa_left:.6g}")
			print(f"\tSearching for right boundary kappa ({self.right_threshold:.3f}% dynamics)...")
			
		self.kappa_right = self.find_kappa_for_dynamics(
			self.right_threshold, self.kappa_left, kappa_max, verbose=verbose
		)
		
		if verbose:
			print(f"\tRight boundary kappa found: {self.kappa_right:.6g}")
			
		return self.kappa_left, self.kappa_right
		
	def find_optimal_kappa_focused(self, 
								  kappa_min=1e-5, 
								  kappa_max=1.0,
								  eta=0,
								  # Island detection parameters
								  min_occupancy=1e-3,
								  std_multiplier=1,
								  max_distance=4,
								  min_size=3,
								  occupancy_threshold=5,
								  verbose=True,
								  save_results=True):
		"""
		Find optimal kappa using focused search approach.
		This replaces the broad kappa sweep method.
		
		Parameters:
		-----------
		kappa_min : float, default=1e-5
			Minimum kappa value for search range
		kappa_max : float, default=1.0
			Maximum kappa value for search range
		eta : float, default=0
			Eta parameter for deconvolution
		min_occupancy : float, default=1e-3
			Minimum occupancy threshold for island detection
		std_multiplier : float, default=1
			Standard deviation multiplier for outlier detection
		max_distance : int, default=4
			Maximum distance for island clustering
		min_size : int, default=3
			Minimum island size
		occupancy_threshold : float, default=5
			Occupancy threshold for total island detection
		verbose : bool, default=True
			Print progress information
		save_results : bool, default=True
			Save dataframe and plot to disk
			
		Returns:
		--------
		tuple
			(optimal_kappa, results_df)
		"""
		if self.chrom is None or self.mnase_span is None:
			raise ValueError("Must set chromosome and span before running kappa optimization")
			
		# Update island detection parameters
		self.island_params.update({
			'min_occupancy': min_occupancy,
			'std_multiplier': std_multiplier,
			'max_distance': max_distance,
			'min_size': min_size,
			'occupancy_threshold': occupancy_threshold
		})
		
		if verbose:
			print(f"Running focused kappa optimization on chr{self.chrom}:{self.mnase_span[0]}-{self.mnase_span[1]}")
			print(f"Using gamma={self.gamma}")
		
		timer = Timer() if verbose else None
		
		# Step 1: Calculate dynamic range
		if verbose and timer:
			timer.print_time("Step 1: Calculating dynamic range")
		self.calculate_dynamic_range(kappa_min, kappa_max, verbose)
		
		# Step 2: Calculate boundary thresholds
		if verbose and timer:
			timer.print_time("Step 2: Calculating boundary thresholds")
		self.calculate_boundary_thresholds(verbose)
		
		# Step 3: Find boundary kappas
		if verbose and timer:
			timer.print_time("Step 3: Finding boundary kappas")
		self.find_boundary_kappas(kappa_min, kappa_max, verbose)
		
		# Step 4: Focused search between boundaries
		if verbose and timer:
			timer.print_time("Step 4: Focused search between boundaries")
		self.optimal_kappa, results_df = self._focused_search(verbose)
		
		# Step 5: Save results
		if save_results:
			self._save_focused_results(results_df)
			
		if verbose:
			print(f"\nFocused kappa optimization completed.")
			print(f"Optimal kappa: {self.optimal_kappa:.6g}")
			print(f"Results shape: {results_df.shape}")
			
		return self.optimal_kappa, results_df
		
	def _focused_search(self, verbose=True):
		"""
		Perform focused search between boundary kappas.
		Similar to find_elbow() in GammaOptimizer.
		
		Parameters:
		-----------
		verbose : bool, default=True
			Print progress information
			
		Returns:
		--------
		tuple
			(optimal_kappa, results_df)
		"""
		# Create logarithmically spaced kappa array between boundaries
		log_kp_start, log_kp_end = np.log2(self.kappa_left), np.log2(self.kappa_right)
		log_kp_vals = np.linspace(log_kp_start, log_kp_end, self.FOCUSED_SEARCH_BINS)
		kappa_array = 2**log_kp_vals
		
		if verbose:
			print(f"\tFocused search using {self.FOCUSED_SEARCH_BINS} points between κ={self.kappa_left:.6g} and κ={self.kappa_right:.6g}")
		
		# Collect results
		dynamic_percentages = []
		sns = []
		rns = []
		solutions = []
		
		for i, kappa in enumerate(kappa_array):
			solution, sn, rn, dynamic_percentage = self.compute_solution(kappa)
			
			dynamic_percentages.append(dynamic_percentage)
			sns.append(sn)
			rns.append(rn)
			solutions.append(solution)
			
			if verbose:
				print(f"\t\tκ: {kappa:.6g}\tdynamics: {dynamic_percentage:.3f}%\trn: {rn:.4f}\tsn: {sn:.4f}")
		
		# Create results dataframe
		results_df = pd.DataFrame({
			'kappa': kappa_array,
			'm_d_perc': dynamic_percentages,
			'rn': rns,
			'sn': sns,
		})
		results_df.set_index('kappa', inplace=True)
		
		# Find optimal kappa: highest kappa yielding ≥2% dynamics
		valid_kappas = results_df[(results_df['m_d_perc'] >= self.TARGET_DYNAMICS_MIN) &
			(results_df['m_d_perc'] <= self.TARGET_DYNAMICS_MAX)]
		
		if len(valid_kappas) == 0:
			# Fallback: use kappa giving closest to target minimum
			closest_idx = np.argmin(np.abs(results_df['m_d_perc'] - self.TARGET_DYNAMICS_MIN))
			optimal_kappa = results_df.index[closest_idx]
			if verbose:
				print(f"\t\tNo kappa found yielding ≥{self.TARGET_DYNAMICS_MIN}% dynamics")
				print(f"\t\tUsing closest match: κ={optimal_kappa:.6g} ({results_df.loc[optimal_kappa, 'm_d_perc']:.3f}%)")
		else:
			# Select lowest kappa < 5% dynamic differences
			optimal_kappa = valid_kappas.index.min()
			if verbose:
				print(f"\t\tOptimal kappa found: κ={optimal_kappa:.6g} ({results_df.loc[optimal_kappa, 'm_d_perc']:.3f}% dynamics)")
		
		self.current_results = results_df
		
		return optimal_kappa, results_df
		
	def _save_focused_results(self, results_df):
		"""Save focused search results dataframe and plot to disk."""
		# Generate filename based on current analysis
		base_filename = f"kappa_focused_chr{self.chrom}_{self.mnase_span[0]}_{self.mnase_span[1]}_gamma{self.gamma}"
		
		# Save dataframe
		csv_path = self.save_dir / f"{base_filename}.csv"
		results_df.to_csv(csv_path)
		print(f"Saved results to: {csv_path}")
		
		# Create and save plot
		fig_path = self.save_dir / f"{base_filename}_plot.png"
		self._create_focused_plot(results_df, save_path=fig_path)
		print(f"Saved plot to: {fig_path}")
		
		# Save summary statistics
		summary_path = self.save_dir / f"{base_filename}_summary.txt"
		self._save_summary_stats(summary_path)
		print(f"Saved summary to: {summary_path}")
		
	def _create_focused_plot(self, results_df, save_path=None, figsize=(12, 4)):
		"""
		Create focused kappa search visualization plot.
		
		Parameters:
		-----------
		results_df : pd.DataFrame
			Results from focused kappa search
		save_path : str or Path, optional
			Path to save the plot
		figsize : tuple, default=(12, 4)
			Figure size
		"""
		fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
		
		# Plot 1: Dynamic percentage vs kappa
		ax1.plot(results_df.index, results_df['m_d_perc'], 'o-b', linewidth=2, markersize=4)
		ax1.set_xscale('log')
		ax1.set_xlabel('DG1-MG1 regularization, κ')
		ax1.set_ylabel('% Mother or\nDaughter-specific')
		ax1.grid(True, alpha=0.3)
		ax1.set_title('Dynamic Chromatin vs κ')
		
		# Highlight optimal kappa
		if self.optimal_kappa:
			optimal_dynamics = results_df.loc[self.optimal_kappa, 'm_d_perc']
			ax1.scatter(self.optimal_kappa, optimal_dynamics, c='red', s=50, zorder=5)
			ax1.annotate(f'Optimal κ={self.optimal_kappa:.1e}\n{optimal_dynamics:.1f}%', 
						(self.optimal_kappa, optimal_dynamics),
						xytext=(10, 10), textcoords='offset points',
						bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.3))
		
		# Add target range
		ax1.axhline(y=self.TARGET_DYNAMICS_MIN, color='green', linestyle='--', alpha=0.5, 
				   label=f'Target min ({self.TARGET_DYNAMICS_MIN}%)')
		ax1.axhline(y=self.TARGET_DYNAMICS_MAX, color='orange', linestyle='--', alpha=0.5,
				   label=f'Target max ({self.TARGET_DYNAMICS_MAX}%)')
		ax1.legend(fontsize=8)
		
		# Plot 2: Smoothing norm vs Residual norm
		ax2.plot(results_df['sn'], results_df['rn'], 'o-r', linewidth=2, markersize=4)
		ax2.set_xlabel('Smoothing norm (sn)')
		ax2.set_ylabel('Residual norm (rn)')
		ax2.grid(True, alpha=0.3)
		ax2.set_title('SN vs RN Trade-off')
		
		# Highlight optimal point
		if self.optimal_kappa:
			optimal_sn = results_df.loc[self.optimal_kappa, 'sn']
			optimal_rn = results_df.loc[self.optimal_kappa, 'rn']
			ax2.scatter(optimal_sn, optimal_rn, c='red', s=50, zorder=5)
		
		# Plot 3: Kappa vs both SN and RN (normalized)
		ax3_twin = ax3.twinx()
		
		# Normalize values for dual y-axis plotting
		sn_norm = (results_df['sn'] - results_df['sn'].min()) / (results_df['sn'].max() - results_df['sn'].min())
		rn_norm = (results_df['rn'] - results_df['rn'].min()) / (results_df['rn'].max() - results_df['rn'].min())
		
		line1 = ax3.plot(results_df.index, sn_norm, 'g-o', linewidth=2, markersize=3, label='SN (norm)')
		line2 = ax3_twin.plot(results_df.index, rn_norm, 'r-s', linewidth=2, markersize=3, label='RN (norm)')
		
		ax3.set_xscale('log')
		ax3.set_xlabel('Kappa (κ)')
		ax3.set_ylabel('Normalized Smoothing Norm', color='green')
		ax3_twin.set_ylabel('Normalized Residual Norm', color='red')
		ax3.grid(True, alpha=0.3)
		ax3.set_title('Regularization Trade-off')
		
		# Highlight optimal kappa
		if self.optimal_kappa:
			ax3.axvline(x=self.optimal_kappa, color='blue', linestyle='--', alpha=0.7, 
					   label=f'Optimal κ')
		
		# Combine legends
		lines1, labels1 = ax3.get_legend_handles_labels()
		lines2, labels2 = ax3_twin.get_legend_handles_labels()
		ax3.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=8)
		
		plt.suptitle(f'Focused Kappa Search: Chr{self.chrom}:{self.mnase_span[0]}-{self.mnase_span[1]} (γ={self.gamma})')
		plt.tight_layout()
		
		if save_path:
			plt.savefig(save_path, dpi=150, bbox_inches='tight')
			plt.close()
		else:
			plt.show()
			
	def _save_summary_stats(self, summary_path):
		"""Save summary statistics to text file."""
		with open(summary_path, 'w') as f:
			f.write(f"Focused Kappa Optimization Summary\n")
			f.write(f"===================================\n\n")
			f.write(f"Region: Chr{self.chrom}:{self.mnase_span[0]}-{self.mnase_span[1]}\n")
			f.write(f"Gamma: {self.gamma}\n\n")
			
			f.write(f"Dynamic Range:\n")
			f.write(f"  Max dynamics: {self.max_dynamics:.3f}%\n")
			f.write(f"  Min dynamics: {self.min_dynamics:.3f}%\n\n")
			
			f.write(f"Boundary Thresholds:\n")
			f.write(f"  Left (5%): {self.left_threshold:.3f}%\n")
			f.write(f"  Right (over-smoothing): {self.right_threshold:.3f}%\n\n")
			
			f.write(f"Boundary Kappas:\n")
			f.write(f"  Left boundary: {self.kappa_left:.6g}\n")
			f.write(f"  Right boundary: {self.kappa_right:.6g}\n\n")
			
			f.write(f"Optimal Selection:\n")
			f.write(f"  Optimal kappa: {self.optimal_kappa:.6g}\n")
			if self.current_results is not None:
				optimal_dynamics = self.current_results.loc[self.optimal_kappa, 'm_d_perc']
				optimal_sn = self.current_results.loc[self.optimal_kappa, 'sn']
				optimal_rn = self.current_results.loc[self.optimal_kappa, 'rn']
				f.write(f"  Dynamic percentage: {optimal_dynamics:.3f}%\n")
				f.write(f"  Smoothing norm: {optimal_sn:.6f}\n")
				f.write(f"  Residual norm: {optimal_rn:.6f}\n")
			
			f.write(f"\nTarget Range: {self.TARGET_DYNAMICS_MIN}-{self.TARGET_DYNAMICS_MAX}%\n")
			f.write(f"Selection Criterion: Highest kappa yielding ≥{self.TARGET_DYNAMICS_MIN}%\n")
			
	def plot_current_results(self, figsize=(12, 4)):
		"""Plot results from the most recent focused kappa search."""
		if self.current_results is None:
			raise ValueError("No results to plot. Run focused kappa search first.")
		self._create_focused_plot(self.current_results, figsize=figsize)
		
	def get_summary_stats(self):
		"""
		Get summary statistics from the current focused search results.
		
		Returns:
		--------
		dict
			Summary statistics including optimal kappa, boundary information, etc.
		"""
		if self.current_results is None:
			raise ValueError("No results available. Run focused kappa search first.")
			
		df = self.current_results
		
		# Find kappa ranges for different dynamic thresholds
		thresholds = [2, 5, 10]
		kappa_ranges = {}
		
		for threshold in thresholds:
			above_threshold = df[df['m_d_perc'] >= threshold]
			if len(above_threshold) > 0:
				kappa_ranges[f'above_{threshold}pct'] = {
					'min_kappa': above_threshold.index.min(),
					'max_kappa': above_threshold.index.max(),
					'count': len(above_threshold)
				}
		
		summary = {
			'optimal_kappa': self.optimal_kappa,
			'max_dynamics_achieved': self.max_dynamics,
			'min_dynamics_achieved': self.min_dynamics,
			'left_boundary_kappa': self.kappa_left,
			'right_boundary_kappa': self.kappa_right,
			'search_range': (self.kappa_left, self.kappa_right),
			'total_points_tested': len(df),
			'dynamic_percentage_range': (df['m_d_perc'].min(), df['m_d_perc'].max()),
			'kappa_ranges_by_threshold': kappa_ranges,
			'target_range': (self.TARGET_DYNAMICS_MIN, self.TARGET_DYNAMICS_MAX)
		}
		
		if self.optimal_kappa:
			summary['optimal_dynamics'] = df.loc[self.optimal_kappa, 'm_d_perc']
			summary['optimal_sn'] = df.loc[self.optimal_kappa, 'sn']
			summary['optimal_rn'] = df.loc[self.optimal_kappa, 'rn']
		
		return summary
		
	def load_previous_results(self, csv_path):
		"""
		Load previously saved results from CSV.
		
		Parameters:
		-----------
		csv_path : str or Path
			Path to the CSV file with previous results
		"""
		self.current_results = pd.read_csv(csv_path, index_col='kappa')
		print(f"Loaded previous results from: {csv_path}")
		return self.current_results
