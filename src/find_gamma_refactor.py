import numpy as np
import pandas as pd
from src.utils import print_fl
from scipy.signal import savgol_filter
from matplotlib import pyplot as plt
from src.timer import Timer

class GammaOptimizer:

	# Micro-array error terms for reference, with the log-TPM values,
	# these ratios become too stringent, the right side specifically.
	
	# LEFT_ERROR_RATIO = 1.05
	# LEFT_ERROR_OFFSET = 0.04
	# RIGHT_ERROR_RATIO = 1.40 
	# RIGHT_ERROR_OFFSET = 0.32

	EXP_LEFT_ERROR_RATIO = 1.05
	EXP_LEFT_ERROR_OFFSET = 0.04

	# Higher for gene expression to allow for greater smoothing
	EXP_RIGHT_ERROR_RATIO = 4.0
	EXP_RIGHT_ERROR_OFFSET = 11.32

	# Higher right error offset for chromatin
	CHROM_LEFT_ERROR_RATIO = 1.05
	CHROM_LEFT_ERROR_OFFSET = 1.0

	# Higher right error offset for chromatin
	CHROM_RIGHT_ERROR_RATIO = 1.40 
	CHROM_RIGHT_ERROR_OFFSET = 15.0

	ELBOW_BINS = 20

	def __init__(self, compute_solution, gamma_min=0.00001, gamma_max=0.001, verbose=False,
		mode='expression', early_stop_enabled=False, early_stop_threshold=0.1):

		self.mode = mode
		self.timer = Timer()

		if mode == 'expression':
			self.LEFT_ERROR_RATIO = self.EXP_LEFT_ERROR_RATIO
			self.LEFT_ERROR_OFFSET = self.EXP_LEFT_ERROR_OFFSET
			self.RIGHT_ERROR_RATIO = self.EXP_RIGHT_ERROR_RATIO
			self.RIGHT_ERROR_OFFSET = self.EXP_RIGHT_ERROR_OFFSET
		elif mode == 'chromatin':
			self.LEFT_ERROR_RATIO = self.CHROM_LEFT_ERROR_RATIO
			self.LEFT_ERROR_OFFSET = self.CHROM_LEFT_ERROR_OFFSET
			self.RIGHT_ERROR_RATIO = self.CHROM_RIGHT_ERROR_RATIO
			self.RIGHT_ERROR_OFFSET = self.CHROM_RIGHT_ERROR_OFFSET
		else:
			raise ValueError("Invalid mode", mode)

		self.compute_solution = compute_solution
		self.gamma_min = gamma_min
		self.gamma_max = gamma_max
		self.verbose = verbose
		self.early_stop_enabled = early_stop_enabled
		self.early_stop_threshold = early_stop_threshold  # Percentage threshold (e.g., 1.0 for 1%)

	def compute_solution_dictionary(self, gamma):
		result_dictionary = self.compute_solution(gamma)
		solution, sn, rn = result_dictionary['solution_F'], \
			result_dictionary['sn'], result_dictionary['rn']

		# Return the mean of the entire window's solution norm, if
		# deconvolving the chromatin
		if self.mode == 'chromatin':
			mean_sn = result_dictionary['mean_sn']

		# For expression, use sn as a duplicate column for ease
		# of minimizing the amount of code changes
		else:
			mean_sn = sn

		return solution, sn, rn, mean_sn
		
	def calculate_base_error(self):

		base_solution, base_sn, self.base_error, mean_sn = self.compute_solution_dictionary(0)

		if self.verbose:
			print_fl(f"\tBase error (γ=0):", end="")
			print_fl(f"\trn: {self.base_error:.6f}", end="")
			print_fl(f"\tsn: {base_sn:.6f}")
			print_fl(f"\tmean sn: {mean_sn:.6f}")
		return self.base_error
		
	def calculate_error_boundaries(self):
		e0 = self.base_error
		relative_left = self.LEFT_ERROR_RATIO * e0
		absolute_left = e0 + self.LEFT_ERROR_OFFSET
		relative_right = self.RIGHT_ERROR_RATIO * e0
		absolute_right = e0 + self.RIGHT_ERROR_OFFSET
		
		self.left_error = min(relative_left, absolute_left)
		self.right_error = max(relative_right, absolute_right)
		
		if self.verbose:
			print_fl(f"\tError boundaries:")
			print_fl(f"\t\tLeft:")
			print_fl(f"\t\t\tRelative ({self.LEFT_ERROR_RATIO:.2f} * e0): {relative_left:.6g}")
			print_fl(f"\t\t\tAbsolute (e0 + {self.LEFT_ERROR_OFFSET:.2f}): {absolute_left:.6g}")
			print_fl(f"\t\t\tChosen: {self.left_error:.6f}")
			print_fl(f"\t\tRight:")
			print_fl(f"\t\t\tRelative ({self.RIGHT_ERROR_RATIO:.2f} * e0): {relative_right:.6g}")
			print_fl(f"\t\t\tAbsolute (e0 + {self.RIGHT_ERROR_OFFSET:.2f}): {absolute_right:.6g}")
			print_fl(f"\t\t\tChosen: {self.right_error:.6f}")
			
		return self.left_error, self.right_error

	def find_gamma_for_error(self, target_error, gamma_low, gamma_high, 
		tolerance=1e-6):
		"""Binary search to find gamma that achieves target error"""
		previous_error = None
		iteration_count = 0
		
		while (gamma_high - gamma_low) > tolerance:
			gamma_mid = (gamma_low + gamma_high) / 2

			current_solution, current_sn, current_error, current_mean_sn = self.compute_solution_dictionary(gamma_mid)

			iteration_count += 1
			
			if self.verbose:
				print_fl(f"\t\tγ: {gamma_mid:.6g}", end="\t")
				print_fl(f"rn: {current_error:.4f}", end="\t")
				print_fl(f"sn: {current_sn:.4f}", end="\t")
				print_fl(f"mean sn: {current_mean_sn:.4f}", end="\t")
				print_fl(f"- {self.timer.get_time()}")
			
			# Check for early stopping condition
			if self.early_stop_enabled and previous_error is not None:
				# Calculate percentage change in rn (residual norm)
				percent_change = abs((current_error - previous_error) / previous_error) * 100
				
				if percent_change < self.early_stop_threshold:
					if self.verbose:
						print_fl(f"\t\tEarly stopping: rn change ({percent_change:.3f}%) < threshold ({self.early_stop_threshold:.1f}%)")
					return gamma_mid
			
			# Check if we've reached the target error within tolerance
			if abs(current_error - target_error) < tolerance:
				return gamma_mid
			elif current_error < target_error:
				gamma_low = gamma_mid
			else:
				gamma_high = gamma_mid
			
			previous_error = current_error
				
		return (gamma_low + gamma_high) / 2

	def find_boundary_gammas(self):
		"""Find gamma values achieving left and right error boundaries"""
		if self.verbose:
			print_fl("\tSearching for left boundary gamma...")
			if self.early_stop_enabled:
				print_fl(f"\t\tEarly stopping enabled (threshold: {self.early_stop_threshold:.1f}%)")
				
		self.gamma_left = self.find_gamma_for_error(self.left_error, self.gamma_min, self.gamma_max)
		
		if self.verbose:
			print_fl(f"\tLeft boundary gamma found: {self.gamma_left:.6g}")
			print_fl("\tSearching for right boundary gamma...")
			
		self.gamma_right = self.find_gamma_for_error(self.right_error, self.gamma_left, self.gamma_max)
		
		if self.verbose:
			print_fl(f"\tRight boundary gamma found: {self.gamma_right:.6g}")
			
		return self.gamma_left, self.gamma_right


	def find_elbow(self):
		"""Find elbow point between gamma boundaries"""

		# Select gamma values to search using a log transformation, this will select
		# lower values earlier in the array to more accurately identify the elbow
		log_gm_start, log_gm_end = np.log2(self.gamma_left), np.log2(self.gamma_right)
		log_gm_vals = np.linspace(log_gm_start, log_gm_end, self.ELBOW_BINS)
		gamma_array = 2**log_gm_vals
		
		# Collect RN/SN pairs
		rn_values = []
		sn_values = []
		solutions = []
		mean_sn_values = []
		
		if self.verbose:
			print_fl(f"Finding elbow in the curve, selecting {self.ELBOW_BINS} bins from the left and right boundaries.")

		for gamma in gamma_array:

			solution, sn, rn, mean_sn = self.compute_solution_dictionary(gamma_mid)

			if self.verbose:
				print_fl(f"\t\tγ: {gamma:.6g}\trn: {rn:.4f}\tsn: {sn:.4f}\tmean_sn: {mean_sn:.4f} - {self.timer.get_time()}")
			rn_values.append(rn)
			sn_values.append(sn)
			mean_sn_values.append(mean_sn)

			solutions.append(solution)
			
		# Convert to numpy arrays for gradient calculation
		rn_values = np.array(rn_values)
		sn_values = np.array(sn_values)

		# https://stackoverflow.com/questions/51762514/find-the-elbow-point-on-an-optimization-curve-with-python
		# Kneedle algorithm
		# https://github.com/arvkevi/kneed
		# https://raghavan.usc.edu/papers/kneedle-simplex11.pdf
		from kneed import KneeLocator
		kn = KneeLocator(rn_values, sn_values, curve='convex', direction='decreasing')

		self.optimal_gamma = gamma_array[kn.maxima_indices[0]]

		if self.verbose:
			print_fl(f"\tOptimal gamma found at elbow: {self.optimal_gamma:.6g}")

		self.elbow_results_df = pd.DataFrame({
			'rn': rn_values,
			'sn': sn_values,
			'mean_sn': mean_sn_values,
			'solution_index': np.arange(len(gamma_array)),
			}, index=gamma_array)
		self.elbow_solutions = np.array(solutions)
		
		return self.optimal_gamma


	def plot_elbow(self):

		plt.figure(figsize=(11, 2))
		plt.subplot(1, 3, 1)
		plt.plot(self.elbow_results_df.rn, self.elbow_results_df.sn, '-o')
		plt.xlabel("Residual norm, rn")
		plt.ylabel("Smoothing norm, sn")

		optimal_gamma = self.optimal_gamma
		plt.scatter(self.elbow_results_df.loc[optimal_gamma].rn,
					self.elbow_results_df.loc[optimal_gamma].sn,
				   c='red', zorder=2)