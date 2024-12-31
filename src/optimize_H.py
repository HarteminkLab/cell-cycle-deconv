import numpy as np
from scipy.optimize import minimize
from src.calcH_single_g1 import calcH

class ParameterOptimizer:
	def __init__(self, initial_values, model_intervals, timepoints, N, F, B, G, bounds_scale=0.2):
		"""
		Initialize optimizer with initial parameter values and matrices.
		
		Args:
			initial_values: Dictionary with initial parameter values
			model_intervals: Required for calcH
			timepoints: Required for calcH
			N: Diagonal matrix (t x t)
			F: Matrix (m x u)
			B: Diagonal matrix (u x u)
			G: Target matrix (t x u)
			bounds_scale: Scale factor for parameter bounds
		"""
		self.initial_values = initial_values
		self.model_intervals = model_intervals
		self.timepoints = timepoints
		self.N = N
		self.F = F
		self.B = B
		self.G = G
		
		# Create parameter bounds as percentage of initial values
		self.bounds = {}
		for param, value in initial_values.items():
			if param == 'halted':
				# Special case: halted must be between 0 and 1
				self.bounds[param] = (max(0, value - 0.1), min(1, value + 0.1))
			elif param == 'beta':
				# Special case: beta seems to be fixed at 0
				self.bounds[param] = (0, 0)
			else:
				# For other parameters, create bounds around initial value
				lower = max(0, value * (1 - bounds_scale))
				upper = value * (1 + bounds_scale)
				self.bounds[param] = (lower, upper)
				
		self.bounds_list = [self.bounds[param] for param in self.initial_values.keys()]

		# Custom bounds list
		# todo: determine if bounds are appropriate
		self.bounds_list[0] = (18, 28)
		self.bounds_list[1] = (55, 75)
		self.bounds_list[2] = (0, 16)
		self.bounds_list[3] = (1, 12)
		self.bounds_list[4] = (0.04, 0.25)
		self.bounds_list[5] = (0, 26)
		self.bounds_list[-1] = (0.1, 0.25)

		print("Bounds: ", self.bounds_list)

	
	def _pack_parameters(self, params_dict):
		"""Convert parameters dictionary to list in consistent order."""
		return [params_dict[param] for param in self.initial_values.keys()]
	
	def _unpack_parameters(self, params_list):
		"""Convert parameters list back to dictionary."""
		return {param: value for param, value 
				in zip(self.initial_values.keys(), params_list)}
	
	def compute_loss(self, params_dict):
		"""
		Compute loss ||NHFB - G|| where H is computed using calcH with given parameters.
		Returns infinity if parameters are invalid.
		"""
		# Pack parameters into list for calcH
		params = self._pack_parameters(params_dict)
		
		try:
			# Update model_intervals with new parameters
			model_intervals_updated = list(self.model_intervals)
			model_intervals_updated[0] = params
		
			# Compute H with new parameters
			H, _ = calcH(model_intervals_updated, self.timepoints)
			
			# Compute NHFB
			NHFB = self.N @ H @ self.F @ self.B
			loss = np.mean((NHFB - self.G)**2)

			self.current_params = params
			self.current_H = H
			self.current_loss = loss
			
			# Squared error loss
			return loss

		except Exception as e:
			print(f"Error in compute_loss: {e}")
			return float('inf')
	
	def optimize(self, method='Nelder-Mead', maxiter=1000, verbose=True, tol=1e-5):
		"""
		Optimize parameters to minimize ||NHFB - G||.
		
		Args:
			method: Optimization method (default: Nelder-Mead)
			maxiter: Maximum iterations
			verbose: Whether to print progress
		"""
		# Convert bounds to list for scipy.optimize
		bounds_list = self.bounds_list
		
		# Initial parameter values
		x0 = list(self.initial_values.values())
		
		# Define objective function
		def objective(x):
			params_dict = self._unpack_parameters(x)
			loss = self.compute_loss(params_dict)
			if verbose:
				print(f"Current loss: {loss}")
			return loss
		
		# Run optimization
		result = minimize(
			objective,
			x0,
			method=method,
			bounds=bounds_list,
			options={
				'maxiter': maxiter,
				'disp': verbose,
				'xatol': tol
			}
		)
		
		# Return optimized parameters as dictionary and final loss
		return self._unpack_parameters(result.x), result.fun
	
	def check_gradient(self, eps=1e-4):
		"""
		Compute numerical gradient to check sensitivity to each parameter.
		Useful for understanding which parameters are most important.
		"""
		x0 = list(self.initial_values.values())
		params_dict = self._unpack_parameters(x0)
		base_loss = self.compute_loss(params_dict)
		
		gradients = {}
		for i, param in enumerate(self.initial_values.keys()):
			x_plus = x0.copy()
			x_plus[i] += eps
			params_plus = self._unpack_parameters(x_plus)
			
			# Forward difference approximation
			grad = (self.compute_loss(params_plus) - base_loss) / eps
			gradients[param] = grad
			
		return gradients
