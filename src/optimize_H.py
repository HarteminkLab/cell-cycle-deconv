import numpy as np
from scipy.optimize import minimize
from src.calcH_single_g1 import calcH
from src.calcH_single_g1 import get_parameter_indices

ITERATION = 0

class ParameterOptimizer:
	def __init__(self, init_params_df, config, 
			N, F, B, G):
		"""
		Initialize optimizer with initial parameter values and matrices.
		"""
		self.config = config
		self.N = N
		self.F = F
		self.B = B
		self.G = G

		# Get the indices of the parameters to be updated
		# this is needed because both config and the optimizer require
		# the parameters to be in the form of a list, we will keep track of
		# which parameters we are updating with this field
		self.parameter_indices = get_parameter_indices(init_params_df.index)

		# Keep track of the paramater values to update in this dataframe
		self.params_df = init_params_df.copy()
		self.params_df['param_index'] = self.parameter_indices
		self.params_df = self.params_df.sort_values('param_index')

		self.update_config_parameters(self.params_df)

	def update_params_df(self, parameter_values):
		# Parameters will update through optimization as a vector, so translate
		# changes to the data frame object

		# Should be a 1:1 mapping to the values
		self.params_df.loc[:, 'value'] = parameter_values


	def update_config_parameters(self, params_df):

		params_dic = self.config.params_dic

		for parameter_name, row in params_df.iterrows():
			params_dic[parameter_name] = row.value

		# After parameter updates, the config needs to recompute
		# the corresponding timepoints for calculating H
		self.config.update_timepoints()


	def compute_loss(self, params):

		if params is None:
			H = self.config.H
		else:
			self.update_params_df(params)
			self.update_config_parameters(self.params_df)
		
			# Compute H with new parameters
			H = self.config.calculate_H()
		
		# Compute NHFB
		NHFB = self.N @ H @ self.F @ self.B
		loss = np.mean((NHFB - self.G)**2)

		self.current_params = params
		self.current_H = H
		self.current_loss = loss

		# Squared error loss
		return loss


	def optimize(self, method='Nelder-Mead', maxiter=1000, verbose=True, tol=1e-2):
		"""
		Optimize parameters to minimize ||NHFB - G||.
		
		Args:
			method: Optimization method (default: Nelder-Mead)
			maxiter: Maximum iterations
			verbose: Whether to print progress
		"""

		bounds_list = self.params_df[['min' , 'max']].values
		initial_parameter_values = self.params_df['value'].values

		global ITERATION
		ITERATION = 0
		
		# Define objective function
		def objective_function(parameters):

			global ITERATION

			loss = self.compute_loss(parameters)

			if ITERATION % 100 == 0 and verbose:
				print(f"Optimization [{ITERATION+1}]: Current loss: {loss}")

			ITERATION += 1

			return loss
		
		# Run optimization
		result = minimize(
			objective_function,
			initial_parameter_values,
			method=method,
			bounds=bounds_list,
			options={
				'maxiter': maxiter,
				'disp': verbose,
				'xatol': tol
			}
		)

		self.result = result
		self.rn = self.result.fun
