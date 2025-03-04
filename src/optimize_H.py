import numpy as np
import pandas as pd

from scipy.optimize import minimize
from src.calcH_single_g1 import calcH
from src.calcH_single_g1 import get_parameter_indices
from src.timer import Timer
from src.utils import print_fl


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

		# Initiialize H with config's H
		self.current_H = config.calculate_H()

		# Compute initial loss 
		self.compute_loss(params=None)

		# Get the indices of the parameters to be updated
		# this is needed because both config and the optimizer require
		# the parameters to be in the form of a list, we will keep track of
		# which parameters we are updating with this field
		self.parameter_indices = get_parameter_indices(init_params_df.index)

		# Keep track of the paramater values to update in this dataframe
		self.params_df = init_params_df.copy()

		# As long as the indices are not resorted they should
		# be indexed by their position in the dataframe
		self.params_df['param_index'] = np.arange(len(self.params_df))

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

			# Update dataframe from vector of parameters
			self.update_params_df(params)

			# Update config paramaters
			self.update_config_parameters(self.params_df)
		
			# Compute H with new parameters
			H = self.config.calculate_H()

		# Compute NHFB
		from src.RealDataReplication import compute_rn

		F = self.F
		G = self.G
		N = self.N
		B = self.B

		# gamma2 must be greater than gamma1, add a boundary of 5% so 
		# S-phase always has available indices for replication timing estimation
		if (self.config.params_dic['gamma2'] - self.config.params_dic['gamma1'] < 0.05):
			loss = float('inf')
		else:
			loss = compute_rn(N, H, F, B, G)

		self.current_params = params
		self.current_H = H
		self.current_loss = loss
		self.rn = loss

		# Squared error loss
		return loss


	def optimize(self, method='Nelder-Mead', maxiter=1000, verbose=True, tol=1e-5):
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
				print_fl(f"Optimization [{ITERATION+1}]: Current loss: {loss}")

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
				'xatol': tol,
				'adaptive': True
			}
		)

		self.result = result
		self.rn = self.result.fun


def create_bounds_params_from_config(config):
	initial_values = config.params_dic

	init_params_df = pd.DataFrame(initial_values, index=['value']).T
	init_params_df = init_params_df.drop('alpha')
	bounds_dic = {
		'mu0': (-30, 30),
		'lambda': (40, 80),
		'delta': (0, 24),
		'sigma0': (1, 14),
		'sigmav': (0.01, 1.),
		'gamma1': (0., 1.),
		'gamma2': (0., 1.),
		'halted': (0.0, 1.),
	}

	bounds_params_df = pd.DataFrame(bounds_dic, index=['min', 'max']).T
	params_df = init_params_df.join(bounds_params_df)
	return params_df


