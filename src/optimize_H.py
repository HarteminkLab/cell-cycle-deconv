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
				'xatol': tol
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


def run_epochs(real_deconv1, params_df, num_epochs, function_update=None):

	timer = Timer()

	num_iterations_N_B = 20
	update_params_df = pd.DataFrame()

	N = real_deconv1.N
	B = real_deconv1.B

	Hs = np.zeros((num_epochs, *real_deconv1.config.H.shape))
	Ns = np.zeros((num_epochs, *N.shape))
	Bs = np.zeros((num_epochs, *B.shape))
	Fs = np.zeros((num_epochs, *real_deconv1.F.shape))

	optimizer = ParameterOptimizer(
		init_params_df=params_df,
		config=real_deconv1.config,
		N=real_deconv1.N,
		F=real_deconv1.F,
		B=real_deconv1.B,
		G=real_deconv1.G
	)

	for epoch in range(num_epochs):

		print_fl("Epoch: ", epoch)
		
		optimizer.optimize(maxiter=1000, verbose=True)

		real_deconv1.H = optimizer.current_H
			
		real_deconv1.iterative_deconvolution_updates(
			total_iterations=num_iterations_N_B, timer=timer,
			initial_B=B, initial_N=N, verbose=False)
		timer.print_time()
		
		params_row = pd.DataFrame([optimizer.params_df['value']], index=[epoch])
		params_row['opt_H_loss'] = optimizer.rn
		params_row['F_rn'] = real_deconv1.rn

		update_params_df = pd.concat([update_params_df, params_row])
		
		optimizer.N = real_deconv1.N
		optimizer.B = real_deconv1.B
		optimizer.F = real_deconv1.F

		Hs[epoch] = real_deconv1.H
		Fs[epoch] = real_deconv1.F
		Ns[epoch] = real_deconv1.N
		Bs[epoch] = real_deconv1.B
		
		N = real_deconv1.N
		B = real_deconv1.B

		print_fl(update_params_df.iloc[-1])

		if function_update is not None:
			function_update(epoch, update_params_df Hs, Fs, Ns, Bs)

	return update_params_df, Hs, Fs, Ns, Bs

