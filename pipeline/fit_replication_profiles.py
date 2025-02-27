
import sys
sys.path.append('.')


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.RealDataReplication import RealDataReplicationDeconvolution
from src.config import load_default_chrom_configs
from src.optimize_H import create_bounds_params_from_config, ParameterOptimizer


def run_epochs(optimizer, replication_deconvolver, num_epochs, num_iterations_N_B):
	from src.timer import Timer

	timer = Timer()

	update_params_df = pd.DataFrame()

	# Initial N and B
	N = replication_deconvolver.N
	B = replication_deconvolver.B

	for epoch in range(num_epochs):
		print("Epoch: ", epoch)
		
		optimizer.optimize(maxiter=1000, verbose=True)

		replication_deconvolver.H = optimizer.current_H
			
		replication_deconvolver.iterative_deconvolution_updates(
			total_iterations=num_iterations_N_B, timer=timer,
			initial_B=B, initial_N=N, verbose=False)
		timer.print_time()
		
		params_row = pd.DataFrame([optimizer.params_df['value']], index=[epoch])
		params_row['opt_H_loss'] = optimizer.rn
		params_row['F_rn'] = replication_deconvolver.rn

		update_params_df = pd.concat([update_params_df, params_row])
		
		optimizer.N = replication_deconvolver.N
		optimizer.B = replication_deconvolver.B
		optimizer.F = replication_deconvolver.F
		
		N = replication_deconvolver.N
		B = replication_deconvolver.B

		print(update_params_df.iloc[-1])

	return update_params_df


def main(replicate=1, chrom=1, num_epochs=10, num_iterations_N_B=20, output_directory=None):

	np.random.seed(123)

	# Load the default replication chrom configuration from disk
	# use the posterios from the CLOCCS fits to initialize
	print("Loading initial cell cycle parameters from CLOCCS fits.")
	config1, config2 = load_default_chrom_configs(from_CLOCCS=True)
	config = config1 if replicate == 1 else config2




	config.params_dic['mu0'] = 5
	config.params_dic['gamma1'] = 0.7
	config.params_dic['gamma2'] = 1.
	config.update_timepoints()
	config.calculate_H()


	replication_deconvolver = RealDataReplicationDeconvolution(config, replicate=replicate, chr=chrom)

	# Generate initial parameters and boundaries for optimization
	bounds_df = create_bounds_params_from_config(config)

	# todo: Testing speedup of optimization
	print("To do: testing speed of mu0, gamma1, gamma2 search first")
	bounds_df = bounds_df.loc[['mu0', 'gamma1', 'gamma2']]

	# First iteration to settle N, Fr, and B
	print("Running initial iterations...")
	replication_deconvolver.setup_deconvolution(config)
	replication_deconvolver.iterative_deconvolution_updates(num_iterations_N_B, verbose=False)
	print("Done.")

	print("Initial config parameters: ", bounds_df)

	# Parameter Optimizer
	optimizer = ParameterOptimizer(
		init_params_df=bounds_df,
		config=config,
		N=replication_deconvolver.N,
		F=replication_deconvolver.F,
		B=replication_deconvolver.B,
		G=replication_deconvolver.G
	)

	# Parameter updates df
	print(f"Running {num_epochs} epochs...")
	parameter_updates_df = run_epochs(optimizer, replication_deconvolver, 
		num_epochs, num_iterations_N_B)
	print("Done")

	if output_directory is not None:
		replication_deconvolver.save_to_disk(output_directory)
		parameter_updates_df.to_csv(f"{output_directory}/parameter_updates_rep{replicate}_chr{chrom}.csv")

	return parameter_updates_df, optimizer, replication_deconvolver

	
if __name__ == '__main__':
	main()
