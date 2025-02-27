
import sys
sys.path.append('.')


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.RealDataReplication import RealDataReplicationDeconvolution
from src.config import load_default_chrom_configs
from src.optimize_H import create_bounds_params_from_config, ParameterOptimizer
from src.timer import Timer
from src.utils import print_fl


def run_epochs(replication_deconvolver, optimizer, num_epochs, function_update=None):

	timer = Timer()

	num_iterations_N_B = 20
	update_params_df = pd.DataFrame()

	N = replication_deconvolver.N
	B = replication_deconvolver.B

	Hs = np.zeros((num_epochs, *replication_deconvolver.config.H.shape))
	Ns = np.zeros((num_epochs, *N.shape))
	Bs = np.zeros((num_epochs, *B.shape))
	Fs = np.zeros((num_epochs, *replication_deconvolver.F.shape))

	for epoch in range(num_epochs):

		print_fl(f"Epoch: {epoch}")
		
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

		Hs[epoch] = replication_deconvolver.H
		Fs[epoch] = replication_deconvolver.F
		Ns[epoch] = replication_deconvolver.N
		Bs[epoch] = replication_deconvolver.B
		
		N = replication_deconvolver.N
		B = replication_deconvolver.B

		print_fl(update_params_df.iloc[-1])

		if function_update is not None:
			function_update(epoch, update_params_df, Hs, Fs, Ns, Bs)

	return update_params_df, Hs, Fs, Ns, Bs


def main(replicate=1, chrom=1, num_epochs=10, num_iterations_N_B=20, output_directory=None,
	from_CLOCCS=True):

	np.random.seed(123)

	# Load the default replication chrom configuration from disk
	# use the posterios from the CLOCCS fits to initialize
	print_fl("Loading initial cell cycle parameters from CLOCCS fits.")
	config1, config2 = load_default_chrom_configs(from_CLOCCS=from_CLOCCS)
	config = config1 if replicate == 1 else config2

	replication_deconvolver = RealDataReplicationDeconvolution(config, replicate=replicate, chr=chrom)

	# Generate initial parameters and boundaries for optimization
	bounds_df = create_bounds_params_from_config(config)

	# First iteration to settle N, Fr, and B
	print_fl("Running initial iterations...")
	replication_deconvolver.setup_deconvolution(config)
	replication_deconvolver.iterative_deconvolution_updates(num_iterations_N_B, verbose=False)
	print_fl("Done.")

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
	print_fl(f"Running {num_epochs} epochs...")
	def epoch_updates(epoch, update_params_df, Hs, Fs, Ns, Bs):
		"""Update function"""

		# Periodic saving to disk
		if epoch % 10 == 0 or epoch == num_epochs-1:

			if output_directory is not None:
				print_fl(f"[{epoch}]Saving to output_directory...")
				replication_deconvolver.save_to_disk(output_directory)
				update_params_df.to_csv(f"{output_directory}/parameter_updates_rep{replicate}_chr{chrom}.csv")

	# Run the optimizer
	update_params_df, Hs, Fs, Ns, Bs = run_epochs(replication_deconvolver, 
		optimizer, num_epochs, function_update=epoch_updates)

	print_fl("Done")

	return update_params_df, optimizer, replication_deconvolver

	
if __name__ == '__main__':
	main()
