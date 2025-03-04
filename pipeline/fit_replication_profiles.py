
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


def run_epochs(replication_deconvolver, bounds_df, num_epochs, function_update=None):

	# Parameter Optimizer
	optimizer = ParameterOptimizer(
		init_params_df=bounds_df,
		config=replication_deconvolver.config,
		N=replication_deconvolver.N,
		F=replication_deconvolver.F,
		B=replication_deconvolver.B,
		G=replication_deconvolver.G
	)

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
	from_CLOCCS=True, config=None):

	np.random.seed(123)

	# Load the default replication chrom configuration from disk
	# use the posterios from the CLOCCS fits to initialize

	if config is None:
		print_fl("Loading initial cell cycle parameters from CLOCCS fits.")
		config1, config2 = load_default_chrom_configs(from_CLOCCS=from_CLOCCS)
		config = config1 if replicate == 1 else config2

	replication_deconvolver = RealDataReplicationDeconvolution(config, replicate=replicate, chr=chrom)

	# Generate initial parameters and boundaries for optimization
	# Subset the parameters to learn
	subset_parameters = ['mu0', 'gamma1', 'gamma2', 'sigma0']
	bounds_df = create_bounds_params_from_config(config)
	bounds_df = bounds_df.loc[subset_parameters]
	print(f"Subsetting the cell cycle parameters to learn: ", subset_parameters)

	# First iteration to settle N, Fr, and B
	print_fl("Running initial iterations...")
	replication_deconvolver.setup_deconvolution(config)
	replication_deconvolver.iterative_deconvolution_updates(num_iterations_N_B, verbose=False)
	print_fl("Done.")

	print("Initial config parameters: ", bounds_df)

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
		bounds_df, num_epochs, function_update=epoch_updates)

	print_fl("Done")

	return update_params_df, optimizer, replication_deconvolver

	
def plot_parameter_updates(output_directory, replicate, chrom):
	filepath = f'{output_directory}/parameter_updates_rep{replicate}_chr{chrom}.csv'
	rep1_updates = pd.read_csv(filepath)

	rows, cols = 1, 5


	rename_parameters = {
		'mu0': "$\\mu_0$",
		'gamma1': "$\\gamma_1$",
		'gamma2': "$\\gamma_2$",
		'sigma0': "$\\sigma_0$",
		'F_rn': "Loss",
	}

	skip_parameters = ['opt_H_loss']

	ylabels = [
		"Recovery offset, min",
		"S start, proportion",
		"S end, proportion",
		"Initial population variation",
		"",
		"Residual norm, ($\\times$1e-4)",
	]

	plt.figure(figsize=(16, 3))

	plot_index = 1
	for i, col in enumerate(rep1_updates.columns[1:]):

		if col in skip_parameters: continue

		values = rep1_updates[col][10:]
		xs = np.arange(len(values))

		if col == 'F_rn':
			values = values * 1000

		plt.subplot(rows, cols, plot_index)
		plt.plot(xs, values)

		plt.ylabel(ylabels[i])

		# Rename the parameter for the title
		if col in rename_parameters.keys():
			plt.title(rename_parameters[col])
		else:
			plt.title(col)

		plt.xlabel("Epoch")

		plot_index += 1

	plt.subplots_adjust(hspace=0.5, wspace=0.45, top=0.77)
	plt.suptitle(f"Parameter convergence, replicate {replicate}, chrom {chrom}", fontsize=16)


if __name__ == '__main__':
	main()
