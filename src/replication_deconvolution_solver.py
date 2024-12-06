

import numpy as np
import cvxpy as cp
from src.timer import Timer
from matplotlib import pyplot as plt


def deconvolve_avg_copy_curve(g, config, H, replication_index):
	"""
	This function attempts to deconvolve the average copy number
	curve from a given g, and replication index.

	The expectation is
	the true replication curve is known, and we are interested in
	identifying the best copy curve that allows us to fit to the
	given raw data curve g. 

	This is an estimate, but is powerful when we compute many
	of these across the genome. The median average copy curve
	is closer to the true average copy curve.

	The average copy number curve is important because it handles
	the effects caused by equal sample normalization of the input
	data.
	"""

	cg1_indices = config.get_Hpositions_for_phase('CG1')
	rg1_indices = config.get_Hpositions_for_phase('RG1')
	postg1_indices = config.get_Hpositions_for_phase('postG1')

	n, m = H.shape

	# Let's try the DCP error thing again, division by a variable is not allowed
	# but we can try to multiply by a variable....
	solver = cp.MOSEK

	# Create f from replication index
	f = np.ones(m)
	f[replication_index:-1] = 2

	inv_learn_avg_copies = cp.Variable(m)

	normalized_f = cp.multiply((f), inv_learn_avg_copies)
	predicted_g = H@normalized_f

	elementwise_result = cp.multiply(predicted_g, 1.0/g) - 1

	objective = cp.Minimize(
		cp.sum(cp.norm(elementwise_result, 'fro')**2)
	)
	constraints = [inv_learn_avg_copies >=0.5, inv_learn_avg_copies <=1]

	# Enforce average copy number of 1 for CG1 and DG1
	for i in range(0, len(cg1_indices)):
		current = cg1_indices[i]
		constraints.append(inv_learn_avg_copies[current] == 1)

	# Enforce average copy number of 1 for RG1
	for i in range(0, len(rg1_indices)):
		current = rg1_indices[i]
		constraints.append(inv_learn_avg_copies[current] == 1)

	# Enforce monotonic decrease during postG1
	# Inverse of the average curve from 1 to 0.5
	for i in range(1, len(postg1_indices)):
		prev = postg1_indices[i-1]
		current = postg1_indices[i]
		constraints.append(inv_learn_avg_copies[prev] >= inv_learn_avg_copies[current])

		if i == len(postg1_indices)-1:
			constraints.append(inv_learn_avg_copies[current] == 0.5)
		elif i == 1:
			constraints.append(inv_learn_avg_copies[prev] == 1.0)

	# Halted cells, copy number of 1
	constraints.append(inv_learn_avg_copies[m-1] == 1.0)

	prob = cp.Problem(objective, constraints)
	result = prob.solve(solver=solver, warm_start=True, verbose=False, eps=1e-5)

	return result, f, inv_learn_avg_copies.value, predicted_g.value


def deconvolve_replication(config, H, G, avg_copies_per_time):
	"""
	Deconvolve the replication curve
	"""

	cg1_indices = config.get_Hpositions_for_phase('CG1')
	rg1_indices = config.get_Hpositions_for_phase('RG1')
	postg1_indices = config.get_Hpositions_for_phase('postG1')

	timer = Timer()

	n, m = H.shape
	n, v = G.shape
	F = np.zeros((m, v))

	for g_index in range(G.shape[1]):

		if g_index % 50 == 0:
			timer.print_time(f"{g_index}/{G.shape[1]}")

		g = G[:, g_index]

		solver = cp.MOSEK
		f = cp.Variable(m, boolean=True)

		normalized_f = (f+1) * 1/avg_copies_per_time
		predicted_g = H@normalized_f

		#elementwise_result = cp.multiply(predicted_g, 1.0/g) - 1

		# Can we do additive minimization now?
		elementwise_result = predicted_g - g

		objective = cp.Minimize(
			cp.sum(cp.norm(elementwise_result, 'fro')**2)
		)
		constraints = []
		
		# Enforce values of 0, during G1
		for i in range(0, len(cg1_indices)):
			current = cg1_indices[i]
			constraints.append(f[current] == 0)

		for i in range(0, len(rg1_indices)):
			current = rg1_indices[i]
			constraints.append(f[current] == 0)

		# Enforce monotonic increase to 1 in postG1
		for i in range(1, len(postg1_indices)):
			prev = postg1_indices[i-1]
			current = postg1_indices[i]
			constraints.append(f[prev] <= f[current])

			# Enforce start and end of postG1 has
			# a transition from 0 to 1
			if i == len(postg1_indices)-1:
				constraints.append(f[current] == 1)
			elif i == 1:
				constraints.append(f[prev] == 0)

		# Halted cells, copy number of 1
		constraints.append(f[m-1] == 0)

		prob = cp.Problem(objective, constraints)
		result = prob.solve(solver=solver, warm_start=True, 
			verbose=False, eps=1e-4)
		F[:, g_index] = f.value

	return F


def estimate_rough_average_copy_curve_fit(config, H, G, num_skip_sites=10):

	from src.RealDataReplication import deconvolve_avg_copy_curve
	from src.helpers import normalize_max_min
	from src.timer import Timer

	num_sites = G.shape[1]
	S_indices = config.get_Hpositions_for_phase('S')
	postG1_indices = config.get_Hpositions_for_phase('postG1')

	n, m = H.shape
	all_avg_copy_curves = np.zeros((num_sites, m))
	found_replication_indices = np.zeros((num_sites, 1))

	timer = Timer()

	for genomic_idx in range(0, num_sites, num_skip_sites):
		
		def determine_optimal_g(G, genomic_idx):
			
			rns = np.zeros(len(S_indices))
			sweep_all_avg_copy_curves = np.zeros((len(S_indices), m))
					
			# Try all replication indices to compute the optimal copy curve
			for index, replication_index in enumerate(S_indices):
				
				# Get g and normalize to a known good range for deconvolution
				g = G[:, genomic_idx]
				normalized_g = normalize_max_min(g)*.1 + .95

				result, f, inv_learn_avg_copies, predicted_g = \
					deconvolve_avg_copy_curve(normalized_g, config, 
						H, replication_index)
				
				rns[index] = result
				sweep_all_avg_copy_curves[index] = inv_learn_avg_copies

			min_idx = np.argmin(rns)
			S_indices[min_idx]

			return min_idx, S_indices[min_idx], rns[min_idx], \
				1./sweep_all_avg_copy_curves[min_idx]
		
		(min_idx, repl_index, rn, \
		 found_avg_copy_curve) = determine_optimal_g(G, genomic_idx)
		
		all_avg_copy_curves[genomic_idx] = found_avg_copy_curve
		found_replication_indices[genomic_idx] = repl_index
		
		if genomic_idx % 20 == 0:
			timer.print_time(f"{genomic_idx}/{num_sites}")

	selected_copy_curves = all_avg_copy_curves[
		(np.quantile(all_avg_copy_curves, axis=1, q=0.5) > 0), :]
	rough_average_copy_curve = np.median(selected_copy_curves.T, 
		axis=1)

	return all_avg_copy_curves, found_replication_indices, \
		selected_copy_curves, rough_average_copy_curve
