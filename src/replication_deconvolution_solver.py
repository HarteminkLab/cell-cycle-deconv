

import numpy as np
import cvxpy as cp
from src.timer import Timer
from matplotlib import pyplot as plt


CONST_1_COPY = 1
CONST_2_COPY = 2


def deconvolve_replication(config, H, G, N, B, timer=None):
	"""
	Deconvolve the replication curve
	"""

	cg1_indices = config.get_Hpositions_for_phase('CG1')
	rg1_indices = config.get_Hpositions_for_phase('RG1')
	postg1_indices = config.get_Hpositions_for_phase('postG1')
	s_indices = config.get_Hpositions_for_phase('S')
	g2m_indices = config.get_Hpositions_for_phase('G2M')

	if timer is None:
		timer = Timer()

	n, m = H.shape
	n, v = G.shape
	F = np.zeros((m, v))
	rns = np.zeros(v)

	for g_index in range(G.shape[1]):

		if g_index % 200 == 0:
			timer.print_time(f"{g_index}/{G.shape[1]}")

		g = G[:, g_index]

		solver = cp.MOSEK

		# f is modeled as a boolean variable. We are interested
		# in the copy number so add 1
		f_0 = cp.Variable(m, boolean=True)
		f = f_0+1

		# Get the relevant baseline occupancy
		b = B[g_index, g_index]

		predicted_g = N@H@(f*b)

		elementwise_result = predicted_g - g

		objective = cp.Minimize(
			cp.sum(cp.norm(elementwise_result, 'fro')**2)
		)

		# constraints = define_constraints(f, config)
		constraints = define_constraints(f, config)

		prob = cp.Problem(objective, constraints)
		result = prob.solve(solver=solver, warm_start=True, 
			verbose=False, eps=1e-4)

		elementwise_result = cp.multiply(predicted_g, 1.0/g) - 1
		rn = cp.sum(cp.norm(elementwise_result, 'fro')**2)

		F[:, g_index] = f.value
		rns[g_index] = rn.value

	return F, rns.mean()


def deconvolve_replication_brute_force(config, H, G, N, B, timer=None):
	"""
	Deconvolve the replication curve
	"""

	cg1_indices = config.get_Hpositions_for_phase('CG1')
	rg1_indices = config.get_Hpositions_for_phase('RG1')
	postg1_indices = config.get_Hpositions_for_phase('postG1')
	s_indices = config.get_Hpositions_for_phase('S')
	g2m_indices = config.get_Hpositions_for_phase('G2M')

	if timer is None:
		timer = Timer()

	n, m = H.shape
	n, v = G.shape
	F = np.zeros((m, v))
	rns = np.zeros(v)

	for g_index in range(G.shape[1]):

		if g_index % 100 == 0:
			timer.print_time(f"{g_index}/{G.shape[1]}")

		g = G[:, g_index]

		# Get the relevant baseline occupancy
		b = B[g_index, g_index]

		def minimization_objective(f, g):
			predicted_g = N@H@(f*b)
			diff = predicted_g - g
			rn = np.mean(diff**2)
			return rn

		best_rn = float('inf')
		best_f = None
		for repl_index in postg1_indices:
			# Create f vector for replication timing and compute rn
			f = np.ones(m)
			f[repl_index:-1] = 2
			rn = minimization_objective(f, g)

			if rn < best_rn:
				best_rn = rn
				best_f = f

		F[:, g_index] = best_f
		rns[g_index] = best_rn

	NHFB = N @ H @ F @ B
	loss = np.mean((NHFB - G)**2)

	return F, rns.mean()


def define_constraints_any_repl_time(f, config):

	cg1_indices = config.get_Hpositions_for_phase('CG1')
	rg1_indices = config.get_Hpositions_for_phase('RG1')
	postg1_indices = config.get_Hpositions_for_phase('postG1')
	s_indices = config.get_Hpositions_for_phase('S')
	g2m_indices = config.get_Hpositions_for_phase('G2M')

	constraints = []
	
	# Enforce monotonic increase during indices
	# designated for replication (may be S only
	# or postG1)
	allow_replication_indices = postg1_indices

	# For every set of indices, enforce monotonic increase
	index_sets = [rg1_indices, cg1_indices, postg1_indices]
	for indices in index_sets:
		for i in range(1, len(indices)):
			prev = indices[i-1]
			current = indices[i]
			constraints.append(f[prev] <= f[current])

	# Enforce G1 to S monotonic increase
	constraints.append(f[cg1_indices[-1]] <= f[postg1_indices[0]])
	constraints.append(f[rg1_indices[-1]] <= f[postg1_indices[0]])

	# Enforce that G2M ends with two copies
	# And G1 starts with 1
	constraints.append(f[postg1_indices[-1]] == CONST_2_COPY)
	constraints.append(f[cg1_indices[0]] == CONST_1_COPY)
	constraints.append(f[rg1_indices[0]] == CONST_1_COPY)

	# Halted cells, copy number of 1
	constraints.append(f[postg1_indices[-1]+1] == CONST_1_COPY)

	return constraints


def define_constraints(f, config):

	cg1_indices = config.get_Hpositions_for_phase('CG1')
	rg1_indices = config.get_Hpositions_for_phase('RG1')
	postg1_indices = config.get_Hpositions_for_phase('postG1')
	s_indices = config.get_Hpositions_for_phase('S')
	g2m_indices = config.get_Hpositions_for_phase('G2M')

	constraints = []
	
	# Enforce values of 0, during G1
	for i in range(0, len(cg1_indices)):
		current = cg1_indices[i]
		constraints.append(f[current] == CONST_1_COPY)

	for i in range(0, len(rg1_indices)):
		current = rg1_indices[i]
		constraints.append(f[current] == CONST_1_COPY)

	# Enforce monotonic increase during indices
	# designated for replication (may be S only
	# or postG1)
	allow_replication_indices = postg1_indices

	for i in range(1, len(allow_replication_indices)):
		prev = allow_replication_indices[i-1]
		current = allow_replication_indices[i]
		constraints.append(f[prev] <= f[current])

		# Enforce start and end of replication timing
		# window starts as 1 and ends as 2
		if i == 1:
			constraints.append(f[prev] == CONST_1_COPY)
		elif i == len(allow_replication_indices)-1:
			constraints.append(f[current] == CONST_2_COPY)

	# If allowing replication timings in S only,
	# Enforce that all of G2M must be copy number 2
	if allow_replication_indices[-1] == s_indices[-1]:
		# Enforce two copies of DNA in G2M
		for i in range(0, len(g2m_indices)):
			current = g2m_indices[i]
			constraints.append(f[current] == CONST_2_COPY)

	# Halted cells, copy number of 1
	constraints.append(f[postg1_indices[-1]+1] == CONST_1_COPY)

	return constraints
