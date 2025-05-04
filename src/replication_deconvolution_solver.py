

import numpy as np
import cvxpy as cp
from src.timer import Timer
from matplotlib import pyplot as plt


CONST_1_COPY = 1
CONST_2_COPY = 2


def deconvolve_replication_brute_force(config, H, G, N, B, timer=None,
	mode='lateG1G2M', verbose=True):
	"""
	Deconvolve the replication curve by computing the rn for every possible
	replication index, this is faster than any optimizer method (if there 
	is a large number if indices, this could be refined using a binary
	search)
	"""

	cg1_indices = config.get_Hpositions_for_phase('CG1')
	dg1_indices = config.get_Hpositions_for_phase('DG1')
	rg1_indices = config.get_Hpositions_for_phase('RG1')
	postg1_indices = config.get_Hpositions_for_phase('postG1')
	s_indices = config.get_Hpositions_for_phase('S')
	g2m_indices = config.get_Hpositions_for_phase('G2M')

	if mode == 'S':
		replication_indices = s_indices
	elif mode == 'postG1':
		replication_indices = postg1_indices
	elif mode == 'lateG1G2M':
		replication_indices = np.concatenate([np.arange(-22, 0), postg1_indices])
	else:
		raise ValueError("Unknown mode", mode)

	if timer is None:
		timer = Timer()

	n, m = H.shape
	n, v = G.shape
	F = np.zeros((m, v))
	rns = np.zeros(v)

	for g_index in range(G.shape[1]):

		if verbose and g_index % 100 == 0:
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

		for repl_index in replication_indices:

			f = np.ones(m)

			# If in S/G2/M, set all all values until the end to 2
			if repl_index in postg1_indices:
				# Create f vector for replication timing and compute rn
				f[repl_index:-1] = 2

			# Otherwise, we will estimate replication times in G1
			# negative values mean we'll set a replication time in RG1, DG1, and CG1 
			elif repl_index < 0:
				s_start = postg1_indices[0]

				# All of post G1
				f[s_start:-1] = 2

				# All of the G1's with the same replication indices
				# a little wonky with the different times, the negative value
				# of the repl index will represent starting from the end of these phases
				f[rg1_indices[repl_index:]] = 2
				f[cg1_indices[repl_index:]] = 2
				f[dg1_indices[repl_index:]] = 2

			rn = minimization_objective(f, g)

			if rn < best_rn:
				best_rn = rn
				best_f = f

		F[:, g_index] = best_f
		rns[g_index] = best_rn

	NHFB = N @ H @ F @ B
	loss = np.mean((NHFB - G)**2)

	return F, rns.mean()


# Deprecated, no longer using constraints for the brute force problem

# def define_constraints_any_repl_time(f, config):

# 	cg1_indices = config.get_Hpositions_for_phase('CG1')
# 	dg1_indices = config.get_Hpositions_for_phase('DG1')
# 	rg1_indices = config.get_Hpositions_for_phase('RG1')
# 	postg1_indices = config.get_Hpositions_for_phase('postG1')
# 	s_indices = config.get_Hpositions_for_phase('S')
# 	g2m_indices = config.get_Hpositions_for_phase('G2M')

# 	constraints = []
	
# 	# Enforce monotonic increase during indices
# 	# designated for replication (may be S only
# 	# or postG1)
# 	allow_replication_indices = postg1_indices

# 	# For every set of indices, enforce monotonic increase
# 	index_sets = [rg1_indices, dg1_indices, cg1_indices, postg1_indices]
# 	for indices in index_sets:
# 		for i in range(1, len(indices)):
# 			prev = indices[i-1]
# 			current = indices[i]
# 			constraints.append(f[prev] <= f[current])

# 	# Enforce G1 to S monotonic increase
# 	constraints.append(f[cg1_indices[-1]] <= f[postg1_indices[0]])
# 	constraints.append(f[dg1_indices[-1]] <= f[postg1_indices[0]])
# 	constraints.append(f[rg1_indices[-1]] <= f[postg1_indices[0]])

# 	# Enforce that G2M ends with two copies
# 	# And G1 starts with 1
# 	constraints.append(f[postg1_indices[-1]] == CONST_2_COPY)
# 	constraints.append(f[cg1_indices[0]] == CONST_1_COPY)
# 	constraints.append(f[dg1_indices[0]] == CONST_1_COPY)
# 	constraints.append(f[rg1_indices[0]] == CONST_1_COPY)

# 	# Halted cells, copy number of 1
# 	constraints.append(f[postg1_indices[-1]+1] == CONST_1_COPY)

# 	return constraints


# def define_constraints(f, config):

# 	cg1_indices = config.get_Hpositions_for_phase('CG1')
# 	rg1_indices = config.get_Hpositions_for_phase('RG1')
# 	postg1_indices = config.get_Hpositions_for_phase('postG1')
# 	s_indices = config.get_Hpositions_for_phase('S')
# 	g2m_indices = config.get_Hpositions_for_phase('G2M')

# 	constraints = []
	
# 	# Enforce values of 0, during G1
# 	for i in range(0, len(cg1_indices)):
# 		current = cg1_indices[i]
# 		constraints.append(f[current] == CONST_1_COPY)

# 	for i in range(0, len(rg1_indices)):
# 		current = rg1_indices[i]
# 		constraints.append(f[current] == CONST_1_COPY)

# 	# Enforce monotonic increase during indices
# 	# designated for replication (may be S only
# 	# or postG1)
# 	allow_replication_indices = postg1_indices

# 	for i in range(1, len(allow_replication_indices)):
# 		prev = allow_replication_indices[i-1]
# 		current = allow_replication_indices[i]
# 		constraints.append(f[prev] <= f[current])

# 		# Enforce start and end of replication timing
# 		# window starts as 1 and ends as 2
# 		if i == 1:
# 			constraints.append(f[prev] == CONST_1_COPY)
# 		elif i == len(allow_replication_indices)-1:
# 			constraints.append(f[current] == CONST_2_COPY)

# 	# If allowing replication timings in S only,
# 	# Enforce that all of G2M must be copy number 2
# 	if allow_replication_indices[-1] == s_indices[-1]:
# 		# Enforce two copies of DNA in G2M
# 		for i in range(0, len(g2m_indices)):
# 			current = g2m_indices[i]
# 			constraints.append(f[current] == CONST_2_COPY)

# 	# Halted cells, copy number of 1
# 	constraints.append(f[postg1_indices[-1]+1] == CONST_1_COPY)

# 	return constraints
