

import numpy as np
import cvxpy as cp
from src.timer import Timer
from matplotlib import pyplot as plt


def deconvolve_replication(config, H, G, N, B, prev_F=None, timer=None,
	smoothness_weight=0.01):
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
	sns = np.zeros(v)

	for g_index in range(G.shape[1]):

		if g_index % 50 == 0:
			timer.print_time(f"{g_index}/{G.shape[1]}")

		g = G[:, g_index]

		solver = cp.MOSEK

		# f is modeled as a boolean variable. We are interested
		# in the copy number so add 1
		f_0 = cp.Variable(m, boolean=True)
		f = f_0+1

		if prev_F is None or g_index == 0 or g_index == v-1:
			avg_diff = 0
		else:

			neighbor_f_left = np.array(np.nan)
			neighbor_f_right = np.array(np.nan)
			compare_f = f

			sum_f = cp.sum(compare_f)

			neighbor_f_left = prev_F[:, g_index-1]
			neighbor_f_right = prev_F[:, g_index+1]
				
			left_sum = neighbor_f_left.sum()
			right_sum = neighbor_f_right.sum()
			average_sum = (left_sum+right_sum)/2

			# Minimize the difference between the current f
			# and the average of the left and right
			# (We want f to be favor being a middle step between
			# the left and right sums)
			avg_diff = average_sum-sum_f

		neighbor_smoothing_norm = cp.abs(avg_diff)

		# Get the relevant baseline occupancy
		b = B[g_index, g_index]

		predicted_g = N@H@(f*b)

		CONST_1_COPY = 1
		CONST_2_COPY = 2

		elementwise_result = cp.multiply(predicted_g, 1.0/g) - 1

		objective = cp.Minimize(
			cp.sum(cp.norm(elementwise_result, 'fro')**2) +
			smoothness_weight*neighbor_smoothing_norm
		)
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
		allow_replication_indices = s_indices

		for i in range(1, len(allow_replication_indices)):
			prev = allow_replication_indices[i-1]
			current = allow_replication_indices[i]
			constraints.append(f[prev] <= f[current])

			# Enforce start and end of S
			if i == 1:
				constraints.append(f[prev] == CONST_1_COPY)
			elif i == len(allow_replication_indices)-1:
				constraints.append(f[current] == CONST_2_COPY)

		if allow_replication_indices[0] == s_indices[0]:
			# Enforce two copies of DNA in G2M
			for i in range(0, len(g2m_indices)):
				current = g2m_indices[i]
				constraints.append(f[current] == CONST_2_COPY)

		# Halted cells, copy number of 1
		constraints.append(f[m-1] == CONST_1_COPY)

		prob = cp.Problem(objective, constraints)
		result = prob.solve(solver=solver, warm_start=True, 
			verbose=False, eps=1e-4)

		elementwise_result = cp.multiply(predicted_g, 1.0/g) - 1
		rn = cp.sum(cp.norm(elementwise_result, 'fro')**2)
		sn = smoothness_weight*neighbor_smoothing_norm

		F[:, g_index] = f.value
		rns[g_index] = rn.value
		sns[g_index] = sn.value

	return F, rns.mean(), sns.mean()


