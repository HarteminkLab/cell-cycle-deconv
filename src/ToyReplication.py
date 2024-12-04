
import numpy as np
import cvxpy as cp
from src.timer import Timer
from matplotlib import pyplot as plt


class AverageCopyNumberDeconvolution(object):
	"""Determine the optimal average copy number curve. Given a single
	occupancy curve that has been normalized, we can determine the copy number curve
	that has been encoded into the occupancy curve. We do not know the replication 
	timing, but we can sweep/brute force to identify the optimal replication timing
	and occupancy curve.

	todo: the idea of sweeping possible replication indices may be easier and more
	straightforward than the deconvolution done in ReplicationDeconvolution. to be continued
	"""

	def __init__(self, H):
		self.H = H


	def deconvolve(self, g):

		H = self.H
		n, m = H.shape
		self.g = g

		possible_f_replication_indices = np.zeros((m, m))

		for replication_index in range(m):
			
			f_replication_curve = np.ones(m)
			f_replication_curve[replication_index:] = 2
			possible_f_replication_indices[replication_index] = f_replication_curve

		# Let's try all possible indices for replication:
		learned_inv_avg_copy_curves = np.zeros((m, m))
		timer = Timer()

		rns = np.zeros(m)

		for replication_index in range(m):
			
			# Let's try the DCP error thing again, division by a variable is not allowed
			# but we can try to multiply by a variable....
			solver = cp.MOSEK

			f = possible_f_replication_indices[replication_index]

			inv_learn_avg_copies = cp.Variable(m)

			normalized_f = cp.multiply((f), inv_learn_avg_copies)
			predicted_g = H@normalized_f

			elementwise_result = cp.multiply(predicted_g, 1.0/g) - 1

			objective = cp.Minimize(
				cp.sum(cp.norm(elementwise_result, 'fro')**2)
			)
			constraints = [inv_learn_avg_copies >=0.5, inv_learn_avg_copies <=1]
			for i in range(1, m):
				prev = i-1
				current = i
				constraints.append(inv_learn_avg_copies[prev] >= inv_learn_avg_copies[current])
			constraints.append(inv_learn_avg_copies[0] == 1)
			constraints.append(inv_learn_avg_copies[-1] == 0.5)

			prob = cp.Problem(objective, constraints)
			result = prob.solve(solver=solver, warm_start=True, verbose=False, eps=1e-5)

			learned_inv_avg_copy_curves[replication_index] = inv_learn_avg_copies.value
			rns[replication_index] = result

		self.rns = rns
		self.learned_inv_avg_copy_curves = learned_inv_avg_copy_curves
		self.min_sol_idx = np.argmin((rns))
		self.f_solution = possible_f_replication_indices[self.min_sol_idx]
		self.avg_curve_solution = learned_inv_avg_copy_curves[self.min_sol_idx]


	def plot_solution(self):
		plt.figure(figsize=(13, 2))
		plt.subplot(1, 3, 1)
		plt.plot(self.H @ (self.f_solution * self.avg_curve_solution))
		plt.plot(self.g)

		plt.subplot(1, 3, 2)
		plt.plot(self.f_solution)

		plt.subplot(1, 3, 3)
		plt.plot(1/self.avg_curve_solution)


	def plot_rn_curve(self):
		rns = self.rns
		learned_inv_avg_copy_curves = self.learned_inv_avg_copy_curves
		min_sol_idx = self.min_sol_idx

		plt.figure(figsize=(8, 1))
		plt.subplot(1, 2, 1)
		plt.plot(rns)
		plt.title("Residual norms for\npossible replication timings")
		plt.axvline(min_sol_idx, c='red')

		plt.subplot(1, 2, 2)
		plt.plot(1/self.avg_curve_solution.T)
		plt.title("Average copy number for minimum solution")


class ReplicationDeconvolution(object):
	"""
	Deconvolution to determine the optimal replication timing for a given 

	G.

	todo: This may not need a deconvolution anymore, it's possible brute force may
	be just as fast... test this. There are only so many possible indices in S (postG1)
	"""

	def __init__(self, true_F, H, G, avg_copies_per_time):

		self.true_F = true_F
		self.H = H
		self.G = G
		self.avg_copies_per_time = avg_copies_per_time
		

	def deconvolve(self):

		timer = Timer()

		H = self.H
		G = self.G
		avg_copies_per_time = self.avg_copies_per_time

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

			elementwise_result = cp.multiply(predicted_g, 1.0/g) - 1

			objective = cp.Minimize(
				cp.sum(cp.norm(elementwise_result, 'fro')**2)
			)
			constraints = []
			for i in range(1, m):
				prev = i-1
				current = i
				constraints.append(f[prev] <= f[current])

			prob = cp.Problem(objective, constraints)
			result = prob.solve(solver=solver, warm_start=True, verbose=False, eps=1e-4)
			F[:, g_index] = f.value

		self.F = F


	def plot_result(self):
		avg_copies_mat = self.avg_copies_per_time.reshape((-1, 1))

		plt.figure(figsize=(9, 2))
		plt.subplot(1, 2, 1)
		plt.plot(self.H@((self.F+1) * 1/avg_copies_mat))
		plt.plot(self.G, ls='dotted')

		plt.subplot(1, 2, 2)
		plt.imshow(self.F, aspect='auto', cmap='Blues')


