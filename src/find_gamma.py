
from src.timer import Timer
import numpy as np
import sys


DEFAULT_GM = 0.0004


def print_fl(*args, **kwargs):
	"""On the cluster, it is helpful to flush after printing
	for live updates"""

	print(*args, **kwargs)
	sys.stdout.flush()


class FindOptimalGamma:

	def __init__(self, model):
		self.model = model
		self.gamma = model.gamma


	def conv_optim(self):
		self.model.gamma = self.gamma
		self.model.deconvolve()
		self.rn = self.model.rn
		self.sn = self.model.sn


	def find_optimal(self, silence=False):
		self.timer = Timer()
		if not silence:
			print_fl('findOptimal')
		ELBOW_BINS = 10
		SMALL = 5e-5

		# some settings

		# Note the Gamma min and max were decreased by a power of 10
		# The motivation is that for Xin/Orlando data was microarrays
		# We are now using RNA-seq normalized by VST so the scale and variation
		# appears to be lower. For a few of the genes this seemed to have a positive
		# effect on the found gamma values. Previously: gamma min and max were (0.001, and 0.01)
		#
		# Also, if the RN is too high, let's end and use the default gamma.
		# Changing the maximum rn: from 10 to 1
		#

		DEFAULT_RN_CUTOFF = 1

		# 11/21/24 - Testing adding an independent recovery S/G2M phase, and removal of mirroring
		# With this addition higher gamma values are needed, thus the increase from 
		# 0.0001, 0.001 as the gamma boundaries

		# Some selected gammas that are predefined: the find gamma curve has some issue
		# finding an appropriate smoothing:
		# CLB2: 0.02
		# CLB5: 0.002

		GAMMA_MIN = 0.0001
		GAMMA_MAX = 0.01

		# left boundary
		rn_rate_left = 1.10
		left_rn = 0.08

		# right boundary
		rn_rate_right = 1.40
		right_rn = 0.32

		# find best fit
		self.gamma = 0
		self.conv_optim()
		base_rn = self.rn
		self.base_rn = base_rn

		if not silence:
			print_fl(f'  ... base_rn = {base_rn:.4f}')

		flag = 1  # not using the default_gm
		if base_rn >= DEFAULT_RN_CUTOFF:
			self.gamma = DEFAULT_GM
			flag = 0
			if not silence:
				print_fl(f'  ... step1: base_rn is too large, use default {self.gamma:.4f}')

		# left boundary search
		if flag:
			gm_left = GAMMA_MIN
			gm_right = GAMMA_MAX

			if not silence:
				print_fl(f'  ... gamma in [{gm_left:.4f}, {gm_right:.4f}]')

			rn_left = min(rn_rate_left * base_rn, base_rn + left_rn)
			leftr = (rn_left / base_rn - 1) * 100
			if not silence:
				print_fl(f'  ...  search left, rn_goal = {rn_left:.4f}, rate = {leftr:.1f}')
			self.gamma = gm_left
			self.conv_optim()

			if self.rn >= DEFAULT_RN_CUTOFF:
				self.gamma = DEFAULT_GM
				flag = 0
				if not silence:
					print_fl(f'  ... left: base_rn is too large, use default {self.gamma:.4f}')

		if flag:
			bs_flag = 1
			if self.rn < rn_left:
				runs, bs_flag = self.binarysearch(gm_left, gm_right, rn_left, silence, DEFAULT_RN_CUTOFF)

			if bs_flag == 0:
				flag = 0
				self.gamma = DEFAULT_GM
				if not silence:
					print_fl(f'  ... left_boundary: base_rn is too large in search, use default {self.gamma:.4f}')
			else:
				gm_left = self.gamma
				rn_left = self.rn

		# right boundary search
		if flag:
			rn_right = max(rn_rate_right * base_rn, base_rn + right_rn)
			rightr = (rn_right / base_rn - 1) * 100
			if not silence:
				print_fl(f'  ...  search right, rn_goal = {rn_right:.4f}, rate = {rightr:.1f}')
			self.gamma = gm_right
			self.conv_optim()
			if self.rn >= DEFAULT_RN_CUTOFF:
				self.gamma = DEFAULT_GM
				flag = 0
				if not silence:
					print_fl(f'  ... right: base_rn is too large, use default {self.gamma:.4f}')

		if flag:
			bs_flag = 1
			if self.rn > rn_right:
				runs, bs_flag = self.binarysearch(gm_left, gm_right, rn_right, silence, DEFAULT_RN_CUTOFF)

			if bs_flag == 0:
				flag = 0
				self.gamma = DEFAULT_GM
				if not silence:
					print_fl(f'  ... right_boundary: base_rn is too large in search, use default {self.gamma:.4f}')
			else:
				gm_right = self.gamma
				rn_right = self.rn

		if flag:
			if not silence:
				print_fl(f'  ... rn range: [{rn_left:.4f}, {rn_right:.4f}]')
				print_fl(f'  ... search gamma in [{gm_left:.4f} {gm_right:.4f}] for elbow')

			bs_flag = 1
			if abs(gm_right - gm_left) < SMALL:  # gm_right == gm_left
				self.gamma = (gm_right + gm_left) / 2
			else:
				step = (gm_right - gm_left) / ELBOW_BINS
				gamma_array = np.arange(gm_left, gm_right + step, step)
				elbow_gamma, flag, gammas, rn, sn = self.find_elbow(gamma_array, silence, DEFAULT_RN_CUTOFF)
				self.gamma = elbow_gamma

			if bs_flag == 0:
				flag = 0
				self.gamma = DEFAULT_GM
				if not silence:
					print_fl(f'  ... findElbow: base_rn is too large or something wrong in search, use default {self.gamma:.4f}')

		if not silence:
			print_fl(f'{self.model.orf_name}: ... final gamma = {self.gamma:.5f}')

		self.conv_optim()

		if not silence:
			print_fl(f"Time to find optimal gamma: {self.timer.get_time()}")

		return flag, rn, sn, gammas, elbow_gamma

	def binarysearch(self, gamma_min, gamma_max, rn_goal, silence, DEFAULT_RN_CUTOFF):
		RN_SMALL = 2e-4
		LR_SMALL = 5e-4
		flag = 1

		left = gamma_min
		right = gamma_max
		runs = 0

		# find the fit_left point
		while right - left > LR_SMALL:
			cur_gamma = (left + right) / 2
			runs += 1

			self.gamma = cur_gamma
			self.conv_optim()

			if self.rn >= DEFAULT_RN_CUTOFF:
				flag = 0
				return runs, flag

			if abs(self.rn - rn_goal) <= RN_SMALL:
				break
			elif self.rn > rn_goal:
				right = cur_gamma
			else:
				left = cur_gamma

			rn_rate = (self.rn / self.base_rn - 1) * 100
			if not silence:
				print_fl(f'  ...   gm = {self.gamma:.4f}, rn = {self.rn:.4f}, rate = {rn_rate:.1f}')

		return runs, flag

	def find_elbow(self, gammas, silence, DEFAULT_RN_CUTOFF):
		flag = 1

		# residual norm (x)
		rn = []
		# solution norm (y)
		sn = []

		# check monotonicity
		for gamma in gammas:
			self.gamma = gamma
			self.conv_optim()

			if self.rn >= DEFAULT_RN_CUTOFF:
				print_fl("Something wrong 1")
				flag = 0
				elbow_gamma = 0
				return elbow_gamma, flag, gammas, rn, sn

			# stop if rn (residual norm) > rn_limit
			if hasattr(self, 'rn_limit') and self.rn > self.rn_limit:
				break

			if not silence:
				print_fl(f'  ...   gamma = {gamma:.4g}, rn = {self.rn:.4g}, sn = {self.sn:.4g}')

			rn.append(self.rn)
			sn.append(self.sn)

		# check monotonicity
		all_idx = [0]
		last_idx = 0

		for idx in range(1, len(sn)):
			# delete if not monotonicity
			if rn[idx] >= rn[last_idx] and sn[idx] <= sn[last_idx]:
				# OK; update
				all_idx.append(idx)
				last_idx = idx

		rn = np.array(rn)[all_idx]
		sn = np.array(sn)[all_idx]
		gammas = np.array(gammas)[all_idx]


		if len(rn) < 2:
			print_fl("Not enough rn values to compute a gradient, using default")
			flag = 0
			elbow_gamma = DEFAULT_GM
			return elbow_gamma, flag, gammas, rn, sn

		# calculate curvature
		x_grad1 = np.gradient(rn)
		x_grad2 = np.gradient(x_grad1)
		y_grad1 = np.gradient(sn)
		y_grad2 = np.gradient(y_grad1)
		curvature = (x_grad1 * y_grad2 - y_grad1 * x_grad2) / ((x_grad1**2 + y_grad1**2)**(1.5))

		boundary = 1

		if not silence:
			print_fl("The x_grad1 is:", x_grad1)
			print_fl("The x_grad2 is:", x_grad2)
			print_fl("The curvature is:", curvature)

		max_val, max_pos = max((val, idx) for idx, val in enumerate(curvature[boundary:len(rn)]))
		max_pos = max_pos + boundary
		pos_left = max_pos - boundary
		pos_right = max_pos + boundary
		elbow_gamma = gammas[max_pos]

		return elbow_gamma, flag, gammas, rn, sn