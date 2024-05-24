
import numpy as np
from matplotlib import pyplot as plt


class ToyReplication:
	"""
	Class to create an example replication timing profile from predefined origin locations.
	"""

	def __init__(self, n, origins, step_min, length_timecourse):

		self.n = n
		self.length_timecourse = length_timecourse
		self.step_min = step_min
		self.origins = origins
		self.timepoints = np.arange(0, length_timecourse+step_min, step_min)
		self.m = len(self.timepoints)

		# Replication speed
		self.rep_speed = 1 # 10kb / min
		self.rep_growth_per_min = self.rep_speed * step_min

		# Start copy number as 1 everywhere
		self.replication_matrix = (np.zeros((self.m, n))+1).astype(int)

	def replicate(self):

		# Loop through each timepoint, 
		# a. Iterate through genome and grow replicated region by replication speed 
		# b. Check if an origin should fire, if so double
		# the copy number at the origin location, if it is already doubled, ignore
		for i in range(0, self.m):
			time = self.timepoints[i]

			# Iterate through genome and grow copy number
			# at replicated position boundaries
			row = self.replication_matrix[i, :]
			updated_row = row.copy()

			for j in range(1, self.n-1):
				
				# Assume replication speed of 10kb / min
				prev_copy = row[j-1]
				cur_copy = row[j]
				next_copy = row[j+1]
				
				# Found a left end of the replication fork
				if prev_copy == 1 and cur_copy == 2:
					left_ind = j-self.rep_growth_per_min
					left_ind = max(left_ind, 0)
					updated_row[left_ind:j] = 2
				
				# Found a right end of the replication fork
				if next_copy == 1 and cur_copy == 2:
					right_ind = j+self.rep_growth_per_min
					right_ind = max(right_ind, 0)
					updated_row[j:right_ind] = 2
			
			# Check if an origin should fire, if so double
			if time in self.origins.keys():
				for origin_pos in self.origins[time]:
					updated_row[origin_pos] = 2

			# Update the replication matrix row
			self.replication_matrix[i:, :] = updated_row
			self.avg_copy_num = self.replication_matrix.mean(axis=1)

	def plot_replication(self):

		plt.figure(figsize=(10, 2))
		plt.subplot(1, 2, 1)
		plt.imshow(self.replication_matrix, aspect='auto', origin='lower', 
			cmap='inferno',
		    extent=[0, self.n, 0, self.timepoints[-1]])
		plt.title("Simulated Replication")

		plt.subplot(1, 2, 2)
		# We should now have the expected copy number per genome per each row in the replication 
		# matrix

		plt.plot(self.timepoints, self.avg_copy_num)
		plt.xlabel("Time, min")
		plt.ylabel("Average copies / genome")
