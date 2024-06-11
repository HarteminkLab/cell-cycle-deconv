
import numpy as np
from matplotlib import pyplot as plt


class ToyReplication:
	"""
	Class to create an example replication timing profile from predefined origin locations.
	"""

	def __init__(self, n, origins, timepoints):

		self.n = n
		self.origins = origins
		self.timepoints = timepoints
		self.m = len(self.timepoints)
		self.std = 2.5

		# Replication speed
		self.rep_speed = 1 # 10kb / min
		self.rep_growth_per_min = self.rep_speed

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
					right_ind = j+1+self.rep_growth_per_min
					right_ind = max(right_ind, 0)
					updated_row[j:right_ind] = 2
			
			# Check if an origin should fire, if so double
			if time in self.origins.keys():
				for origin_pos in self.origins[time]:
					updated_row[origin_pos] = 2

			# Update the replication matrix row
			self.replication_matrix[i:, :] = updated_row
			self.avg_copy_num = self.replication_matrix.mean(axis=1)

		# Create a replication timing profile based on the index from 1 to 2
		# todo: index == time for this example
		self.replication_times = self.replication_matrix.argmax(axis=0)

	def set_reads(self, reads, tps):

		self.reads = reads
		self.tps = tps

	def copy_number_correction(self):
		from src.CopyNumberCorrection import copy_number_correct

		def normalize_read_counts(read_counts, norm_scale=5000., axis=1):
			"""Normalize reads across columns"""
			normalized_reads = (read_counts / \
				read_counts.sum(axis=axis).reshape((-1, 1))) * norm_scale
			return normalized_reads

		self.copy_num_2_props = create_copy_number_2_prop_mat(self.tps, self.replication_times, std=self.std)

		# Generate reads as constant vector of ones, and normalize
		self.normalized_reads = normalize_read_counts(self.reads)

		# Copy number correct and normalize the corrected reads
		self.corrected_copy_num_reads = copy_number_correct(self.copy_num_2_props, self.normalized_reads)
		self.normalized_corrected_reads = normalize_read_counts(self.corrected_copy_num_reads, axis=1)


	def plot_correction(self):
		extent = [0, self.n, -5, self.tps[-1]+5]
		 
		plt.figure(figsize=(13, 2))

		plt.subplot(1, 4, 1)
		plt.imshow(self.copy_num_2_props+1, origin='lower', aspect='auto', vmax=2.,
			extent=extent)
		plt.colorbar()
		plt.title("Copies of DNA")
		plt.ylabel("Time, min")
		plt.xlabel("Genomic position")

		plt.subplot(1, 4, 2)
		plt.imshow(self.normalized_reads, aspect='auto', origin='lower',
		          cmap='RdBu_r', vmin=70, vmax=130, extent=extent)
		plt.colorbar()
		plt.title("Uncorrected reads")

		plt.subplot(1, 4, 3)
		plt.imshow(self.corrected_copy_num_reads, aspect='auto', origin='lower',
		          cmap='RdBu_r', vmin=70, vmax=130, extent=extent)
		plt.colorbar()
		plt.title("Corrected reads")

		plt.subplot(1, 4, 4)
		plt.imshow(self.normalized_corrected_reads, aspect='auto', origin='lower', 
		    cmap='RdBu_r', vmin=70, vmax=130, extent=extent)
		plt.colorbar()
		plt.title("Normalized+Corrected\nfor copy number")


	def plot_replication(self):

		plt.figure(figsize=(7, 1.5))
		plt.subplots_adjust(wspace=0.5)
		plt.subplot(1, 2, 1)
		plt.imshow(self.replication_matrix, aspect='auto', origin='lower', 
			cmap='inferno',
			extent=[-0.5, self.n-.5, 0, self.timepoints[-1]])
		plt.title("Simulated Replication")
		plt.ylabel("Genome position, 10kb")
		plt.ylabel("Time")
		plt.xlabel("Genomic position")
		# plt.xticks(np.arange(0, self.replication_matrix.shape[1], 1))


		plt.subplot(1, 2, 2)
		# We should now have the expected copy number per genome per each row in the replication 
		# matrix

		plt.plot(self.timepoints, self.avg_copy_num)
		plt.xlabel("Time")
		plt.ylabel("Avg copy #")
		plt.title("Average copy number over time")

	def normalize_reads_matrix(self, reads_mat, plot=True):

		# Compute the original depth (number of reads per timepoint)
		self.original_depth = reads_mat.mean(axis=1)

		# Scale the number of reads per timepoint by the replication matrix (reads with two
		# copies are multiplied by 2). And compute the updated depth (increased)

		# Invert the replication matrix, regions with 2 copies need to be downscaled by half
		copy_mult_reads = (1./self.replication_matrix * reads_mat)
		self.scaled_depth = copy_mult_reads.mean(axis=1)

		# Now scaled the copy adjusted reads by the read depth to 
		# acheive the original read depth
		self.rescaled_reads = copy_mult_reads / self.scaled_depth.reshape((-1, 1)) \
			* self.original_depth.reshape(-1, 1)

		plt.figure(figsize=(13, 6))
		plt.subplot(2, 3, 1)
		plt.subplots_adjust(hspace=0.5, wspace=0.3)

		plt.imshow(reads_mat, aspect='auto', origin='lower', 
			cmap='RdBu_r', vmin=0, vmax=2.,
			extent=[0, self.n, 0, self.timepoints[-1]])
		plt.title("Reads")
		plt.ylabel("Genome position, 10kb")
		plt.ylabel("Time, min")
		plt.colorbar()

		plt.subplot(2, 3, 2)
		plt.imshow(copy_mult_reads, aspect='auto', origin='lower', 
			cmap='RdBu_r', vmin=0, vmax=2.,
			extent=[0, self.n, 0, self.timepoints[-1]])
		plt.title("Copy adjusted reads")
		plt.ylabel("Genome position, 10kb")
		plt.ylabel("Time, min")
		plt.colorbar()

		plt.subplot(2, 3, 3)
		plt.imshow(self.rescaled_reads, aspect='auto', origin='lower', 
			cmap='RdBu_r', vmin=0, vmax=2.,
			extent=[0, self.n, 0, self.timepoints[-1]])
		plt.title("Normalized reads")
		plt.ylabel("Genome position, 10kb")
		plt.ylabel("Time, min")
		plt.colorbar()


		plt.subplot(2, 3, 4)
		plt.plot(self.timepoints, self.original_depth)
		plt.xlabel("Time, min")
		plt.ylabel("Average reads")
		plt.title("Average reads")

		plt.subplot(2, 3, 5)
		plt.plot(self.timepoints, self.scaled_depth)
		plt.xlabel("Time, min")
		plt.ylabel("Average reads")
		plt.title("Average reads scaled by copy number")

		plt.subplot(2, 3, 6)
		plt.plot(self.timepoints, self.rescaled_reads.mean(axis=1))
		plt.xlabel("Time, min")
		plt.ylabel("Average reads")
		plt.title("Average reads normalized")

# Approximate a normal distribution of cells moving through the time course
# the proportion of replicated DNA is determined by the variation in the cells
# that have crossed the replication time

def create_copy_number_2_prop_mat(tps, replication_times, std=5):
	"""Create a copy number proportional matrix from replication times
	and timepoints we are interested in"""
	
	from scipy.stats.distributions import norm

	copy_num_2_props = np.zeros((len(tps), len(replication_times)))

	for j in range(len(replication_times)):
		replication_time = replication_times[j]    
		for i in range(len(tps)):
			t = tps[i]
			copy_num_2_prop = (1 - norm.cdf(replication_time, loc=t, scale=std))
			copy_num_2_props[i, j] = copy_num_2_prop

	return copy_num_2_props
