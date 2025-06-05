import numpy as np

class IslandDetector:
	def __init__(self, max_distance=1):
		"""
		Simple island detector using DFS with Manhattan distance.
		
		Parameters:
		- max_distance: Maximum Manhattan distance for connectivity
		"""
		self.max_distance = max_distance
	
	def find_islands(self, binary_mask, min_size=1):
		"""
		Find all islands in a binary mask.
		
		Parameters:
		- binary_mask: 2D boolean array where True indicates candidate pixels
		- min_size: Minimum number of pixels required for an island
		
		Returns:
		- List of boolean masks, one for each island
		"""
		self.binary_mask = binary_mask
		self.visited = np.zeros_like(binary_mask, dtype=bool)
		self.rows, self.cols = binary_mask.shape
		
		islands = []
		
		# Scan through every pixel
		for row in range(self.rows):
			for col in range(self.cols):
				# If pixel is part of mask and not yet visited
				if binary_mask[row, col] and not self.visited[row, col]:
					
					# Start new island - create empty mask for this island
					island_mask = np.zeros_like(binary_mask, dtype=bool)
					
					# Use DFS to find all connected pixels
					self._dfs(row, col, island_mask)
					
					# Only keep islands that meet minimum size
					if island_mask.sum() >= min_size:
						islands.append(island_mask)
		
		return islands
	
	def _dfs(self, row, col, island_mask):
		"""
		Depth-first search to find all pixels connected to (row, col).
		
		Parameters:
		- row, col: Starting position
		- island_mask: Boolean mask to fill with this island's pixels
		"""
		# Mark this pixel as visited and part of current island
		self.visited[row, col] = True
		island_mask[row, col] = True
		
		# Check all potential neighbors within max_distance
		for dr in range(-self.max_distance, self.max_distance + 1):
			for dc in range(-self.max_distance, self.max_distance + 1):
				
				# Skip center pixel (already processed)
				if dr == 0 and dc == 0:
					continue
				
				# Calculate neighbor position
				new_row, new_col = row + dr, col + dc
				
				# Check if neighbor is valid
				if self._is_valid_neighbor(new_row, new_col, dr, dc):
					# Recursively explore this neighbor
					self._dfs(new_row, new_col, island_mask)
	
	def _is_valid_neighbor(self, row, col, dr, dc):
		"""
		Check if a neighbor pixel is valid for inclusion.
		
		Parameters:
		- row, col: Neighbor position
		- dr, dc: Row and column offsets from current pixel
		
		Returns:
		- True if neighbor should be included in island
		"""
		# Check bounds
		if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
			return False
		
		# Check Manhattan distance
		manhattan_dist = abs(dr) + abs(dc)
		if manhattan_dist > self.max_distance:
			return False
		
		# Check if pixel is part of mask and not already visited
		if not self.binary_mask[row, col] or self.visited[row, col]:
			return False
		
		return True
	
	def visualize_islands(self, islands, figsize=(12, 4)):
		"""
		Visualize detected islands.
		
		Parameters:
		- islands: List of boolean masks from find_islands()
		"""
		import matplotlib.pyplot as plt
		
		if not islands:
			print("No islands to visualize")
			return
		
		# Create combined visualization
		fig, axes = plt.subplots(1, min(3, len(islands) + 1), figsize=figsize)
		if len(islands) == 0:
			axes = [axes]
		elif len(islands) == 1:
			axes = [axes[0], axes[1]] if hasattr(axes, '__len__') else [axes]
		
		# Plot original mask
		combined_mask = np.zeros_like(islands[0], dtype=int)
		for i, island in enumerate(islands):
			combined_mask[island] = i + 1
		
		if len(axes) > 0:
			im = axes[0].imshow(combined_mask, cmap='tab10', aspect='auto', 
							  origin='lower', interpolation='none')
			axes[0].set_title(f'All Islands (n={len(islands)})')
			axes[0].set_xlabel('Genomic Position')
			axes[0].set_ylabel('Fragment Length')
			plt.colorbar(im, ax=axes[0])
		
		# Plot individual islands
		for i, island in enumerate(islands[:min(2, len(islands))]):
			if i + 1 < len(axes):
				axes[i + 1].imshow(island.astype(int), cmap='Blues', 
								 aspect='auto', origin='lower', interpolation='none')
				axes[i + 1].set_title(f'Island {i+1} (size: {island.sum()})')
				axes[i + 1].set_xlabel('Genomic Position')
				axes[i + 1].set_ylabel('Fragment Length')
		
		plt.tight_layout()
		plt.show()
	
	def island_statistics(self, islands):
		"""
		Calculate statistics for detected islands.
		
		Parameters:
		- islands: List of boolean masks from find_islands()
		
		Returns:
		- Dictionary with island statistics
		"""
		if not islands:
			return {'count': 0}
		
		sizes = [island.sum() for island in islands]
		
		stats = {
			'count': len(islands),
			'total_pixels': sum(sizes),
			'size_mean': np.mean(sizes),
			'size_std': np.std(sizes),
			'size_min': min(sizes),
			'size_max': max(sizes),
			'sizes': sizes
		}
		
		return stats

# Usage example:
# detector = IslandDetector(max_distance=2)
# islands = detector.find_islands(binary_mask, min_size=3)
# detector.visualize_islands(islands)
# stats = detector.island_statistics(islands)
# print(f"Found {stats['count']} islands with mean size {stats['size_mean']:.1f}")
