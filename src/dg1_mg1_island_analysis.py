import numpy as np
import matplotlib.pyplot as plt
from src.f_island_detector import IslandDetector

class ChromatinOccupancyIslandDetector:
	def __init__(self, config1, config2, F):
		"""
		Initialize the chromatin occupancy island detector.
		
		Parameters:
		- config1: Configuration object for getting phase positions
		- config2: Second configuration (not used currently)
		- F: Deconvolution solution array
		"""
		self.config1 = config1
		self.config2 = config2
		self.F = F
		
		# Reshape F to image format and extract phase-specific arrays
		self._setup_phase_arrays()
		
		# Initialize storage for processed data
		self.mg1_avg = None
		self.dg1_avg = None
		self.filtered_differences = None
		self.differences_2d = None
		self.filter_mask = None
		self.thresholds = {}
		self.islands = {}
	
	def _setup_phase_arrays(self):
		"""Extract mother and daughter G1 arrays from F"""
		# Reshape F to (time, fragment_length, genomic_position)
		self.F_imgs = self.F.reshape((self.F.shape[0], 26, -1))
		print(f"F_imgs shape: {self.F_imgs.shape}")
		
		# Get phase-specific indices
		self.mg1_indices = self.config1.get_Hpositions_for_phase("CG1")
		self.dg1_indices = self.config1.get_Hpositions_for_phase('DG1')
		
		# Extract phase-specific arrays
		self.mg1_F = self.F_imgs[self.mg1_indices]  # (time, fragment_length, genomic_position)
		self.dg1_F = self.F_imgs[self.dg1_indices]  # (time, fragment_length, genomic_position)
		
		print(f"MG1 shape: {self.mg1_F.shape}")
		print(f"DG1 shape: {self.dg1_F.shape}")
	
	def step1_preprocess_data(self, min_occupancy=1):
		"""
		Step 1: Average over time, filter low values, compute differences
		
		Parameters:
		- min_occupancy: Minimum value threshold for filtering
		"""
		print("=== Step 1: Data Preprocessing ===")
		
		# Average over time dimension (axis=0)
		self.mg1_avg = self.mg1_F.mean(axis=0)  # (fragment_length, genomic_position)
		self.dg1_avg = self.dg1_F.mean(axis=0)  # (fragment_length, genomic_position)
		
		print(f"After time averaging - MG1: {self.mg1_avg.shape}, DG1: {self.dg1_avg.shape}")
		
		# Create filter mask: keep positions where at least one array > min_occupancy
		self.filter_mask = (self.mg1_avg > min_occupancy) | (self.dg1_avg > min_occupancy)
		
		# Apply filter to both arrays
		mg1_filtered = self.mg1_avg[self.filter_mask]
		dg1_filtered = self.dg1_avg[self.filter_mask]
		
		# Compute differences for filtered data (for distribution analysis)
		self.filtered_differences = dg1_filtered - mg1_filtered
		
		# Compute 2D differences (for island detection)
		self.differences_2d = self.dg1_avg - self.mg1_avg
		
		print(f"Total positions: {self.mg1_avg.size}")
		print(f"Filtered positions: {len(self.filtered_differences)}")
		print(f"Filtered percentage: {len(self.filtered_differences)/self.mg1_avg.size*100:.1f}%")
		
		return self.filtered_differences, self.differences_2d
	
	def step2_analyze_distribution(self, std_multiplier=1, 
			bins=80, xlim=(-50, 50), plot=True):
		"""
		Step 2: Analyze distribution and calculate thresholds
		
		Parameters:
		- std_multiplier: Number of standard deviations for threshold
		- bins: Number of histogram bins
		- xlim: X-axis limits for plotting
		"""
		print("=== Step 2: Distribution Analysis ===")
		
		if self.filtered_differences is None:
			raise ValueError("Must run step1_preprocess_data() first")
		
		# Calculate statistics
		mean_diff = self.filtered_differences.mean()
		std_diff = self.filtered_differences.std()

		# Calculate symmetric thresholds with minimum threshold
		min_threshold = 0.1
		threshold_magnitude = max(std_diff * std_multiplier, min_threshold)

		lower_threshold = mean_diff - threshold_magnitude
		upper_threshold = mean_diff + threshold_magnitude
		
		self.thresholds['std'] = {
			'lower': lower_threshold,
			'upper': upper_threshold,
			'mean': mean_diff,
			'std': std_diff,
			'multiplier': std_multiplier
		}
		
		print(f"Mean difference: {mean_diff:.3f}")
		print(f"Std difference: {std_diff:.3f}")
		print(f"Threshold range: [{lower_threshold:.3f}, {upper_threshold:.3f}]")
		
		# Plot distribution
		if plot:
			fig, ax = plt.subplots(figsize=(4, 3))
			
			ax.hist(self.filtered_differences, bins=np.linspace(*xlim, bins), 
					alpha=0.7, edgecolor='black', linewidth=0.5)
			
			ax.axvline(lower_threshold, color='red', linestyle='--', 
					  label=f'Lower threshold: {lower_threshold:.2f}')
			ax.axvline(upper_threshold, color='red', linestyle='--', 
					  label=f'Upper threshold: {upper_threshold:.2f}')
			ax.axvline(mean_diff, color='green', linestyle='-', alpha=0.7,
					  label=f'Mean: {mean_diff:.2f}')
			
			ax.set_xlim(xlim)
			ax.set_ylim(0, 50)
			ax.set_xlabel('DG1 - MG1 Occupancy Difference')
			ax.set_ylabel('Frequency')
			ax.set_title(f'Distribution of Occupancy Differences (±{std_multiplier}σ thresholds)')
			ax.legend()
			plt.tight_layout()
		
		return self.thresholds['std']
	
	def step3_detect_islands(self, threshold_method='std', max_distance=2, min_size=3):
		"""
		Step 3: Detect islands using thresholds on 2D differences
		
		Parameters:
		- threshold_method: Which threshold to use ('std')
		- max_distance: Maximum distance for connecting pixels into islands
		- min_size: Minimum island size (number of pixels)
		"""
		print("=== Step 3: Island Detection ===")
		
		if threshold_method not in self.thresholds:
			raise ValueError(f"Threshold method '{threshold_method}' not calculated. Run step2 first.")
		
		if self.differences_2d is None:
			raise ValueError("Must run step1_preprocess_data() first")
		
		thresh = self.thresholds[threshold_method]
		
		# Create binary masks for positive and negative islands
		positive_mask = self.differences_2d > thresh['upper']
		negative_mask = self.differences_2d < thresh['lower']
		
		print(f"Positive outlier pixels: {positive_mask.sum()}")
		print(f"Negative outlier pixels: {negative_mask.sum()}")
		
		# Detect islands for each direction
		positive_islands = self._detect_islands_from_mask(
			positive_mask, 'positive', max_distance, min_size)
		negative_islands = self._detect_islands_from_mask(
			negative_mask, 'negative', max_distance, min_size)
		
		self.islands[threshold_method] = {
			'positive': positive_islands,
			'negative': negative_islands,
			'parameters': {
				'max_distance': max_distance,
				'min_size': min_size,
				'threshold_method': threshold_method
			}
		}
		
		print(f"Detected {len(positive_islands)} positive islands")
		print(f"Detected {len(negative_islands)} negative islands")
		
		return positive_islands, negative_islands
	
	def _detect_islands_from_mask(self, mask, direction, max_distance, min_size):
		"""Use manual island detection instead of scipy"""
		
		# Create island detector
		detector = IslandDetector(max_distance=max_distance)
		
		# Find islands as boolean masks
		island_masks = detector.find_islands(mask, min_size=min_size)
		
		islands = []
		for i, island_mask in enumerate(island_masks):
			# Get positions and calculate statistics
			positions = np.where(island_mask)
			island_values = self.differences_2d[positions]
			
			island_info = {
				'id': i + 1,
				'direction': direction,
				'size': island_mask.sum(),
				'mean_signal': np.mean(island_values),
				'positions': positions,
				'mask': island_mask,  # Include the boolean mask
				'fragment_span': (np.min(positions[0]), np.max(positions[0])),
				'genomic_span': (np.min(positions[1]), np.max(positions[1]))
			}
			islands.append(island_info)
		
		return islands
	
	def plot_islands_overview(self, threshold_method='std', figsize=(15, 5)):
		"""Plot overview of detected islands"""
		if threshold_method not in self.islands:
			raise ValueError(f"Islands for '{threshold_method}' not detected. Run step3 first.")
		
		islands_data = self.islands[threshold_method]
		
		fig, axes = plt.subplots(1, 3, figsize=figsize)
		
		# Plot 1: Original differences
		im1 = axes[0].imshow(self.differences_2d, cmap='RdBu_r', aspect='auto', 
							origin='lower', interpolation='none')
		axes[0].set_title('Original Differences (DG1-MG1)')
		axes[0].set_xlabel('Genomic Position')
		axes[0].set_ylabel('Fragment Length')
		plt.colorbar(im1, ax=axes[0])
		
		# Plot 2: Positive islands
		pos_overlay = np.zeros_like(self.differences_2d)
		for island in islands_data['positive']:
			pos_overlay[island['positions']] = island['mean_signal']
		
		im2 = axes[1].imshow(pos_overlay, cmap='Reds', aspect='auto', 
							origin='lower', interpolation='none')
		axes[1].set_title(f'Positive Islands (n={len(islands_data["positive"])})')
		axes[1].set_xlabel('Genomic Position')
		axes[1].set_ylabel('Fragment Length')
		plt.colorbar(im2, ax=axes[1])
		
		# Plot 3: Negative islands
		neg_overlay = np.zeros_like(self.differences_2d)
		for island in islands_data['negative']:
			neg_overlay[island['positions']] = island['mean_signal']
		
		im3 = axes[2].imshow(neg_overlay, cmap='Blues_r', aspect='auto', 
							origin='lower', interpolation='none')
		axes[2].set_title(f'Negative Islands (n={len(islands_data["negative"])})')
		axes[2].set_xlabel('Genomic Position')
		axes[2].set_ylabel('Fragment Length')
		plt.colorbar(im3, ax=axes[2])
		
		plt.tight_layout()
		plt.show()
	
	def summary_statistics(self, threshold_method='std'):
		"""Print summary statistics for detected islands"""
		if threshold_method not in self.islands:
			print(f"No islands detected for method '{threshold_method}'")
			return
		
		islands_data = self.islands[threshold_method]
		
		print(f"\n=== Island Summary ({threshold_method}) ===")
		
		for direction in ['positive', 'negative']:
			islands = islands_data[direction]
			if islands:
				sizes = [isl['size'] for isl in islands]
				signals = [abs(isl['mean_signal']) for isl in islands]
				
				print(f"\n{direction.capitalize()} Islands:")
				print(f"  Count: {len(islands)}")
				print(f"  Size - Mean: {np.mean(sizes):.1f}, Range: [{min(sizes)}, {max(sizes)}]")
				print(f"  Signal - Mean: {np.mean(signals):.3f}, Range: [{min(signals):.3f}, {max(signals):.3f}]")
				
				# Show largest island
				largest = max(islands, key=lambda x: x['size'])
				print(f"  Largest: {largest['size']} pixels, signal: {largest['mean_signal']:.3f}")
			else:
				print(f"\n{direction.capitalize()} Islands: None detected")
	
	def detect_total_occupancy_islands(self, occupancy_threshold=1, max_distance=2, min_size=3):
		"""
		Step 4a: Detect islands in mean occupancy (mg1 + dg1)/2
		
		Parameters:
		- occupancy_threshold: Minimum mean occupancy for island detection
		- max_distance: Maximum distance for island connectivity
		- min_size: Minimum island size
		"""
		print("=== Step 4a: Total Occupancy Islands ===")
		
		if self.mg1_avg is None or self.dg1_avg is None:
			raise ValueError("Must run step1_preprocess_data() first")
		
		# Compute mean occupancy
		mean_occupancy = (self.mg1_avg + self.dg1_avg) / 2
		
		# Create binary mask for occupancy
		occupancy_mask = mean_occupancy > occupancy_threshold
		
		print(f"Occupancy threshold: {occupancy_threshold}")
		print(f"Pixels above threshold: {occupancy_mask.sum()}")
		
		# Use IslandDetector to find islands
		detector = IslandDetector(max_distance=max_distance)
		island_masks = detector.find_islands(occupancy_mask, min_size=min_size)
		
		# Convert to island info format
		total_islands = []
		for i, island_mask in enumerate(island_masks):
			positions = np.where(island_mask)
			island_occupancy = mean_occupancy[positions]
			
			island_info = {
				'id': i + 1,
				'direction': 'total',
				'size': island_mask.sum(),
				'mean_signal': np.mean(island_occupancy),
				'positions': positions,
				'mask': island_mask,
				'fragment_span': (np.min(positions[0]), np.max(positions[0])),
				'genomic_span': (np.min(positions[1]), np.max(positions[1]))
			}
			total_islands.append(island_info)
		
		print(f"Detected {len(total_islands)} total occupancy islands")
		
		return total_islands, occupancy_mask
	
	def detect_all_island_types(self, threshold_method='std', occupancy_threshold=1, 
							   max_distance=2, min_size=3):
		"""
		Step 4: Master method to detect all four island types
		
		Parameters:
		- threshold_method: Which threshold to use for M/D islands
		- occupancy_threshold: Threshold for total occupancy islands
		- max_distance: Maximum distance for connectivity  
		- min_size: Minimum island size
		"""
		print("=== Step 4: Comprehensive Island Detection ===")
		
		# Step 4a: Find total occupancy islands
		total_islands, total_occupancy_mask = self.detect_total_occupancy_islands(
			occupancy_threshold, max_distance, min_size)
		
		# Step 4b: Find mother/daughter islands (reuse existing logic)
		if threshold_method not in self.islands:
			print(f"Running step3 to detect {threshold_method} islands...")
			self.step3_detect_islands(threshold_method, max_distance, min_size)
		
		mother_islands = self.islands[threshold_method]['negative']  # negative differences = mother-specific
		daughter_islands = self.islands[threshold_method]['positive']  # positive differences = daughter-specific
		
		# Step 4c: Create combined M/D pixel mask
		md_pixels_mask = np.zeros_like(total_occupancy_mask, dtype=bool)
		
		# Add mother island pixels
		for island in mother_islands:
			md_pixels_mask[island['positions']] = True
			
		# Add daughter island pixels  
		for island in daughter_islands:
			md_pixels_mask[island['positions']] = True
		
		print(f"Mother/Daughter pixels: {md_pixels_mask.sum()}")
		
		# Step 4d: Subtract M/D pixels from total occupancy and re-detect
		remaining_mask = total_occupancy_mask & ~md_pixels_mask
		print(f"Remaining pixels after M/D subtraction: {remaining_mask.sum()}")
		
		# Re-run island detection on remaining mask
		detector = IslandDetector(max_distance=max_distance)
		unchanging_island_masks = detector.find_islands(remaining_mask, min_size=min_size)
		
		# Convert unchanging islands to info format
		unchanging_islands = []
		for i, island_mask in enumerate(unchanging_island_masks):
			positions = np.where(island_mask)
			island_differences = self.differences_2d[positions]  # Use differences for signal
			
			island_info = {
				'id': i + 1,
				'direction': 'unchanging',
				'size': island_mask.sum(),
				'mean_signal': np.mean(island_differences),  # Should be close to 0
				'positions': positions,
				'mask': island_mask,
				'fragment_span': (np.min(positions[0]), np.max(positions[0])),
				'genomic_span': (np.min(positions[1]), np.max(positions[1]))
			}
			unchanging_islands.append(island_info)
		
		print(f"Detected {len(unchanging_islands)} unchanging islands")
		
		# Store all island types
		self.all_islands = {
			'total': total_islands,
			'mother': mother_islands,
			'daughter': daughter_islands, 
			'unchanging': unchanging_islands,
			'parameters': {
				'threshold_method': threshold_method,
				'occupancy_threshold': occupancy_threshold,
				'max_distance': max_distance,
				'min_size': min_size
			}
		}
		
		return self.all_islands
	
	def plot_all_islands_overview(self, figsize=(20, 4)):
		"""Plot overview of all four island types"""
		if not hasattr(self, 'all_islands'):
			raise ValueError("Must run detect_all_island_types() first")
		
		fig, row_axes = plt.subplots(2, 3, figsize=figsize)
		
		all_axes = np.array(row_axes).flatten()

		island_types = ['total', 'unchanging', 'mother', 'daughter']
		colors = ['viridis', 'Greys', 'Blues', 'Reds']

		ax = all_axes[0]
		im = ax.imshow(self.F_imgs.mean(0), cmap='magma_r', vmin=0, vmax=40,
			origin='lower', aspect='auto')
		plt.colorbar(im, ax=ax)
		ax.set_title("Average MG1/DG1 data")

		ax = all_axes[2]
		im = ax.imshow(self.differences_2d, cmap='RdBu_r', vmin=-20, vmax=20,
			origin='lower', aspect='auto')
		plt.colorbar(im, ax=ax)
		ax.set_title("Difference DG1 - MG1")

		axes = np.concatenate([all_axes[1:2], all_axes[3:]])
		for i, (island_type, cmap) in enumerate(zip(island_types, colors)):
			islands = self.all_islands[island_type]
			
			# Create overlay
			overlay = np.zeros_like(self.differences_2d)
			for island in islands:
				overlay[island['positions']] = abs(island['mean_signal'])
			
			im = axes[i].imshow(overlay, cmap=cmap, aspect='auto', 
							  origin='lower', interpolation='none')
			axes[i].set_title(f'{island_type.capitalize()} Islands\n(n={len(islands)})')
			axes[i].set_xlabel('Genomic Position')
			if i == 0:
				axes[i].set_ylabel('Fragment Length')
			plt.colorbar(im, ax=axes[i])
		
		plt.tight_layout()
		plt.show()
	
	def comprehensive_summary(self, detail=False):
		"""Print comprehensive statistics for all island types"""
		if not hasattr(self, 'all_islands'):
			print("No comprehensive islands detected. Run detect_all_island_types() first.")
			return
		
		# Calculate all metrics
		composition_stats = self.calculate_chromatin_composition()
		
		print("\n" + "="*60)
		print("CHROMATIN LANDSCAPE COMPOSITION ANALYSIS")
		print("="*60)
		
		# Key Summary (Venn diagram style)
		print(f"\n🧬 CHROMATIN BREAKDOWN:")
		print(f"   Total chromatin regions detected: {composition_stats['total_islands']['count']} islands")
		print(f"   Genome coverage by chromatin: {composition_stats['genome_coverage']:.1f}%")
		
		print(f"\n📊 CHROMATIN COMPOSITION (by island count):")
		print(f"   • Mother-specific:    {composition_stats['island_proportions']['mother']:.1f}% ({composition_stats['mother_islands']['count']} islands)")
		print(f"   • Daughter-specific:  {composition_stats['island_proportions']['daughter']:.1f}% ({composition_stats['daughter_islands']['count']} islands)")
		print(f"   • Unchanging:         {composition_stats['island_proportions']['unchanging']:.1f}% ({composition_stats['unchanging_islands']['count']} islands)")
		
		print(f"\n📊 CHROMATIN COMPOSITION (by pixel coverage):")
		print(f"   • Mother-specific:    {composition_stats['pixel_proportions']['mother']:.1f}%")
		print(f"   • Daughter-specific:  {composition_stats['pixel_proportions']['daughter']:.1f}%")
		print(f"   • Unchanging:         {composition_stats['pixel_proportions']['unchanging']:.1f}%")
		
		if detail:
			# Detailed breakdown
			print(f"\n" + "-"*50)
			print("DETAILED ISLAND STATISTICS")
			print("-"*50)
			
			for island_type in ['mother', 'daughter', 'unchanging']:
				stats = composition_stats[f'{island_type}_islands']
				
				if stats['count'] > 0:
					print(f"\n{island_type.capitalize()} Islands:")
					print(f"  Count: {stats['count']}")
					print(f"  Total pixels: {stats['total_pixels']:,}")
					print(f"  Size - Mean: {stats['mean_size']:.1f}, Range: [{stats['min_size']}, {stats['max_size']}]")
					print(f"  Signal - Mean: {stats['mean_signal']:.3f}, Range: [{stats['min_signal']:.3f}, {stats['max_signal']:.3f}]")
					print(f"  Largest island: {stats['largest_size']} pixels (signal: {stats['largest_signal']:.3f})")
				else:
					print(f"\n{island_type.capitalize()} Islands: None detected")
		
		return composition_stats

	def calculate_chromatin_composition(self):
		"""Calculate comprehensive composition statistics"""
		if not hasattr(self, 'all_islands'):
			raise ValueError("Must run detect_all_island_types() first")
		
		total_genome_pixels = self.mg1_avg.size
		
		# Count and pixel statistics for each type
		stats = {}
		
		for island_type in ['total', 'mother', 'daughter', 'unchanging']:
			islands = self.all_islands[island_type]
			
			if islands:
				sizes = [isl['size'] for isl in islands]
				signals = [abs(isl['mean_signal']) for isl in islands]
				
				stats[f'{island_type}_islands'] = {
					'count': len(islands),
					'total_pixels': sum(sizes),
					'mean_size': np.mean(sizes),
					'min_size': min(sizes),
					'max_size': max(sizes),
					'mean_signal': np.mean(signals),
					'min_signal': min(signals),
					'max_signal': max(signals),
					'largest_size': max(sizes),
					'largest_signal': max(islands, key=lambda x: x['size'])['mean_signal']
				}
			else:
				stats[f'{island_type}_islands'] = {
					'count': 0, 'total_pixels': 0, 'mean_size': 0,
					'min_size': 0, 'max_size': 0, 'mean_signal': 0,
					'min_signal': 0, 'max_signal': 0, 'largest_size': 0, 'largest_signal': 0
				}
		
		# Calculate proportions
		total_chromatin_pixels = stats['total_islands']['total_pixels']
		total_chromatin_islands = stats['total_islands']['count']
		
		# Functional islands (M + D + Unchanging should sum to total)
		functional_islands = (stats['mother_islands']['count'] + 
							stats['daughter_islands']['count'] + 
							stats['unchanging_islands']['count'])
		
		functional_pixels = (stats['mother_islands']['total_pixels'] + 
						   stats['daughter_islands']['total_pixels'] + 
						   stats['unchanging_islands']['total_pixels'])
		
		# Island count proportions
		if functional_islands > 0:
			island_proportions = {
				'mother': (stats['mother_islands']['count'] / functional_islands) * 100,
				'daughter': (stats['daughter_islands']['count'] / functional_islands) * 100,
				'unchanging': (stats['unchanging_islands']['count'] / functional_islands) * 100
			}
		else:
			island_proportions = {'mother': 0, 'daughter': 0, 'unchanging': 0}
		
		# Pixel coverage proportions  
		if functional_pixels > 0:
			pixel_proportions = {
				'mother': (stats['mother_islands']['total_pixels'] / functional_pixels) * 100,
				'daughter': (stats['daughter_islands']['total_pixels'] / functional_pixels) * 100,
				'unchanging': (stats['unchanging_islands']['total_pixels'] / functional_pixels) * 100
			}
		else:
			pixel_proportions = {'mother': 0, 'daughter': 0, 'unchanging': 0}
		
		# Dynamic vs static analysis
		dynamic_islands = stats['mother_islands']['count'] + stats['daughter_islands']['count']
		dynamic_pixels = stats['mother_islands']['total_pixels'] + stats['daughter_islands']['total_pixels']
		
		if functional_islands > 0 and functional_pixels > 0:
			dynamics = {
				'dynamic_islands': dynamic_islands,
				'static_islands': stats['unchanging_islands']['count'],
				'dynamic_pixels': dynamic_pixels,
				'static_pixels': stats['unchanging_islands']['total_pixels'],
				'dynamic_percent': (dynamic_pixels / functional_pixels) * 100,
				'static_percent': (stats['unchanging_islands']['total_pixels'] / functional_pixels) * 100,
				'dynamic_ratio': dynamic_pixels / max(stats['unchanging_islands']['total_pixels'], 1)
			}
		else:
			dynamics = {
				'dynamic_islands': 0, 'static_islands': 0, 'dynamic_pixels': 0, 'static_pixels': 0,
				'dynamic_percent': 0, 'static_percent': 0, 'dynamic_ratio': 0
			}

		self.composition_stats = {
			**stats,
			'genome_coverage': (total_chromatin_pixels / total_genome_pixels) * 100,
			'island_proportions': island_proportions,
			'pixel_proportions': pixel_proportions,
			'dynamics': dynamics,
			'functional_islands_total': functional_islands,
			'functional_pixels_total': functional_pixels
		}

		return self.composition_stats
