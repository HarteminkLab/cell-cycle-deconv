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
	
	def detect_all_island_types(self, threshold_method='std', 

							   # Total islands (step 1)
							   occupancy_threshold=5, total_max_distance=2, total_min_size=3,

							   # M/D islands (step 2) 
							   md_max_distance=8, md_min_size=2):
		"""
		3-step comprehensive island detection:
		1. Find total chromatin islands (occupancy-based noise removal)
		2. Find M/D signal islands within chromatin pixels  
		3. Classify all chromatin pixels as mother/daughter/unchanging
		"""
		print("=== Step 4: 3-Step Comprehensive Island Detection ===")
		
		# Step 1: Find total chromatin landscape
		print("Step 1: Detecting total chromatin islands...")
		total_islands, total_chromatin_mask = self.detect_total_occupancy_islands(
			occupancy_threshold, total_max_distance, total_min_size)
		
		print(f"  Found {len(total_islands)} total chromatin islands")
		print(f"  Total chromatin pixels: {total_chromatin_mask.sum()}")
		
		# Step 2: Detect M/D signal islands within chromatin regions
		print("Step 2: Detecting mother/daughter signal islands...")
		
		if threshold_method not in self.thresholds:
			print(f"  Running step2 to calculate {threshold_method} thresholds...")
			self.step2_analyze_distribution()
		
		thresh = self.thresholds[threshold_method]
		print(f"  Using thresholds: lower={thresh['lower']:.3f}, upper={thresh['upper']:.3f}")
		
		# Classify pixels within chromatin regions only
		mother_pixel_mask = np.zeros_like(total_chromatin_mask, dtype=bool)
		daughter_pixel_mask = np.zeros_like(total_chromatin_mask, dtype=bool)
		
		# Get chromatin pixel positions and their difference values
		chromatin_positions = np.where(total_chromatin_mask)
		chromatin_differences = self.differences_2d[chromatin_positions]
		
		# Classify chromatin pixels by threshold
		mother_indices = chromatin_differences < thresh['lower']
		daughter_indices = chromatin_differences > thresh['upper']
		
		# Create M/D pixel masks (only within chromatin regions)
		mother_pixel_mask[chromatin_positions[0][mother_indices], 
						  chromatin_positions[1][mother_indices]] = True
		daughter_pixel_mask[chromatin_positions[0][daughter_indices], 
							chromatin_positions[1][daughter_indices]] = True
		
		print(f"  Mother pixels: {mother_pixel_mask.sum()}")
		print(f"  Daughter pixels: {daughter_pixel_mask.sum()}")
		print(f"  Unchanging pixels: {total_chromatin_mask.sum() - mother_pixel_mask.sum() - daughter_pixel_mask.sum()}")
		
		# Group M/D pixels into islands for noise removal
		detector = IslandDetector(max_distance=md_max_distance)
		
		mother_island_masks = detector.find_islands(mother_pixel_mask, min_size=md_min_size)
		mother_islands = self._convert_masks_to_islands(mother_island_masks, 'mother')
		
		daughter_island_masks = detector.find_islands(daughter_pixel_mask, min_size=md_min_size)
		daughter_islands = self._convert_masks_to_islands(daughter_island_masks, 'daughter')
		
		print(f"  → Mother islands: {len(mother_islands)} (after size/distance filtering)")
		print(f"  → Daughter islands: {len(daughter_islands)} (after size/distance filtering)")
		
		# Step 3: Classify remaining chromatin pixels as unchanging
		print("Step 3: Classifying remaining pixels as unchanging...")
		
		# Create mask of all M/D island pixels (post-filtering)
		md_islands_mask = np.zeros_like(total_chromatin_mask, dtype=bool)
		for island in mother_islands + daughter_islands:
			md_islands_mask[island['positions']] = True
		
		# Remaining chromatin pixels = unchanging
		unchanging_pixel_mask = total_chromatin_mask & ~md_islands_mask
		
		# Group unchanging pixels into islands
		unchanging_island_masks = detector.find_islands(unchanging_pixel_mask, min_size=md_min_size)
		unchanging_islands = self._convert_masks_to_islands(unchanging_island_masks, 'unchanging')
		
		print(f"  → Unchanging islands: {len(unchanging_islands)}")
		
		# Verification: Check that all chromatin pixels are classified
		total_classified = md_islands_mask.sum() + unchanging_pixel_mask.sum()
		print(f"\nVerification:")
		print(f"  Total chromatin pixels: {total_chromatin_mask.sum()}")
		print(f"  Classified pixels: {total_classified}")
		print(f"  Coverage: {total_classified/total_chromatin_mask.sum()*100:.1f}%")
		
		# Store results
		self.all_islands = {
			'total': total_islands,
			'mother': mother_islands,
			'daughter': daughter_islands,
			'unchanging': unchanging_islands,
			'parameters': {
				'threshold_method': threshold_method,
				'occupancy_threshold': occupancy_threshold,
				'total_max_distance': total_max_distance,
				'total_min_size': total_min_size,
				'md_max_distance': md_max_distance,
				'md_min_size': md_min_size,
				'classification_method': '3_step_pixel_based'
			}
		}
		
		return self.all_islands

	def _convert_masks_to_islands(self, island_masks, direction):
		"""Convert island masks to island info format"""
		islands = []
		for i, island_mask in enumerate(island_masks):
			positions = np.where(island_mask)
			island_values = self.differences_2d[positions]
			
			island_info = {
				'id': i + 1,
				'direction': direction,
				'size': island_mask.sum(),
				'mean_signal': np.mean(island_values),
				'positions': positions,
				'mask': island_mask,
				'fragment_span': (np.min(positions[0]), np.max(positions[0])),
				'genomic_span': (np.min(positions[1]), np.max(positions[1]))
			}
			islands.append(island_info)
		
		return islands
	
	def plot_all_islands_overview(self, figsize=(16, 4)):
		"""Plot overview of all four island types"""
		if not hasattr(self, 'all_islands'):
			raise ValueError("Must run detect_all_island_types() first")
		
		fig, row_axes = plt.subplots(2, 3, figsize=figsize)
		
		all_axes = np.array(row_axes).flatten()

		island_types = ['total', 'unchanging', 'mother', 'daughter']
		colors = ['viridis', 'Greys', 'Blues', 'Reds']

		ax = all_axes[0]
		mean_f_img = self.F_imgs.mean(0)
		extent = [0, mean_f_img.shape[1]*10, 0, 260]

		im = ax.imshow(mean_f_img, cmap='magma_r', vmin=0, vmax=40,
			origin='lower', aspect='auto', 
			extent=extent)
		plt.colorbar(im, ax=ax)
		ax.set_title("Average MG1/DG1 data")
		ax.set_xticks([])

		ax = all_axes[2]
		im = ax.imshow(self.differences_2d, cmap='RdBu_r', vmin=-20, vmax=20,
			origin='lower', aspect='auto', extent=extent)
		plt.colorbar(im, ax=ax)
		ax.set_title("Difference DG1 - MG1")
		ax.set_yticks([])
		ax.set_xticks([])

		axes = np.concatenate([all_axes[1:2], all_axes[3:]])
		for i, (island_type, cmap) in enumerate(zip(island_types, colors)):

			ax = axes[i]
			islands = self.all_islands[island_type]
			
			# Create overlay
			overlay = np.zeros_like(self.differences_2d)
			for island in islands:
				overlay[island['positions']] = abs(island['mean_signal'])
			
			im = ax.imshow(overlay > 0, cmap=cmap, aspect='auto', extent=extent,
							  origin='lower', interpolation='none', vmin=0, vmax=1)
			ax.set_title(f'{island_type.capitalize()}')
			plt.colorbar(im, ax=ax)

			if i < 1:
				ax.set_xticks([])

			if not i == 1:
				ax.set_yticks([])
		
		plt.tight_layout()
		plt.show()
	
	def comprehensive_summary(self, detail=False):
		"""Print comprehensive statistics for all island types"""
		if not hasattr(self, 'all_islands'):
			print("No comprehensive islands detected. Run detect_all_island_types() first.")
			return

		# Add this line:
		classification_method = self.all_islands['parameters'].get('classification_method', 'pixel_first')
		
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

		print("\n" + "="*60)
		print("CHROMATIN LANDSCAPE COMPOSITION ANALYSIS")
		print(f"Classification method: {classification_method}")  # Add this line
		print("="*60)
		
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


	def detect_all_island_types_strategy_b(self, threshold_method='std', 
										   # Total islands (step 1)
										   occupancy_threshold=5, total_max_distance=2, total_min_size=3,
										   # M/D islands (step 2) 
										   md_max_distance=8, md_min_size=2):
		"""
		Strategy B: Classify M/D/unchanging ONLY within significant total islands
		
		1. Find total chromatin islands (with size/distance filtering)
		2. Classify M/D/unchanging ONLY within those island pixels
		3. Group classified pixels into islands
		"""
		print("=== Strategy B: Island-First Classification ===")
		
		# Step 1: Find significant total chromatin islands
		print("Step 1: Detecting significant total chromatin islands...")
		total_islands, total_chromatin_mask = self.detect_total_occupancy_islands(
			occupancy_threshold, total_max_distance, total_min_size)
		
		# Create mask of pixels that made it into total islands (post-filtering)
		total_islands_mask = np.zeros_like(total_chromatin_mask, dtype=bool)
		for island in total_islands:
			total_islands_mask[island['positions']] = True
		
		print(f"  Found {len(total_islands)} total chromatin islands")
		print(f"  Raw chromatin pixels: {total_chromatin_mask.sum()}")
		print(f"  Significant island pixels: {total_islands_mask.sum()}")
		print(f"  Filtering removed: {total_chromatin_mask.sum() - total_islands_mask.sum()} pixels")
		
		# Step 2: Classify pixels ONLY within significant total islands
		print("Step 2: Classifying pixels within significant islands...")
		
		if threshold_method not in self.thresholds:
			print(f"  Running step2 to calculate {threshold_method} thresholds...")
			self.step2_analyze_distribution()
		
		thresh = self.thresholds[threshold_method]
		print(f"  Using thresholds: lower={thresh['lower']:.3f}, upper={thresh['upper']:.3f}")
		
		# Get ONLY the island pixel positions and their difference values
		island_positions = np.where(total_islands_mask)
		island_differences = self.differences_2d[island_positions]
		
		# Classify island pixels by threshold
		mother_indices = island_differences < thresh['lower']
		daughter_indices = island_differences > thresh['upper']
		unchanging_indices = ~mother_indices & ~daughter_indices
		
		# Create pixel classification masks (only within total islands)
		mother_pixel_mask = np.zeros_like(total_chromatin_mask, dtype=bool)
		daughter_pixel_mask = np.zeros_like(total_chromatin_mask, dtype=bool)
		unchanging_pixel_mask = np.zeros_like(total_chromatin_mask, dtype=bool)
		
		mother_pixel_mask[island_positions[0][mother_indices], 
						  island_positions[1][mother_indices]] = True
		daughter_pixel_mask[island_positions[0][daughter_indices], 
							island_positions[1][daughter_indices]] = True
		unchanging_pixel_mask[island_positions[0][unchanging_indices], 
							  island_positions[1][unchanging_indices]] = True
		
		print(f"  Pixel classification within islands:")
		print(f"    Mother pixels: {mother_pixel_mask.sum()}")
		print(f"    Daughter pixels: {daughter_pixel_mask.sum()}")
		print(f"    Unchanging pixels: {unchanging_pixel_mask.sum()}")
		print(f"    Total classified: {mother_pixel_mask.sum() + daughter_pixel_mask.sum() + unchanging_pixel_mask.sum()}")
		
		# Verification: All classified pixels should be within total islands
		all_classified = mother_pixel_mask | daughter_pixel_mask | unchanging_pixel_mask
		assert np.array_equal(all_classified, total_islands_mask), "Classification error: not all island pixels classified!"
		print(f"  ✅ Verification passed: All island pixels classified exactly once")
		
		# Step 3: Group classified pixels into islands (for analysis)
		print("Step 3: Grouping classified pixels into islands...")
		
		detector = IslandDetector(max_distance=md_max_distance)
		
		# Find islands within each pixel type
		mother_island_masks = detector.find_islands(mother_pixel_mask, min_size=md_min_size)
		mother_islands = self._convert_masks_to_islands(mother_island_masks, 'mother')
		
		daughter_island_masks = detector.find_islands(daughter_pixel_mask, min_size=md_min_size)
		daughter_islands = self._convert_masks_to_islands(daughter_island_masks, 'daughter')
		
		unchanging_island_masks = detector.find_islands(unchanging_pixel_mask, min_size=md_min_size)
		unchanging_islands = self._convert_masks_to_islands(unchanging_island_masks, 'unchanging')
		
		print(f"  Island detection results:")
		print(f"    Mother islands: {len(mother_islands)}")
		print(f"    Daughter islands: {len(daughter_islands)}")
		print(f"    Unchanging islands: {len(unchanging_islands)}")
		
		# Step 4: Calculate island coverage (pixels that made it into M/D/unchanging islands)
		island_pixels = {
			'mother': sum(island['size'] for island in mother_islands),
			'daughter': sum(island['size'] for island in daughter_islands),
			'unchanging': sum(island['size'] for island in unchanging_islands)
		}
		
		total_pixel_counts = {
			'mother': mother_pixel_mask.sum(),
			'daughter': daughter_pixel_mask.sum(),
			'unchanging': unchanging_pixel_mask.sum()
		}
		
		print(f"  Island coverage:")
		for pixel_type in ['mother', 'daughter', 'unchanging']:
			coverage = island_pixels[pixel_type] / max(total_pixel_counts[pixel_type], 1) * 100
			lost = total_pixel_counts[pixel_type] - island_pixels[pixel_type]
			print(f"    {pixel_type.capitalize()}: {island_pixels[pixel_type]}/{total_pixel_counts[pixel_type]} pixels ({coverage:.1f}% coverage, {lost} lost to size filtering)")
		
		# Final verification
		total_island_pixels_analyzed = sum(total_pixel_counts.values())
		print(f"\nFinal verification:")
		print(f"  Total island pixels: {total_islands_mask.sum()}")
		print(f"  Pixels analyzed: {total_island_pixels_analyzed}")
		print(f"  Match: {total_islands_mask.sum() == total_island_pixels_analyzed}")
		
		# Store results
		self.all_islands = {
			'total': total_islands,
			'mother': mother_islands,
			'daughter': daughter_islands,
			'unchanging': unchanging_islands,
			'pixel_masks': {
				'total_chromatin': total_chromatin_mask,  # Raw chromatin (pre-filtering)
				'total_islands': total_islands_mask,      # Significant islands (post-filtering)
				'mother': mother_pixel_mask,
				'daughter': daughter_pixel_mask,
				'unchanging': unchanging_pixel_mask
			},
			'pixel_counts': total_pixel_counts,
			'island_coverage': island_pixels,
			'filtering_stats': {
				'raw_chromatin_pixels': total_chromatin_mask.sum(),
				'significant_island_pixels': total_islands_mask.sum(),
				'pixels_lost_to_total_filtering': total_chromatin_mask.sum() - total_islands_mask.sum()
			},
			'parameters': {
				'threshold_method': threshold_method,
				'occupancy_threshold': occupancy_threshold,
				'total_max_distance': total_max_distance,
				'total_min_size': total_min_size,
				'md_max_distance': md_max_distance,
				'md_min_size': md_min_size,
				'classification_method': 'strategy_b_island_first'
			}
		}
		
		return self.all_islands

	def plot_all_islands_strategy_b(self, show_filtering=True, figsize=(18, 4)):
		"""
		Plot overview for Strategy B with optional filtering comparison
		"""
		if not hasattr(self, 'all_islands'):
			raise ValueError("Must run detect_all_island_types_strategy_b() first")
		
		n_rows = 2 if show_filtering else 1
		n_cols = 5 if show_filtering else 4
		fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
		
		if n_rows == 1:
			axes = [axes]
		
		# Common extent
		mean_f_img = self.F_imgs.mean(0)
		extent = [0, mean_f_img.shape[1]*10, 0, 260]
		
		# Row 1: Show filtering process (if requested)
		if show_filtering:
			# Raw chromatin
			ax = axes[0][0]
			mask = self.all_islands['pixel_masks']['total_chromatin']
			im = ax.imshow(mean_f_img, cmap='magma_r', aspect='auto', extent=extent,
						  origin='lower', interpolation='none', vmin=0, vmax=40)
			ax.set_title(f'Raw Chromatin')
			plt.colorbar(im, ax=ax)
			ax.set_xticks([])
			
			# Filtered total islands
			ax = axes[0][1]
			mask = self.all_islands['pixel_masks']['total_islands']
			im = ax.imshow(mask, cmap='viridis', aspect='auto', extent=extent,
						  origin='lower', interpolation='none')
			lost = self.all_islands['filtering_stats']['pixels_lost_to_total_filtering']
			ax.set_title(f'Significant Islands')
			plt.colorbar(im, ax=ax)
			ax.set_xticks([])
			ax.set_yticks([])
			
			# M/D/unchanging classification
			pixel_types = ['mother', 'daughter', 'unchanging']
			pixel_colors = ['Blues', 'Reds', 'Greys']
			
			for i, (pixel_type, cmap) in enumerate(zip(pixel_types, pixel_colors)):
				ax = axes[0][i+2]
				mask = self.all_islands['pixel_masks'][pixel_type]
				im = ax.imshow(mask, cmap=cmap, aspect='auto', extent=extent,
							  origin='lower', interpolation='none')
				count = mask.sum()
				ax.set_title(f'{pixel_type.capitalize()}\n({count} pixels)')
				plt.colorbar(im, ax=ax)
				ax.set_xticks([])
				ax.set_yticks([])
		
		# Bottom row: Island detection results
		island_row = 1 if show_filtering else 0
		col_offset = 2 if show_filtering else 0
		

		# Differences
		ax = axes[island_row][0]
		im1 = ax.imshow(self.differences_2d, cmap='RdBu_r', aspect='auto', 
							origin='lower', interpolation='none', vmin=-5, vmax=5,
							extent=extent)
		ax.set_title('Mean G1 differences')
		ax.set_xlabel('Genomic Position')
		ax.set_ylabel('Fragment Length')
		plt.colorbar(im1, ax=ax)

		# Total islands
		# ax = axes[island_row][1]
		# total_overlay = np.zeros_like(self.differences_2d)
		# for island in self.all_islands['total']:
		# 	total_overlay[island['positions']] = 1
		
		# im = ax.imshow(total_overlay, cmap='viridis', aspect='auto', extent=extent,
		# 			   origin='lower', interpolation='none')
		# ax.set_title(f'Total Islands\n({len(self.all_islands["total"])} islands)')
		# plt.colorbar(im, ax=ax)
		# if show_filtering:
		# 	ax.set_xticks([])
		
		# M/D/unchanging islands
		island_types = ['mother', 'daughter', 'unchanging']
		colors = ['Blues', 'Reds', 'Greys']
		
		for i, (island_type, cmap) in enumerate(zip(island_types, colors)):
			ax = axes[island_row][i+col_offset]
			islands = self.all_islands[island_type]
			
			overlay = np.zeros_like(self.differences_2d)
			for island in islands:
				overlay[island['positions']] = 1
			
			im = ax.imshow(overlay, cmap=cmap, aspect='auto', extent=extent,
						   origin='lower', interpolation='none', vmin=0, vmax=1)
			
			# Show both island count and pixel coverage
			island_pixels = self.all_islands['island_coverage'][island_type]
			total_pixels = self.all_islands['pixel_counts'][island_type]
			coverage = island_pixels / max(total_pixels, 1) * 100
			
			ax.set_title(f'{island_type.capitalize()} Islands\n({len(islands)} islands, {coverage:.0f}% coverage)')
			plt.colorbar(im, ax=ax)
			
			ax.set_yticks([])
			if show_filtering:
				ax.set_xticks([])
		
		plt.tight_layout()
		plt.show()
