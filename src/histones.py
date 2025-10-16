import pandas as pd
import re

class HistoneModificationOrganizer:
	"""
	A class to organize and retrieve histone modifications using pandas DataFrame as underlying storage.
	"""
	
	def __init__(self):
		"""
		Initialize with a list of histone modification strings.
		
		Args:
			histone_modifications_list (list): List of histone modification strings
		"""
		self.df = self._create_dataframe(histones_ordering())
	
	def _parse_modification(self, modification_name):
		"""
		Parse a histone modification string to extract histone name, modification type, and other details.
		
		Args:
			modification_name (str): Histone modification string (e.g., 'H3K4ac', 'H3K4me3')
			
		Returns:
			dict: Dictionary with parsed information
		"""
		# Handle special case for histone variants
		if modification_name == 'Htz1':
			return {
				'modification_name': modification_name,
				'histone_name': 'Htz1',
				'modification_type': 'Histone Variant',
				'residue': None,
				'position': None,
				'degree': None
			}
		
		# Initialize result dictionary
		result = {
			'modification_name': modification_name,
			'histone_name': None,
			'modification_type': None,
			'residue': None,
			'position': None,
			'degree': None
		}
		
		# Extract modification type and degree from the end
		if modification_name.endswith('ac'):
			result['modification_type'] = 'Acetylation'
			histone_part = modification_name[:-2]
		elif modification_name.endswith('ph'):
			result['modification_type'] = 'Phosphorylation'
			histone_part = modification_name[:-2]
		elif modification_name.endswith('me3'):
			result['modification_type'] = 'Methylation'
			result['degree'] = 'tri'
			histone_part = modification_name[:-3]
		elif modification_name.endswith('me2s'):
			result['modification_type'] = 'Methylation'
			result['degree'] = 'di-symmetric'
			histone_part = modification_name[:-4]
		elif modification_name.endswith('me2'):
			result['modification_type'] = 'Methylation'
			result['degree'] = 'di'
			histone_part = modification_name[:-3]
		elif modification_name.endswith('me'):
			result['modification_type'] = 'Methylation'
			result['degree'] = 'mono'
			histone_part = modification_name[:-2]
		else:
			# Fallback for unknown modifications
			result['modification_type'] = 'Unknown'
			histone_part = modification_name
		
		# Extract histone name
		if histone_part.startswith('H2A'):
			result['histone_name'] = 'H2A'
			residue_part = histone_part[3:]
		elif histone_part.startswith('H3'):
			result['histone_name'] = 'H3'
			residue_part = histone_part[2:]
		elif histone_part.startswith('H4'):
			result['histone_name'] = 'H4'
			residue_part = histone_part[2:]
		elif histone_part.startswith('H2'):
			result['histone_name'] = 'H2'
			residue_part = histone_part[2:]
		else:
			result['histone_name'] = histone_part
			residue_part = ''
		
		# Extract residue and position (e.g., 'K4' -> residue='K', position=4)
		if residue_part:
			match = re.match(r'([A-Z])(\d+)', residue_part)
			if match:
				result['residue'] = match.group(1)
				result['position'] = int(match.group(2))
		
		return result
	
	def _create_dataframe(self, histone_modifications_list):
		"""
		Create a pandas DataFrame from the list of histone modifications.
		
		Args:
			histone_modifications_list (list): List of histone modification strings
			
		Returns:
			pd.DataFrame: DataFrame with parsed modification information
		"""
		parsed_data = []
		for modification in histone_modifications_list:
			parsed_data.append(self._parse_modification(modification))
		
		return pd.DataFrame(parsed_data)
	
	def get_all_modifications(self):
		"""
		Get the complete ungrouped list of histone modifications.
		
		Returns:
			list: All histone modifications
		"""
		return self.df['modification_name'].tolist()
	
	def get_dataframe(self):
		"""
		Get the underlying DataFrame.
		
		Returns:
			pd.DataFrame: Complete DataFrame with all modification information
		"""
		return self.df.copy()
	
	def get_grouped_modifications(self):
		"""
		Get all modifications grouped by modification type and histone name.
		
		Returns:
			dict: Nested dictionary of grouped modifications
		"""
		grouped = {}
		
		for mod_type in self.df['modification_type'].unique():
			grouped[mod_type] = {}
			mod_df = self.df[self.df['modification_type'] == mod_type]
			
			for histone in mod_df['histone_name'].unique():
				modifications = mod_df[mod_df['histone_name'] == histone]['modification_name'].tolist()
				grouped[mod_type][histone] = modifications
		
		return grouped

	def get_modifications_flattened_by_group_name(self, modification_type):
		"""
		Get modifications for a specific modification type, ordered correctly.
		
		Parameters:
		-----------
		modification_type : str
			The modification type to filter by
			
		Returns:
		--------
		list
			Ordered list of modification names for this type
		"""
		group_mods = self.get_by_modification_type(modification_type).modification_name.values
		# Apply the global histones ordering but filter to only this group
		all_ordered = histones_ordering()
		modifications = [mod for mod in all_ordered if mod in group_mods]

		return modifications
	
	def get_by_modification_type(self, modification_type):
		"""
		Get all modifications of a specific type.
		
		Args:
			modification_type (str): Type of modification ('Acetylation', 'Methylation', 'Phosphorylation', 'Histone Variant')
			
		Returns:
			pd.DataFrame: DataFrame containing modifications of the specified type
		"""
		return self.df[self.df['modification_type'] == modification_type].copy()
	
	def get_by_histone(self, histone_name):
		"""
		Get all modifications for a specific histone.
		
		Args:
			histone_name (str): Name of histone ('H2A', 'H3', 'H4', 'Htz1', etc.)
			
		Returns:
			pd.DataFrame: DataFrame containing modifications for the specified histone
		"""
		return self.df[self.df['histone_name'] == histone_name].copy()
	
	def get_by_histone_and_modification_type(self, histone_name, modification_type):
		"""
		Get modifications for a specific histone and modification type.
		
		Args:
			histone_name (str): Name of histone
			modification_type (str): Type of modification
			
		Returns:
			pd.DataFrame: DataFrame containing modifications matching both criteria
		"""
		return self.df[
			(self.df['histone_name'] == histone_name) & 
			(self.df['modification_type'] == modification_type)
		].copy()
	
	def get_by_residue(self, residue):
		"""
		Get all modifications for a specific amino acid residue.
		
		Args:
			residue (str): Amino acid residue ('K', 'S', 'R', etc.)
			
		Returns:
			pd.DataFrame: DataFrame containing modifications for the specified residue
		"""
		return self.df[self.df['residue'] == residue].copy()
	
	def get_by_position(self, position):
		"""
		Get all modifications at a specific position.
		
		Args:
			position (int): Position number
			
		Returns:
			pd.DataFrame: DataFrame containing modifications at the specified position
		"""
		return self.df[self.df['position'] == position].copy()
	
	def get_by_methylation_degree(self, degree):
		"""
		Get methylation modifications by degree.
		
		Args:
			degree (str): Methylation degree ('mono', 'di', 'tri', 'di-symmetric')
			
		Returns:
			pd.DataFrame: DataFrame containing methylation modifications of specified degree
		"""
		return self.df[
			(self.df['modification_type'] == 'Methylation') & 
			(self.df['degree'] == degree)
		].copy()
	
	def get_modification_types(self):
		"""
		Get all available modification types.
		
		Returns:
			list: List of modification types
		"""
		return self.df['modification_type'].unique().tolist()
	
	def get_histone_names(self):
		"""
		Get all available histone names.
		
		Returns:
			list: List of unique histone names
		"""
		return self.df['histone_name'].unique().tolist()
	
	def get_residues(self):
		"""
		Get all available amino acid residues.
		
		Returns:
			list: List of unique residues
		"""
		return self.df['residue'].dropna().unique().tolist()
	
	def get_positions(self):
		"""
		Get all available positions.
		
		Returns:
			list: List of unique positions
		"""
		return sorted(self.df['position'].dropna().unique().tolist())
	
	def filter_modifications(self, **kwargs):
		"""
		Filter modifications using multiple criteria.
		
		Args:
			**kwargs: Keyword arguments for filtering (e.g., histone_name='H3', modification_type='Acetylation')
			
		Returns:
			pd.DataFrame: Filtered DataFrame
		"""
		filtered_df = self.df.copy()
		
		for column, value in kwargs.items():
			if column in filtered_df.columns:
				if isinstance(value, list):
					filtered_df = filtered_df[filtered_df[column].isin(value)]
				else:
					filtered_df = filtered_df[filtered_df[column] == value]
		
		return filtered_df
	
	def get_summary_stats(self):
		"""
		Get summary statistics of the modifications.
		
		Returns:
			dict: Dictionary with summary statistics
		"""
		stats = {
			'total_modifications': len(self.df),
			'by_modification_type': self.df['modification_type'].value_counts().to_dict(),
			'by_histone': self.df['histone_name'].value_counts().to_dict(),
			'by_residue': self.df['residue'].value_counts().dropna().to_dict(),
			'methylation_degrees': self.df[self.df['modification_type'] == 'Methylation']['degree'].value_counts().dropna().to_dict()
		}
		return stats
	
	def print_summary(self):
		"""
		Print a formatted summary of all modifications.
		"""
		stats = self.get_summary_stats()
		
		print("Histone Modification Summary:")
		print("=" * 50)
		print(f"Total modifications: {stats['total_modifications']}")
		
		print(f"\nBy modification type:")
		for mod_type, count in stats['by_modification_type'].items():
			print(f"  {mod_type}: {count}")
		
		print(f"\nBy histone:")
		for histone, count in stats['by_histone'].items():
			print(f"  {histone}: {count}")
		
		print(f"\nBy residue:")
		for residue, count in stats['by_residue'].items():
			print(f"  {residue}: {count}")
		
		if stats['methylation_degrees']:
			print(f"\nMethylation degrees:")
			for degree, count in stats['methylation_degrees'].items():
				print(f"  {degree}: {count}")

# Function to get the ordered list of histone modifications
def histones_ordering():
	histone_modifications_ordered = [
		# H2A acetylation sites
		'H2AK5ac',  # Transcriptional activation
		
		# ACETYLATION MARKS (Generally Activating)
		# H3 acetylation sites
		'H3K4ac',   # Promoter acetylation
		'H3K9ac',   # Active promoter mark
		'H3K14ac',  # Transcriptional activation
		'H3K18ac',  # Gene activation
		'H3K23ac',  # Active chromatin
		'H3K27ac',  # Active enhancers/promoters
		'H3K56ac',  # DNA replication/repair, transcription
		
		# H4 acetylation sites
		'H4K5ac',   # Transcriptional activation
		'H4K8ac',   # Active chromatin
		'H4K12ac',  # Transcriptional activation
		'H4K16ac',  # Active transcription, chromatin opening
		
		# ACTIVATING METHYLATION MARKS
		# H3K4 methylation (promoter marks)
		'H3K4me',   # Poised/weak promoter activity
		'H3K4me2',  # Promoter activity
		'H3K4me3',  # Active promoters
		
		# H3K36 methylation (transcription elongation)
		'H3K36me',  # Transcription elongation
		'H3K36me2', # Active transcription
		'H3K36me3', # Active gene bodies
		
		# H3K79 methylation (active transcription)
		'H3K79me',  # Active transcription
		'H3K79me3', # Active transcription, telomeric silencing
		
		# OTHER METHYLATION MARKS
		'H4K20me',     # Heterochromatin/gene silencing
		'H4R3me',      # Can be activating or repressing
		'H4R3me2s',    # Symmetric dimethylation, context-dependent
		
		# PHOSPHORYLATION MARKS (Signaling/Stress Response)
		'H3S10ph',     # Mitosis, immediate early gene activation
		'H2AS129ph',   # DNA damage response (yeast-specific)
		
		# HISTONE VARIANTS
		'Htz1',        # H2A.Z variant - chromatin boundaries, gene regulation
	]
	return histone_modifications_ordered
