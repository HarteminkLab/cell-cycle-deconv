import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
import warnings

class HistonesNucleosomesDataset:
	"""
	A class to integrate Chereji nucleosome positioning data with Weiner histone modification data.
	
	The Chereji dataset provides precise positioning of functionally important -1 and +1 nucleosomes
	flanking gene promoters, while the Weiner dataset contains comprehensive histone modification
	data for ~66K nucleosomes genome-wide. This class performs fuzzy matching to combine these
	datasets and provides analysis capabilities for promoter-associated nucleosome modifications.
	"""
	
	def __init__(self, fuzzy_window_size: int = 80):
		"""
		Initialize the integrator with dataset paths and parameters.
		
		Args:
			weiner_nucleosome_path: Path to Weiner nucleosome positioning CSV
			weiner_histones_path: Path to Weiner histone modifications CSV  
			chereji_nucleosome_path: Path to Chereji nucleosome positioning CSV
			fuzzy_window_size: Window size (bp) for fuzzy matching nucleosomes
		"""
		self.weiner_nucleosome_path =  'output/large_histones_dataset_weiner_2015/mmc3.csv'
		self.weiner_histones_path = 'output/large_histones_dataset_weiner_2015/molcel_5341_mmc4'
		self.chereji_nucleosome_path = 'data/reference_data/Chereji_2018_additional_files_table2_+1_+-1_nucleosomes.csv'
		self.fuzzy_window_size = fuzzy_window_size
		
		# Data containers
		self.weiner_nucleosomes = None
		self.weiner_histones = None
		self.chereji_nucleosomes = None
		self.integrated_data: Optional[pd.DataFrame] = None
		self.matched_nucleosome_ids: Optional[List[int]] = None
		
	def load_weiner_nucleosomes(self) -> pd.DataFrame:
		"""Load Weiner nucleosome positioning data."""
		path = self.weiner_nucleosome_path
			
		df = pd.read_csv(path)
		self.weiner_nucleosomes = df.set_index('nuc_id')
		return self.weiner_nucleosomes
	
	def load_weiner_histones(self, path: str = None, wild_type_only: bool = True) -> pd.DataFrame:
		"""
		Load Weiner histone modification data.
		
		Args:
			path: Path to histone data file
			wild_type_only: If True, load only wild-type columns (exclude mutant data)
		"""
		path = path or self.weiner_histones_path
		if path is None:
			raise ValueError("Weiner histones path not provided")
			
		df = pd.read_csv(path, low_memory=False)
		df = df.set_index('Unnamed: 0')
		df.index.name = 'nuc_id'
		df = df.iloc[1:]  # Remove header row
		df.index = df.index.astype(int)
		
		if wild_type_only:
			# Select only columns without '.' (wild-type data)
			wt_columns = ~df.columns.str.contains('\\.')
			df = df[df.columns[wt_columns]]
			
		self.weiner_histones = df
		return self.weiner_histones
	
	def load_chereji_nucleosomes(self, path: str = None) -> pd.DataFrame:
		"""Load Chereji nucleosome positioning data."""
		path = self.chereji_nucleosome_path
			
		df = pd.read_csv(path).set_index('ORF')
		
		# Convert chromosome notation from Roman to numeric
		chroms = df.Chr.str.replace('chr', '').apply(self._from_roman)
		df.Chr = chroms
		
		self.chereji_nucleosomes = df
		return self.chereji_nucleosomes
	
	def _from_roman(self, roman: str) -> int:
		"""Convert Roman numeral to integer."""
		roman_numerals = {
			'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6, 'VII': 7, 'VIII': 8,
			'IX': 9, 'X': 10, 'XI': 11, 'XII': 12, 'XIII': 13, 'XIV': 14, 'XV': 15, 'XVI': 16
		}
		return roman_numerals.get(roman.upper(), 0)
	
	def fuzzy_join_nucleosomes(self, 
							 weiner_df: pd.DataFrame,
							 chereji_df: pd.DataFrame,
							 nucleosome_key: str,
							 window_size: int = None) -> pd.DataFrame:
		"""
		Perform fuzzy matching between Weiner and Chereji nucleosome datasets.
		
		Args:
			weiner_df: Weiner nucleosome DataFrame
			chereji_df: Chereji nucleosome DataFrame  
			nucleosome_key: Column name in Chereji data ('+1 nucleosome' or '-1 nucleosome')
			window_size: Matching window size in bp
			
		Returns:
			Chereji DataFrame with added 'matched_nuc_id' column
		"""
		window_size = window_size or self.fuzzy_window_size
		half_window = window_size // 2
		
		# Prepare data for merge_asof
		weiner_prep = weiner_df.reset_index()[['nuc_id', 'chr', 'center']].copy()
		weiner_prep = weiner_prep.sort_values(['chr', 'center'])
		
		chereji_prep = chereji_df.reset_index().copy()
		chereji_prep = chereji_prep.rename(columns={'Chr': 'chr', nucleosome_key: 'canonical_pos'})
		chereji_prep = chereji_prep.dropna(subset=['canonical_pos'])
		chereji_prep = chereji_prep.sort_values(['chr', 'canonical_pos'])
		
		# Perform chromosome-wise fuzzy matching
		results = []
		
		for chr_num in chereji_prep['chr'].unique():
			weiner_chr = weiner_prep[weiner_prep['chr'] == chr_num]
			chereji_chr = chereji_prep[chereji_prep['chr'] == chr_num]
			
			if weiner_chr.empty or chereji_chr.empty:
				chereji_chr = chereji_chr.copy()
				chereji_chr['matched_nuc_id'] = np.nan
				results.append(chereji_chr)
				continue
			
			# Use merge_asof for nearest neighbor matching within tolerance
			matched = pd.merge_asof(
				chereji_chr,
				weiner_chr[['center', 'nuc_id']],
				left_on='canonical_pos',
				right_on='center',
				tolerance=half_window,
				direction='nearest'
			)
			
			matched = matched.rename(columns={'nuc_id': 'matched_nuc_id'})
			matched = matched.drop(columns=['center'])
			results.append(matched)
		
		# Combine results
		final_result = pd.concat(results, ignore_index=True)
		final_result = final_result.rename(columns={'chr': 'Chr', 'canonical_pos': nucleosome_key})
		final_result = final_result.set_index(chereji_df.index.name)
		
		# Reorder columns
		original_cols = list(chereji_df.columns)
		final_result = final_result[original_cols + ['matched_nuc_id']]
		
		return final_result.loc[chereji_df.index]
	
	def integrate_datasets(self) -> pd.DataFrame:
		"""
		Main integration method that combines all datasets.
		
		Returns:
			Integrated DataFrame with Chereji positioning and matched nucleosome IDs
		"""
		# Ensure all datasets are loaded
		if self.weiner_nucleosomes is None:
			self.load_weiner_nucleosomes()
		if self.chereji_nucleosomes is None:
			self.load_chereji_nucleosomes()
			
		# Perform fuzzy matching for +1 and -1 nucleosomes
		plus_one_matched = self.fuzzy_join_nucleosomes(
			self.weiner_nucleosomes, 
			self.chereji_nucleosomes,
			'+1 nucleosome'
		)
		
		minus_one_matched = self.fuzzy_join_nucleosomes(
			self.weiner_nucleosomes,
			self.chereji_nucleosomes, 
			'-1 nucleosome'
		)
		
		# Combine matched IDs
		matched_ids = plus_one_matched[['matched_nuc_id']].join(
			minus_one_matched[['matched_nuc_id']], 
			lsuffix='_p1', 
			rsuffix='_m1'
		)
		
		# Create integrated dataset
		self.integrated_data = self.chereji_nucleosomes.join(matched_ids)
		
		# Extract unique matched nucleosome IDs for subsetting
		p1_ids = self.integrated_data['matched_nuc_id_p1'].dropna().astype(int)
		m1_ids = self.integrated_data['matched_nuc_id_m1'].dropna().astype(int)
		self.matched_nucleosome_ids = sorted(list(set(p1_ids).union(set(m1_ids))))
		
		return self.integrated_data
	
	def get_subset_histones(self, load_if_needed: bool = True) -> pd.DataFrame:
		"""
		Get histone modification data for only the matched nucleosomes.
		
		Args:
			load_if_needed: Whether to load histone data if not already loaded
			
		Returns:
			Subset of histone modification data for matched nucleosomes
		"""
		if self.matched_nucleosome_ids is None:
			self.integrate_datasets()
			
		if self.weiner_histones is None and load_if_needed:
			self.load_weiner_histones()
		elif self.weiner_histones is None:
			raise ValueError("Histone data not loaded. Set load_if_needed=True or load manually.")
			
		self.subset_chereji_weiner_histones = self.weiner_histones.loc[self.matched_nucleosome_ids]
		return self.subset_chereji_weiner_histones
	
	def get_summary_stats(self) -> Dict:
		"""Get summary statistics about the integrated dataset."""
		if self.integrated_data is None:
			self.integrate_datasets()
			
		stats = {
			'total_genes': len(self.integrated_data),
			'genes_with_plus1_match': self.integrated_data['matched_nuc_id_p1'].notna().sum(),
			'genes_with_minus1_match': self.integrated_data['matched_nuc_id_m1'].notna().sum(),
			'genes_with_both_matches': (
				self.integrated_data['matched_nuc_id_p1'].notna() & 
				self.integrated_data['matched_nuc_id_m1'].notna()
			).sum(),
			'unique_matched_nucleosomes': len(self.matched_nucleosome_ids) if self.matched_nucleosome_ids else 0,
			'fuzzy_window_size': self.fuzzy_window_size
		}
		
		return stats
	
	def get_gene_nucleosome_data(self, gene_id: str) -> Dict:
		"""
		Get comprehensive data for a specific gene.
		
		Args:
			gene_id: Gene identifier (ORF name)
			
		Returns:
			Dictionary with gene positioning and histone modification data
		"""
		if self.integrated_data is None:
			self.integrate_datasets()
			
		if gene_id not in self.integrated_data.index:
			raise ValueError(f"Gene {gene_id} not found in dataset")
			
		gene_data = self.integrated_data.loc[gene_id]
		result = {'gene_info': gene_data.to_dict()}
		
		# Get histone data for matched nucleosomes
		if self.weiner_histones is not None:
			for nuc_type in ['p1', 'm1']:
				nuc_id = gene_data.get(f'matched_nuc_id_{nuc_type}')
				if pd.notna(nuc_id):
					nuc_id = int(nuc_id)
					if nuc_id in self.weiner_histones.index:
						result[f'{nuc_type}_histone_mods'] = self.weiner_histones.loc[nuc_id].to_dict()
						
		return result

	def load_all(self):
		print("Loading all Chereji nucleosomes (-1 and +1)...")
		self.load_chereji_nucleosomes()
		print(f"Done. {len(self.chereji_nucleosomes)} genes with +1 and -1 nucleosomes loaded")

		print("Loading all ~66k Weiner nucleosomes with histone mods...")
		self.load_weiner_nucleosomes()
		print(f"Done. {len(self.weiner_nucleosomes)} nucleosomes loaded")

		# Merge sets and retrieve histone modifications for relevant nucleosomes
		print("Merging Chereji and Weiner nucleosomes, keeping Chereji as canonical...")
		self.integrate_datasets()
		print(f"Done. Merged into a set of {len(self.matched_nucleosome_ids)} nucleosomes")

		print("Loading these histones data for these nucleosomes")
		self.get_subset_histones()
		print(f"Done. Loaded histone data for {len(self.subset_chereji_weiner_histones)} nucleosomes")
	
	def __repr__(self) -> str:
		stats = self.get_summary_stats() if self.integrated_data is not None else {}
		return f"""NucleosomeDatasetIntegrator(
	Fuzzy window: {self.fuzzy_window_size}bp
	Datasets loaded: Weiner nucleosomes={self.weiner_nucleosomes is not None}, 
					Weiner histones={self.weiner_histones is not None},
					Chereji={self.chereji_nucleosomes is not None}
	Integration status: {stats}
)"""
