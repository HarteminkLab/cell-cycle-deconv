
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

