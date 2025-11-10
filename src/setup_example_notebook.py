
# Code to setup intermediate files for rna-seq saving
def subset_and_save(mnase_span, strand, replicate, chrom):
    load_dir = 'output/draft4_run/'
    save_dir = 'output/example/'
    filename = f'{strand}_rna_seq_pileup_rep{replicate}_chr{chrom}.h5'
    load_path = f"{load_dir}/rna_seq_intermediate/pileups/{filename}"
    save_path = f"{save_dir}/rna_seq_intermediate/pileups/{filename}"
    
    chr16_w_reads = pd.read_hdf(load_path, key='pileup')
    selected_reads = chr16_w_reads[range(mnase_span[0], mnase_span[1])]
    
    selected_reads.to_hdf(save_path, key='pileup', mode='w', complevel=9, complib='zlib')

subset_and_save(mnase_span, 'watson', 1, 16)
subset_and_save(mnase_span, 'crick', 1, 16)

subset_and_save(mnase_span, 'watson', 2, 16)
subset_and_save(mnase_span, 'crick', 2, 16)

# -----------

# Following loading from the default mnase path (output/mnase), save the loaded reads
# for the example span to the example output directory.

save_path = 'output/example/mnase/yl_rep1_mnase_reads/yl_rep1_mnase_reads_chr16.h5'
combined_model.chrom1_model.locus_reads.to_hdf(save_path, key='mnase_data', mode='w', complevel=9, complib='zlib')

save_path = 'output/example/mnase/yl_rep2_mnase_reads/yl_rep2_mnase_reads_chr16.h5'
combined_model.chrom2_model.locus_reads.to_hdf(save_path, key='mnase_data', mode='w', complevel=9, complib='zlib')
