
import pandas as pd

YL_REP2_RNA_PATH = 'output/yl_rep2_rna_reads/yl_rep2_rna_reads_chr{}.h5'
DATA_KEY = 'rna_data'

def save_rna_reads(all_fragments, chrom):
    """
    Save RNA reads to disk
    """
    save_path = YL_REP2_RNA_PATH.format(chrom)
    with pd.HDFStore(save_path, complevel=9, complib='zlib') as store:
        store[DATA_KEY] = all_fragments
        print(f"Wrote file: {save_path} to disk.")
    
def load_rna_reads(chrom):
    """
    Load RNA reads to disk
    """

    save_path = YL_REP2_RNA_PATH.format(chrom)
    print(f"Retrieving file: {save_path} from disk.")
    with pd.HDFStore(save_path, complevel=9, complib='zlib') as store:
        data_retrieved = store[DATA_KEY]
    return data_retrieved

