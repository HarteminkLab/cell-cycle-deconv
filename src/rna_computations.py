
import pysam
import pandas as pd


def compute_read_count_gene(filename, gene):
    
    samfile = pysam.AlignmentFile(filename, "rb")

    # get chromosome reads

    chrom, strand, start, stop = gene.chr, gene.strand, gene.start, gene.stop
    itr = samfile.fetch(str(chrom), start, stop)
    
    count = 0
    for read in itr:

        # skip unmapped reads, i.e. -F 4
        if read.is_unmapped: continue

        length = read.reference_length
        position = read.pos+1 # first base begins at 1
        strand = '-'
        if read.is_reverse: strand = '+'

        if strand is not gene.strand: continue

        count += 1

    samfile.close()

    return count


def compute_read_counts(filename, genes):

    read_counts = genes.copy().set_index('orf_name')
    read_counts['read_counts'] = 0

    idx = 0
    for index, gene in read_counts.iterrows():    
        
        count = 0
        try:
            count = compute_read_count_gene(filename, gene)
        except ValueError:
            count = 0

        read_counts.loc[index, 'read_counts'] = count

        idx += 1

    return read_counts

def compute_tpm():
    # Drop chrMito and 2-micron genes
    read_counts = read_counts[(read_counts['chr'] != 'chrMito') & (read_counts['chr'] != '2-micron')]

    # read_per_kb = read_counts['read_counts'] / read_counts['length'] * 1000
    # total_reads = read_per_kb.sum()
    # transcripts_per_million = read_per_kb / total_reads * 1e6
    # read_counts['tpm'] = transcripts_per_million

    return read_counts
