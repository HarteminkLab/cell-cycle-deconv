
import numpy as np
import pandas as pd


def filter_rna_seq(rna_seq, start=None, end=None, chrom=None,
    time=None, strand=None):
    """
    Filter RNA-seq data given the argument parameters, do not filter if
    not specified
    """

    if strand is not None:
        select = rna_seq.strand == strand
    else:
        # dummy select statement
        select = ~(rna_seq.strand == None)

    if time is not None:
        select = select & (rna_seq.time == time)

    if chrom is not None:
        select = (rna_seq.chr == chrom) & select
 
    if start is not None and end is not None:

        # Entire read must be inside window
        # select = ((rna_seq.stop <= end) & 
        #           (rna_seq.start >= start)) & select

        # Read intersects any portion of the window
        select = ((rna_seq.stop > start) & 
                  (rna_seq.start <= end)) & select

    return rna_seq[select].copy()


def filter_rna_seq_pileup(pileup, start=None, end=None, chrom=None,
    time=None, strand=None):
    """
    Filter RNA-seq data given the argument parameters, do not filter if
    not specified
    """

    if strand is not None:
        select = pileup.strand == strand
    else:
        # dummy select
        select = ~(pileup.strand == None)

    if time is not None:
        select = select & (pileup.time == time)

    if chrom is not None:
        select = (pileup.chr == chrom) & select
 
    if start is not None and end is not None:
        select = ((pileup.position < end) & 
                  (pileup.position >= start)) & select

    return pileup[select].copy()


def calculate_read_counts(orfs, rna_seq, sample_key='count'):
    """Get RNA-seq read counts and TPM

        When introns are counted, equivalent to using to R package Rsubread::featuresCounts:

        featureCounts(files=<list of files>, annot.ext=<annotation.gff file>,
                        
                        # specify that reads are on the reverses strand
                        strandSpecific=2, 
                        
                        # allow reads to map to multiple ORFs
                        allowMultiOverlap=TRUE, 
                        
                        # GTF/GFF settings, count genes and use the "ID" attribute
                        # as the unique identifier
                        isGTFAnnotationFile=TRUE,
                        GTF.featureType='gene', 
                        GTF.attrType='ID')
    """

    read_counts = orfs[[]].copy()
    read_counts[sample_key] = 0

    for chrom in range(1, 17):

        chrom_orfs = orfs[orfs.chr == chrom]
        chrom_rna_seq = filter_rna_seq(rna_seq, chrom=chrom)
        
        for idx, orf in chrom_orfs.iterrows():
            orf_rna_seq = filter_rna_seq(chrom_rna_seq, start=orf.start, end=orf.stop)
            strand_select = (orf_rna_seq.strand == orf.strand)

            cur_orf_read_counts = orf_rna_seq[strand_select]
            read_counts.loc[idx, sample_key] = len(cur_orf_read_counts)

    return read_counts


def convert_to_TPM_all_times(data, lengths):

    times = data.columns
    TPMs = data.copy()

    for time in times:
        TPMs.loc[:, time] = convert_to_TPM(TPMs[time], lengths)

    return TPMs
    

def convert_to_TPM(data, lengths):
    """Convert to TPM, normalize by length of ORF, then divide 
    by sum of each time, multiply by 1 mil so each time point 
    sums to 1e6"""
    
    data = data.copy()

    # normalize by ORF length
    data['read_per_bp'] = data / lengths

    scaling = data['read_per_bp'].sum()
    
    TPM = (data['read_per_bp'] / scaling * 1e6) # scale to 1 million
    return TPM

