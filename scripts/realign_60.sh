#/bin/bash

# Script to realign the replicate 1 60 minute timepoint in the 2019 cell cycle dataset
# In post-processing it appears this timepoint has some odd artifacts.
# The original alignment was done on HARDAC with bowtie 1.1.1


# This is bowtie 1.3.1, previously used on HARDAC as 1.1.1
BOWTIE=/bin/bowtie

DMAH_FILE="DMAH70"


BOWTIE_INDEX=/usr/project/compbio/tqtran/aligned_data/yeast_UCSC-sacCer3_SGD-R64/bowtie_index/yeast_UCSC-sacCer3_SGD-R64
CURRENT_DATE=$(date "+%Y-%m-%d-%H-%M")
FASTQ_DIR=/usr/project/compbio/tqtran/aligned_data/cell-cycle/fastq/${DMAH_FILE}

# Lane 1
OUTPUT_SAM_FILE="${DMAH_FILE}_sacCer3_m1_${CURRENT_DATE}_0.sam"
LANE="L001"

$BOWTIE --wrapper basic-0 --time -p 32 -n 2 -l 20 --phred33-quals -m 1 --best --strata -S -y -x $BOWTIE_INDEX -1 $FASTQ_DIR/${DMAH_FILE}_S7_${LANE}_R1_001.fastq -2 $FASTQ_DIR/${DMAH_FILE}_S7_${LANE}_R2_001.fastq $OUTPUT_SAM_FILE 2>logfile1.txt

# Lane 2
OUTPUT_SAM_FILE="${DMAH_FILE}_sacCer3_m1_${CURRENT_DATE}_1.sam"
LANE="L002"

$BOWTIE --wrapper basic-0 --time -p 32 -n 2 -l 20 --phred33-quals -m 1 --best --strata -S -y -x $BOWTIE_INDEX -1 $FASTQ_DIR/${DMAH_FILE}_S7_${LANE}_R1_001.fastq -2 $FASTQ_DIR/${DMAH_FILE}_S7_${LANE}_R2_001.fastq $OUTPUT_SAM_FILE 2>logfile2.txt

# Lane 3
OUTPUT_SAM_FILE="${DMAH_FILE}_sacCer3_m1_${CURRENT_DATE}_2.sam"
LANE="L003"

$BOWTIE --wrapper basic-0 --time -p 32 -n 2 -l 20 --phred33-quals -m 1 --best --strata -S -y -x $BOWTIE_INDEX -1 $FASTQ_DIR/${DMAH_FILE}_S7_${LANE}_R1_001.fastq -2 $FASTQ_DIR/${DMAH_FILE}_S7_${LANE}_R2_001.fastq $OUTPUT_SAM_FILE 2>logfile3.txt

# Lane 4
OUTPUT_SAM_FILE="${DMAH_FILE}_sacCer3_m1_${CURRENT_DATE}_3.sam"
LANE="L004"

$BOWTIE --wrapper basic-0 --time -p 32 -n 2 -l 20 --phred33-quals -m 1 --best --strata -S -y -x $BOWTIE_INDEX -1 $FASTQ_DIR/${DMAH_FILE}_S7_${LANE}_R1_001.fastq -2 $FASTQ_DIR/${DMAH_FILE}_S7_${LANE}_R2_001.fastq $OUTPUT_SAM_FILE 2>logfile4.txt

# ------------------ Convert the sam files to bam ----------------------------------

# Initialize an empty array to hold the names of the BAM files
bam_files=()

# Iterate over all .sam files in the current directory
for samfile in *.sam; do
    # Extract the base name by removing the .sam extension
    basename=$(basename "$samfile" .sam)
    
    # Define the name of the resulting BAM file
    bamfile="${basename}.bam"
    
    # Convert SAM to BAM and keep the base name
    samtools view -bS "$samfile" > "$bamfile"
    
    # Add the BAM file name to the array
    bam_files+=("$bamfile")
done


# ----------------- Merge the bam files ----------------------------

# todo: we don't need to use the wildcard if we can collect the names

# Now, use the array to merge the BAM files
# Check if there are at least two BAM files to merge
if [ ${#bam_files[@]} -gt 1 ]; then
    samtools merge -@ nProc -l 9 merged_output.bam "${bam_files[@]}"
else