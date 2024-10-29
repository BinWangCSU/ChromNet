#!/bin/bash

FILE_STORE_FOLDER=$1
FILE_NAME=$2
BAM_DIR_PATH=$3

BAM_DIR="${BAM_DIR_PATH}/${FILE_NAME}"

# export PATH=/packages/samtools-1.17:$PATH

mkdir -p ${FILE_STORE_FOLDER}

echo "${FILE_NAME}"

# Merge all BAM files in the provided directory
MERGED_BAM_FILES=""
for BAM_FILE in ${BAM_DIR}/*.bam; 
do 
    MERGED_BAM_FILES="${MERGED_BAM_FILES} ${BAM_FILE}"
done

samtools merge -o "${FILE_STORE_FOLDER}"/"${FILE_NAME}"_merged.bam ${MERGED_BAM_FILES}
samtools sort -o "${FILE_STORE_FOLDER}"/"${FILE_NAME}"_merged_sorted.bam "${FILE_STORE_FOLDER}"/"${FILE_NAME}"_merged.bam
samtools index "${FILE_STORE_FOLDER}"/"${FILE_NAME}"_merged_sorted.bam

TARGET_READS=40000000

# Calculate total number of reads
TOTAL_READS=$(samtools view -@ 8 -c "${FILE_STORE_FOLDER}/${FILE_NAME}_merged_sorted.bam")

# Calculate scaling factor
SCALE_FACTOR=$(bc <<< "scale=10; $TARGET_READS / $TOTAL_READS")

# Generate bigWig file
bamCoverage --bam "${FILE_STORE_FOLDER}/${FILE_NAME}_merged_sorted.bam" \
-o "${FILE_STORE_FOLDER}/${FILE_NAME}.bw" \
--scaleFactor $SCALE_FACTOR \
--normalizeUsing RPKM \
--binSize 10
