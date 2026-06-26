#!/bin/bash
# Prepare mini-QIITA per-sample FASTQ datasets into dnadb format.
#
# For each dataset subdirectory, creates:
#   $DATASETS_DIR/mini-qiita-{id}/sequences.fasta.db
#   $DATASETS_DIR/mini-qiita-{id}/sequences.fasta.mapping.db
#
# Uses only forward reads (R1). Sample names are the FASTQ filenames
# minus extension (e.g. CR-813_S258_L001_R1_001).
#
# Usage:
#   bash scripts/dataset/prepare_mini_qiita.sh [DATASET_IDS ...]
#
# Environment variables required:
#   DATA_DIR      — root containing mini-qiita/per_sample_FASTQ/
#   DATASETS_DIR  — output datasets directory
#
# Example:
#   bash scripts/dataset/prepare_mini_qiita.sh 163012 177266

set -euo pipefail

if [ $# -eq 0 ]; then
    DATASET_IDS=("108759" "147774" "163012" "177266")
else
    DATASET_IDS=("$@")
fi
FASTQ_ROOT="${DATA_DIR}/mini-qiita/per_sample_FASTQ"
MIN_LENGTH=150

for DATASET_ID in "${DATASET_IDS[@]}"; do
    INPUT_DIR="${FASTQ_ROOT}/${DATASET_ID}"
    OUTPUT_DIR="${DATASETS_DIR}/mini-qiita-${DATASET_ID}"
    FASTA_DB="${OUTPUT_DIR}/sequences.fasta.db"
    MAPPING_DB="${OUTPUT_DIR}/sequences.fasta.mapping.db"

    if [ -d "${FASTA_DB}" ] && [ -d "${MAPPING_DB}" ]; then
        echo "[${DATASET_ID}] Already prepared, skipping."
        continue
    fi

    mkdir -p "${OUTPUT_DIR}"

    R1_FILES=("${INPUT_DIR}"/*_R1_001.fastq.gz)
    if [ ${#R1_FILES[@]} -eq 0 ]; then
        echo "[${DATASET_ID}] No R1 files found, skipping."
        continue
    fi
    echo "[${DATASET_ID}] ${#R1_FILES[@]} samples -> ${OUTPUT_DIR}"

    dnadb fasta import-multiplexed \
        --min-length "${MIN_LENGTH}" \
        --output-sequences-path "${FASTA_DB}" \
        --output-mapping-path "${MAPPING_DB}" \
        "${R1_FILES[@]}"

    echo "[${DATASET_ID}] Done."
done

echo "All datasets prepared."
