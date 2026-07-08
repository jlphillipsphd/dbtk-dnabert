#!/bin/bash
# Run Naive Bayes taxonomy prediction using a trained QIIME2 classifier.
#
# Exports test sequences from a dnadb FASTA DB, runs QIIME2 classify-sklearn
# in parallel, exports the taxonomy TSV, then calls format_nb_predictions.py
# to produce a .pt file compatible with evaluate_predictions.py.
#
# Usage:
#   predict_naive_bayes.sh [options]
#
# Options:
#   --classifier-qza PATH     Path to the trained classifier .qza artifact  (required)
#   --sequences-db PATH       Test sequences FASTA DB (.fasta.db)            (required)
#   --taxonomy-db PATH        Training taxonomy DB (.tax.db) for rank labels (required)
#   --output PATH             Output .pt file                                 (required)
#   --num-jobs N              Parallel workers for classify-sklearn           (default: 1)
#   --top-k K                 K for top-K stored in .pt output               (default: 1)
#   --qiime2-env ENV          Conda environment for QIIME2                    (default: qiime2-amplicon-2026.4)
#   --work-dir DIR            Temporary working directory                     (default: auto)
#   --keep-work-dir           Do not delete the working directory on exit
#   --overwrite               Re-run all steps even if outputs exist
#   --help                    Show this message
#
# Example:
#   bash scripts/predict_naive_bayes.sh \
#       --classifier-qza $DATA_DIR/qiime/silva-138.2-99-515f-806r-nb-classifier.qza \
#       --sequences-db   $DATASETS_DIR/silva_nr99_filtered_515f_806r_test/sequences.fasta.db \
#       --taxonomy-db    $DATASETS_DIR/silva_nr99_filtered_515f_806r/taxonomy.tax.db \
#       --output         $DATASETS_DIR/silva_nr99_filtered_515f_806r_test/predictions_nb_qiime.pt \
#       --num-jobs       32 \
#       --work-dir       $DATA_DIR/qiime/nb_predict_work \
#       --keep-work-dir

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
CLASSIFIER_QZA=""
SEQUENCES_PATH=""
TAXONOMY_PATH=""
OUTPUT_PATH=""
NUM_JOBS=1
TOP_K=1
QIIME2_ENV="qiime2-amplicon-2026.4"
WORK_DIR=""
KEEP_WORK_DIR=false
OVERWRITE=false

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --classifier-qza)  CLASSIFIER_QZA="$2";   shift 2 ;;
        --sequences-db)    SEQUENCES_PATH="$2";   shift 2 ;;
        --taxonomy-db)     TAXONOMY_PATH="$2";    shift 2 ;;
        --output)          OUTPUT_PATH="$2";      shift 2 ;;
        --num-jobs)        NUM_JOBS="$2";         shift 2 ;;
        --top-k)           TOP_K="$2";            shift 2 ;;
        --qiime2-env)      QIIME2_ENV="$2";       shift 2 ;;
        --work-dir)        WORK_DIR="$2";         shift 2 ;;
        --keep-work-dir)   KEEP_WORK_DIR=true;    shift   ;;
        --overwrite)       OVERWRITE=true;        shift   ;;
        --help)            sed -n '2,/^set /p' "$0" | grep '^#' | sed 's/^# \?//'; exit 0 ;;
        *) echo "ERROR: Unknown argument: $1"; exit 1 ;;
    esac
done

[ -z "${CLASSIFIER_QZA}"  ] && { echo "ERROR: --classifier-qza is required"; exit 1; }
[ -z "${SEQUENCES_PATH}" ] && { echo "ERROR: --sequences-db is required";   exit 1; }
[ -z "${TAXONOMY_PATH}"  ] && { echo "ERROR: --taxonomy-db is required";    exit 1; }
[ -z "${OUTPUT_PATH}"    ] && { echo "ERROR: --output is required";          exit 1; }

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function step() {
    local primary="$1"; shift
    local desc="$1";    shift
    if [ -e "${primary}" ] && [ "${OVERWRITE}" = "false" ]; then
        echo "  [skip] ${desc}"
        return 0
    fi
    [ -e "${primary}" ] && rm -rf "${primary}"
    echo "  ${desc}..."
    "$@"
}

# ---------------------------------------------------------------------------
# Early exit if final output already exists
# ---------------------------------------------------------------------------
if [ -e "${OUTPUT_PATH}" ] && [ "${OVERWRITE}" = "false" ]; then
    echo "Output already exists: ${OUTPUT_PATH}"
    echo "Use --overwrite to re-run."
    exit 0
fi

# ---------------------------------------------------------------------------
# Working directory
# ---------------------------------------------------------------------------
if [ -z "${WORK_DIR}" ]; then
    WORK_DIR="$(mktemp -d)"
    if [ "${KEEP_WORK_DIR}" = "false" ]; then
        trap 'rm -rf "${WORK_DIR}"' EXIT
    fi
fi
mkdir -p "${WORK_DIR}"

FASTA_FILE="${WORK_DIR}/sequences.fasta"
SEQS_QZA="${WORK_DIR}/sequences.qza"
TAXONOMY_QZA="${WORK_DIR}/taxonomy.qza"
TAXONOMY_TSV="${WORK_DIR}/taxonomy.tsv"

echo "Naive Bayes Taxonomy Prediction"
echo "  Classifier QZA   : ${CLASSIFIER_QZA}"
echo "  Sequences DB     : ${SEQUENCES_PATH}"
echo "  Training taxonomy: ${TAXONOMY_PATH}"
echo "  Output           : ${OUTPUT_PATH}"
echo "  Num jobs         : ${NUM_JOBS}"
echo "  Top-K            : ${TOP_K}"
echo "  QIIME2 env       : ${QIIME2_ENV}"
echo "  Work dir         : ${WORK_DIR}"
echo "  Overwrite        : ${OVERWRITE}"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Export sequences from dnadb FASTA DB to plain FASTA
# ---------------------------------------------------------------------------
step "${FASTA_FILE}" "Exporting sequences from dnadb to FASTA" \
    python3 - "${SEQUENCES_PATH}" "${FASTA_FILE}" <<'PYEOF'
import sys
from dnadb import fasta

sequences_path, fasta_path = sys.argv[1], sys.argv[2]
with fasta.FastaDb(sequences_path) as db, open(fasta_path, "w") as out:
    for entry in db:
        out.write(f">{entry.identifier}\n{entry.sequence}\n")
print(f"  Exported {fasta_path}")
PYEOF

# ---------------------------------------------------------------------------
# Step 2: Import FASTA as QIIME2 FeatureData[Sequence] artifact
# ---------------------------------------------------------------------------
step "${SEQS_QZA}" "Importing sequences into QIIME2" \
    conda run -n "${QIIME2_ENV}" qiime tools import \
        --type 'FeatureData[Sequence]' \
        --input-path  "${FASTA_FILE}" \
        --output-path "${SEQS_QZA}"

# ---------------------------------------------------------------------------
# Step 3: Classify with classify-sklearn
# ---------------------------------------------------------------------------
step "${TAXONOMY_QZA}" "Running classify-sklearn (n_jobs=${NUM_JOBS})" \
    conda run -n "${QIIME2_ENV}" qiime feature-classifier classify-sklearn \
        --i-classifier "${CLASSIFIER_QZA}" \
        --i-reads      "${SEQS_QZA}" \
        --p-n-jobs     "${NUM_JOBS}" \
        --p-confidence 'disable' \
        --o-classification "${TAXONOMY_QZA}" \
        --verbose

# ---------------------------------------------------------------------------
# Step 4: Export taxonomy TSV
# ---------------------------------------------------------------------------
step "${TAXONOMY_TSV}" "Exporting taxonomy TSV" \
    bash -c "conda run -n '${QIIME2_ENV}' qiime tools export \
        --input-path  '${TAXONOMY_QZA}' \
        --output-path '${WORK_DIR}/taxonomy_export' \
    && mv '${WORK_DIR}/taxonomy_export/taxonomy.tsv' '${TAXONOMY_TSV}'"

# ---------------------------------------------------------------------------
# Step 5: Format as .pt
# ---------------------------------------------------------------------------
OVERWRITE_FLAG=""
[ "${OVERWRITE}" = "true" ] && OVERWRITE_FLAG="--overwrite"

step "${OUTPUT_PATH}" "Formatting predictions as .pt" \
    python3 "${SCRIPT_DIR}/format_nb_predictions.py" \
        --predictions   "${TAXONOMY_TSV}" \
        --sequences-db  "${SEQUENCES_PATH}" \
        --taxonomy-db   "${TAXONOMY_PATH}" \
        --output        "${OUTPUT_PATH}" \
        --top-k         "${TOP_K}" \
        ${OVERWRITE_FLAG}

echo ""
echo "Done. Predictions saved to: ${OUTPUT_PATH}"
