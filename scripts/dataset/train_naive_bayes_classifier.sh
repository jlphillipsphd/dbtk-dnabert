#!/bin/bash
# Train a QIIME2 Naive Bayes taxonomy classifier on SILVA reference sequences.
#
# Uses the .qza artifacts produced by prepare_silva.sh as inputs.  The output
# classifier .qza is extracted so that sklearn_pipeline.pkl is available for
# direct use in prediction scripts without a full QIIME2 installation.
#
# Environment variables (with defaults if unset):
#   DATA_DIR      Root directory for raw QIIME2 artifacts  (default: ./data)
#   DATASETS_DIR  Root directory for dnadb datasets         (default: ./datasets)
#
# Usage:
#   train_naive_bayes_classifier.sh --silva-version 138.2 [options]
#
# Options:
#   --silva-version VERSION   SILVA release version, e.g. 138.2   (required)
#   --target TARGET           SSURef_NR99 | SSURef                 (default: SSURef_NR99)
#   --amplicons               Use 515f/806r amplicon sequences      (default: on)
#   --no-amplicons            Use full-length dereplicated sequences
#   --qiime2-env ENV          Conda environment for QIIME2           (default: qiime2-amplicon-2026.4)
#   --overwrite               Remove and rerun each step
#   --help                    Show this message

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DATA_DIR="${DATA_DIR:-./data}"
DATASETS_DIR="${DATASETS_DIR:-./datasets}"

SILVA_VERSION="138.2"
TARGET="SSURef_NR99"
DO_AMPLICONS=true
QIIME2_ENV="qiime2-amplicon-2026.4"
OVERWRITE=false

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --silva-version) SILVA_VERSION="$2";  shift 2 ;;
        --target)        TARGET="$2";          shift 2 ;;
        --amplicons)     DO_AMPLICONS=true;    shift   ;;
        --no-amplicons)  DO_AMPLICONS=false;   shift   ;;
        --qiime2-env)    QIIME2_ENV="$2";      shift 2 ;;
        --overwrite)     OVERWRITE=true;       shift   ;;
        --help)          sed -n '2,/^set /p' "$0" | grep '^#' | sed 's/^# \?//'; exit 0 ;;
        *) echo "ERROR: Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -z "${SILVA_VERSION}" ]; then
    echo "ERROR: --silva-version is required"
    exit 1
fi

case "${TARGET}" in
    SSURef|SSURef_NR99) ;;
    *) echo "ERROR: --target must be SSURef or SSURef_NR99"; exit 1 ;;
esac

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
# Paths derived from the prepare_silva.sh conventions
# ---------------------------------------------------------------------------
variant_suffix=""
[ "${TARGET}" = "SSURef_NR99" ] && variant_suffix="-99"

tmp_dir="${DATA_DIR}/silva/${SILVA_VERSION}${variant_suffix}/tmp"
t="${tmp_dir}/silva-${SILVA_VERSION}${variant_suffix}"

if [ "${DO_AMPLICONS}" = "true" ]; then
    seq_qza="${t}-seqs-515f-806r-derep.qza"
    tax_qza="${t}-tax-515f-806r-derep.qza"
    amplicon_tag="-515f-806r"
else
    seq_qza="${t}-seqs-derep.qza"
    tax_qza="${t}-tax-derep.qza"
    amplicon_tag=""
fi

if [ ! -f "${seq_qza}" ] || [ ! -f "${tax_qza}" ]; then
    echo "ERROR: Input QZA files not found:"
    echo "  sequences: ${seq_qza}"
    echo "  taxonomy:  ${tax_qza}"
    echo "Run prepare_silva.sh first to generate the SILVA QIIME2 artifacts."
    exit 1
fi

classifier_name="silva-${SILVA_VERSION}${variant_suffix}${amplicon_tag}-nb-classifier"
qiime_dir="${DATA_DIR}/qiime"
classifier_qza="${qiime_dir}/${classifier_name}.qza"
# Extracted directory: sklearn_pipeline.pkl lives here for use in prediction scripts
extracted_dir="${qiime_dir}/${classifier_name}"

mkdir -p "${qiime_dir}"

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
echo "QIIME2 Naive Bayes Classifier Training"
echo "  SILVA version    : ${SILVA_VERSION}"
echo "  Target           : ${TARGET}"
echo "  Amplicons        : ${DO_AMPLICONS}"
echo "  QIIME2 env       : ${QIIME2_ENV}"
echo "  Sequences QZA    : ${seq_qza}"
echo "  Taxonomy QZA     : ${tax_qza}"
echo "  Classifier QZA   : ${classifier_qza}"
echo "  Extracted dir    : ${extracted_dir}"
echo "  Overwrite        : ${OVERWRITE}"
echo ""

step "${classifier_qza}" "Training Naive Bayes classifier (this may take several hours)" \
    conda run -n "${QIIME2_ENV}" qiime feature-classifier fit-classifier-naive-bayes \
        --i-reference-reads    "${seq_qza}" \
        --i-reference-taxonomy "${tax_qza}" \
        --o-classifier         "${classifier_qza}"

step "${extracted_dir}/sklearn_pipeline.tar" "Extracting classifier artifact" \
    conda run -n "${QIIME2_ENV}" qiime tools export \
        --input-path  "${classifier_qza}" \
        --output-path "${extracted_dir}"

echo ""
echo "Done."
echo ""
echo "Classifier artifacts:"
echo "  ${classifier_qza}"
echo "    -> QIIME2 TaxonomicClassifier artifact"
echo "  ${extracted_dir}/sklearn_pipeline.tar"
echo "    -> scikit-learn Pipeline (tar) for prediction scripts"
