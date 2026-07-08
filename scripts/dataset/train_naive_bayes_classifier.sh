#!/bin/bash
# Train a QIIME2 Naive Bayes taxonomy classifier on SILVA reference sequences,
# then export it as a portable NaiveBayesForTaxonomy model directory for use
# with 'dbtk model predict'.
#
# Uses the .qza artifacts produced by prepare_silva.sh as inputs.
#
# Environment variables (with defaults if unset):
#   DATA_DIR      Root directory for raw QIIME2 artifacts  (default: ./data)
#   DATASETS_DIR  Root directory for dnadb datasets         (default: ./datasets)
#   MODELS_DIR    Root directory for exported models        (default: ./models)
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
#   --taxonomy-db PATH        Taxonomy .tax.db for export step      (default: DATASETS_DIR/silva_nr99_filtered_515f_806r/taxonomy.tax.db)
#   --overwrite               Remove and rerun each step
#   --help                    Show this message

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DATA_DIR="${DATA_DIR:-./data}"
DATASETS_DIR="${DATASETS_DIR:-./datasets}"
MODELS_DIR="${MODELS_DIR:-./models}"

SILVA_VERSION="138.2"
TARGET="SSURef_NR99"
DO_AMPLICONS=true
QIIME2_ENV="qiime2-amplicon-2026.4"
TAXONOMY_DB_OVERRIDE=""
OVERWRITE=false

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --silva-version) SILVA_VERSION="$2";      shift 2 ;;
        --target)        TARGET="$2";            shift 2 ;;
        --amplicons)     DO_AMPLICONS=true;      shift   ;;
        --no-amplicons)  DO_AMPLICONS=false;     shift   ;;
        --qiime2-env)    QIIME2_ENV="$2";        shift 2 ;;
        --taxonomy-db)   TAXONOMY_DB_OVERRIDE="$2"; shift 2 ;;
        --overwrite)     OVERWRITE=true;         shift   ;;
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
extracted_dir="${qiime_dir}/${classifier_name}"
nb_model_dir="${MODELS_DIR}/nb-silva${SILVA_VERSION}${variant_suffix}${amplicon_tag}"
taxonomy_db="${TAXONOMY_DB_OVERRIDE:-${DATASETS_DIR}/silva_nr99_filtered_515f_806r/taxonomy.tax.db}"

mkdir -p "${qiime_dir}" "${MODELS_DIR}"

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
echo "  Taxonomy DB      : ${taxonomy_db}"
echo "  Model output dir : ${nb_model_dir}"
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

step "${extracted_dir}/nb_meta.json" "Extracting NB parameters from pipeline (QIIME2 env)" \
    conda run -n "${QIIME2_ENV}" python3 "${SCRIPT_DIR}/export_nb_model.py" extract \
        --pipeline-tar "${extracted_dir}/sklearn_pipeline.tar" \
        --output       "${extracted_dir}"

step "${nb_model_dir}/config.json" "Assembling NaiveBayesForTaxonomy model directory" \
    python3 "${SCRIPT_DIR}/export_nb_model.py" assemble \
        --params-dir  "${extracted_dir}" \
        --taxonomy-db "${taxonomy_db}" \
        --output      "${nb_model_dir}"

echo ""
echo "Done."
echo ""
echo "Artifacts:"
echo "  ${classifier_qza}"
echo "    -> QIIME2 TaxonomicClassifier artifact"
echo "  ${extracted_dir}/sklearn_pipeline.tar"
echo "    -> scikit-learn Pipeline (tar)"
echo "  ${nb_model_dir}/"
echo "    -> NaiveBayesForTaxonomy model directory (use with 'dbtk model predict')"
