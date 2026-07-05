#!/bin/bash
# Prepare SILVA reference datasets for DNABERT training.
#
# Downloads and processes SILVA data through QIIME2/rescript, then creates
# dnadb FASTA and taxonomy databases.  Each step is skipped if its output
# already exists; use --overwrite to force re-running.
#
# Environment variables (with defaults if unset):
#   DATA_DIR      Root directory for raw QIIME2 artifacts  (default: ./data)
#   DATASETS_DIR  Root directory for dnadb datasets         (default: ./datasets)
#
# Usage:
#   prepare_silva.sh --silva-version 138.2 [options]
#
# Options:
#   --silva-version VERSION   SILVA release version, e.g. 138.2  (required)
#   --target TARGET           SSURef | SSURef_NR99 | both         (default: both)
#   --amplicons               Extract 515f/806r amplicons         (default: on)
#   --no-amplicons            Skip amplicon extraction
#   --full-length             Also create full-length datasets     (default: off)
#   --num-jobs N              Parallel jobs for extract-reads      (default: 1)
#   --qiime2-env ENV          Conda environment for QIIME2         (default: qiime2)
#   --taxonomy-depth N        Taxonomy depth for dnadb import      (default: 6)
#   --min-sequence-length N   Minimum bp for amplicon/test filter  (default: 150)
#   --overwrite               Remove and rerun each step
#   --help                    Show this message

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DATA_DIR="${DATA_DIR:-./data}"
DATASETS_DIR="${DATASETS_DIR:-./datasets}"

SILVA_VERSION=""
TARGET="both"
DO_AMPLICONS=true
DO_FULL_LENGTH=false
NUM_JOBS=1
QIIME2_ENV="qiime2"
TAXONOMY_DEPTH=6
MIN_SEQUENCE_LENGTH=150
OVERWRITE=false

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --silva-version)       SILVA_VERSION="$2";        shift 2 ;;
        --target)              TARGET="$2";               shift 2 ;;
        --amplicons)           DO_AMPLICONS=true;         shift   ;;
        --no-amplicons)        DO_AMPLICONS=false;        shift   ;;
        --full-length)         DO_FULL_LENGTH=true;       shift   ;;
        --num-jobs)            NUM_JOBS="$2";             shift 2 ;;
        --qiime2-env)          QIIME2_ENV="$2";           shift 2 ;;
        --taxonomy-depth)      TAXONOMY_DEPTH="$2";       shift 2 ;;
        --min-sequence-length) MIN_SEQUENCE_LENGTH="$2";  shift 2 ;;
        --overwrite)           OVERWRITE=true;            shift   ;;
        --help)                sed -n '2,/^set /p' "$0" | grep '^#' | sed 's/^# \?//'; exit 0 ;;
        *) echo "ERROR: Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -z "${SILVA_VERSION}" ]; then
    echo "ERROR: --silva-version is required"
    exit 1
fi

case "${TARGET}" in
    SSURef|SSURef_NR99|both) ;;
    *) echo "ERROR: --target must be SSURef, SSURef_NR99, or both"; exit 1 ;;
esac

if [ "${DO_AMPLICONS}" = "false" ] && [ "${DO_FULL_LENGTH}" = "false" ]; then
    echo "ERROR: at least one of --amplicons or --full-length must be enabled"
    exit 1
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# step PRIMARY_OUTPUT DESCRIPTION COMMAND...
# Skips the command if PRIMARY_OUTPUT already exists (unless --overwrite).
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

# also_remove FILE...
# Removes extra outputs before the next step when --overwrite is active.
function also_remove() {
    if [ "${OVERWRITE}" = "true" ]; then
        for f in "$@"; do
            [ -e "${f}" ] && rm -rf "${f}"
        done
    fi
}

# export_qza QZA_PATH OUTPUT_FILE ARTIFACT_FILENAME
# Exports a QIIME2 artifact to a single file.
function export_qza() {
    local qza="$1"
    local output="$2"
    local artifact_filename="$3"
    local tmp_dir="${output%.fasta}.export"
    tmp_dir="${tmp_dir%.tsv}.export"
    conda run -n "${QIIME2_ENV}" qiime tools export \
        --input-path "${qza}" \
        --output-path "${tmp_dir}"
    mv "${tmp_dir}/${artifact_filename}" "${output}"
    rm -rf "${tmp_dir}"
}

# ---------------------------------------------------------------------------
# QIIME2 pipeline
# Processes one SILVA target (SSURef or SSURef_NR99) through all QIIME2 steps
# and exports the final FASTA/TSV files to DATA_DIR.
# ---------------------------------------------------------------------------
function run_qiime2_pipeline() {
    local qiime_target="$1"   # "SSURef" or "SSURef_NR99"

    local variant_suffix=""
    [ "${qiime_target}" = "SSURef_NR99" ] && variant_suffix="-99"

    local data_dir="${DATA_DIR}/silva/${SILVA_VERSION}${variant_suffix}"
    local tmp_dir="${data_dir}/tmp"
    local p="${data_dir}/silva-${SILVA_VERSION}${variant_suffix}"
    local t="${tmp_dir}/silva-${SILVA_VERSION}${variant_suffix}"

    mkdir -p "${data_dir}" "${tmp_dir}"

    # Download
    also_remove "${t}-tax.qza"
    step "${t}-rna-seqs.qza" "Downloading SILVA ${SILVA_VERSION} ${qiime_target}" \
        conda run -n "${QIIME2_ENV}" qiime rescript get-silva-data \
            --p-version "${SILVA_VERSION}" \
            --p-target  "${qiime_target}" \
            --o-silva-sequences "${t}-rna-seqs.qza" \
            --o-silva-taxonomy  "${t}-tax.qza"

    # RNA → DNA
    step "${t}-seqs.qza" "Reverse-transcribing RNA to DNA" \
        conda run -n "${QIIME2_ENV}" qiime rescript reverse-transcribe \
            --i-rna-sequences "${t}-rna-seqs.qza" \
            --o-dna-sequences "${t}-seqs.qza"

    # Cull
    step "${t}-seqs-culled.qza" "Culling degenerate/homopolymer sequences" \
        conda run -n "${QIIME2_ENV}" qiime rescript cull-seqs \
            --i-sequences          "${t}-seqs.qza" \
            --p-num-degenerates    5 \
            --p-homopolymer-length 8 \
            --o-clean-sequences    "${t}-seqs-culled.qza"

    # Filter by length per domain
    also_remove "${t}-seqs-filtered-discarded.qza"
    step "${t}-seqs-filtered.qza" "Filtering by length per domain" \
        conda run -n "${QIIME2_ENV}" qiime rescript filter-seqs-length-by-taxon \
            --i-sequences   "${t}-seqs-culled.qza" \
            --i-taxonomy    "${t}-tax.qza" \
            --p-labels      Archaea Bacteria Eukaryota \
            --p-min-lens    900 1200 1400 \
            --o-filtered-seqs   "${t}-seqs-filtered.qza" \
            --o-discarded-seqs  "${t}-seqs-filtered-discarded.qza"

    # Dereplicate
    also_remove "${t}-tax-derep.qza"
    step "${t}-seqs-derep.qza" "Dereplicating" \
        conda run -n "${QIIME2_ENV}" qiime rescript dereplicate \
            --i-sequences   "${t}-seqs-filtered.qza" \
            --i-taxa        "${t}-tax.qza" \
            --p-mode        "uniq" \
            --p-rank-handles "domain" "phylum" "class" "order" "family" "genus" \
            --o-dereplicated-sequences "${t}-seqs-derep.qza" \
            --o-dereplicated-taxa      "${t}-tax-derep.qza"

    # Export full-length FASTA + TSV
    if [ "${DO_FULL_LENGTH}" = "true" ]; then
        step "${p}-seqs-derep.fasta" "Exporting full-length sequences" \
            export_qza "${t}-seqs-derep.qza" "${p}-seqs-derep.fasta" "dna-sequences.fasta"
        step "${p}-tax-derep.tsv" "Exporting full-length taxonomy" \
            export_qza "${t}-tax-derep.qza" "${p}-tax-derep.tsv" "taxonomy.tsv"
    fi

    if [ "${DO_AMPLICONS}" = "true" ]; then
        # Extract 515f/806r
        step "${t}-seqs-515f-806r.qza" "Extracting 515f/806r amplicons" \
            conda run -n "${QIIME2_ENV}" qiime feature-classifier extract-reads \
                --i-sequences       "${t}-seqs-derep.qza" \
                --p-f-primer        GTGCCAGCMGCCGCGGTAA \
                --p-r-primer        GGACTACHVGGGTWTCTAAT \
                --p-trunc-len       0 \
                --p-trim-left       0 \
                --p-identity        0.8 \
                --p-min-length      "${MIN_SEQUENCE_LENGTH}" \
                --p-max-length      0 \
                --p-n-jobs          "${NUM_JOBS}" \
                --p-read-orientation both \
                --o-reads           "${t}-seqs-515f-806r.qza"

        # Dereplicate amplicons
        also_remove "${t}-tax-515f-806r-derep.qza"
        step "${t}-seqs-515f-806r-derep.qza" "Dereplicating 515f/806r amplicons" \
            conda run -n "${QIIME2_ENV}" qiime rescript dereplicate \
                --i-sequences   "${t}-seqs-515f-806r.qza" \
                --i-taxa        "${t}-tax-derep.qza" \
                --p-mode        "uniq" \
                --p-rank-handles "domain" "phylum" "class" "order" "family" "genus" \
                --o-dereplicated-sequences "${t}-seqs-515f-806r-derep.qza" \
                --o-dereplicated-taxa      "${t}-tax-515f-806r-derep.qza"

        # Export amplicon FASTA + TSV
        step "${p}-seqs-515f-806r-derep.fasta" "Exporting 515f/806r sequences" \
            export_qza "${t}-seqs-515f-806r-derep.qza" "${p}-seqs-515f-806r-derep.fasta" "dna-sequences.fasta"
        step "${p}-tax-515f-806r-derep.tsv" "Exporting 515f/806r taxonomy" \
            export_qza "${t}-tax-515f-806r-derep.qza" "${p}-tax-515f-806r-derep.tsv" "taxonomy.tsv"
    fi
}

# ---------------------------------------------------------------------------
# dnadb dataset creation
# ---------------------------------------------------------------------------
function create_dnadb_dataset() {
    local dataset_name="$1"
    local fasta_path="$2"
    local tax_path="$3"
    local test_fasta_path="${4:-}"   # empty = skip test set
    local test_tax_path="${5:-}"

    local out="${DATASETS_DIR}/${dataset_name}"
    mkdir -p "${out}"

    echo "  Dataset: ${dataset_name}"

    step "${out}/sequences.fasta.db" "Importing sequences" \
        dnadb fasta import "${fasta_path}" "${out}/sequences.fasta.db"

    step "${out}/taxonomy.tax.db" "Importing taxonomy" \
        dnadb taxonomy import \
            --depth    "${TAXONOMY_DEPTH}" \
            --fasta-db "${out}/sequences.fasta.db" \
            "${tax_path}" \
            "${out}/taxonomy.tax.db"

    if [ -n "${test_fasta_path}" ]; then
        step "${out}/sequences.test.fasta.db" "Creating test dataset" \
            python3 "${SCRIPT_DIR}/prepare_silva_test_dataset.py" \
                --reference-tax-db "${out}/taxonomy.tax.db" \
                --sequences-path   "${test_fasta_path}" \
                --taxonomy-path    "${test_tax_path}" \
                --output-path      "${out}" \
                --min-length       "${MIN_SEQUENCE_LENGTH}"
    fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
echo "SILVA Data Preparation"
echo "  Version         : ${SILVA_VERSION}"
echo "  Target          : ${TARGET}"
echo "  Amplicons       : ${DO_AMPLICONS}"
echo "  Full-length     : ${DO_FULL_LENGTH}"
echo "  Num jobs        : ${NUM_JOBS}"
echo "  QIIME2 env      : ${QIIME2_ENV}"
echo "  Taxonomy depth  : ${TAXONOMY_DEPTH}"
echo "  Min seq length  : ${MIN_SEQUENCE_LENGTH} bp"
echo "  Data dir        : ${DATA_DIR}"
echo "  Datasets dir    : ${DATASETS_DIR}"
echo "  Overwrite       : ${OVERWRITE}"
echo ""

# SSURef is always processed — its exports are the test-set source for NR99.
echo "=== QIIME2: SILVA ${SILVA_VERSION} SSURef ==="
run_qiime2_pipeline "SSURef"

if [ "${TARGET}" = "SSURef_NR99" ] || [ "${TARGET}" = "both" ]; then
    echo ""
    echo "=== QIIME2: SILVA ${SILVA_VERSION} SSURef_NR99 ==="
    run_qiime2_pipeline "SSURef_NR99"
fi

# Shorthand path prefixes for the exported files
ssuref="${DATA_DIR}/silva/${SILVA_VERSION}/silva-${SILVA_VERSION}"
nr99="${DATA_DIR}/silva/${SILVA_VERSION}-99/silva-${SILVA_VERSION}-99"

echo ""
echo "=== dnadb datasets ==="

if [ "${TARGET}" = "SSURef" ] || [ "${TARGET}" = "both" ]; then
    if [ "${DO_AMPLICONS}" = "true" ]; then
        create_dnadb_dataset \
            "silva_515f_806r" \
            "${ssuref}-seqs-515f-806r-derep.fasta" \
            "${ssuref}-tax-515f-806r-derep.tsv"
        # No test set: SSURef is itself the reference, leaving nothing held out.
    fi
    if [ "${DO_FULL_LENGTH}" = "true" ]; then
        create_dnadb_dataset \
            "silva_filtered" \
            "${ssuref}-seqs-derep.fasta" \
            "${ssuref}-tax-derep.tsv"
    fi
fi

if [ "${TARGET}" = "SSURef_NR99" ] || [ "${TARGET}" = "both" ]; then
    if [ "${DO_AMPLICONS}" = "true" ]; then
        create_dnadb_dataset \
            "silva_nr99_filtered_515f_806r" \
            "${nr99}-seqs-515f-806r-derep.fasta" \
            "${nr99}-tax-515f-806r-derep.tsv" \
            "${ssuref}-seqs-515f-806r-derep.fasta" \
            "${ssuref}-tax-515f-806r-derep.tsv"
    fi
    if [ "${DO_FULL_LENGTH}" = "true" ]; then
        create_dnadb_dataset \
            "silva_nr99_filtered" \
            "${nr99}-seqs-derep.fasta" \
            "${nr99}-tax-derep.tsv" \
            "${ssuref}-seqs-derep.fasta" \
            "${ssuref}-tax-derep.tsv"
    fi
fi

echo ""
echo "Done."
