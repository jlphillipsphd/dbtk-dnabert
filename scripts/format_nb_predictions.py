#!/usr/bin/env python3
"""
Convert a QIIME2 taxonomy prediction TSV into the .pt format consumed by
evaluate_predictions.py.

Reads the taxonomy TSV produced by classify-sklearn, loads rank_labels and
parent_indices from the training taxonomy DB, and writes a .pt file whose
structure matches the output of predict_taxonomy.py.

Called automatically by predict_naive_bayes.sh, but can also be run directly
to reformat without re-running the QIIME2 classify-sklearn step.

Usage:
    python format_nb_predictions.py --predictions TSV --sequences-db FASTA_DB
                                    --taxonomy-db TAX_DB --output OUTPUT [options]

Example:
    python scripts/format_nb_predictions.py \\
        --predictions  $DATA_DIR/qiime/nb_predict_work/taxonomy.tsv \\
        --sequences-db $DATASETS_DIR/silva_nr99_filtered_515f_806r_test/sequences.fasta.db \\
        --taxonomy-db  $DATASETS_DIR/silva_nr99_filtered_515f_806r/taxonomy.tax.db \\
        --output       $DATASETS_DIR/silva_nr99_filtered_515f_806r_test/predictions_nb_qiime.pt \\
        --overwrite
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from dnadb import fasta
from dnabert.models import load_taxonomy
from rich.progress import track


UNASSIGNED = "Unassigned"


def load_predictions(
    predictions_path: Path,
) -> Dict[str, Tuple[List[str], float]]:
    """
    Parse a QIIME2 taxonomy TSV into a dict: seq_id → (per-rank labels, confidence).

    The TSV has columns: Feature ID, Taxon, Confidence.
    Taxon is a ';'-delimited string like 'd__Bacteria; p__Proteobacteria; ...'.
    Confidence is a float in [0, 1] or -1 when disabled.
    """
    predictions: Dict[str, Tuple[List[str], float]] = {}
    with open(predictions_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            seq_id = row["Feature ID"]
            taxon_str = row["Taxon"].strip()
            try:
                confidence = float(row["Confidence"])
            except (ValueError, KeyError):
                confidence = -1.0
            if taxon_str == UNASSIGNED:
                rank_parts = []
            else:
                rank_parts = [p.strip() for p in taxon_str.split(";")]
            predictions[seq_id] = (rank_parts, confidence)
    return predictions


def main():
    parser = argparse.ArgumentParser(
        description="Convert QIIME2 classify-sklearn TSV to evaluate_predictions.py .pt format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--predictions",  type=Path, required=True,
                        help="Taxonomy TSV from qiime tools export (classify-sklearn output)")
    parser.add_argument("--sequences-db", type=Path, required=True,
                        help="Test sequences FASTA DB (.fasta.db) — sets sequence order")
    parser.add_argument("--taxonomy-db",  type=Path, required=True,
                        help="Training taxonomy DB (.tax.db) — source of rank_labels")
    parser.add_argument("--output",       type=Path, required=True,
                        help="Output .pt file")
    parser.add_argument("--top-k",        type=int, default=1,
                        help="K for top-K fields in output (default: 1, NB gives only top-1)")
    parser.add_argument("--overwrite",    action="store_true",
                        help="Overwrite output if it already exists")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        print(f"Output already exists: {args.output}")
        print("Use --overwrite to re-run.")
        return

    print(f"Loading training taxonomy from {args.taxonomy_db}...")
    rank_labels, parent_indices, _ = load_taxonomy(args.taxonomy_db)
    num_ranks = len(rank_labels)
    # rank_labels is now alphabetically sorted (same as train_id_to_taxon)
    train_id_to_taxon = rank_labels
    label_to_id = [
        {label: i for i, label in enumerate(labels)}
        for labels in rank_labels
    ]
    print(f"  {num_ranks} ranks, "
          + ", ".join(f"rank{r}: {len(train_id_to_taxon[r])} taxa"
                      for r in range(num_ranks)))

    print(f"Loading predictions from {args.predictions}...")
    predictions = load_predictions(args.predictions)
    print(f"  {len(predictions):,} predictions loaded")

    print(f"Reading sequence order from {args.sequences_db}...")
    with fasta.FastaDb(str(args.sequences_db)) as fasta_db:
        seq_ids = [entry.identifier for entry in track(fasta_db, description="  Reading")]
    print(f"  {len(seq_ids):,} sequences")

    missing = sum(1 for sid in seq_ids if sid not in predictions)
    if missing:
        print(f"  Warning: {missing:,} sequences have no prediction (will use taxon_id -1)")

    k = args.top_k
    n = len(seq_ids)
    pred_ids    = torch.full((n, num_ranks),    -1, dtype=torch.long)
    topk_ids    = torch.full((n, num_ranks, k), -1, dtype=torch.long)
    topk_scores = torch.full((n, num_ranks, k), float("-inf"), dtype=torch.float)

    n_unknown = [0] * num_ranks
    for i, seq_id in enumerate(track(seq_ids, description="Formatting predictions")):
        if seq_id not in predictions:
            continue
        rank_parts, confidence = predictions[seq_id]
        if not rank_parts:
            continue
        for r in range(min(num_ranks, len(rank_parts))):
            label = rank_parts[r]
            taxon_id = label_to_id[r].get(label, -1)
            if taxon_id == -1:
                n_unknown[r] += 1
            pred_ids[i, r] = taxon_id
            topk_ids[i, r, 0] = taxon_id
            topk_scores[i, r, 0] = confidence

    for r in range(num_ranks):
        if n_unknown[r]:
            print(f"  Warning: rank {r}: {n_unknown[r]:,} predictions not in training taxonomy")

    payload = {
        "seq_ids":           seq_ids,
        "pred_ids":          pred_ids,
        "topk_ids":          topk_ids,
        "topk_scores":       topk_scores,
        "rank_labels":       rank_labels,
        "parent_indices":    parent_indices,
        "train_id_to_taxon": train_id_to_taxon,
        "top_k":             k,
        "model_path":        str(args.taxonomy_db),
        "sequences_path":    str(args.sequences_db),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"\nSaved {n:,} predictions → {args.output}")


if __name__ == "__main__":
    main()
