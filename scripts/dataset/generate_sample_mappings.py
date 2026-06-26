#!/usr/bin/env python3
"""
Generate SILVA-based sample mapping files for SetBERT pretraining.

For each mini-QIITA dataset, runs the fine-tuned DNABERT TopDown model on every
unique real-world sequence to predict its genus-level taxon, then builds a mapping
from each real-world sample to SILVA reference sequences of the predicted genera.
Read counts are preserved: if a sample has N reads mapping to genus G, N SILVA
sequences are randomly drawn from G and written to the mapping entry.

Input per dataset:
    DATASETS_DIR/mini-qiita-{id}/sequences.fasta.db
    DATASETS_DIR/mini-qiita-{id}/sequences.fasta.mapping.db

Output per dataset:
    DATASETS_DIR/mini-qiita-{id}/sequences.topdown.{REFERENCE_DATASET}.fasta.mapping.db

Usage:
    python generate_sample_mappings.py MODEL_DIR DATASETS_DIR [DATASET_IDS ...]

Example:
    python scripts/dataset/generate_sample_mappings.py \\
        $MODELS_DIR/dnabert-finetune-topdown-64d-250bp-1mer-silva138.2 \\
        $DATASETS_DIR \\
        --reference-dataset silva_nr99_filtered_515f_806r \\
        163012 177266
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from dnadb import fasta, taxonomy
from tqdm import tqdm

from dnabert.models import DnaBertForTaxonomy

DATASET_IDS = ["108759", "147774", "163012", "177266"]
GENUS_RANK = -1  # last rank = genus in 6-rank SILVA taxonomy


def build_genus_index(silva_tax_db: taxonomy.TaxonomyDb, num_genera: int) -> list[np.ndarray]:
    """Return a list where index i holds all SILVA sequence indices for genus i."""
    return [silva_tax_db.sequence_indices_with_taxonomy_id(i) for i in range(num_genera)]


def predict_genus_for_all(
    model: DnaBertForTaxonomy,
    fasta_db: fasta.FastaDb,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """
    Run DNABERT TopDown on every unique sequence in fasta_db.
    Returns genus predictions as int64 array of shape [len(fasta_db)].
    """
    tokenizer = model.tokenizer
    max_length = model.base.config.max_length
    pad_id = tokenizer.vocab["[PAD]"]
    kmer, stride = tokenizer.kmer, tokenizer.kmer_stride
    max_bp = max_length * stride - stride + kmer

    n = len(fasta_db)
    all_preds = np.empty(n, dtype=np.int64)

    model.eval()
    with torch.no_grad():
        for start in tqdm(range(0, n, batch_size), desc="  Classifying sequences", leave=False):
            end = min(start + batch_size, n)
            tokens_list = []
            for i in range(start, end):
                seq = fasta_db.entry(i).sequence
                t = torch.tensor(tokenizer(seq[:max_bp]))
                t = F.pad(t, (0, max_length - len(t)), value=pad_id)
                tokens_list.append(t)
            tokens = torch.stack(tokens_list).to(device)
            logits_list = model(tokens)
            genus_ids = logits_list[GENUS_RANK].argmax(-1).cpu().numpy()
            all_preds[start:end] = genus_ids

    return all_preds


def generate_mapping(
    model: DnaBertForTaxonomy,
    genus_index: list[np.ndarray],
    dataset_dir: Path,
    output_path: Path,
    silva_fasta_db: fasta.FastaDb,
    batch_size: int,
    device: torch.device,
    rng: np.random.Generator,
):
    fasta_db_path = dataset_dir / "sequences.fasta.db"
    mapping_db_path = dataset_dir / "sequences.fasta.mapping.db"

    if not fasta_db_path.exists() or not mapping_db_path.exists():
        print(f"  Skipping {dataset_dir.name}: sequences.fasta.db or sequences.fasta.mapping.db not found.")
        return

    real_fasta_db = fasta.FastaDb(fasta_db_path)
    # load_into_memory=True so sequence_index() is a numpy array lookup, not an lmdb hit
    real_samples = real_fasta_db.mappings(mapping_db_path, load_into_memory=True)
    print(f"  {len(real_samples)} samples, {len(real_fasta_db):,} unique sequences")

    # Classify every unique sequence once
    genus_preds = predict_genus_for_all(model, real_fasta_db, batch_size, device)

    # Count sequences mapping to empty genera
    empty = sum(1 for gid in genus_preds if len(genus_index[gid]) == 0)
    if empty:
        print(f"  Warning: {empty:,} unique sequences predict a genus with no SILVA sequences (reads will be dropped)")

    # Build SILVA mapping: for each sample, translate every read occurrence to a SILVA sequence
    silva_mapping_factory = fasta.FastaMappingDbFactory(output_path, silva_fasta_db)

    for sample in tqdm(real_samples, desc="  Building SILVA mappings"):
        silva_entry = silva_mapping_factory.create_entry(sample.name)

        # Get all sequence index occurrences as a numpy array (fast: in-memory lookup)
        raw_indices = np.array([sample.sequence_index(i) for i in range(len(sample))], dtype=np.int64)

        # Group by unique real-world sequence to batch the random SILVA sampling
        unique_indices, counts = np.unique(raw_indices, return_counts=True)
        for real_idx, count in zip(unique_indices, counts):
            candidates = genus_index[int(genus_preds[real_idx])]
            if len(candidates) == 0:
                continue
            # Sample 'count' SILVA sequences (with replacement) for this real-world sequence
            chosen = rng.choice(candidates, size=int(count), replace=True)
            for silva_idx in chosen:
                silva_entry.write_sequence_index(int(silva_idx))

        silva_mapping_factory.write_entry(silva_entry)

    silva_mapping_factory.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate SILVA-based sample mappings for SetBERT pretraining"
    )
    parser.add_argument(
        "model_dir", type=Path,
        help="Exported DNABERT TopDown model directory"
    )
    parser.add_argument(
        "datasets_dir", type=Path,
        help="Datasets directory ($DATASETS_DIR)"
    )
    parser.add_argument(
        "dataset_ids", nargs="*", default=DATASET_IDS,
        help="mini-qiita dataset IDs to process (default: all four)"
    )
    parser.add_argument(
        "--reference-dataset", default="silva_nr99_filtered_515f_806r",
        help="Reference dataset name (default: silva_nr99_filtered_515f_806r)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="DNABERT inference batch size (default: 256)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for SILVA sequence sampling (default: 42)"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing mapping files"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading model from {args.model_dir}...")
    model = DnaBertForTaxonomy.from_pretrained(args.model_dir).to(device)
    num_genera = len(model.config.rank_labels[GENUS_RANK])

    reference_dir = args.datasets_dir / args.reference_dataset
    print(f"Loading SILVA reference from {reference_dir}...")
    silva_fasta_db = fasta.FastaDb(reference_dir / "sequences.fasta.db")
    silva_tax_db = taxonomy.TaxonomyDb(
        reference_dir / "taxonomy.tax.db",
        in_memory=taxonomy.TaxonomyDb.InMemory.SequencesWithTaxonomy,
    )
    print(f"  {len(silva_fasta_db):,} sequences, {num_genera:,} genera")

    print("Building genus → SILVA sequence index...")
    genus_index = build_genus_index(silva_tax_db, num_genera)

    rng = np.random.default_rng(args.seed)

    for dataset_id in args.dataset_ids:
        dataset_dir = args.datasets_dir / f"mini-qiita-{dataset_id}"
        output_name = f"sequences.topdown.{args.reference_dataset}.fasta.mapping.db"
        output_path = dataset_dir / output_name

        if output_path.exists() and not args.overwrite:
            print(f"\n[{dataset_id}] Mapping already exists, skipping (use --overwrite to redo).")
            continue

        print(f"\n[{dataset_id}] Generating SILVA mapping...")
        generate_mapping(
            model=model,
            genus_index=genus_index,
            dataset_dir=dataset_dir,
            output_path=output_path,
            silva_fasta_db=silva_fasta_db,
            batch_size=args.batch_size,
            device=device,
            rng=rng,
        )
        print(f"[{dataset_id}] Written to {output_path}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
