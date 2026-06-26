#!/usr/bin/env python3
"""
Generate SILVA-based sample mapping files for SetBERT pretraining.

For each dataset listed in the config, classifies every unique real-world sequence
at genus level, then builds a per-sample mapping to SILVA reference sequences of the
predicted genera. Read counts are preserved: if a sample has N reads mapping to genus
G, N SILVA sequences are randomly drawn from G and written to the mapping entry.

Output per dataset:
    {dataset.path}/sequences.{classifier.model_name}.{reference_dataset.name}.fasta.mapping.db

Usage:
    python generate_sample_mappings.py --config configs/generate_mappings/topdown_64d.yaml
    python generate_sample_mappings.py --config ... --overwrite
    python generate_sample_mappings.py --config ... --seed 0

Config schema (YAML):
    classifier:
      class_path: dnabert.classifiers.DnaBertTopDownClassifier
      config:
        model_path: ${env:MODELS_DIR}/...
        batch_size: 256

    reference_dataset:
      name: silva_nr99_filtered_515f_806r
      path: ${env:DATASETS_DIR}/silva_nr99_filtered_515f_806r

    datasets:
      - name: mini-qiita-108759
        path: ${env:DATASETS_DIR}/mini-qiita-108759

    seed: 42
    overwrite: false
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from dnadb import fasta, taxonomy
from dbtk.config import load as load_config
from tqdm import tqdm


def build_genus_index(silva_tax_db: taxonomy.TaxonomyDb, num_genera: int) -> list[np.ndarray]:
    """Return a list where index i holds all SILVA sequence indices for genus i."""
    return [silva_tax_db.sequence_indices_with_taxonomy_id(i) for i in range(num_genera)]


def generate_mapping(
    classifier,
    genus_index: list[np.ndarray],
    dataset_dir: Path,
    output_path: Path,
    silva_fasta_db: fasta.FastaDb,
    device: torch.device,
    rng: np.random.Generator,
) -> None:
    fasta_db_path = dataset_dir / "sequences.fasta.db"
    mapping_db_path = dataset_dir / "sequences.fasta.mapping.db"

    if not fasta_db_path.exists() or not mapping_db_path.exists():
        print(f"  Skipping {dataset_dir.name}: sequences.fasta.db or .mapping.db not found.")
        return

    real_fasta_db = fasta.FastaDb(fasta_db_path)
    real_samples = real_fasta_db.mappings(mapping_db_path, load_into_memory=True)
    print(f"  {len(real_samples)} samples, {len(real_fasta_db):,} unique sequences")

    genus_preds = classifier.predict_genus(real_fasta_db, device)

    empty = sum(1 for gid in genus_preds if len(genus_index[gid]) == 0)
    if empty:
        print(f"  Warning: {empty:,} sequences predict a genus with no SILVA entries (reads will be dropped)")

    silva_mapping_factory = fasta.FastaMappingDbFactory(output_path, silva_fasta_db)

    for sample in tqdm(real_samples, desc="  Building SILVA mappings"):
        silva_entry = silva_mapping_factory.create_entry(sample.name)
        raw_indices = np.array([sample.sequence_index(i) for i in range(len(sample))], dtype=np.int64)
        unique_indices, counts = np.unique(raw_indices, return_counts=True)
        for real_idx, count in zip(unique_indices, counts):
            candidates = genus_index[int(genus_preds[real_idx])]
            if len(candidates) == 0:
                continue
            chosen = rng.choice(candidates, size=int(count), replace=True)
            for silva_idx in chosen:
                silva_entry.write_sequence_index(int(silva_idx))
        silva_mapping_factory.write_entry(silva_entry)

    silva_mapping_factory.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate SILVA-based sample mappings for SetBERT pretraining"
    )
    parser.add_argument("--config", type=Path, required=True, help="YAML config file")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing mapping files")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed from config")
    args = parser.parse_args()

    cfg = load_config(args.config)

    classifier = cfg["classifier"]
    ref_name = cfg["reference_dataset"]["name"]
    ref_path = Path(cfg["reference_dataset"]["path"])
    datasets = [(d["name"], Path(d["path"])) for d in cfg["datasets"]]
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    overwrite = args.overwrite or cfg.get("overwrite", False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Classifier: {classifier.model_name}")

    print(f"Loading SILVA reference from {ref_path}...")
    silva_fasta_db = fasta.FastaDb(ref_path / "sequences.fasta.db")
    silva_tax_db = taxonomy.TaxonomyDb(
        ref_path / "taxonomy.tax.db",
        in_memory=taxonomy.TaxonomyDb.InMemory.SequencesWithTaxonomy,
    )
    num_genera = silva_tax_db.num_labels
    print(f"  {len(silva_fasta_db):,} sequences, {num_genera:,} genera")

    print("Building genus → SILVA sequence index...")
    genus_index = build_genus_index(silva_tax_db, num_genera)

    rng = np.random.default_rng(seed)

    for dataset_name, dataset_path in datasets:
        output_name = f"sequences.{classifier.model_name}.{ref_name}.fasta.mapping.db"
        output_path = dataset_path / output_name

        if output_path.exists() and not overwrite:
            print(f"\n[{dataset_name}] Already exists, skipping (use --overwrite to redo).")
            continue

        print(f"\n[{dataset_name}] Generating SILVA mapping...")
        generate_mapping(
            classifier=classifier,
            genus_index=genus_index,
            dataset_dir=dataset_path,
            output_path=output_path,
            silva_fasta_db=silva_fasta_db,
            device=device,
            rng=rng,
        )
        print(f"[{dataset_name}] Written to {output_path}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
