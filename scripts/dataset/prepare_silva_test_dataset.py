#!/usr/bin/env python3
"""
Create held-out test FASTA and taxonomy DBs from a full SILVA dataset.

Sequences are selected from the full SILVA source that meet all three criteria:
  1. NOT present in the reference taxonomy DB (e.g. NR99 is the reference)
  2. Taxonomy recognized by the reference tree
  3. At or above the minimum length

The test taxonomy DB is built against the same tree as the reference, ensuring
taxon_ids align between reference and test sets.

Usage:
    python prepare_silva_test_dataset.py \\
        --reference-tax-db $DATASETS_DIR/silva_nr99_filtered_515f_806r/taxonomy.tax.db \\
        --sequences-path   $DATA_DIR/silva/138.2/silva-138.2-seqs-515f-806r-derep.fasta \\
        --taxonomy-path    $DATA_DIR/silva/138.2/silva-138.2-tax-515f-806r-derep.tsv \\
        --output-path      $DATASETS_DIR/silva_nr99_filtered_515f_806r \\
        --min-length       150
"""

import argparse
from pathlib import Path
from typing import Generator, Iterable, Tuple

from dnadb import fasta, taxonomy
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn


def paired_entries(
    sequences_path: Path,
    taxonomy_path: Path,
) -> Generator[Tuple[fasta.FastaEntry, taxonomy.TaxonomyEntry], None, None]:
    """Yield (FastaEntry, TaxonomyEntry) pairs matched by sequence ID."""
    pending = {}
    tax_iter = iter(taxonomy.entries(str(taxonomy_path)))
    for seq_entry in fasta.entries(str(sequences_path)):
        while seq_entry.identifier not in pending:
            try:
                tax_entry = next(tax_iter)
                pending[tax_entry.sequence_id] = tax_entry
            except StopIteration:
                return
        yield seq_entry, pending.pop(seq_entry.identifier)


def main():
    parser = argparse.ArgumentParser(
        description="Build held-out test FASTA and taxonomy DBs from full SILVA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--reference-tax-db", type=Path, required=True,
        help="Reference taxonomy DB (e.g. NR99) — sequences present here are excluded",
    )
    parser.add_argument(
        "--sequences-path", type=Path, required=True,
        help="Full SILVA sequences FASTA file (source for test sequences)",
    )
    parser.add_argument(
        "--taxonomy-path", type=Path, required=True,
        help="Full SILVA taxonomy TSV file matching the sequences",
    )
    parser.add_argument(
        "--output-path", type=Path, required=True,
        help="Directory to write sequences.test.fasta.db and taxonomy.test.tax.db",
    )
    parser.add_argument(
        "--min-length", type=int, default=150,
        help="Minimum sequence length in bp (default: 150)",
    )
    args = parser.parse_args()

    args.output_path.mkdir(parents=True, exist_ok=True)
    fasta_out = args.output_path / "sequences.test.fasta.db"
    tax_out   = args.output_path / "taxonomy.test.tax.db"

    print(f"Reference taxonomy DB : {args.reference_tax_db}")
    print(f"Source sequences      : {args.sequences_path}")
    print(f"Source taxonomy       : {args.taxonomy_path}")
    print(f"Output path           : {args.output_path}")
    print(f"Min sequence length   : {args.min_length} bp")
    print()

    ref_tax_db = taxonomy.TaxonomyDb(str(args.reference_tax_db))
    depth = ref_tax_db.tree.depth

    n_total = n_in_ref = n_short = n_unknown = n_written = 0
    written_ids: set[str] = set()

    print("Writing test sequences...")
    with Progress(SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn()) as progress:
        task = progress.add_task("Filtering", total=None)
        with fasta.FastaDbFactory(str(fasta_out)) as fasta_factory:
            for seq_entry, tax_entry in paired_entries(args.sequences_path, args.taxonomy_path):
                n_total += 1
                progress.update(task, advance=1)
                if len(seq_entry.sequence) < args.min_length:
                    n_short += 1
                    continue
                if not ref_tax_db.has_taxonomy(tax_entry):
                    n_unknown += 1
                    continue
                if tax_entry.sequence_id in ref_tax_db:
                    n_in_ref += 1
                    continue
                fasta_factory.write_entry(seq_entry)
                written_ids.add(seq_entry.identifier)
                n_written += 1

    print(f"  {n_total:>10,}  sequences scanned")
    print(f"  {n_in_ref:>10,}  skipped — present in reference DB")
    print(f"  {n_short:>10,}  skipped — below {args.min_length} bp")
    print(f"  {n_unknown:>10,}  skipped — taxonomy not in reference tree")
    print(f"  {n_written:>10,}  written to {fasta_out.name}")
    print()

    print("Writing test taxonomy DB...")
    test_fasta_db = fasta.FastaDb(str(fasta_out))
    with Progress(SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn()) as progress:
        task = progress.add_task("Writing taxonomy", total=None)
        with taxonomy.TaxonomyDbFactory(str(tax_out), test_fasta_db, depth, ref_tax_db.tree) as tax_factory:
            for tax_entry in taxonomy.entries(str(args.taxonomy_path)):
                if tax_entry.sequence_id not in written_ids:
                    continue
                tax_factory.write_entry(tax_entry)
                progress.update(task, advance=1)

    ref_tax_db.close()
    print(f"  Written to {tax_out.name}")


if __name__ == "__main__":
    main()
