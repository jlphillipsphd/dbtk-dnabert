#!/usr/bin/env python3
"""
Evaluate a fine-tuned DnaBert taxonomy classification model.

Accuracy is computed by comparing taxonomy label strings. The model's embedded
rank_labels (stored in taxonomy_id / DFS order) are sorted alphabetically to
recover taxon_id order, which matches the integer IDs produced by the model.
Ground-truth IDs from the evaluation taxonomy DB are resolved via
tree.id_to_taxon_map (also alphabetically ordered). String comparison then
works correctly even when the eval taxonomy DB has a broader label space than
the training taxonomy (e.g. full SILVA 138.2 vs NR99).

Usage:
    python evaluate_taxonomy.py MODEL_PATH SEQUENCES_DB TAXONOMIES_DB [options]

Example:
    python evaluate_taxonomy.py $MODELS_DIR/dnabert-topdown-exported \\
        $DATASETS_DIR/silva_nr99_filtered_515f_806r/sequences.test.fasta.db \\
        $DATASETS_DIR/silva_nr99_filtered_515f_806r/taxonomy.test.tax.db \\
        --output results.tsv
"""

import argparse
import importlib
import sys
from pathlib import Path

from rich.progress import track

import lightning as L
import torch
from dnadb import taxonomy
from transformers import PretrainedConfig

from dnabert.datamodules import DnaBertTaxonomyPredictDataModule

MODEL_TYPE_MAP = {
    "dnabert_for_taxonomy": "dnabert.models.DnaBertForTaxonomy",
}


def load_model(model_path: Path):
    model_path = Path(model_path).resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(
            f"Model directory not found: {model_path}\n"
            "Make sure $MODELS_DIR is set and the model has been exported with:\n"
            "  dbtk model export <checkpoint.ckpt> <output_dir>"
        )
    config = PretrainedConfig.from_pretrained(model_path)
    module_name, class_name = MODEL_TYPE_MAP[config.model_type].rsplit(".", 1)
    model_class = getattr(importlib.import_module(module_name), class_name)
    return model_class.from_pretrained(model_path)


def load_eval_id_to_taxon(tax_path: Path) -> tuple[list[list[str]], ...]:
    """Load taxon_id → label map from the evaluation taxonomy DB (alphabetical order)."""
    with taxonomy.TaxonomyDb(str(tax_path)) as tax_db:
        return tax_db.tree.id_to_taxon_map


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned DnaBert taxonomy model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("model_path",      type=Path, help="Exported HF model directory")
    parser.add_argument("sequences_path",  type=Path, help="Sequences FASTA DB path")
    parser.add_argument("taxonomies_path", type=Path, help="Taxonomy DB path")
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Save per-sequence results to a TSV file"
    )
    parser.add_argument(
        "--show-predictions", action="store_true",
        help="Print per-sequence predictions to stdout"
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="Inference batch size (default: 256)"
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="K for top-K accuracy (default: 5)"
    )
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="DataLoader workers (default: 0)"
    )
    args = parser.parse_args()

    # Load model
    model = load_model(args.model_path)
    model._predict_top_k = args.top_k
    num_ranks = model.num_ranks

    # Build pred_id → label map from the model's embedded rank_labels.
    # rank_labels is stored in taxonomy_id (DFS) order; sorting each rank
    # alphabetically recovers taxon_id order, matching model prediction IDs.
    train_id_to_taxon = [
        sorted(set(label.strip() for label in labels))
        for labels in model.config.rank_labels
    ]

    # Load taxon_id → label map from the evaluation taxonomy DB.
    # LMDB cannot be opened twice in one process, so we load into memory
    # and close before trainer.predict() reopens it via the datamodule.
    print("Loading eval taxonomy label map...")
    eval_id_to_taxon = load_eval_id_to_taxon(args.taxonomies_path)

    # Predict datamodule
    dm = DnaBertTaxonomyPredictDataModule(
        tokenizer=model.tokenizer,
        sequences_path=args.sequences_path,
        taxonomies_path=args.taxonomies_path,
        max_length=model.base.config.max_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Run inference via Lightning Trainer
    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
        logger=False,
        enable_model_summary=False,
    )
    outputs = trainer.predict(model, datamodule=dm)

    # Aggregate results across batches
    all_seq_ids = []
    pred_ids_list, true_ids_list, topk_ids_list = [], [], []
    for seq_ids, pred_ids, true_ids, topk_ids in outputs:
        all_seq_ids.extend(seq_ids)
        pred_ids_list.append(pred_ids)
        true_ids_list.append(true_ids)
        topk_ids_list.append(topk_ids)

    all_pred_ids = torch.cat(pred_ids_list, dim=0)   # [N_local, R]
    all_true_ids = torch.cat(true_ids_list, dim=0)   # [N_local, R]
    all_topk_ids = torch.cat(topk_ids_list, dim=0)   # [N_local, R, k]

    # In DDP each rank holds only its shard. Gather everything onto rank 0
    # so we print a single merged table; non-rank-0 processes exit early.
    if trainer.world_size > 1:
        import torch.distributed as dist
        local_payload = (
            all_seq_ids,
            all_pred_ids.cpu(),
            all_true_ids.cpu(),
            all_topk_ids.cpu(),
        )
        gathered = [None] * trainer.world_size
        dist.all_gather_object(gathered, local_payload)
        if trainer.global_rank != 0:
            return
        seq_lists, pred_list, true_list, topk_list = zip(*gathered)
        all_seq_ids  = [sid for sids in seq_lists for sid in sids]
        all_pred_ids = torch.cat(pred_list, dim=0)
        all_true_ids = torch.cat(true_list, dim=0)
        all_topk_ids = torch.cat(topk_list, dim=0)

    n = len(all_seq_ids)

    def pred_label(r: int, tid: int) -> str:
        labels = train_id_to_taxon[r]
        return labels[tid] if tid < len(labels) else ""

    def true_label(r: int, tid: int) -> str:
        labels = eval_id_to_taxon[r]
        return labels[tid].strip() if tid < len(labels) else ""

    # Single pass: resolve labels and compute top-1 / top-k correctness together.
    true_parts, pred_parts, top1_correct, topk_correct = [], [], [], []
    for i in track(range(n), description="Computing accuracy"):
        tp = [true_label(r, all_true_ids[i, r].item()) for r in range(num_ranks)]
        pp = [pred_label(r, all_pred_ids[i, r].item()) for r in range(num_ranks)]
        true_parts.append(tp)
        pred_parts.append(pp)
        top1_correct.append([pp[r] == tp[r] for r in range(num_ranks)])
        topk_correct.append([
            tp[r] in {pred_label(r, all_topk_ids[i, r, k].item()) for k in range(args.top_k)}
            for r in range(num_ranks)
        ])

    top1_acc = [sum(top1_correct[i][r] for i in range(n)) / n for r in range(num_ranks)]
    topk_acc = [sum(topk_correct[i][r] for i in range(n)) / n for r in range(num_ranks)]

    # Per-sequence output
    if args.output or args.show_predictions:
        out = open(args.output, "w") if args.output else sys.stdout
        rank_headers = "\t".join(f"rank{r}_correct" for r in range(num_ranks))
        out.write(f"sequence_id\tground_truth\tpredicted\t{rank_headers}\n")
        for i, seq_id in track(enumerate(all_seq_ids), total=n, description="Writing results"):
            gt   = ";".join(true_parts[i])
            pred = ";".join(pred_parts[i])
            rank_cols = "\t".join("Y" if top1_correct[i][r] else "N" for r in range(num_ranks))
            out.write(f"{seq_id}\t{gt}\t{pred}\t{rank_cols}\n")
        if out is not sys.stdout:
            out.close()

    # Summary table
    k = args.top_k
    print(f"\nResults ({n} sequences):")
    print(f"{'Rank':<8}  {'Top-1':>8}  {f'Top-{k}':>8}")
    print("-" * 30)
    for rank in range(num_ranks):
        print(f"rank {rank:<3}  {top1_acc[rank]:>8.4f}  {topk_acc[rank]:>8.4f}")


if __name__ == "__main__":
    main()
