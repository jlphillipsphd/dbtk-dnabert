#!/usr/bin/env python3
"""
Evaluate a fine-tuned DnaBert taxonomy classification model.

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

import lightning as L
import torch
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


def taxon_string(rank_labels, taxon_ids) -> str:
    return ";".join(rank_labels[rank][int(tid)] for rank, tid in enumerate(taxon_ids))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned DnaBert taxonomy model"
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

    # Load model and set top-k for predict_step
    model = load_model(args.model_path)
    model._predict_top_k = args.top_k
    rank_labels = model.config.rank_labels
    num_ranks = model.num_ranks

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

    all_pred_ids = torch.cat(pred_ids_list, dim=0)   # [N, R]
    all_true_ids = torch.cat(true_ids_list, dim=0)   # [N, R]
    all_topk_ids = torch.cat(topk_ids_list, dim=0)   # [N, R, k]
    n = len(all_seq_ids)

    # Compute accuracy
    top1_acc = (all_pred_ids == all_true_ids).float().mean(dim=0)         # [R]
    topk_acc = (all_topk_ids == all_true_ids.unsqueeze(-1)).any(dim=-1).float().mean(dim=0)  # [R]

    # Per-sequence output
    if args.output or args.show_predictions:
        out = open(args.output, "w") if args.output else sys.stdout
        rank_headers = "\t".join(f"rank{r}_correct" for r in range(num_ranks))
        out.write(f"sequence_id\tground_truth\tpredicted\t{rank_headers}\n")
        top1_per_seq = (all_pred_ids == all_true_ids)  # [N, R]
        for i, seq_id in enumerate(all_seq_ids):
            gt   = taxon_string(rank_labels, all_true_ids[i].tolist())
            pred = taxon_string(rank_labels, all_pred_ids[i].tolist())
            rank_cols = "\t".join("Y" if top1_per_seq[i, r].item() else "N" for r in range(num_ranks))
            out.write(f"{seq_id}\t{gt}\t{pred}\t{rank_cols}\n")
        if out is not sys.stdout:
            out.close()

    # Summary table
    k = args.top_k
    print(f"\nResults ({n} sequences):")
    print(f"{'Rank':<8}  {'Top-1':>8}  {f'Top-{k}':>8}")
    print("-" * 30)
    for rank in range(num_ranks):
        print(f"rank {rank:<3}  {top1_acc[rank].item():>8.4f}  {topk_acc[rank].item():>8.4f}")


if __name__ == "__main__":
    main()
