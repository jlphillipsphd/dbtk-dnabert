#!/usr/bin/env python3
"""
Run taxonomy prediction and cache results to disk for later evaluation.

Processes a FASTA DB through a fine-tuned DnaBert taxonomy model and saves
top-k predictions and all model metadata needed for offline evaluation to a
single .pt file.  Ground-truth labels are not involved at inference time;
use evaluate_predictions.py with a taxonomy DB to compute accuracy metrics.

Usage:
    python predict_taxonomy.py MODEL_PATH SEQUENCES_DB OUTPUT [options]

Example:
    python predict_taxonomy.py $MODELS_DIR/dnabert-topdown-exported \\
        $DATASETS_DIR/silva_nr99_filtered_515f_806r/sequences.test.fasta.db \\
        predictions.pt
"""

import argparse
import importlib
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


def main():
    parser = argparse.ArgumentParser(
        description="Cache DnaBert taxonomy predictions to disk",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("model_path",     type=Path, help="Exported HF model directory")
    parser.add_argument("sequences_path", type=Path, help="Sequences FASTA DB path")
    parser.add_argument("output_path",    type=Path, help="Output .pt file")
    parser.add_argument("--batch-size",  type=int, default=256,
                        help="Inference batch size (default: 256)")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers (default: 0)")
    parser.add_argument("--top-k",       type=int, default=5,
                        help="K for top-K predictions stored in output (default: 5)")
    args = parser.parse_args()

    print("Loading model...")
    model = load_model(args.model_path)
    model._predict_top_k = args.top_k

    train_id_to_taxon = [
        sorted(set(label.strip() for label in labels))
        for labels in model.config.rank_labels
    ]

    dm = DnaBertTaxonomyPredictDataModule(
        tokenizer=model.tokenizer,
        sequences_path=args.sequences_path,
        max_length=model.base.config.max_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
        logger=False,
        enable_model_summary=False,
    )

    print("Running inference...")
    outputs = trainer.predict(model, datamodule=dm)

    all_seq_ids = []
    pred_ids_list, topk_ids_list, topk_scores_list = [], [], []
    for seq_ids, pred_ids, topk_ids, topk_scores in outputs:
        all_seq_ids.extend(seq_ids)
        pred_ids_list.append(pred_ids)
        topk_ids_list.append(topk_ids)
        topk_scores_list.append(topk_scores)

    all_pred_ids    = torch.cat(pred_ids_list,    dim=0)
    all_topk_ids    = torch.cat(topk_ids_list,    dim=0)
    all_topk_scores = torch.cat(topk_scores_list, dim=0)

    if trainer.world_size > 1:
        import torch.distributed as dist
        local_payload = (
            all_seq_ids,
            all_pred_ids.cpu(),
            all_topk_ids.cpu(),
            all_topk_scores.cpu(),
        )
        gathered = [None] * trainer.world_size
        dist.all_gather_object(gathered, local_payload)
        if trainer.global_rank != 0:
            return
        seq_lists, pred_list, topk_list, scores_list = zip(*gathered)
        all_seq_ids     = [sid for sids in seq_lists for sid in sids]
        all_pred_ids    = torch.cat(pred_list,  dim=0)
        all_topk_ids    = torch.cat(topk_list,  dim=0)
        all_topk_scores = torch.cat(scores_list, dim=0)

    payload = {
        'seq_ids':           all_seq_ids,
        'pred_ids':          all_pred_ids.cpu(),
        'topk_ids':          all_topk_ids.cpu(),
        'topk_scores':       all_topk_scores.cpu(),
        'rank_labels':       model.config.rank_labels,
        'parent_indices':    model.config.parent_indices,
        'train_id_to_taxon': train_id_to_taxon,
        'top_k':             args.top_k,
        'model_path':        str(args.model_path),
        'sequences_path':    str(args.sequences_path),
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output_path)
    n = len(all_seq_ids)
    print(f"Saved {n:,} predictions ({args.top_k}-best per rank) → {args.output_path}")


if __name__ == "__main__":
    main()
