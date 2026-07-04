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

Anchor mode (--anchor-rank K):
    Rank K is trusted. All coarser ranks (< K) are resolved by walking up
    the taxonomy tree from the rank-K prediction rather than using independent
    model predictions. Ranks finer than K (> K) remain independent.
    When a taxon appears under multiple parents (shared name), the parent with
    the highest model score at that rank is selected; if no candidate appears
    in the top-k, the alphabetically first candidate is used as a tiebreak.
    --anchor-rank 0 is equivalent to independent mode (no coarser ranks exist).

Usage:
    python evaluate_taxonomy.py MODEL_PATH SEQUENCES_DB TAXONOMIES_DB [options]

Example:
    python evaluate_taxonomy.py $MODELS_DIR/dnabert-topdown-exported \\
        $DATASETS_DIR/silva_nr99_filtered_515f_806r/sequences.test.fasta.db \\
        $DATASETS_DIR/silva_nr99_filtered_515f_806r/taxonomy.test.tax.db \\
        --output results.tsv --anchor-rank 5
"""

import argparse
import importlib
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def load_eval_id_to_taxon(tax_path: Path) -> Tuple[List[str], ...]:
    """Load taxon_id → label map from the evaluation taxonomy DB (alphabetical order)."""
    with taxonomy.TaxonomyDb(str(tax_path)) as tax_db:
        return tax_db.tree.id_to_taxon_map


def build_ancestor_map(
    rank_labels: List[List[str]],
    parent_indices: List[List[int]],
    train_id_to_taxon: List[List[str]],
) -> List[Optional[Dict[int, List[int]]]]:
    """
    Build a per-rank mapping from taxon_id → list of candidate parent taxon_ids.

    parent_map[r][taxon_id] = sorted list of candidate parent taxon_ids at rank r-1.
    parent_map[0] = None (domain has no parent).
    The list has more than one entry only when the same label appears under
    multiple parents (shared taxon name within a rank).
    """
    taxon_to_id = [
        {label: i for i, label in enumerate(labels)}
        for labels in train_id_to_taxon
    ]

    parent_map: List[Optional[Dict[int, List[int]]]] = [None]
    for r in range(1, len(rank_labels)):
        mapping: Dict[int, List[int]] = {}
        for taxonomy_id_i, label in enumerate(rank_labels[r]):
            taxon_id = taxon_to_id[r].get(label)
            if taxon_id is None:
                continue
            parent_taxonomy_id = parent_indices[r][taxonomy_id_i]
            parent_label = rank_labels[r - 1][parent_taxonomy_id]
            parent_taxon_id = taxon_to_id[r - 1].get(parent_label)
            if parent_taxon_id is None:
                continue
            candidates = mapping.setdefault(taxon_id, [])
            if parent_taxon_id not in candidates:
                candidates.append(parent_taxon_id)
        parent_map.append(mapping)
    return parent_map


def resolve_anchored(
    pred_taxon_id_k: int,
    anchor_rank: int,
    parent_map: List[Optional[Dict[int, List[int]]]],
    topk_ids_seq: torch.Tensor,    # [R, K]
    topk_scores_seq: torch.Tensor, # [R, K]
) -> Dict[int, int]:
    """
    Walk up the tree from rank anchor_rank, resolving taxon_ids for ranks 0..anchor_rank-1.
    Returns {rank: resolved_taxon_id} for all ranks 0..anchor_rank.
    When a taxon appears under multiple parents, the parent with the highest model
    score at that rank is selected; ties fall back to the alphabetically first candidate.
    """
    resolved: Dict[int, int] = {anchor_rank: pred_taxon_id_k}
    current = pred_taxon_id_k
    for r in range(anchor_rank, 0, -1):
        candidates = parent_map[r].get(current, [])
        if not candidates:
            current = -1
        elif len(candidates) == 1:
            current = candidates[0]
        else:
            parent_rank = r - 1
            score_dict = {
                int(topk_ids_seq[parent_rank, j]): float(topk_scores_seq[parent_rank, j])
                for j in range(topk_ids_seq.shape[1])
            }
            current = max(candidates, key=lambda c: score_dict.get(c, float('-inf')))
        resolved[r - 1] = current
    return resolved


def resolve_first(
    taxon_id: int,
    from_rank: int,
    to_rank: int,
    parent_map: List[Optional[Dict[int, List[int]]]],
) -> int:
    """
    Walk up the tree from from_rank to to_rank, always picking the first (alphabetically
    lowest) candidate parent.  Used to build the set of reachable ancestors for top-k.
    """
    current = taxon_id
    for r in range(from_rank, to_rank, -1):
        candidates = parent_map[r].get(current, [])
        current = candidates[0] if candidates else -1
    return current


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
        help="K for top-K accuracy and anchor disambiguation (default: 5)"
    )
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="DataLoader workers (default: 0)"
    )
    parser.add_argument(
        "--anchor-rank", type=int, default=None, metavar="K",
        help="Trust model prediction at rank K; resolve coarser ranks from the tree. "
             "Ranks finer than K remain independent. "
             "--anchor-rank 0 is equivalent to independent mode."
    )
    args = parser.parse_args()

    anchor_rank = args.anchor_rank
    anchoring = anchor_rank is not None and anchor_rank > 0

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

    # Build ancestor map for anchor mode
    parent_map = None
    if anchoring:
        parent_map = build_ancestor_map(
            model.config.rank_labels,
            model.config.parent_indices,
            train_id_to_taxon,
        )

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
    pred_ids_list, true_ids_list, topk_ids_list, topk_scores_list = [], [], [], []
    for seq_ids, pred_ids, true_ids, topk_ids, topk_scores in outputs:
        all_seq_ids.extend(seq_ids)
        pred_ids_list.append(pred_ids)
        true_ids_list.append(true_ids)
        topk_ids_list.append(topk_ids)
        topk_scores_list.append(topk_scores)

    all_pred_ids    = torch.cat(pred_ids_list,    dim=0)  # [N, R]
    all_true_ids    = torch.cat(true_ids_list,    dim=0)  # [N, R]
    all_topk_ids    = torch.cat(topk_ids_list,    dim=0)  # [N, R, k]
    all_topk_scores = torch.cat(topk_scores_list, dim=0)  # [N, R, k]

    # In DDP each rank holds only its shard. Gather everything onto rank 0
    # so we print a single merged table; non-rank-0 processes exit early.
    if trainer.world_size > 1:
        import torch.distributed as dist
        local_payload = (
            all_seq_ids,
            all_pred_ids.cpu(),
            all_true_ids.cpu(),
            all_topk_ids.cpu(),
            all_topk_scores.cpu(),
        )
        gathered = [None] * trainer.world_size
        dist.all_gather_object(gathered, local_payload)
        if trainer.global_rank != 0:
            return
        seq_lists, pred_list, true_list, topk_list, scores_list = zip(*gathered)
        all_seq_ids     = [sid for sids in seq_lists for sid in sids]
        all_pred_ids    = torch.cat(pred_list,   dim=0)
        all_true_ids    = torch.cat(true_list,   dim=0)
        all_topk_ids    = torch.cat(topk_list,   dim=0)
        all_topk_scores = torch.cat(scores_list, dim=0)

    n = len(all_seq_ids)

    def pred_label(r: int, tid: int) -> str:
        labels = train_id_to_taxon[r]
        return labels[tid] if 0 <= tid < len(labels) else ""

    def true_label(r: int, tid: int) -> str:
        labels = eval_id_to_taxon[r]
        return labels[tid].strip() if 0 <= tid < len(labels) else ""

    # Determine which ranks are anchored (resolved from tree) vs independent
    # rank r is anchored if anchor mode is active and r <= anchor_rank
    is_anchored = [
        (anchoring and r < anchor_rank)
        for r in range(num_ranks)
    ]

    # Single pass: resolve labels and compute top-1 / top-k correctness.
    true_parts, pred_parts, top1_correct, topk_correct = [], [], [], []
    tree_inconsistencies = 0  # correct anchor in top-k but resolved ancestor is wrong
    for i in track(range(n), description="Computing accuracy"):
        tp = [true_label(r, all_true_ids[i, r].item()) for r in range(num_ranks)]

        if anchoring:
            resolved = resolve_anchored(
                all_pred_ids[i, anchor_rank].item(),
                anchor_rank,
                parent_map,
                all_topk_ids[i],    # [R, k]
                all_topk_scores[i], # [R, k]
            )
            pp = [
                pred_label(r, resolved[r]) if r in resolved else pred_label(r, all_pred_ids[i, r].item())
                for r in range(num_ranks)
            ]
        else:
            pp = [pred_label(r, all_pred_ids[i, r].item()) for r in range(num_ranks)]

        true_parts.append(tp)
        pred_parts.append(pp)
        top1_correct.append([pp[r] == tp[r] for r in range(num_ranks)])

        if anchoring:
            # For ranks coarser than anchor: top-k = is the true ancestor reachable
            # from any of the top-k predictions at the anchor rank?
            # For the anchor rank and finer: standard independent top-k.
            anchor_topk_ids = {all_topk_ids[i, anchor_rank, j].item() for j in range(args.top_k)}
            true_anchor_id  = all_true_ids[i, anchor_rank].item()
            anchor_correct_in_topk = true_anchor_id in anchor_topk_ids
            seq_inconsistent = False

            topk_reachable = []
            for r in range(num_ranks):
                if is_anchored[r] and r < anchor_rank:
                    ancestor_labels = {
                        pred_label(r, resolve_first(
                            all_topk_ids[i, anchor_rank, j].item(),
                            anchor_rank, r, parent_map,
                        ))
                        for j in range(args.top_k)
                    }
                    topk_reachable.append(tp[r] in ancestor_labels)

                    # Sanity check: if the correct anchor taxon is in the top-k, resolving
                    # it upward should always yield the correct ancestor. A mismatch means
                    # the model's embedded taxonomy tree diverges from the eval taxonomy DB.
                    if anchor_correct_in_topk and not seq_inconsistent:
                        resolved_ancestor = pred_label(r, resolve_first(
                            true_anchor_id, anchor_rank, r, parent_map,
                        ))
                        if resolved_ancestor != tp[r]:
                            seq_inconsistent = True
                else:
                    topk_reachable.append(
                        tp[r] in {pred_label(r, all_topk_ids[i, r, j].item()) for j in range(args.top_k)}
                    )
            tree_inconsistencies += seq_inconsistent
            topk_correct.append(topk_reachable)
        else:
            topk_correct.append([
                tp[r] in {pred_label(r, all_topk_ids[i, r, j].item()) for j in range(args.top_k)}
                for r in range(num_ranks)
            ])

    top1_acc = [sum(top1_correct[i][r] for i in range(n)) / n for r in range(num_ranks)]
    topk_acc = [sum(topk_correct[i][r] for i in range(n)) / n for r in range(num_ranks)]

    # Per-sequence output (TSV)
    if args.output or args.show_predictions:
        out = open(args.output, "w") if args.output else sys.stdout

        def col_name(r: int) -> str:
            base = f"rank{r}_correct"
            return base + "_anchored" if is_anchored[r] else base

        rank_headers = "\t".join(col_name(r) for r in range(num_ranks))
        out.write(f"sequence_id\tground_truth\tpredicted\t{rank_headers}\n")
        for i, seq_id in track(enumerate(all_seq_ids), total=n, description="Writing results"):
            gt       = ";".join(true_parts[i])
            pred     = ";".join(pred_parts[i])
            rank_cols = "\t".join("Y" if top1_correct[i][r] else "N" for r in range(num_ranks))
            out.write(f"{seq_id}\t{gt}\t{pred}\t{rank_cols}\n")
        if out is not sys.stdout:
            out.close()

    # Summary table
    k = args.top_k
    if anchoring:
        print(f"\nResults ({n} sequences, anchor-rank={anchor_rank}):")
        header = f"{'Rank':<8}  {'Top-1':>8}  {f'Top-{k}':>8}  {'Anchored':>8}"
        print(header)
        print("-" * len(header))
        for rank in range(num_ranks):
            anchored_col = "Y" if is_anchored[rank] else "N"
            print(f"rank {rank:<3}  {top1_acc[rank]:>8.4f}  {topk_acc[rank]:>8.4f}  {anchored_col:>8}")
        if tree_inconsistencies:
            print(
                f"\nWARNING: {tree_inconsistencies} case(s) where the correct anchor taxon "
                f"was in the top-{k} but its resolved ancestor did not match the ground truth. "
                "This likely indicates that the model was trained on a different taxonomy version "
                "than the evaluation DB (e.g. a taxon was reclassified between SILVA versions)."
            )
    else:
        print(f"\nResults ({n} sequences):")
        print(f"{'Rank':<8}  {'Top-1':>8}  {f'Top-{k}':>8}")
        print("-" * 30)
        for rank in range(num_ranks):
            print(f"rank {rank:<3}  {top1_acc[rank]:>8.4f}  {topk_acc[rank]:>8.4f}")


if __name__ == "__main__":
    main()
