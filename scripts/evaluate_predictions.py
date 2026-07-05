#!/usr/bin/env python3
"""
Evaluate cached taxonomy predictions produced by predict_taxonomy.py.

Computes top-1 and top-k accuracy per rank from a .pt predictions file and a
taxonomy DB (the ground-truth source).  No model is needed at evaluation time.

Supports anchor-rank mode: rank K is trusted; coarser ranks are resolved by
walking up the stored taxonomy tree rather than using independent predictions.
The --top-k flag may be set to any value <= the K stored in the predictions file,
allowing cheaper re-evaluation (e.g. top-1 through top-3 from a top-5 file).

Usage:
    python evaluate_predictions.py PREDICTIONS_FILE TAXONOMIES_DB [options]

Example:
    python evaluate_predictions.py predictions.pt \\
        $DATASETS_DIR/silva_nr99_filtered_515f_806r/taxonomy.test.tax.db \\
        --anchor-rank 5
    python evaluate_predictions.py predictions.pt taxonomy.test.tax.db \\
        --top-k 1 --output results.tsv
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

from rich.progress import track

import torch
from dnadb import taxonomy


def build_ancestor_map(
    rank_labels: List[List[str]],
    parent_indices: List[List[int]],
    train_id_to_taxon: List[List[str]],
) -> List[Optional[Dict[int, List[int]]]]:
    """
    Build a per-rank mapping from taxon_id -> list of candidate parent taxon_ids.

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


def resolve_all_ancestors(
    taxon_id: int,
    from_rank: int,
    to_rank: int,
    parent_map: List[Optional[Dict[int, List[int]]]],
) -> set:
    """
    Walk up the tree from from_rank to to_rank, expanding ALL candidate parents
    at each step.  Returns the set of all reachable ancestor taxon_ids at to_rank.
    Handles shared-name taxa (e.g. Incertae_Sedis) correctly.
    """
    current = {taxon_id}
    for r in range(from_rank, to_rank, -1):
        next_set = set()
        for tid in current:
            next_set.update(parent_map[r].get(tid, []))
        current = next_set
        if not current:
            break
    return current


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate cached taxonomy predictions from predict_taxonomy.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("predictions_path", type=Path,
                        help="Predictions .pt file from predict_taxonomy.py")
    parser.add_argument("taxonomies_path",  type=Path,
                        help="Taxonomy DB path (ground-truth source)")
    parser.add_argument(
        "--top-k", type=int, default=None,
        help="K for top-K accuracy; must be <= stored K (default: use stored K)"
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Save per-sequence results to a TSV file"
    )
    parser.add_argument(
        "--show-predictions", action="store_true",
        help="Print per-sequence predictions to stdout"
    )
    parser.add_argument(
        "--anchor-rank", type=int, default=None, metavar="K",
        help="Trust model prediction at rank K; resolve coarser ranks from the tree. "
             "Ranks finer than K remain independent. "
             "--anchor-rank 0 is equivalent to independent mode."
    )
    args = parser.parse_args()

    print(f"Loading predictions from {args.predictions_path}...")
    data = torch.load(args.predictions_path, map_location="cpu", weights_only=False)

    all_seq_ids       = data['seq_ids']
    all_pred_ids      = data['pred_ids']       # [N, R]
    all_topk_ids      = data['topk_ids']       # [N, R, K_stored]
    all_topk_scores   = data['topk_scores']    # [N, R, K_stored]
    rank_labels       = data['rank_labels']
    parent_indices    = data['parent_indices']
    train_id_to_taxon = data['train_id_to_taxon']
    stored_k          = data['top_k']

    top_k = args.top_k if args.top_k is not None else stored_k
    if top_k > stored_k:
        print(f"ERROR: --top-k {top_k} exceeds the K={stored_k} stored in the predictions file")
        sys.exit(1)

    # Trim the K dimension if a smaller top-k was requested
    all_topk_ids    = all_topk_ids[:, :, :top_k]
    all_topk_scores = all_topk_scores[:, :, :top_k]

    n, num_ranks = all_pred_ids.shape

    # Load ground-truth labels from the taxonomy DB
    print(f"Loading ground-truth labels from {args.taxonomies_path}...")
    with taxonomy.TaxonomyDb(str(args.taxonomies_path)) as tax_db:
        eval_id_to_taxon = list(tax_db.tree.id_to_taxon_map)
        true_ids_list = [
            tax_db[seq_id].taxonomy.taxon_ids
            for seq_id in track(all_seq_ids, description="  Reading labels")
        ]
    all_true_ids = torch.tensor(true_ids_list, dtype=torch.long)  # [N, R]

    anchor_rank = args.anchor_rank
    anchoring = anchor_rank is not None and anchor_rank > 0

    parent_map = None
    if anchoring:
        parent_map = build_ancestor_map(rank_labels, parent_indices, train_id_to_taxon)

    def pred_label(r: int, tid: int) -> str:
        labels = train_id_to_taxon[r]
        return labels[tid] if 0 <= tid < len(labels) else ""

    def true_label(r: int, tid: int) -> str:
        labels = eval_id_to_taxon[r]
        return labels[tid].strip() if 0 <= tid < len(labels) else ""

    is_anchored = [
        (anchoring and r < anchor_rank)
        for r in range(num_ranks)
    ]

    true_parts, pred_parts, top1_correct, topk_correct = [], [], [], []
    tree_inconsistencies = 0
    for i in track(range(n), description="Computing accuracy"):
        tp = [true_label(r, all_true_ids[i, r].item()) for r in range(num_ranks)]

        if anchoring:
            resolved = resolve_anchored(
                all_pred_ids[i, anchor_rank].item(),
                anchor_rank,
                parent_map,
                all_topk_ids[i],
                all_topk_scores[i],
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
            anchor_topk_ids = {all_topk_ids[i, anchor_rank, j].item() for j in range(top_k)}
            true_anchor_id  = all_true_ids[i, anchor_rank].item()
            anchor_correct_in_topk = true_anchor_id in anchor_topk_ids
            seq_inconsistent = False

            topk_reachable = []
            for r in range(num_ranks):
                if is_anchored[r] and r < anchor_rank:
                    true_r_id = all_true_ids[i, r].item()
                    reachable_ids: set = set()
                    for j in range(top_k):
                        reachable_ids.update(resolve_all_ancestors(
                            all_topk_ids[i, anchor_rank, j].item(),
                            anchor_rank, r, parent_map,
                        ))
                    topk_reachable.append(true_r_id in reachable_ids)

                    if anchor_correct_in_topk and not seq_inconsistent:
                        reachable_from_true = resolve_all_ancestors(
                            true_anchor_id, anchor_rank, r, parent_map,
                        )
                        if true_r_id not in reachable_from_true:
                            seq_inconsistent = True
                else:
                    topk_reachable.append(
                        tp[r] in {pred_label(r, all_topk_ids[i, r, j].item()) for j in range(top_k)}
                    )
            tree_inconsistencies += seq_inconsistent
            topk_correct.append(topk_reachable)
        else:
            topk_correct.append([
                tp[r] in {pred_label(r, all_topk_ids[i, r, j].item()) for j in range(top_k)}
                for r in range(num_ranks)
            ])

    top1_acc = [sum(top1_correct[i][r] for i in range(n)) / n for r in range(num_ranks)]
    topk_acc = [sum(topk_correct[i][r] for i in range(n)) / n for r in range(num_ranks)]

    if args.output or args.show_predictions:
        out = open(args.output, "w") if args.output else sys.stdout

        def col_name(r: int) -> str:
            base = f"rank{r}_correct"
            return base + "_anchored" if is_anchored[r] else base

        rank_headers = "\t".join(col_name(r) for r in range(num_ranks))
        out.write(f"sequence_id\tground_truth\tpredicted\t{rank_headers}\n")
        for i, seq_id in track(enumerate(all_seq_ids), total=n, description="Writing results"):
            gt        = ";".join(true_parts[i])
            pred      = ";".join(pred_parts[i])
            rank_cols = "\t".join("Y" if top1_correct[i][r] else "N" for r in range(num_ranks))
            out.write(f"{seq_id}\t{gt}\t{pred}\t{rank_cols}\n")
        if out is not sys.stdout:
            out.close()

    k = top_k
    if anchoring:
        print(f"\nResults ({n:,} sequences, anchor-rank={anchor_rank}):")
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
        print(f"\nResults ({n:,} sequences):")
        print(f"{'Rank':<8}  {'Top-1':>8}  {f'Top-{k}':>8}")
        print("-" * 30)
        for rank in range(num_ranks):
            print(f"rank {rank:<3}  {top1_acc[rank]:>8.4f}  {topk_acc[rank]:>8.4f}")


if __name__ == "__main__":
    main()
