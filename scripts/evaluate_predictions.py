#!/usr/bin/env python3
"""
Evaluate cached taxonomy predictions produced by predict_taxonomy.py.

Computes top-1 and top-k accuracy per rank from a .pt predictions file and a
taxonomy DB (the ground-truth source).  No model is needed at evaluation time.

The accuracy computation loop runs in parallel across sequences using Python's
multiprocessing module.  Use --num-workers to control the pool size (default: 4,
set to 0 for sequential execution).

Supports anchor-rank mode: rank K is trusted; coarser ranks are resolved by
walking up the stored taxonomy tree rather than using independent predictions.
The --top-k flag may be set to any value <= the K stored in the predictions file,
allowing cheaper re-evaluation (e.g. top-1 through top-3 from a top-5 file).

Usage:
    python evaluate_predictions.py PREDICTIONS_FILE TAXONOMIES_DB [options]

Example:
    python evaluate_predictions.py predictions.pt \\
        $DATASETS_DIR/silva_nr99_filtered_515f_806r_test/taxonomy.tax.db \\
        --anchor-rank 5
    python evaluate_predictions.py predictions.pt taxonomy.tax.db \\
        --top-k 1 --output results.tsv
"""

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rich.progress import track

import torch
from dnadb import taxonomy


# ---------------------------------------------------------------------------
# Tree traversal helpers
# ---------------------------------------------------------------------------

def build_ancestor_map(
    rank_labels: List[List[str]],
    parent_indices: List[List[List[int]]],
    train_id_to_taxon: List[List[str]],
) -> List[Optional[Dict[int, List[int]]]]:
    """
    Build a per-rank mapping from taxon_id -> list of candidate parent taxon_ids.

    parent_map[r][taxon_id] = list of candidate parent taxon_ids at rank r-1.
    parent_map[0] = None (domain has no parent).
    The list has more than one entry only when the same label appears under
    multiple parents (shared taxon name within a rank).

    parent_indices[r][i] is a list of parent alpha IDs for child alpha ID i.
    rank_labels and train_id_to_taxon are both alphabetically ordered, so
    child alpha ID i corresponds directly to rank_labels[r][i].
    """
    parent_map: List[Optional[Dict[int, List[int]]]] = [None]
    for r in range(1, len(rank_labels)):
        mapping: Dict[int, List[int]] = {}
        for child_id, parents in enumerate(parent_indices[r]):
            if parents:
                mapping[child_id] = list(parents)
        parent_map.append(mapping)
    return parent_map


def resolve_anchored(
    pred_taxon_id_k: int,
    anchor_rank: int,
    parent_map: List[Optional[Dict[int, List[int]]]],
    topk_ids_seq,    # List[List[int]]  shape [R][K]
    topk_scores_seq, # List[List[float]] shape [R][K]
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
                int(topk_ids_seq[parent_rank][j]): float(topk_scores_seq[parent_rank][j])
                for j in range(len(topk_ids_seq[parent_rank]))
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


# ---------------------------------------------------------------------------
# Per-sequence computation — module level so multiprocessing can pickle it
# ---------------------------------------------------------------------------

_w_train_id_to_taxon: Optional[List[List[str]]] = None
_w_eval_id_to_taxon:  Optional[List[List[str]]] = None
_w_parent_map:        Optional[List] = None
_w_anchor_rank:       Optional[int] = None
_w_top_k:             Optional[int] = None
_w_num_ranks:         Optional[int] = None
_w_anchoring:         Optional[bool] = None


def _worker_init(train_id_to_taxon, eval_id_to_taxon, parent_map,
                 anchor_rank, top_k, num_ranks, anchoring):
    global _w_train_id_to_taxon, _w_eval_id_to_taxon, _w_parent_map
    global _w_anchor_rank, _w_top_k, _w_num_ranks, _w_anchoring
    _w_train_id_to_taxon = train_id_to_taxon
    _w_eval_id_to_taxon  = eval_id_to_taxon
    _w_parent_map        = parent_map
    _w_anchor_rank       = anchor_rank
    _w_top_k             = top_k
    _w_num_ranks         = num_ranks
    _w_anchoring         = anchoring


def _process_sequence(seq_args: Tuple) -> Tuple:
    """
    Compute accuracy metrics for a single sequence.
    Input args are plain Python lists (converted from tensors by the caller).
    Returns (tp, pp, top1, topk_reachable, inconsistent).
    """
    true_ids_i, pred_ids_i, topk_ids_i, topk_scores_i = seq_args

    num_ranks   = _w_num_ranks
    top_k       = _w_top_k
    anchoring   = _w_anchoring
    anchor_rank = _w_anchor_rank

    def pred_lbl(r: int, tid: int) -> str:
        labels = _w_train_id_to_taxon[r]
        return labels[tid] if 0 <= tid < len(labels) else ""

    def true_lbl(r: int, tid: int) -> str:
        labels = _w_eval_id_to_taxon[r]
        return labels[tid].strip() if 0 <= tid < len(labels) else ""

    tp = [true_lbl(r, true_ids_i[r]) for r in range(num_ranks)]

    if anchoring:
        resolved = resolve_anchored(
            pred_ids_i[anchor_rank], anchor_rank, _w_parent_map,
            topk_ids_i, topk_scores_i,
        )
        pp = [
            pred_lbl(r, resolved[r]) if r in resolved else pred_lbl(r, pred_ids_i[r])
            for r in range(num_ranks)
        ]
    else:
        pp = [pred_lbl(r, pred_ids_i[r]) for r in range(num_ranks)]

    top1 = [pp[r] == tp[r] for r in range(num_ranks)]

    if anchoring:
        anchor_topk_set  = set(topk_ids_i[anchor_rank])
        true_anchor_id   = true_ids_i[anchor_rank]
        anchor_in_topk   = true_anchor_id in anchor_topk_set
        seq_inconsistent = False
        topk_reachable   = []

        for r in range(num_ranks):
            if r < anchor_rank:
                true_r_id = true_ids_i[r]
                reachable_ids: set = set()
                for j in range(top_k):
                    reachable_ids.update(resolve_all_ancestors(
                        topk_ids_i[anchor_rank][j], anchor_rank, r, _w_parent_map,
                    ))
                topk_reachable.append(true_r_id in reachable_ids)
                if anchor_in_topk and not seq_inconsistent:
                    if true_r_id not in resolve_all_ancestors(
                            true_anchor_id, anchor_rank, r, _w_parent_map):
                        seq_inconsistent = True
            else:
                topk_reachable.append(
                    tp[r] in {pred_lbl(r, topk_ids_i[r][j]) for j in range(top_k)}
                )
        return tp, pp, top1, topk_reachable, seq_inconsistent

    topk_reachable = [
        tp[r] in {pred_lbl(r, topk_ids_i[r][j]) for j in range(top_k)}
        for r in range(num_ranks)
    ]
    return tp, pp, top1, topk_reachable, False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="Parallel workers for the accuracy computation loop (default: 4, 0=sequential)"
    )
    cli = parser.parse_args()

    print(f"Loading predictions from {cli.predictions_path}...")
    data = torch.load(cli.predictions_path, map_location="cpu", weights_only=False)

    all_seq_ids       = data['seq_ids']
    all_pred_ids      = data['pred_ids']       # [N, R]
    all_topk_ids      = data['topk_ids']       # [N, R, K_stored]
    all_topk_scores   = data['topk_scores']    # [N, R, K_stored]
    rank_labels       = data['rank_labels']
    parent_indices    = data['parent_indices']
    train_id_to_taxon = data['train_id_to_taxon']
    stored_k          = data['top_k']

    top_k = cli.top_k if cli.top_k is not None else stored_k
    if top_k > stored_k:
        print(f"ERROR: --top-k {top_k} exceeds the K={stored_k} stored in the predictions file")
        sys.exit(1)

    all_topk_ids    = all_topk_ids[:, :, :top_k]
    all_topk_scores = all_topk_scores[:, :, :top_k]

    n, num_ranks = all_pred_ids.shape

    print(f"Loading ground-truth labels from {cli.taxonomies_path}...")
    with taxonomy.TaxonomyDb(str(cli.taxonomies_path)) as tax_db:
        eval_id_to_taxon = list(tax_db.tree.id_to_taxon_map)
        true_ids_list = [
            tax_db[seq_id].taxonomy.taxon_ids
            for seq_id in track(all_seq_ids, description="  Reading labels")
        ]
    all_true_ids = torch.tensor(true_ids_list, dtype=torch.long)  # [N, R]

    anchor_rank = cli.anchor_rank
    anchoring = anchor_rank is not None and anchor_rank > 0

    parent_map = None
    if anchoring:
        parent_map = build_ancestor_map(rank_labels, parent_indices, train_id_to_taxon)

    is_anchored = [(anchoring and r < anchor_rank) for r in range(num_ranks)]

    # Shared initialisation args for workers (or sequential setup)
    init_args = (train_id_to_taxon, eval_id_to_taxon, parent_map,
                 anchor_rank, top_k, num_ranks, anchoring)

    def seq_args_iter():
        for i in range(n):
            yield (
                all_true_ids[i].tolist(),
                all_pred_ids[i].tolist(),
                all_topk_ids[i].tolist(),
                all_topk_scores[i].tolist(),
            )

    true_parts, pred_parts, top1_correct, topk_correct = [], [], [], []
    tree_inconsistencies = 0

    n_workers = cli.num_workers
    if n_workers > 0:
        with Pool(n_workers, initializer=_worker_init, initargs=init_args) as pool:
            result_iter = pool.imap(_process_sequence, seq_args_iter(), chunksize=512)
            for tp, pp, top1, topk, incon in track(
                    result_iter, total=n, description="Computing accuracy"):
                true_parts.append(tp)
                pred_parts.append(pp)
                top1_correct.append(top1)
                topk_correct.append(topk)
                tree_inconsistencies += incon
    else:
        _worker_init(*init_args)
        for seq_arg in track(seq_args_iter(), total=n, description="Computing accuracy"):
            tp, pp, top1, topk, incon = _process_sequence(seq_arg)
            true_parts.append(tp)
            pred_parts.append(pp)
            top1_correct.append(top1)
            topk_correct.append(topk)
            tree_inconsistencies += incon

    top1_acc = [sum(top1_correct[i][r] for i in range(n)) / n for r in range(num_ranks)]
    topk_acc = [sum(topk_correct[i][r] for i in range(n)) / n for r in range(num_ranks)]

    if cli.output or cli.show_predictions:
        out = open(cli.output, "w") if cli.output else sys.stdout

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
