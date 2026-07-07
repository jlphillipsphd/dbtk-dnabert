from dbtk._utils import export
from dbtk.nn.models import BaseModelType, BaseModelClassType, DbtkModel
from dbtk.nn import layers
from deprecated import deprecated
import lightning as L
from pathlib import Path
from transformers import PretrainedConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

from .tokenizers import DnaTokenizer


def _topk_padded(logits: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Top-k indices and scores, padded to length k when a rank has fewer than k classes."""
    k_actual = min(k, logits.shape[-1])
    result = logits.topk(k_actual, dim=-1)
    indices = result.indices  # [B, k_actual]
    values  = result.values   # [B, k_actual]
    if k_actual < k:
        pad_idx = indices[:, -1:].expand(-1, k - k_actual)
        pad_val = torch.full((indices.shape[0], k - k_actual), float('-inf'), device=logits.device)
        indices = torch.cat([indices, pad_idx], dim=-1)
        values  = torch.cat([values,  pad_val],  dim=-1)
    return indices, values  # [B, k], [B, k]


@export
class DnaBert(DbtkModel):
    class Config(PretrainedConfig):
        model_type = "dnabert"

        def __init__(
            self,
            kmer: int = 6,
            kmer_stride: int = 1,
            normalize_sequences: bool = True,
            embed_dim: int = 768,
            num_heads: int = 12,
            num_layers: int = 6,
            feedforward_dim: int = 2048,
            activation: str = "gelu",
            max_length: int = 250,
            **kwargs
        ):
            super().__init__(**kwargs)
            self.kmer = kmer
            self.kmer_stride = kmer_stride
            self.normalize_sequences = normalize_sequences
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.num_layers = num_layers
            self.feedforward_dim = feedforward_dim
            self.activation = activation
            self.max_length = max_length

    config_class = Config

    def __init__(self, config: Optional[Union[Config, dict]] = None):
        super().__init__(config)

        self.tokenizer = DnaTokenizer(
            kmer=self.config.kmer,
            kmer_stride=self.config.kmer_stride,
            normalize_sequences=self.config.normalize_sequences
        )

        if isinstance(self.config.activation, str):
            activation = getattr(F, self.config.activation)
        else:
            activation = self.config.activation

        self.transformer = layers.TransformerEncoder(
            layers.TransformerEncoderBlock(
                mha=layers.RelativeMultiHeadAttention(
                    embed_dim=self.config.embed_dim,
                    num_heads=self.config.num_heads,
                    max_length=self.config.max_length
                ),
                feedforward_dim=self.config.feedforward_dim,
                feedforward_activation=activation
            ),
            num_layers=self.config.num_layers
        )

        self.embeddings = nn.Embedding(
            len(self.tokenizer),
            self.config.embed_dim,
            padding_idx=self.tokenizer.vocab["[PAD]"]
        )

    def forward(
        self,
        kmers: torch.Tensor
    ):
        # Prepend class token and append separator token
        kmers = F.pad(kmers, (1, 0), mode="constant", value=self.tokenizer.vocab["[CLS]"])
        tokens = self.embeddings(kmers)

        # Pass through transformer
        output = self.transformer(tokens)

        # Separate embeddings
        transformed_class_tokens = output[:, 0]
        transformed_kmers = output[:, 1:]

        return {
            "class": transformed_class_tokens,
            "tokens": transformed_kmers
        }

@export
class DnaBertForPretraining(DbtkModel):
    class Config(PretrainedConfig):
        # Enable nesting
        is_composition = True
        model_type = "dnabert_for_pretraining"

        def __init__(
            self,
            base: Optional[BaseModelType[DnaBert]] = None,
            base_class: Optional[BaseModelClassType[DnaBert]] = "dnabert.models.DnaBert",
            min_mask_ratio: float = 0.15,
            max_mask_ratio: float = 0.15,
            **kwargs
        ):
            super().__init__(**kwargs)
            self.base = base
            self.base_class = base_class
            self.min_mask_ratio = min_mask_ratio
            self.max_mask_ratio = max_mask_ratio

    config_class = Config
    base_model_prefix = "base"
    sub_models = ["base"]

    base: DnaBert

    def __init__(self, config: Optional[Union[Config, dict]] = None):
        super().__init__(config)

        # Setup base model
        self.mask_head = nn.Linear(self.base.config.embed_dim, self.base.tokenizer.num_token_ids)

    def _apply_random_masking(self, kmers: torch.Tensor, inplace: bool = False):
        """
        Randomly replace contiguous blocks of kmers with mask tokens.
        """
        # Compute mask regions
        lengths = torch.sum(kmers != 0, dim=1)
        mask_ratios = self.config.min_mask_ratio + torch.rand(lengths.shape[0], device=kmers.device)*(self.config.max_mask_ratio - self.config.min_mask_ratio)
        num_mask_tokens = torch.clamp(torch.round(lengths * mask_ratios).long(), min=1)
        offsets = torch.rand(size=(lengths.shape[0],), device=kmers.device)
        offsets = torch.round(offsets * (lengths - num_mask_tokens)).long()

        # Compute mask
        indices = torch.arange(kmers.shape[-1], device=kmers.device).expand(lengths.shape[0], -1)
        mask = (offsets.unsqueeze(-1) <= indices) & ((offsets+num_mask_tokens).unsqueeze(-1) > indices)

        # Apply masking
        if not inplace:
            kmers = kmers.clone()
        targets = kmers[mask]
        kmers[mask] = self.base.tokenizer.vocab["[MASK]"]
        return kmers, mask, targets

    def forward(self, kmers: torch.Tensor):
        kmers, mask, targets = self._apply_random_masking(kmers, inplace=True)
        output = self.base(kmers)["tokens"]
        masked_output = output[mask]
        return self.mask_head(masked_output), targets

    def _step(self, mode: str, batch: Dict[str, torch.Tensor]):
        kmers = batch["kmers"]
        output, targets = self(kmers)
        loss = F.cross_entropy(output, targets)
        accuracy = (output.argmax(dim=-1) == targets).float().mean()
        self.log(f"{mode}/loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log(f"{mode}/accuracy", accuracy, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def training_step(self, batch):
        return self._step("train", batch)

    def validation_step(self, batch):
        return self._step("val", batch)

    def test_step(self, batch):
        return self._step("test", batch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def to_embedding_model(self) -> "DnaBertForEmbedding":
        """Return a DnaBertForEmbedding carrying this model's encoder weights."""
        return DnaBertForEmbedding(DnaBertForEmbedding.Config(base=self.base))

    @property
    def tokenizer(self):
        return self.base.tokenizer

    def save_pretrained(self, *args, **kwargs):
        return self.base.save_pretrained(*args, **kwargs)

def _load_taxonomy_from_lmdb(
    path: Path,
) -> Tuple[List[List[str]], List[List[List[int]]], List[List[int]]]:
    from dnadb import taxonomy as tax_module
    with tax_module.TaxonomyDb(str(path)) as tax_db:
        tree = tax_db.tree

    num_ranks = tree.depth
    # rank_labels from id_to_taxon_map: unique, alphabetically sorted at every rank
    rank_labels: List[List[str]] = [list(labels) for labels in tree.id_to_taxon_map]
    num_leaf_taxa = len(rank_labels[-1])

    # Collect full paths in alphabetical taxon_id space from each DFS leaf node.
    # Each DFS leaf carries taxon_id (alpha index) and taxon_ids (root→leaf alpha tuple).
    # Shared-name genera yield multiple distinct paths for the same taxon_id.
    leaf_paths_all: List[List[Tuple]] = [[] for _ in range(num_leaf_taxa)]
    for leaf in tree.taxonomy_id_map[-1]:
        path_tuple = leaf.taxon_ids  # (domain_alpha, phylum_alpha, ..., genus_alpha)
        if path_tuple not in leaf_paths_all[leaf.taxon_id]:
            leaf_paths_all[leaf.taxon_id].append(path_tuple)

    # Canonical path per genus: lexicographically smallest (deterministic).
    leaf_paths: List[List[int]] = [list(sorted(paths)[0]) for paths in leaf_paths_all]

    # parent_indices[r][child_alpha_id] = sorted list of parent alpha IDs.
    # Derived from all leaf paths so multi-parent shared-name taxa are fully captured.
    parent_indices: List[List[List[int]]] = [
        [[] for _ in range(len(rank_labels[r]))] for r in range(num_ranks)
    ]
    for paths in leaf_paths_all:
        for path_tuple in paths:
            for r in range(1, num_ranks):
                child_id = path_tuple[r]
                parent_id = path_tuple[r - 1]
                if parent_id not in parent_indices[r][child_id]:
                    parent_indices[r][child_id].append(parent_id)
    for r in range(num_ranks):
        for i in range(len(parent_indices[r])):
            parent_indices[r][i].sort()

    return rank_labels, parent_indices, leaf_paths


def _load_taxonomy_from_greengenes(
    path: Path,
) -> Tuple[List[List[str]], List[List[List[int]]], List[List[int]]]:
    from dnadb import taxonomy as tax_module
    all_labels = sorted(set(e.label for e in tax_module.entries(str(path))))
    parsed = [label.split("; ") for label in all_labels]
    num_ranks = len(parsed[0])

    prefix_to_idx: List[Dict[str, int]] = [dict() for _ in range(num_ranks)]
    for parts in parsed:
        for rank in range(num_ranks):
            prefix = "; ".join(parts[:rank + 1])
            if prefix not in prefix_to_idx[rank]:
                prefix_to_idx[rank][prefix] = len(prefix_to_idx[rank])

    rank_labels: List[List[str]] = []
    parent_indices: List[List[List[int]]] = []
    for rank in range(num_ranks):
        by_idx = sorted(prefix_to_idx[rank].items(), key=lambda kv: kv[1])
        rank_labels.append([label for label, _ in by_idx])
        if rank == 0:
            parent_indices.append([[] for _ in range(len(by_idx))])
        else:
            pindices: List[List[int]] = []
            for label, _ in by_idx:
                parent_prefix = "; ".join(label.split("; ")[:-1])
                pindices.append([prefix_to_idx[rank - 1][parent_prefix]])
            parent_indices.append(pindices)

    # Full path per unique leaf lineage (Greengenes has no shared names).
    leaf_paths: List[List[int]] = [
        [prefix_to_idx[r]["; ".join(parts[:r + 1])] for r in range(num_ranks)]
        for parts in parsed
    ]

    return rank_labels, parent_indices, leaf_paths


@export
def load_taxonomy(
    path: Union[str, Path],
) -> Tuple[List[List[str]], List[List[List[int]]], List[List[int]]]:
    """Load rank_labels, parent_indices, and leaf_paths from a taxonomy file.

    .txt → Greengenes flat-file format; otherwise → LMDB TaxonomyDb.
    Returns (rank_labels, parent_indices, leaf_paths).
      rank_labels[r]       — sorted unique taxon labels at rank r
      parent_indices[r][i] — sorted list of parent alpha IDs for child i at rank r
      leaf_paths[i]        — [domain_alpha, ..., genus_alpha] for leaf genus i
    """
    path = Path(path)
    if path.suffix == ".txt":
        return _load_taxonomy_from_greengenes(path)
    return _load_taxonomy_from_lmdb(path)


@export
class TaxonomyHead(nn.Module):
    """Base class for per-rank taxonomy prediction heads."""

    def __init__(
        self,
        embed_dim: int,
        rank_labels: List[List[str]],
        parent_indices: List[List[List[int]]],
        leaf_paths: List[List[int]],
    ):
        super().__init__()
        if not rank_labels:
            raise ValueError("rank_labels must be non-empty")
        self.rank_labels = rank_labels

    @property
    def num_ranks(self) -> int:
        return len(self.rank_labels)


@export
class TopDownTaxonomyHead(TaxonomyHead):
    def __init__(
        self,
        embed_dim: int,
        rank_labels: List[List[str]],
        parent_indices: List[List[List[int]]],
        leaf_paths: List[List[int]],
    ):
        super().__init__(embed_dim, rank_labels, parent_indices, leaf_paths)
        self.projections = nn.ModuleList([
            nn.Linear(embed_dim, len(labels))
            for labels in rank_labels
        ])
        # Build (child, parent) edge index buffers per rank from parent_indices.
        # parent_indices[r][i] is a sorted list of parent alpha IDs for child i.
        # Unique-parent taxa produce one edge; shared-name taxa produce multiple.
        for rank, edges in enumerate(parent_indices):
            if not edges:
                continue
            child_ids, parent_ids = [], []
            for child_id, parents in enumerate(edges):
                for parent_id in parents:
                    child_ids.append(child_id)
                    parent_ids.append(parent_id)
            if child_ids:
                self.register_buffer(f"edge_child_{rank}", torch.tensor(child_ids, dtype=torch.long))
                self.register_buffer(f"edge_parent_{rank}", torch.tensor(parent_ids, dtype=torch.long))

    def forward(self, embedding: torch.Tensor) -> List[torch.Tensor]:
        logits_list = []
        prev_logits = None
        for rank, projection in enumerate(self.projections):
            logits = projection(embedding)
            if rank > 0 and hasattr(self, f"edge_child_{rank}"):
                edge_child = getattr(self, f"edge_child_{rank}")   # [E]
                edge_parent = getattr(self, f"edge_parent_{rank}")  # [E]
                B = embedding.shape[0]
                # Gather parent logits for every edge: [B, E]
                parent_scores = prev_logits.index_select(1, edge_parent)
                # Scatter-max onto children: for each child take max over its parent edges.
                child_bias = parent_scores.new_full((B, logits.shape[-1]), float('-inf'))
                child_bias.scatter_reduce_(
                    1,
                    edge_child.unsqueeze(0).expand(B, -1),
                    parent_scores,
                    reduce='amax',
                    include_self=True,
                )
                logits = logits + child_bias
            logits_list.append(logits)
            prev_logits = logits
        return logits_list


@export
class DnaBertForTaxonomy(DbtkModel):
    class Config(PretrainedConfig):
        is_composition = True
        model_type = "dnabert_for_taxonomy"

        def __init__(
            self,
            base: Optional[BaseModelType["DnaBert"]] = None,
            base_class: Optional[BaseModelClassType["DnaBert"]] = "dnabert.models.DnaBert",
            rank_labels: Optional[List[List[str]]] = None,
            parent_indices: Optional[List[List[List[int]]]] = None,
            leaf_paths: Optional[List[List[int]]] = None,
            taxonomy_db_path: Optional[str] = None,
            head_type: str = "topdown",
            **kwargs
        ):
            super().__init__(**kwargs)
            self.base = base
            self.base_class = base_class
            self.rank_labels = rank_labels or []
            self.parent_indices = parent_indices or []
            self.leaf_paths = leaf_paths or []
            self.taxonomy_db_path = taxonomy_db_path
            self.head_type = head_type

    config_class = Config
    base_model_prefix = "base"
    sub_models = ["base"]
    base: "DnaBert"

    def __init__(self, config: Optional[Union[Config, dict]] = None):
        super().__init__(config)
        if not self.config.rank_labels and self.config.taxonomy_db_path:
            rank_labels, parent_indices, leaf_paths = load_taxonomy(self.config.taxonomy_db_path)
            self.config.rank_labels = rank_labels
            self.config.parent_indices = parent_indices
            self.config.leaf_paths = leaf_paths
        head_types = {
            "topdown": TopDownTaxonomyHead,
            "naive": NaiveTaxonomyHead,
            "bertax": BertaxTaxonomyHead,
        }
        if self.config.head_type not in head_types:
            raise ValueError(
                f"Unknown head_type '{self.config.head_type}'. Choose from: {list(head_types)}"
            )
        self.taxonomy_head = head_types[self.config.head_type](
            self.base.config.embed_dim,
            self.config.rank_labels,
            self.config.parent_indices,
            self.config.leaf_paths,
        )

    @property
    def num_ranks(self) -> int:
        return self.taxonomy_head.num_ranks

    @property
    def tokenizer(self):
        return self.base.tokenizer

    def forward(self, kmers: torch.Tensor):
        return self.taxonomy_head(self.base(kmers)["class"])

    def _step(self, mode: str, batch):
        sequences, taxonomies = batch  # taxonomies: [num_ranks, batch_size]
        output = self(sequences)
        if isinstance(output, list):
            loss = sum(F.cross_entropy(logits, taxonomies[rank]) for rank, logits in enumerate(output))
            self.log(f"{mode}/loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
            with torch.no_grad():
                for rank, logits in enumerate(output):
                    acc = (logits.argmax(dim=-1) == taxonomies[rank]).float().mean()
                    self.log(f"{mode}/rank{rank}_acc", acc, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        else:
            loss = F.cross_entropy(output, taxonomies[-1])
            self.log(f"{mode}/loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
            with torch.no_grad():
                predicted_leaves = output.argmax(dim=-1)
                # A correctly predicted genus unambiguously determines the organism,
                # so credit all coarser ranks when the leaf is correct — even for
                # shared-name genera whose canonical ancestor path may differ from
                # the true path for some training examples.
                correct_leaf = (predicted_leaves == taxonomies[-1])
                for rank in range(self.num_ranks):
                    pred_at_rank = self.taxonomy_head.ancestor_at_rank(predicted_leaves, rank)
                    acc = (correct_leaf | (pred_at_rank == taxonomies[rank])).float().mean()
                    self.log(f"{mode}/rank{rank}_acc", acc, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def training_step(self, batch):
        return self._step("train", batch)

    def validation_step(self, batch):
        return self._step("val", batch)

    def test_step(self, batch):
        return self._step("test", batch)

    def predict_step(self, batch, batch_idx):
        top_k = getattr(self, '_predict_top_k', 1)

        seq_ids, tokens = batch

        output = self(tokens)
        if isinstance(output, list):
            pred_ids = torch.stack([l.argmax(-1) for l in output], dim=1)  # [B, R]
            topk_results = [_topk_padded(l, top_k) for l in output]
            topk_ids    = torch.stack([r[0] for r in topk_results], dim=1)  # [B, R, k]
            topk_scores = torch.stack([r[1] for r in topk_results], dim=1)  # [B, R, k]
        else:
            pred_leaf = output.argmax(-1)  # [B]
            k = min(top_k, output.shape[-1])
            leaf_topk = output.topk(k, dim=-1)
            top_k_leaves  = leaf_topk.indices  # [B, k]
            top_k_values  = leaf_topk.values   # [B, k]
            pred_ids = torch.stack([
                self.taxonomy_head.ancestor_at_rank(pred_leaf, r)
                for r in range(self.num_ranks)
            ], dim=1)  # [B, R]
            topk_ids = torch.stack([
                torch.stack([
                    self.taxonomy_head.ancestor_at_rank(top_k_leaves[:, j], r)
                    for j in range(k)
                ], dim=1)
                for r in range(self.num_ranks)
            ], dim=1)  # [B, R, k]
            # Broadcast leaf scores across ranks (each ancestor's score = its leaf's score)
            topk_scores = top_k_values.unsqueeze(1).expand(-1, self.num_ranks, -1)  # [B, R, k]

        return seq_ids, pred_ids.cpu(), topk_ids.cpu(), topk_scores.cpu()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def setup(self, stage: str):
        if not self.config.rank_labels:
            return
        num_genera = len(self.config.rank_labels[-1])
        datamodule = getattr(self.trainer, 'datamodule', None)
        datamodule_num_taxa = getattr(datamodule, 'num_taxa', None)
        if datamodule_num_taxa is not None and datamodule_num_taxa != num_genera:
            raise ValueError(
                f"Mapping database has {datamodule_num_taxa} genera but model "
                f"taxonomy has {num_genera} — ensure both were generated from the "
                "same reference taxonomy database"
            )

    def to_embedding_model(self) -> "DnaBertForEmbedding":
        """Return a DnaBertForEmbedding carrying this model's encoder weights."""
        return DnaBertForEmbedding(DnaBertForEmbedding.Config(base=self.base))

    @classmethod
    def from_taxonomy_db(
        cls,
        taxonomy_db_path: Union[str, Path],
        base: Optional[Union["DnaBert", BaseModelType["DnaBert"]]] = None,
        **config_kwargs
    ) -> "DnaBertForTaxonomy":
        return cls(cls.Config(base=base, taxonomy_db_path=str(taxonomy_db_path), **config_kwargs))


@export
class NaiveTaxonomyHead(TaxonomyHead):
    def __init__(
        self,
        embed_dim: int,
        rank_labels: List[List[str]],
        parent_indices: List[List[List[int]]],
        leaf_paths: List[List[int]],
    ):
        super().__init__(embed_dim, rank_labels, parent_indices, leaf_paths)
        num_leaf_taxa = len(rank_labels[-1])
        self.projection = nn.Linear(embed_dim, num_leaf_taxa)

        # leaf_paths[i][r] = alpha taxon_id at rank r for leaf genus i.
        # Register one buffer per rank so ancestor_at_rank is a direct lookup.
        paths_tensor = torch.tensor(leaf_paths, dtype=torch.long)  # [num_leaf_taxa, num_ranks]
        for rank in range(len(rank_labels)):
            self.register_buffer(f"leaf_ancestors_{rank}", paths_tensor[:, rank].clone())

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.projection(embedding)

    def ancestor_at_rank(self, predicted_leaves: torch.Tensor, rank: int) -> torch.Tensor:
        return getattr(self, f"leaf_ancestors_{rank}")[predicted_leaves]


@export
class BertaxTaxonomyHead(TaxonomyHead):
    def __init__(
        self,
        embed_dim: int,
        rank_labels: List[List[str]],
        parent_indices: List[List[List[int]]],
        leaf_paths: List[List[int]],  # accepted for API consistency, unused
    ):
        super().__init__(embed_dim, rank_labels, parent_indices, leaf_paths)
        taxon_counts = [len(labels) for labels in rank_labels]
        self.projections = nn.ModuleList()
        cumulative = 0
        for count in taxon_counts:
            self.projections.append(nn.Linear(embed_dim + cumulative, count))
            cumulative += count

    def forward(self, embedding: torch.Tensor) -> List[torch.Tensor]:
        logits_list = []
        inp = embedding
        for projection in self.projections:
            logits = projection(inp)
            logits_list.append(logits)
            inp = torch.cat([inp, logits], dim=-1)
        return logits_list




@export
class DnaBertForEmbedding(DbtkModel):
    class Config(PretrainedConfig):
        is_composition = True
        model_type = "dnabert_for_embedding"

        base: Optional[BaseModelType[DnaBert]] = None
        base_class: Optional[BaseModelClassType[DnaBert]] = "dnabert.models.DnaBert"

    config_class = Config
    base_model_prefix = "base"
    sub_models = ["base"]

    base: DnaBert

    def __init__(self, config: Optional[Union[Config, dict]] = None):
        super().__init__(config)

    def forward(self, kmers: torch.Tensor):
        return self.base(kmers)["class"]

    @property
    def tokenizer(self):
        return self.base.tokenizer

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        base = DnaBert.from_pretrained(*args, **kwargs)
        return cls(cls.Config(base=base))

    def save_pretrained(self, *args, **kwargs):
        return self.base.save_pretrained(*args, **kwargs)
