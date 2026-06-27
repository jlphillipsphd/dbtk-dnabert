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
from typing import Dict, List, Optional, Union

from .tokenizers import DnaTokenizer


def _topk_padded(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k indices padded to length k when a rank has fewer than k classes."""
    k_actual = min(k, logits.shape[-1])
    indices = logits.topk(k_actual, dim=-1).indices  # [B, k_actual]
    if k_actual < k:
        pad = indices[:, -1:].expand(-1, k - k_actual)
        indices = torch.cat([indices, pad], dim=-1)
    return indices  # [B, k]


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
        self.log(f"{mode}/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f"{mode}/accuracy", accuracy, prog_bar=True, on_step=True, on_epoch=True)
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

    @property
    def tokenizer(self):
        return self.base.tokenizer

    def save_pretrained(self, *args, **kwargs):
        return self.base.save_pretrained(*args, **kwargs)

@export
class TopDownTaxonomyHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        rank_labels: List[List[str]],
        parent_indices: List[List[int]]
    ):
        super().__init__()
        self.rank_labels = rank_labels
        self.projections = nn.ModuleList([
            nn.Linear(embed_dim, len(labels))
            for labels in rank_labels
        ])
        for rank, indices in enumerate(parent_indices):
            if indices:
                self.register_buffer(
                    f"parent_idx_{rank}",
                    torch.tensor(indices, dtype=torch.long)
                )

    @property
    def num_ranks(self) -> int:
        return len(self.projections)

    def forward(self, embedding: torch.Tensor) -> List[torch.Tensor]:
        logits_list = []
        prev_logits = None
        for rank, projection in enumerate(self.projections):
            logits = projection(embedding)
            if rank > 0:
                parent_idx = getattr(self, f"parent_idx_{rank}")
                logits = logits + prev_logits[:, parent_idx]
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
            parent_indices: Optional[List[List[int]]] = None,
            taxonomy_db_path: Optional[str] = None,
            **kwargs
        ):
            super().__init__(**kwargs)
            self.base = base
            self.base_class = base_class
            self.rank_labels = rank_labels or []
            self.parent_indices = parent_indices or []
            self.taxonomy_db_path = taxonomy_db_path

    config_class = Config
    base_model_prefix = "base"
    sub_models = ["base"]
    base: "DnaBert"

    def __init__(self, config: Optional[Union[Config, dict]] = None):
        super().__init__(config)
        if not self.config.rank_labels and self.config.taxonomy_db_path:
            self._load_taxonomy_from_db()
        self.taxonomy_head = TopDownTaxonomyHead(
            self.base.config.embed_dim,
            self.config.rank_labels,
            self.config.parent_indices
        )

    @property
    def num_ranks(self) -> int:
        return self.taxonomy_head.num_ranks

    @property
    def tokenizer(self):
        return self.base.tokenizer

    def forward(self, kmers: torch.Tensor) -> List[torch.Tensor]:
        return self.taxonomy_head(self.base(kmers)["class"])

    def _step(self, mode: str, batch):
        sequences, taxonomies = batch  # taxonomies: [num_ranks, batch_size]
        logits_list = self(sequences)

        loss = sum(
            F.cross_entropy(logits, taxonomies[rank])
            for rank, logits in enumerate(logits_list)
        )

        self.log(f"{mode}/loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        with torch.no_grad():
            for rank, logits in enumerate(logits_list):
                acc = (logits.argmax(dim=-1) == taxonomies[rank]).float().mean()
                self.log(f"{mode}/rank{rank}_acc", acc, prog_bar=False, on_step=False, on_epoch=True)

        return loss

    def training_step(self, batch):
        return self._step("train", batch)

    def validation_step(self, batch):
        return self._step("val", batch)

    def test_step(self, batch):
        return self._step("test", batch)

    def predict_step(self, batch, batch_idx):
        top_k = getattr(self, '_predict_top_k', 5)
        seq_ids, tokens, true_ids = batch  # true_ids: [B, R]
        logits_list = self(tokens)
        pred_ids = torch.stack([l.argmax(-1) for l in logits_list], dim=1)  # [B, R]
        topk_ids = torch.stack([_topk_padded(l, top_k) for l in logits_list], dim=1)  # [B, R, k]
        return seq_ids, pred_ids.cpu(), true_ids.cpu(), topk_ids.cpu()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def to_embedding_model(self) -> "DnaBertForEmbedding":
        """Return a DnaBertForEmbedding carrying this model's encoder weights."""
        return DnaBertForEmbedding(DnaBertForEmbedding.Config(base=self.base))

    def _load_taxonomy_from_db(self):
        from dnadb import taxonomy as tax_module
        with tax_module.TaxonomyDb(self.config.taxonomy_db_path) as tax_db:
            tree = tax_db.tree
        for rank, taxons in enumerate(tree.taxonomy_id_map):
            self.config.rank_labels.append([t.taxon_label for t in taxons])
            if rank == 0:
                self.config.parent_indices.append([])
            else:
                self.config.parent_indices.append([t.parent.taxon_id for t in taxons])

    @classmethod
    def from_taxonomy_db(
        cls,
        taxonomy_db_path: Union[str, Path],
        base: Optional[Union["DnaBert", BaseModelType["DnaBert"]]] = None,
        **config_kwargs
    ) -> "DnaBertForTaxonomy":
        return cls(cls.Config(base=base, taxonomy_db_path=str(taxonomy_db_path), **config_kwargs))


@export
class NaiveTaxonomyHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        rank_labels: List[List[str]],
        parent_indices: List[List[int]]
    ):
        super().__init__()
        self.rank_labels = rank_labels
        num_leaf_taxa = len(rank_labels[-1])
        self.projection = nn.Linear(embed_dim, num_leaf_taxa)

        # Build leaf→ancestor index buffers for per-rank accuracy
        num_ranks = len(rank_labels)
        ancestors = list(range(num_leaf_taxa))
        self.register_buffer(f"leaf_ancestors_{num_ranks - 1}", torch.tensor(ancestors, dtype=torch.long))
        for rank in range(num_ranks - 2, -1, -1):
            ancestors = [parent_indices[rank + 1][a] for a in ancestors]
            self.register_buffer(f"leaf_ancestors_{rank}", torch.tensor(ancestors, dtype=torch.long))

    @property
    def num_ranks(self) -> int:
        return len(self.rank_labels)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.projection(embedding)

    def ancestor_at_rank(self, predicted_leaves: torch.Tensor, rank: int) -> torch.Tensor:
        return getattr(self, f"leaf_ancestors_{rank}")[predicted_leaves]


@export
class BertaxTaxonomyHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        rank_labels: List[List[str]],
        parent_indices: List[List[int]]  # accepted for API consistency, unused
    ):
        super().__init__()
        self.rank_labels = rank_labels
        taxon_counts = [len(labels) for labels in rank_labels]
        self.projections = nn.ModuleList()
        cumulative = 0
        for count in taxon_counts:
            self.projections.append(nn.Linear(embed_dim + cumulative, count))
            cumulative += count

    @property
    def num_ranks(self) -> int:
        return len(self.projections)

    def forward(self, embedding: torch.Tensor) -> List[torch.Tensor]:
        logits_list = []
        inp = embedding
        for projection in self.projections:
            logits = projection(inp)
            logits_list.append(logits)
            inp = torch.cat([inp, logits], dim=-1)
        return logits_list


@export
class DnaBertForNaiveTaxonomy(DbtkModel):
    class Config(PretrainedConfig):
        is_composition = True
        model_type = "dnabert_for_naive_taxonomy"

        def __init__(
            self,
            base: Optional[BaseModelType["DnaBert"]] = None,
            base_class: Optional[BaseModelClassType["DnaBert"]] = "dnabert.models.DnaBert",
            rank_labels: Optional[List[List[str]]] = None,
            parent_indices: Optional[List[List[int]]] = None,
            taxonomy_db_path: Optional[str] = None,
            **kwargs
        ):
            super().__init__(**kwargs)
            self.base = base
            self.base_class = base_class
            self.rank_labels = rank_labels or []
            self.parent_indices = parent_indices or []
            self.taxonomy_db_path = taxonomy_db_path

    config_class = Config
    base_model_prefix = "base"
    sub_models = ["base"]
    base: "DnaBert"

    def __init__(self, config: Optional[Union[Config, dict]] = None):
        super().__init__(config)
        if not self.config.rank_labels and self.config.taxonomy_db_path:
            self._load_taxonomy_from_db()
        self.taxonomy_head = NaiveTaxonomyHead(
            self.base.config.embed_dim,
            self.config.rank_labels,
            self.config.parent_indices
        )

    @property
    def num_ranks(self) -> int:
        return self.taxonomy_head.num_ranks

    @property
    def tokenizer(self):
        return self.base.tokenizer

    def forward(self, kmers: torch.Tensor) -> torch.Tensor:
        return self.taxonomy_head(self.base(kmers)["class"])

    def _step(self, mode: str, batch):
        sequences, taxonomies = batch  # taxonomies: [num_ranks, batch_size]
        leaf_targets = taxonomies[-1]
        logits = self(sequences)
        loss = F.cross_entropy(logits, leaf_targets)
        self.log(f"{mode}/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        with torch.no_grad():
            predicted_leaves = logits.argmax(dim=-1)
            for rank in range(self.num_ranks):
                pred_at_rank = self.taxonomy_head.ancestor_at_rank(predicted_leaves, rank)
                acc = (pred_at_rank == taxonomies[rank]).float().mean()
                self.log(f"{mode}/rank{rank}_acc", acc, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch):
        return self._step("train", batch)

    def validation_step(self, batch):
        return self._step("val", batch)

    def test_step(self, batch):
        return self._step("test", batch)

    def predict_step(self, batch, batch_idx):
        top_k = getattr(self, '_predict_top_k', 5)
        seq_ids, tokens, true_ids = batch  # true_ids: [B, R]
        logits = self(tokens)              # [B, num_leaf_taxa]
        pred_leaf = logits.argmax(-1)      # [B]
        k = min(top_k, logits.shape[-1])
        top_k_leaves = logits.topk(k, dim=-1).indices  # [B, k]
        pred_ids = torch.stack([           # [B, R]
            self.taxonomy_head.ancestor_at_rank(pred_leaf, r)
            for r in range(self.num_ranks)
        ], dim=1)
        topk_ids = torch.stack([           # [B, R, k]
            torch.stack([
                self.taxonomy_head.ancestor_at_rank(top_k_leaves[:, j], r)
                for j in range(k)
            ], dim=1)
            for r in range(self.num_ranks)
        ], dim=1)
        return seq_ids, pred_ids.cpu(), true_ids.cpu(), topk_ids.cpu()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def to_embedding_model(self) -> "DnaBertForEmbedding":
        """Return a DnaBertForEmbedding carrying this model's encoder weights."""
        return DnaBertForEmbedding(DnaBertForEmbedding.Config(base=self.base))

    def _load_taxonomy_from_db(self):
        from dnadb import taxonomy as tax_module
        with tax_module.TaxonomyDb(self.config.taxonomy_db_path) as tax_db:
            tree = tax_db.tree
        for rank, taxons in enumerate(tree.taxonomy_id_map):
            self.config.rank_labels.append([t.taxon_label for t in taxons])
            if rank == 0:
                self.config.parent_indices.append([])
            else:
                self.config.parent_indices.append([t.parent.taxon_id for t in taxons])

    @classmethod
    def from_taxonomy_db(
        cls,
        taxonomy_db_path: Union[str, Path],
        base: Optional[Union["DnaBert", BaseModelType["DnaBert"]]] = None,
        **config_kwargs
    ) -> "DnaBertForNaiveTaxonomy":
        return cls(cls.Config(base=base, taxonomy_db_path=str(taxonomy_db_path), **config_kwargs))


@export
class DnaBertForBertaxTaxonomy(DbtkModel):
    class Config(PretrainedConfig):
        is_composition = True
        model_type = "dnabert_for_bertax_taxonomy"

        def __init__(
            self,
            base: Optional[BaseModelType["DnaBert"]] = None,
            base_class: Optional[BaseModelClassType["DnaBert"]] = "dnabert.models.DnaBert",
            rank_labels: Optional[List[List[str]]] = None,
            parent_indices: Optional[List[List[int]]] = None,
            taxonomy_db_path: Optional[str] = None,
            **kwargs
        ):
            super().__init__(**kwargs)
            self.base = base
            self.base_class = base_class
            self.rank_labels = rank_labels or []
            self.parent_indices = parent_indices or []
            self.taxonomy_db_path = taxonomy_db_path

    config_class = Config
    base_model_prefix = "base"
    sub_models = ["base"]
    base: "DnaBert"

    def __init__(self, config: Optional[Union[Config, dict]] = None):
        super().__init__(config)
        if not self.config.rank_labels and self.config.taxonomy_db_path:
            self._load_taxonomy_from_db()
        self.taxonomy_head = BertaxTaxonomyHead(
            self.base.config.embed_dim,
            self.config.rank_labels,
            self.config.parent_indices
        )

    @property
    def num_ranks(self) -> int:
        return self.taxonomy_head.num_ranks

    @property
    def tokenizer(self):
        return self.base.tokenizer

    def forward(self, kmers: torch.Tensor) -> List[torch.Tensor]:
        return self.taxonomy_head(self.base(kmers)["class"])

    def _step(self, mode: str, batch):
        sequences, taxonomies = batch  # taxonomies: [num_ranks, batch_size]
        logits_list = self(sequences)
        loss = sum(
            F.cross_entropy(logits, taxonomies[rank])
            for rank, logits in enumerate(logits_list)
        )
        self.log(f"{mode}/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        with torch.no_grad():
            for rank, logits in enumerate(logits_list):
                acc = (logits.argmax(dim=-1) == taxonomies[rank]).float().mean()
                self.log(f"{mode}/rank{rank}_acc", acc, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch):
        return self._step("train", batch)

    def validation_step(self, batch):
        return self._step("val", batch)

    def test_step(self, batch):
        return self._step("test", batch)

    def predict_step(self, batch, batch_idx):
        top_k = getattr(self, '_predict_top_k', 5)
        seq_ids, tokens, true_ids = batch  # true_ids: [B, R]
        logits_list = self(tokens)
        pred_ids = torch.stack([l.argmax(-1) for l in logits_list], dim=1)  # [B, R]
        topk_ids = torch.stack([_topk_padded(l, top_k) for l in logits_list], dim=1)  # [B, R, k]
        return seq_ids, pred_ids.cpu(), true_ids.cpu(), topk_ids.cpu()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def to_embedding_model(self) -> "DnaBertForEmbedding":
        """Return a DnaBertForEmbedding carrying this model's encoder weights."""
        return DnaBertForEmbedding(DnaBertForEmbedding.Config(base=self.base))

    def _load_taxonomy_from_db(self):
        from dnadb import taxonomy as tax_module
        with tax_module.TaxonomyDb(self.config.taxonomy_db_path) as tax_db:
            tree = tax_db.tree
        for rank, taxons in enumerate(tree.taxonomy_id_map):
            self.config.rank_labels.append([t.taxon_label for t in taxons])
            if rank == 0:
                self.config.parent_indices.append([])
            else:
                self.config.parent_indices.append([t.parent.taxon_id for t in taxons])

    @classmethod
    def from_taxonomy_db(
        cls,
        taxonomy_db_path: Union[str, Path],
        base: Optional[Union["DnaBert", BaseModelType["DnaBert"]]] = None,
        **config_kwargs
    ) -> "DnaBertForBertaxTaxonomy":
        return cls(cls.Config(base=base, taxonomy_db_path=str(taxonomy_db_path), **config_kwargs))


@export
class DnaBertForEmbedding(DbtkModel):
    class Config(PretrainedConfig):
        # Enable nesting
        is_composition = True

        # Model configuration
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
