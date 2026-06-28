"""
Genus-level taxonomy classifier adapter for reference mapping generation.

Wraps a fine-tuned DnaBertForTaxonomy model and exposes a uniform interface:
  - model_name: str  — derived from model config, used in output filenames
  - predict_genus(fasta_db, device) -> np.ndarray[N, int64]

The TaxonomyClassifier Protocol lets the mapping pipeline accept any classifier
without modifying pipeline code — only the YAML config changes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as F
from dnadb import fasta
from tqdm import tqdm

GENUS_RANK = -1  # last rank = genus in the 6-rank SILVA taxonomy


class TaxonomyClassifier(Protocol):
    model_name: str

    def predict_genus(
        self,
        fasta_db: fasta.FastaDb,
        device: torch.device,
    ) -> np.ndarray:
        """Return genus-level taxon IDs, shape [len(fasta_db)], dtype int64."""
        ...


def _classify(
    model,
    fasta_db: fasta.FastaDb,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Batch inference for DnaBertForTaxonomy. Handles both list and single-tensor output."""
    tokenizer = model.tokenizer
    max_length = model.base.config.max_length
    pad_id = tokenizer.vocab["[PAD]"]
    kmer, stride = tokenizer.kmer, tokenizer.kmer_stride
    max_bp = max_length * stride - stride + kmer

    n = len(fasta_db)
    all_preds = np.empty(n, dtype=np.int64)

    model.eval()
    with torch.no_grad():
        for start in tqdm(range(0, n, batch_size), desc="  Classifying sequences", leave=False):
            end = min(start + batch_size, n)
            tokens_list = []
            for i in range(start, end):
                seq = fasta_db.entry(i).sequence
                t = torch.tensor(tokenizer(seq[:max_bp]))
                t = F.pad(t, (0, max_length - len(t)), value=pad_id)
                tokens_list.append(t)
            tokens = torch.stack(tokens_list).to(device)
            output = model(tokens)
            genus_logits = output[GENUS_RANK] if isinstance(output, list) else output
            all_preds[start:end] = genus_logits.argmax(-1).cpu().numpy()

    return all_preds


class DnaBertClassifier:
    """
    Genus-level taxonomy classifier backed by a DnaBertForTaxonomy checkpoint.
    The head type (topdown / naive / bertax) is read from the model's saved config.
    Pass head_type explicitly for checkpoints saved before head_type was introduced.
    """

    def __init__(
        self,
        model_path: str | Path,
        batch_size: int = 256,
        head_type: str | None = None,
    ):
        self._model_path = Path(model_path)
        self.batch_size = batch_size
        self._head_type = head_type
        self._model = None
        self._device = None

    @property
    def model_name(self) -> str:
        if self._head_type is not None:
            return self._head_type
        if self._model is not None:
            return self._model.config.head_type
        config_file = self._model_path / "config.json"
        with open(config_file) as f:
            return json.load(f).get("head_type", "topdown")

    def _load(self, device: torch.device) -> None:
        if self._model is None or self._device != device:
            from dnabert.models import DnaBertForTaxonomy
            self._model = DnaBertForTaxonomy.from_pretrained(self._model_path).to(device)
            if self._head_type is not None and self._model.config.head_type != self._head_type:
                self._model.config.head_type = self._head_type
            self._device = device

    def predict_genus(self, fasta_db: fasta.FastaDb, device: torch.device) -> np.ndarray:
        self._load(device)
        return _classify(self._model, fasta_db, self.batch_size, device)
