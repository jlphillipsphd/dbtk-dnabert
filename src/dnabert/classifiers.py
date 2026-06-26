"""
Genus-level taxonomy classifier adapters for reference mapping generation.

Each adapter wraps a fine-tuned DNABERT model and exposes a uniform interface:
  - model_name: str  — used in output filenames
  - predict_genus(fasta_db, device) -> np.ndarray[N, int64]

The TaxonomyClassifier Protocol lets the mapping pipeline be swapped to any
classifier without modifying pipeline code — only the YAML config changes.
"""

from __future__ import annotations

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


def _classify_list_output(
    model,
    fasta_db: fasta.FastaDb,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """
    Batch inference for models whose forward() returns List[Tensor] (one per rank).
    Extracts genus predictions from the last rank (GENUS_RANK = -1).
    Used by TopDown and BERTax adapters.
    """
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
            logits_list = model(tokens)
            all_preds[start:end] = logits_list[GENUS_RANK].argmax(-1).cpu().numpy()

    return all_preds


def _classify_single_output(
    model,
    fasta_db: fasta.FastaDb,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """
    Batch inference for models whose forward() returns a single Tensor [B, num_leaf_taxa].
    The leaf prediction IS the genus prediction for flat/naive classifiers.
    """
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
            logits = model(tokens)
            all_preds[start:end] = logits.argmax(-1).cpu().numpy()

    return all_preds


class DnaBertTopDownClassifier:
    """
    Genus-level classifier using the DNABERT TopDown hierarchical taxonomy head.
    Genus is the deepest rank (last element of the per-rank logits list).
    """

    model_name = "topdown"

    def __init__(self, model_path: str | Path, batch_size: int = 256):
        self._model_path = Path(model_path)
        self.batch_size = batch_size
        self._model = None
        self._device = None

    def _load(self, device: torch.device) -> None:
        if self._model is None or self._device != device:
            from dnabert.models import DnaBertForTaxonomy
            self._model = DnaBertForTaxonomy.from_pretrained(self._model_path).to(device)
            self._device = device

    def predict_genus(self, fasta_db: fasta.FastaDb, device: torch.device) -> np.ndarray:
        self._load(device)
        return _classify_list_output(self._model, fasta_db, self.batch_size, device)


class DnaBertBertaxClassifier:
    """
    Genus-level classifier using the DNABERT BERTax independent per-rank taxonomy head.
    Genus is the deepest rank (last element of the per-rank logits list).
    """

    model_name = "bertax"

    def __init__(self, model_path: str | Path, batch_size: int = 256):
        self._model_path = Path(model_path)
        self.batch_size = batch_size
        self._model = None
        self._device = None

    def _load(self, device: torch.device) -> None:
        if self._model is None or self._device != device:
            from dnabert.models import DnaBertForBertaxTaxonomy
            self._model = DnaBertForBertaxTaxonomy.from_pretrained(self._model_path).to(device)
            self._device = device

    def predict_genus(self, fasta_db: fasta.FastaDb, device: torch.device) -> np.ndarray:
        self._load(device)
        return _classify_list_output(self._model, fasta_db, self.batch_size, device)


class DnaBertNaiveClassifier:
    """
    Genus-level classifier using the DNABERT Naive flat single-head taxonomy classifier.
    The model predicts leaf taxa (genera) directly, so argmax on the output IS the genus.
    """

    model_name = "naive"

    def __init__(self, model_path: str | Path, batch_size: int = 256):
        self._model_path = Path(model_path)
        self.batch_size = batch_size
        self._model = None
        self._device = None

    def _load(self, device: torch.device) -> None:
        if self._model is None or self._device != device:
            from dnabert.models import DnaBertForNaiveTaxonomy
            self._model = DnaBertForNaiveTaxonomy.from_pretrained(self._model_path).to(device)
            self._device = device

    def predict_genus(self, fasta_db: fasta.FastaDb, device: torch.device) -> np.ndarray:
        self._load(device)
        return _classify_single_output(self._model, fasta_db, self.batch_size, device)
