"""
Genus-level taxonomy classifier config holder for reference mapping generation.

DnaBertClassifier is instantiated from the generate_mappings YAML config and
exposes the model path, batch size, and a human-readable model name used in
output filenames. Actual inference is performed by GenerateMappings via
Lightning Trainer + DnaBertFastaPredictDataModule.
"""

from __future__ import annotations

import json
from pathlib import Path


class DnaBertClassifier:
    """
    Config holder for a DnaBertForTaxonomy checkpoint used in mapping generation.
    Pass head_type explicitly for checkpoints exported before head_type was added
    to the model config.
    """

    def __init__(
        self,
        model_path: str | Path,
        batch_size: int = 256,
        name: str | None = None,
    ):
        self._model_path = Path(model_path)
        self.batch_size = batch_size
        self._name = name

    @property
    def model_path(self) -> Path:
        return self._model_path

    @property
    def model_name(self) -> str:
        if self._name is not None:
            return self._name
        config_file = self._model_path / "config.json"
        with open(config_file) as f:
            return json.load(f).get("head_type", "topdown")
