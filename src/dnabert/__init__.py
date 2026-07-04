import importlib.metadata
from transformers import AutoConfig
from .models import *
from .classifiers import DnaBertClassifier

AutoConfig.register("dnabert", DnaBert.Config)
AutoConfig.register("dnabert_for_pretraining", DnaBertForPretraining.Config)
AutoConfig.register("dnabert_for_taxonomy", DnaBertForTaxonomy.Config)
AutoConfig.register("dnabert_for_embedding", DnaBertForEmbedding.Config)

__version__ = importlib.metadata.version("dbtk-dnabert")

__all__ = [
    "DnaBert",
    "DnaBertForPretraining",
    "load_taxonomy",
    "TaxonomyHead",
    "TopDownTaxonomyHead",
    "NaiveTaxonomyHead",
    "BertaxTaxonomyHead",
    "DnaBertForTaxonomy",
    "DnaBertForEmbedding",
    "DnaBertClassifier",
]