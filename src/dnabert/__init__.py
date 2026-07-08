import importlib.metadata
from transformers import AutoConfig
from .models import *
from .nb_models import NaiveBayesForTaxonomy
from .classifiers import DnaBertClassifier

AutoConfig.register("dnabert", DnaBert.Config)
AutoConfig.register("dnabert_for_pretraining", DnaBertForPretraining.Config)
AutoConfig.register("dnabert_for_taxonomy", DnaBertForTaxonomy.Config)
AutoConfig.register("dnabert_for_embedding", DnaBertForEmbedding.Config)
AutoConfig.register("naive_bayes_for_taxonomy", NaiveBayesForTaxonomy.Config)

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
    "NaiveBayesForTaxonomy",
    "DnaBertClassifier",
]