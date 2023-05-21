# -*- coding: utf-8 -*-

from transformers import (
    BertTokenizer,
    BertConfig,
    T5Tokenizer,
    T5Config,
)

from models.ftbert import FTBertCls, FTBertNer
from models.kdbert import KDBertCls, KDBertNer
from models.pkdbert import PKDBertCls
from models.sparsebert import SparseBertCls, SparseBertNer

from models.ftenct5 import FTEncT5Cls, FTEncT5Ner
from models.sparseenct5 import SparseEncT5Cls, SparseEncT5Ner

def get_model_class(model_type):
    if model_type == "ftbert_cls":
        tokenizer_class = BertTokenizer
        config_class = BertConfig
        model_class = FTBertCls
    elif model_type == "ftbert_ner":
        tokenizer_class = BertTokenizer
        config_class = BertConfig
        model_class = FTBertNer
    elif model_type == "kdbert_cls":
        tokenizer_class = BertTokenizer
        config_class = BertConfig
        model_class = KDBertCls
    elif model_type == "kdbert_ner":
        tokenizer_class = BertTokenizer
        config_class = BertConfig
        model_class = KDBertNer
    elif model_type == "pkdbert_cls":
        tokenizer_class = BertTokenizer
        config_class = BertConfig
        model_class = PKDBertCls
    elif model_type == "sparsebert_cls":
        tokenizer_class = BertTokenizer
        config_class = BertConfig
        model_class = SparseBertCls
    elif model_type == "sparsebert_ner":
        tokenizer_class = BertTokenizer
        config_class = BertConfig
        model_class = SparseBertNer
    elif model_type == "ftenct5_cls":
        tokenizer_class = T5Tokenizer
        config_class = T5Config
        model_class = FTEncT5Cls
    elif model_type == "ftenct5_ner":
        tokenizer_class = T5Tokenizer
        config_class = T5Config
        model_class = FTEncT5Ner
    elif model_type == "sparseenct5_cls":
        tokenizer_class = T5Tokenizer
        config_class = T5Config
        model_class = SparseEncT5Cls
    elif model_type == "sparseenct5_ner":
        tokenizer_class = T5Tokenizer
        config_class = T5Config
        model_class = SparseEncT5Ner
    else:
        raise KeyError(f"Unknown model type {model_type}.")

    return tokenizer_class, config_class, model_class
