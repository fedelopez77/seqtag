"""This file is a wrapper to encapsulate the choice of a tokenizer"""
from transformers import AutoTokenizer


def get_tokenizer(lang_model_name):
    return AutoTokenizer.from_pretrained(lang_model_name)
