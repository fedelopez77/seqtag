"""This file is a wrapper to encapsulate the choice of an embedding (language) model"""
from transformers import AutoModel


def get_embedding_model(lang_model_name):
    return AutoModel.from_pretrained(lang_model_name)
