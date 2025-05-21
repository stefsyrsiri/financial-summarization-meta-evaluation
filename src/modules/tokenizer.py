"""Tokenizer module for different languages using spaCy.
This module provides a Tokenizer class that loads spaCy models for
English, Greek, and Spanish. It allows tokenization and sentence segmentation."""

import spacy


class Tokenizer:
    _LOADED_MODELS = {}

    def __init__(self, lang_code):
        models = {
            "en": "en_core_web_sm",
            "el": "el_core_news_sm",
            "es": "es_core_news_sm"
        }

        if lang_code not in models:
            raise ValueError(f"No spaCy model available for language: {lang_code}")

        if lang_code not in Tokenizer._LOADED_MODELS:
            Tokenizer._LOADED_MODELS[lang_code] = spacy.load(models[lang_code])

        self.nlp = Tokenizer._LOADED_MODELS[lang_code]

    def tokenize(self, text):
        return [token.text for token in self.nlp(text)]

    def sentencize(self, text):
        return [sent.text for sent in self.nlp(text).sents]
