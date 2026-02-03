from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable, List

import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer


_URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
_USER_RE = re.compile(r"@\w+")
_NUM_RE = re.compile(r"\b\d+\b")


@lru_cache(maxsize=1)
def _stemmer():
    return PorterStemmer()


@lru_cache(maxsize=1)
def _lemmatizer():
    return WordNetLemmatizer()


def _basic_clean(text: str) -> str:
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _USER_RE.sub(" ", text)
    text = _NUM_RE.sub(" ", text)
    # garde lettres + apostrophes, remplace le reste par espace
    text = re.sub(r"[^a-z'\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _to_wordnet_pos(treebank_tag: str):
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    if treebank_tag.startswith("N"):
        return wordnet.NOUN
    if treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def preprocess_one(text: str, mode: str = "clean") -> str:
    if mode not in {"clean", "stem", "lemma"}:
        raise ValueError(f"Unknown mode: {mode}")

    cleaned = _basic_clean(text)

    if mode == "clean":
        return cleaned

    tokens = word_tokenize(cleaned)

    if mode == "stem":
        st = _stemmer()
        tokens = [st.stem(t) for t in tokens]
        return " ".join(tokens)

    # lemma
    tags = pos_tag(tokens)
    lem = _lemmatizer()
    tokens = [lem.lemmatize(w, _to_wordnet_pos(tag)) for w, tag in tags]
    return " ".join(tokens)


def preprocess_texts(texts: Iterable[str], mode: str = "clean") -> List[str]:
    return [preprocess_one(t, mode=mode) for t in texts]
