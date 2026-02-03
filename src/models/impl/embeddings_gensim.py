from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class GensimEmbeddingConfig:
    method: str  # "word2vec" | "fasttext"
    dim: int = 100
    window: int = 5
    min_count: int = 2
    workers: int = 4
    epochs: int = 10
    seed: int = 42
    sg: int = 1  # 1=skip-gram, 0=cbow


def _tokenize_texts(texts: List[str]) -> List[List[str]]:
    # Cohérent avec ton build_vocab/encode_text: split whitespace
    return [t.split() for t in texts]


def train_gensim_model(texts: List[str], cfg: GensimEmbeddingConfig):
    """
    Entraîne un modèle gensim Word2Vec ou FastText sur les textes tokenisés.
    Retourne le modèle gensim (Word2Vec/FastText).
    """
    tokens = _tokenize_texts(texts)

    if cfg.method == "word2vec":
        from gensim.models import Word2Vec

        model = Word2Vec(
            sentences=tokens,
            vector_size=cfg.dim,
            window=cfg.window,
            min_count=cfg.min_count,
            workers=cfg.workers,
            sg=cfg.sg,
            seed=cfg.seed,
            epochs=cfg.epochs,
        )
        return model

    if cfg.method == "fasttext":
        from gensim.models import FastText

        model = FastText(
            sentences=tokens,
            vector_size=cfg.dim,
            window=cfg.window,
            min_count=cfg.min_count,
            workers=cfg.workers,
            sg=cfg.sg,
            seed=cfg.seed,
            epochs=cfg.epochs,
        )
        return model

    raise ValueError(f"Unknown method: {cfg.method}")


def build_embedding_matrix(
    texts: List[str],
    vocab,
    method: str,
    dim: int,
    seed: int = 42,
    window: int = 5,
    min_count: int = 2,
    workers: int = 4,
    epochs: int = 10,
    sg: int = 1,
) -> np.ndarray:
    """
    Construit une matrice (vocab_size, dim) alignée sur vocab.itos :
    - <PAD> (pad_id) -> vecteur 0
    - tokens présents dans le modèle -> vecteur appris
    - tokens absents -> petit random normal
    """
    cfg = GensimEmbeddingConfig(
        method=method,
        dim=dim,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
        seed=seed,
        sg=sg,
    )
    model = train_gensim_model(texts, cfg)

    # gensim expose les vecteurs via model.wv
    wv = model.wv

    rng = np.random.default_rng(seed)
    matrix = rng.normal(loc=0.0, scale=0.05, size=(len(vocab.itos), dim)).astype(np.float32)

    # PAD = zeros
    matrix[vocab.pad_id] = 0.0

    # UNK: on laisse random (ça marche)
    # Remplissage pour les mots connus
    for i, tok in enumerate(vocab.itos):
        if i in (vocab.pad_id, vocab.unk_id):
            continue
        if tok in wv:
            matrix[i] = wv[tok].astype(np.float32)

    return matrix
