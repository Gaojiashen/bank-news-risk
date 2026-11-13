"""严格对齐语义数据，仅在训练期拟合PCA，供全部日度模型共享。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA


def load_semantics(path: Path, panel):
    with np.load(path, allow_pickle=False) as d:
        dates = d["dates"].astype(str)
        if not len(dates) or np.any(dates[1:] <= dates[:-1]):
            raise ValueError("Semantic dates must be strictly increasing")
        indices = np.searchsorted(dates, panel["dates"])
        if np.any(indices >= len(dates)) or not np.array_equal(dates[indices], panel["dates"]):
            raise ValueError("Semantic coverage is missing daily model dates")
        if not np.array_equal(d["bank_names"].astype(str), panel["banks"]) or not np.array_equal(d["bank_codes"].astype(str), panel["codes"]):
            raise ValueError("Semantic bank order differs from model inputs")
        raw = d["features"][indices].astype(np.float32)
        counts = d["article_counts"][indices]
        metadata = json.loads(str(d["metadata_json"]))
    if raw.shape != (*panel["risk"].shape, 768) or counts.shape != panel["risk"].shape:
        raise ValueError("Expected semantic features [dates,16,768] and counts [dates,16]")
    if not np.isfinite(raw).all() or not np.issubdtype(counts.dtype, np.integer) or np.any(counts < 0):
        raise ValueError("Invalid semantic values or article counts")
    if np.any(raw[counts == 0] != 0):
        raise ValueError("Missing-news semantic vectors must be zero")
    return raw, counts, metadata


def transform_semantics(raw, counts, state):
    """应用已保存的PCA及训练期得分尺度，并追加有无新闻标记。"""
    mean = np.asarray(state["mean"], dtype=np.float32)
    components = np.asarray(state["components"], dtype=np.float32)
    scale = np.asarray(state["score_scale"], dtype=np.float32)
    compressed = ((raw - mean) @ components.T) / scale
    compressed[counts == 0] = 0
    return np.concatenate((compressed, (counts > 0)[..., None].astype(np.float32)), axis=-1).astype(np.float32)


def fit_semantic_pca(raw, counts, visible, dimensions=16):
    if raw.ndim != 3 or counts.shape != raw.shape[:2] or visible.shape != (len(raw),):
        raise ValueError("PCA requires aligned dates, banks, embeddings and training visibility")
    if not np.isfinite(raw).all() or np.any(counts < 0):
        raise ValueError("PCA inputs must be finite with nonnegative counts")
    selected = visible[:, None] & (counts > 0)
    training = raw[selected]
    if not 1 <= dimensions <= min(training.shape):
        raise ValueError("Too few visible training bank-days for requested PCA dimension")
    pca = PCA(n_components=dimensions, svd_solver="randomized", random_state=42)
    scores = pca.fit_transform(training)
    if not np.isfinite(pca.explained_variance_ratio_).all():
        raise ValueError("Training semantics have no usable variance")
    state = dict(mean=pca.mean_.astype(np.float32), components=pca.components_.astype(np.float32),
                 score_scale=np.maximum(scores.std(axis=0), 1e-6).astype(np.float32),
                 explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
                 fit_bank_days=int(selected.sum()), dimensions=dimensions,
                 has_news_indicator=True)
    return transform_semantics(raw, counts, state), state
