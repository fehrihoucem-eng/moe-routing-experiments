"""Transforms of X.

Every experiment is a function of X: [tokens, layers, n_experts]. These are the
few that came up while validating the pipeline -- they exist to fix the calling
convention (take X, return an array of the same or reduced rank, never mutate)
rather than to be a complete library.
"""

from __future__ import annotations

import numpy as np


def binarize(X: np.ndarray) -> np.ndarray:
    """Selection mask: 1.0 where an expert was routed to, 0.0 elsewhere.

    Drops the gate magnitudes, keeping only *which* experts fired -- the right
    input for co-activation and cache-hit questions, where a low-gate expert
    still costs a full expert load.
    """
    return (X != 0).astype(X.dtype)


def layer_histogram(X: np.ndarray) -> np.ndarray:
    """[layers, n_experts] selection counts summed over tokens."""
    return binarize(X).sum(axis=0)


def expert_histogram(X: np.ndarray) -> np.ndarray:
    """[n_experts] selection counts summed over tokens and layers."""
    return binarize(X).sum(axis=(0, 1))


def gate_mass(X: np.ndarray) -> np.ndarray:
    """[layers, n_experts] total gate mass, i.e. histogram weighted by gate."""
    return X.sum(axis=0)


def topk_mask(X: np.ndarray, k: int) -> np.ndarray:
    """Keep only each (token, layer)'s k largest gates, zeroing the rest.

    ``k`` above the captured top-k is a no-op; below it, this is the "what if the
    router had been narrower" ablation.
    """
    if k >= X.shape[-1]:
        return X.copy()
    idx = np.argpartition(X, -k, axis=-1)[..., -k:]
    out = np.zeros_like(X)
    np.put_along_axis(out, idx, np.take_along_axis(X, idx, axis=-1), axis=-1)
    return out * (X != 0)  # never resurrect an expert the router did not pick


def renormalize(X: np.ndarray) -> np.ndarray:
    """Rescale each (token, layer)'s gates to sum to 1.

    The captured gates already sum to 1; this matters after :func:`topk_mask`,
    which removes mass the model had assigned.
    """
    total = X.sum(axis=-1, keepdims=True)
    return np.divide(X, total, out=np.zeros_like(X), where=total != 0)


def cooccurrence(X: np.ndarray, layer: int) -> np.ndarray:
    """[n_experts, n_experts] co-selection counts within one layer."""
    M = binarize(X)[:, layer, :]
    return M.T @ M


def transition(X: np.ndarray, layer: int, delta: int = 1) -> np.ndarray:
    """[n_experts, n_experts] counts of layer-``layer`` experts co-firing with
    layer-``layer+delta`` experts on the same token -- the cross-layer coupling
    behind colibri's own ``.coli_pairs`` prefetch table."""
    M = binarize(X)
    return M[:, layer, :].T @ M[:, layer + delta, :]
