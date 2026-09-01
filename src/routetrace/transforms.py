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


def expert_histogram(X: np.ndarray) -> np.ndarray:
    """[layers, n_experts] selection counts per Expert, summed over tokens.

    An Expert is a (layer, slot) pair, not a slot index: each of the 40 layers
    owns its own 256 experts, which is why colibri accounts for 10,240 of them.
    Slot 151 of layer 0 and slot 151 of layer 20 are different weight matrices,
    so this keeps the layer axis. See :func:`slot_histogram`.
    """
    return binarize(X).sum(axis=0)


def slot_histogram(X: np.ndarray) -> np.ndarray:
    """[n_experts] selection counts per slot index, summed over tokens AND layers.

    This collapses 40 distinct Experts onto each index, so it measures router
    *index* bias, not how busy any expert is. It is almost never the statistic
    you want -- reach for :func:`expert_histogram` unless you specifically mean
    "does the router favour low indices".
    """
    return binarize(X).sum(axis=(0, 1))


def expert_profile(X: np.ndarray) -> np.ndarray:
    """[layers * n_experts] selection frequency per Expert, summing to 1.

    The comparable form: two sets of tokens of different sizes give profiles you
    can take a distance between. Flattened so that each entry is one Expert.
    """
    h = expert_histogram(X).astype(np.float64).ravel()
    total = h.sum()
    return h / total if total else h


def gate_mass(X: np.ndarray) -> np.ndarray:
    """[layers, n_experts] total gate mass per Expert, i.e. the gate-weighted
    counterpart of :func:`expert_histogram`."""
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
