"""Guards on the published Coverage result.

The numbers in ``docs/results/predictor-coverage.md``, ``site/predictors.html``
and the README are all quotations of ``predictor-coverage.json``. Regenerating
that file needs the GPU capture, so it is committed rather than rebuilt in CI --
which means nothing else checks it. These tests are that check: they pin the
identities Coverage must satisfy no matter what the Predictors do, so a stale,
truncated or silently-rescaled result file fails here instead of in a claim.

They deliberately assert *invariants*, not values. Pinning 38.9% would make every
honest improvement to a Predictor look like a regression.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

RESULT = Path(__file__).resolve().parents[1] / "docs/results/predictor-coverage.json"

pytestmark = pytest.mark.skipif(not RESULT.exists(), reason="result not generated yet")


@pytest.fixture(scope="module")
def result() -> dict:
    return json.loads(RESULT.read_text())


@pytest.fixture(scope="module")
def names(result) -> list[str]:
    return [p["name"] for p in result["predictors"]]


def test_the_grid_is_the_one_the_adr_describes(result):
    """Layers 1..n-1, and Layer 0 present only in the popularity footnote."""
    assert result["layers"] == list(range(1, 40))
    assert result["top_k"] == 8
    assert len(result["popularity_layer_0"]) == len(result["budgets"])


def test_coverage_cannot_exceed_the_arithmetic_ceiling(result, names):
    """Coverage@K <= K/8. A Budget below the Router's top-k caps Coverage no
    matter how good the Predictor is -- 62.5% at K=5. A result file violating
    this is measuring something other than what it says."""
    top_k = result["top_k"]
    for name in names:
        for layer_row in result["per_layer"][name]:
            for k, cov in zip(result["budgets"], layer_row):
                assert 0.0 <= cov <= min(1.0, k / top_k) + 1e-9, f"{name} K={k} cov={cov}"


def test_coverage_is_monotone_in_the_budget(result, names):
    """A larger Budget names a superset of Slots, so Coverage cannot fall."""
    for name in names:
        for layer_row in result["per_layer"][name]:
            assert np.all(np.diff(layer_row) >= -1e-9), name
        assert np.all(np.diff(result["pooled"][name]) >= -1e-9), name


def test_pooled_is_the_mean_over_layers(result, names):
    """The pooled row is a summary of the per-Layer curves, not a separate
    measurement -- per ADR-0003, per-Layer is the result."""
    for name in names:
        want = np.mean(result["per_layer"][name], axis=0)
        assert np.allclose(result["pooled"][name], want, atol=1e-9), name


def test_chance_is_k_over_n_experts(result):
    for k, c in zip(result["budgets"], result["chance"]):
        assert c == pytest.approx(k / 256)


def test_every_predictor_beats_chance(result, names):
    """Weak, and deliberately so: popularity clears it by only ~3x, and that
    narrowness is itself a finding. A Predictor *below* chance is a bug."""
    for name in names:
        for cov, c in zip(result["pooled"][name], result["chance"]):
            assert cov > c, name


def test_popularity_is_flat_across_layers(result):
    """A is the control. If it ever developed a strong depth profile, the depth
    story told about the other Predictors would be about the Corpus instead."""
    k8 = result["budgets"].index(8)
    per_layer = np.array(result["per_layer"]["popularity"])[:, k8]
    span = float(per_layer.max() - per_layer.min())
    assert span < 0.10, f"popularity spans {span:.3f} across layers"


def test_fold_spread_has_one_row_per_fold(result, names):
    for name in names:
        spread = np.array(result["fold_spread"][name])
        assert spread.shape == (result["n_folds"], len(result["budgets"])), name


def test_categories_are_the_corpus_five(result, names):
    want = {
        "coding",
        "conversational_creative",
        "factual_expository",
        "math_reasoning",
        "structured_extraction",
    }
    for name in names:
        assert set(result["by_category"][name]) == want, name


def test_the_test_split_is_not_in_this_file(result):
    """The result of a choice must not contain the holdout it did not spend.

    A key named for test appearing here means run_predictors.py grew a leak
    (ADR-0003); confirm_on_test.py writes its own file.
    """
    blob = json.dumps(result)
    assert '"test"' not in blob
    assert "n_test" not in blob
    assert result["n_train_prompts"] == 80


def test_the_alpha_infinity_limit_is_recorded_as_such(result):
    """alpha=inf is not JSON-representable and is written as null. A NaN or a
    string here would silently break confirm_on_test.py's parameter handoff."""
    for params in result["tuned"].values():
        if "alpha" in params:
            assert params["alpha"] is None or isinstance(params["alpha"], (int, float))
        if "beta" in params:
            assert 0.0 <= params["beta"] <= 1.0
