"""Regression tests for ``TDAPipeline``.

The class is meant to be a drop-in upgrade of the functional pipeline that
was validated against the Gidea et al. (2020) Lorenz figure (see
``test_lorenz.py`` and ``tests/output/lorenz_validation.png``). The main
test here re-runs both pipelines on the same Lorenz data and asserts that
``L1_H1`` matches the functional reference to 1e-6.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from tda_oct10.tda_pipeline import (  # noqa: E402
    TDAPipeline,
    landscape_l1_norms,
    sliding_windows,
    takens_embed,
    vietoris_rips_diagrams,
)
from test_lorenz import (  # noqa: E402
    TAKENS_DIM,
    TAKENS_TAU,
    WINDOW_SIZE,
    simulate_lorenz_map,
)


def _reference_l1_h1(x: np.ndarray) -> np.ndarray:
    """L^1 H_1 series produced by the validated functional pipeline."""
    cloud = takens_embed(x, dimension=TAKENS_DIM, time_delay=TAKENS_TAU)
    windows = sliding_windows(cloud, window_size=WINDOW_SIZE, stride=1)
    diagrams = vietoris_rips_diagrams(windows, homology_dimensions=(1,))
    return landscape_l1_norms(diagrams, homology_dimension=1)


def test_tda_pipeline_matches_lorenz_reference_l1_h1():
    x, _ = simulate_lorenz_map()
    ref = _reference_l1_h1(x)

    pipeline = TDAPipeline(
        window_size=WINDOW_SIZE,
        max_edge_length=np.inf,
        homology_dims=[1],
        landscape_layers=5,
        landscape_bins=100,
        n_jobs=-1,
        mode="takens",
        embedding_dim=TAKENS_DIM,
        time_delay=TAKENS_TAU,
    )
    out = pipeline.fit_transform(x)

    assert out["L1_H1"].shape == ref.shape
    np.testing.assert_allclose(out["L1_H1"], ref, atol=1e-6, rtol=0)

    # Sanity: H_0 outputs are NaN-filled since H_0 wasn't requested,
    # H_1 entropy and L^2 are real and finite.
    assert np.all(np.isnan(out["L1_H0"]))
    assert np.all(np.isnan(out["L2_H0"]))
    assert np.all(np.isfinite(out["L2_H1"]))
    assert np.all(np.isfinite(out["persistence_entropy_H1"]))
    assert len(out["diagrams"]) == ref.shape[0]


def test_tda_pipeline_returns_h0_when_requested():
    """With both dims requested, H_0 outputs are real and the same length."""
    x, _ = simulate_lorenz_map()
    pipeline = TDAPipeline(
        window_size=WINDOW_SIZE,
        max_edge_length=2.0,  # ample for the Lorenz x-scale (~0.5)
        homology_dims=[0, 1],
        landscape_layers=5,
        landscape_bins=100,
        n_jobs=-1,
        mode="takens",
        embedding_dim=TAKENS_DIM,
        time_delay=TAKENS_TAU,
    )
    out = pipeline.fit_transform(x)

    n = out["L1_H1"].shape[0]
    for key in ("L1_H0", "L1_H1", "L2_H0", "L2_H1", "persistence_entropy_H1"):
        assert out[key].shape == (n,), f"{key} has wrong shape {out[key].shape}"
        assert np.all(np.isfinite(out[key])), f"{key} contains non-finite values"
