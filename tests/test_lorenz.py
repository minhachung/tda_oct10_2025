"""Lorenz-system sanity test.

Reproduces Figure 5(a) of Gidea, Goldsmith, Katz, Roldan & Shmalo (2020),
"Topological recognition of critical transitions in time series of
cryptocurrencies" (Physica A 548).

System (noisy delayed quadratic map):
    x_{t+1}   = y_t
    y_{t+1}   = z_t
    z_{t+1}   = M_1 + B x_t + M_{2,t} y_t - z_t^2 + epsilon * N(0,1)
    M_{2,t+1} = M_{2,t} + delta_M_2

Parameters (from the paper):
    M_1 = 0, B = 0.7, delta_M_2 = 2.8e-5, epsilon = 1e-3,
    M_2 in [0.75, ~0.81], N = 2100 points, x-coordinate only,
    Takens d = 4, tau = 1, sliding window w = 100, Vietoris-Rips H_1.

Expected qualitative behaviour: the per-window L^1-norm of the H_1
persistence landscape stays near zero for M_2 < ~0.78 and rises sharply as
M_2 approaches the bifurcation near 0.81.

Run:
    pytest tests/test_lorenz.py -v
or
    python tests/test_lorenz.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tda_oct10.tda_pipeline import (  # noqa: E402
    landscape_l1_norms,
    sliding_windows,
    takens_embed,
    vietoris_rips_diagrams,
)

# System parameters (Gidea et al. 2020, Fig. 5a).
M_1 = 0.0
B = 0.7
M_2_START = 0.75
DELTA_M_2 = 2.8e-5
NOISE_EPS = 1e-3
N_POINTS = 2100

# TDA parameters.
TAKENS_DIM = 4
TAKENS_TAU = 1
WINDOW_SIZE = 100

OUT_DIR = Path(__file__).parent / "output"
OUT_PNG = OUT_DIR / "lorenz_validation.png"


def simulate_lorenz_map(
    n: int = N_POINTS,
    m_1: float = M_1,
    b: float = B,
    m_2_start: float = M_2_START,
    delta_m_2: float = DELTA_M_2,
    noise_eps: float = NOISE_EPS,
    seed: int = 0,
):
    """Simulate the noisy delayed quadratic map. Returns (x, M_2) arrays of length n."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    y = np.zeros(n)
    z = np.zeros(n)
    m_2 = np.zeros(n)
    # Non-trivial fixed point of the deterministic map at M_2_START is z* = M_2 - 0.3
    # (solving z = 0 + B z + M_2 z - z^2 with B = 0.7). Start near it to skip transients.
    fp = m_2_start - 0.3
    x[0] = y[0] = z[0] = fp
    m_2[0] = m_2_start
    for t in range(n - 1):
        x[t + 1] = y[t]
        y[t + 1] = z[t]
        z[t + 1] = (
            m_1 + b * x[t] + m_2[t] * y[t] - z[t] ** 2 + noise_eps * rng.standard_normal()
        )
        m_2[t + 1] = m_2[t] + delta_m_2
    return x, m_2


def run_pipeline():
    x, m_2 = simulate_lorenz_map()
    cloud = takens_embed(x, dimension=TAKENS_DIM, time_delay=TAKENS_TAU)
    windows = sliding_windows(cloud, window_size=WINDOW_SIZE, stride=1)
    diagrams = vietoris_rips_diagrams(windows, homology_dimensions=(1,))
    l1 = landscape_l1_norms(diagrams, homology_dimension=1)
    # Each window i covers embedded points i..i+W-1; the latest original-series
    # index touched by window i is (i + W - 1) + (d - 1) * tau.
    n_windows = windows.shape[0]
    window_end_idx = np.arange(n_windows) + WINDOW_SIZE - 1 + (TAKENS_DIM - 1) * TAKENS_TAU
    m_2_at_window = m_2[window_end_idx]
    return x, m_2, m_2_at_window, l1


def plot_result(x, m_2, m_2_at_window, l1, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6))

    axes[0].plot(m_2, x, lw=0.5)
    axes[0].set_xlabel(r"$M_2$")
    axes[0].set_ylabel(r"$x_t$")
    axes[0].set_title(r"Noisy delayed quadratic map: $x_t$ vs drifting $M_2$")

    axes[1].plot(m_2_at_window, l1, lw=1.0, color="C3")
    axes[1].axvspan(0.75, 0.78, alpha=0.10, color="green",
                    label=r"expected near-zero ($M_2 < 0.78$)")
    axes[1].set_xlabel(r"$M_2$ at window end")
    axes[1].set_ylabel(r"$\|\lambda\|_1$  (H$_1$ landscape)")
    axes[1].set_title(r"L$^1$-norm of H$_1$ persistence landscape per sliding window")
    axes[1].legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def test_lorenz_landscape_rises_at_bifurcation():
    """L^1 norm is near-zero for small M_2 and rises sharply as M_2 -> 0.81."""
    x, m_2, m_2_at_window, l1 = run_pipeline()
    plot_result(x, m_2, m_2_at_window, l1, OUT_PNG)

    early = l1[m_2_at_window < 0.78]
    late = l1[m_2_at_window > 0.80]
    early_mean = float(early.mean()) if early.size else float("nan")
    late_mean = float(late.mean()) if late.size else float("nan")
    ratio = late_mean / max(early_mean, 1e-12)

    print(f"\nLorenz validation: early L1 mean = {early_mean:.4g}")
    print(f"Lorenz validation: late  L1 mean = {late_mean:.4g}")
    print(f"Lorenz validation: late/early    = {ratio:.2f}")

    assert ratio > 5.0, (
        f"Expected late/early ratio > 5; got {ratio:.2f} "
        f"(early={early_mean:.4g}, late={late_mean:.4g}). "
        f"Plot saved to {OUT_PNG} for inspection."
    )


if __name__ == "__main__":
    x, m_2, m_2_at_window, l1 = run_pipeline()
    plot_result(x, m_2, m_2_at_window, l1, OUT_PNG)
    early = l1[m_2_at_window < 0.78].mean()
    late = l1[m_2_at_window > 0.80].mean()
    print(f"Saved plot to {OUT_PNG}")
    print(f"Mean L^1 for M_2 < 0.78: {early:.4g}")
    print(f"Mean L^1 for M_2 > 0.80: {late:.4g}")
    print(f"Ratio late/early: {late / max(early, 1e-12):.2f}")
