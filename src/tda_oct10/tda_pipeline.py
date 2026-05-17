"""TDA pipeline for windowed persistence-landscape features.

This module exposes two layers:

1. Small functional helpers (``takens_embed``, ``sliding_windows``,
   ``vietoris_rips_diagrams``, ``landscape_l1_norms``). These are the
   originals used by the validated Lorenz regression test in
   ``tests/test_lorenz.py`` and are kept for backward compatibility.

2. ``TDAPipeline``: a configurable end-to-end pipeline that turns a returns
   series into per-window L^1 / L^2 landscape norms and persistence-entropy
   features for H_0 and H_1, with optional on-disk caching of the
   (expensive) persistence diagrams.

Two embedding modes are supported:

- ``mode='multivariate'``: returns of shape ``(T, d)``. Each sliding window
  of length ``window_size`` is treated directly as a point cloud in R^d
  before Vietoris-Rips. (Ismail et al. 2020 style.)
- ``mode='takens'``: univariate returns of shape ``(T,)``. The series is
  delay-embedded into R^embedding_dim via ``SingleTakensEmbedding``, then
  sliding windows give point clouds in R^embedding_dim. (Gidea et al.
  2020 style.)

H_0 is computed in addition to H_1 because the Oct 10 2025 liquidation
cascade is hypothesised to show up as a collapse of connected components
(H_0 destabilisation), not just an H_1 loop signature. The TDA literature
on critical transitions typically reports only H_1.
"""

from __future__ import annotations

import hashlib
import pickle
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
from gtda.diagrams import PersistenceEntropy, PersistenceLandscape
from gtda.homology import VietorisRipsPersistence
from gtda.time_series import SingleTakensEmbedding, SlidingWindow


# ---------------------------------------------------------------------------
# Functional helpers (kept for backward compatibility with test_lorenz.py).
# ---------------------------------------------------------------------------


def takens_embed(series, dimension: int = 4, time_delay: int = 1) -> np.ndarray:
    """Delay-embed a 1-D series into R^dimension.

    Output shape: ``(len(series) - (dimension - 1) * time_delay, dimension)``.
    """
    series = np.asarray(series).reshape(-1)
    embedder = SingleTakensEmbedding(
        parameters_type="fixed",
        time_delay=time_delay,
        dimension=dimension,
    )
    return embedder.fit_transform(series)


def sliding_windows(point_cloud: np.ndarray, window_size: int = 100, stride: int = 1) -> np.ndarray:
    """Sliding windows over a point cloud of shape ``(n_points, n_features)``.

    Output shape: ``(n_windows, window_size, n_features)``.
    """
    sw = SlidingWindow(size=window_size, stride=stride)
    return sw.fit_transform(point_cloud)


def vietoris_rips_diagrams(
    windows: np.ndarray,
    homology_dimensions=(1,),
    n_jobs: int = -1,
    max_edge_length: float = np.inf,
) -> np.ndarray:
    """Vietoris-Rips persistence diagrams for each window.

    Input shape: ``(n_windows, window_size, n_features)``.
    Output shape: ``(n_windows, n_points, 3)``; the last axis is
    ``(birth, death, homology_dimension)``.
    """
    vr = VietorisRipsPersistence(
        homology_dimensions=list(homology_dimensions),
        max_edge_length=max_edge_length,
        n_jobs=n_jobs,
    )
    return vr.fit_transform(windows)


def landscape_l1_norms(
    diagrams: np.ndarray,
    homology_dimension: int = 1,
    n_layers: int = 5,
    n_bins: int = 100,
) -> np.ndarray:
    """L^1 norm of the persistence landscape for one homology dim, per window.

    The landscape is fit jointly on the batch so the sampling grid is
    consistent across windows; the L^1 norm sums ``|landscape|`` over
    (layers, bins) and rescales by the bin width to approximate the
    integral. Returns a 1-D array of length ``n_windows``.
    """
    landscape = PersistenceLandscape(n_layers=n_layers, n_bins=n_bins)
    grids = landscape.fit_transform(diagrams)
    per_window = _slice_landscape(grids, landscape, homology_dimension, n_layers)
    samplings = landscape.samplings_[homology_dimension]
    bin_width = float(samplings[1] - samplings[0]) if len(samplings) > 1 else 1.0
    return np.sum(np.abs(per_window), axis=(1, 2)) * bin_width


def _slice_landscape(
    grids: np.ndarray,
    landscape: PersistenceLandscape,
    homology_dimension: int,
    n_layers: int,
) -> np.ndarray:
    """Return the ``(n_windows, n_layers, n_bins)`` slice for one H-dim.

    giotto-tda 0.6's ``PersistenceLandscape.fit_transform`` returns a 3-D
    array of shape ``(n_windows, n_homology_dims * n_layers, n_bins)`` —
    the layers for each homology dim are stacked along axis 1 in the
    order of ``landscape.homology_dimensions_``.
    """
    if grids.ndim != 3:
        raise RuntimeError(f"Unexpected landscape grid shape: {grids.shape}")
    dims_present = list(landscape.homology_dimensions_)
    if homology_dimension not in dims_present:
        raise ValueError(
            f"Requested homology_dimension={homology_dimension} not present "
            f"(found {dims_present})."
        )
    h_idx = dims_present.index(homology_dimension)
    start = h_idx * n_layers
    stop = start + n_layers
    return grids[:, start:stop, :]


# ---------------------------------------------------------------------------
# TDAPipeline class.
# ---------------------------------------------------------------------------


# Reasonable upper bound on how long the diagram computation can take before
# we start caching results to disk on the next call.
_DEFAULT_CACHE_THRESHOLD_SECONDS = 60.0

_VALID_MODES = ("multivariate", "takens")


class TDAPipeline:
    """Sliding-window TDA: point cloud → persistence diagram → landscape norms.

    Parameters
    ----------
    window_size :
        Number of consecutive points in each sliding window (the size of
        the point cloud passed to Vietoris-Rips).
    max_edge_length :
        Largest edge length the VR filtration considers. The default
        (``0.10``) is tuned for *raw* daily multi-crypto log-returns
        (typical intra-window pairwise distance ~0.1; see the
        ``sweep_max_edge_length`` study in ``validation.py`` for the
        evidence). The Ismail-style default ``0.05`` only works after
        per-window normalisation. For arbitrarily scaled inputs (e.g.
        Lorenz), pass ``np.inf`` or a value large enough to span the
        cloud diameter.
    homology_dims :
        Homology dimensions to compute. Must include ``1``; ``0`` is
        included by default so the dict result has H_0 features too.
    landscape_layers :
        Number of landscape layers ``lambda_k`` to keep.
    landscape_bins :
        Number of samples per layer used to discretise the integral.
    n_jobs :
        Passed through to each giotto-tda transformer. ``-1`` uses all cores.
    mode :
        ``'multivariate'`` for ``(T, d)`` input passed straight to sliding
        windows, or ``'takens'`` for ``(T,)`` input that is first delay-
        embedded into R^embedding_dim.
    embedding_dim, time_delay :
        Only used when ``mode='takens'``.
    cache_dir :
        If set, persistence diagrams are pickled here when computation
        exceeds ``cache_threshold_seconds``. On subsequent calls with the
        same input + params, the diagrams are loaded from disk instead of
        recomputed.
    cache_threshold_seconds :
        Minimum elapsed seconds of diagram computation before the result
        is written to ``cache_dir``.
    """

    def __init__(
        self,
        window_size: int = 50,
        max_edge_length: float = 0.10,
        homology_dims: Optional[list[int]] = None,
        landscape_layers: int = 5,
        landscape_bins: int = 100,
        n_jobs: int = -1,
        mode: str = "multivariate",
        embedding_dim: int = 4,
        time_delay: int = 1,
        cache_dir: Optional[Union[str, Path]] = None,
        cache_threshold_seconds: float = _DEFAULT_CACHE_THRESHOLD_SECONDS,
    ):
        if mode not in _VALID_MODES:
            raise ValueError(
                f"mode must be one of {_VALID_MODES}, got {mode!r}"
            )
        if homology_dims is None:
            homology_dims = [0, 1]
        homology_dims = sorted(int(d) for d in homology_dims)
        if 1 not in homology_dims:
            raise ValueError(
                "homology_dims must include 1 (the dict result requires H_1)."
            )
        if window_size < 2:
            raise ValueError(f"window_size must be >= 2, got {window_size}")
        if landscape_layers < 1:
            raise ValueError(f"landscape_layers must be >= 1, got {landscape_layers}")
        if landscape_bins < 2:
            raise ValueError(f"landscape_bins must be >= 2, got {landscape_bins}")
        if mode == "takens" and embedding_dim < 1:
            raise ValueError(f"embedding_dim must be >= 1, got {embedding_dim}")
        if mode == "takens" and time_delay < 1:
            raise ValueError(f"time_delay must be >= 1, got {time_delay}")

        self.window_size = window_size
        self.max_edge_length = max_edge_length
        self.homology_dims = homology_dims
        self.landscape_layers = landscape_layers
        self.landscape_bins = landscape_bins
        self.n_jobs = n_jobs
        self.mode = mode
        self.embedding_dim = embedding_dim
        self.time_delay = time_delay
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.cache_threshold_seconds = cache_threshold_seconds

    # ------- internals --------------------------------------------------

    def _build_windows(self, returns: np.ndarray) -> np.ndarray:
        """Return point-cloud windows of shape ``(n_windows, W, d_embed)``."""
        if self.mode == "takens":
            series = returns.reshape(-1)
            embedder = SingleTakensEmbedding(
                parameters_type="fixed",
                time_delay=self.time_delay,
                dimension=self.embedding_dim,
            )
            cloud = embedder.fit_transform(series)
        else:  # multivariate
            cloud = returns
            if cloud.ndim == 1:
                cloud = cloud.reshape(-1, 1)
            elif cloud.ndim != 2:
                raise ValueError(
                    "multivariate mode expects 1-D or 2-D input, "
                    f"got shape {cloud.shape}"
                )

        if cloud.shape[0] < self.window_size:
            raise ValueError(
                f"Not enough points ({cloud.shape[0]}) for window_size="
                f"{self.window_size} after embedding."
            )
        sw = SlidingWindow(size=self.window_size, stride=1)
        return sw.fit_transform(cloud)

    def _cache_key(self, returns: np.ndarray) -> str:
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(returns).tobytes())
        h.update(repr(returns.shape).encode())
        h.update(returns.dtype.str.encode())
        params = (
            self.window_size,
            float(self.max_edge_length),
            tuple(self.homology_dims),
            self.mode,
            self.embedding_dim,
            self.time_delay,
        )
        h.update(repr(params).encode())
        return h.hexdigest()[:24]

    def _diagram_cache_path(self, key: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"diagrams_{key}.pkl"

    def _load_cached_diagrams(self, key: str) -> Optional[np.ndarray]:
        path = self._diagram_cache_path(key)
        if path is None or not path.exists():
            return None
        with open(path, "rb") as f:
            return pickle.load(f)

    def _save_diagrams(self, key: str, diagrams: np.ndarray) -> None:
        path = self._diagram_cache_path(key)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(diagrams, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _compute_diagrams(self, windows: np.ndarray, cache_key: str) -> np.ndarray:
        cached = self._load_cached_diagrams(cache_key)
        if cached is not None:
            return cached

        vr = VietorisRipsPersistence(
            homology_dimensions=self.homology_dims,
            max_edge_length=self.max_edge_length,
            n_jobs=self.n_jobs,
        )
        t0 = time.perf_counter()
        diagrams = vr.fit_transform(windows)
        elapsed = time.perf_counter() - t0
        if elapsed > self.cache_threshold_seconds:
            self._save_diagrams(cache_key, diagrams)
        return diagrams

    def _landscape_norms_by_dim(
        self, diagrams: np.ndarray
    ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Per requested H-dim: ``(L1_per_window, L2_per_window)``."""
        landscape = PersistenceLandscape(
            n_layers=self.landscape_layers,
            n_bins=self.landscape_bins,
            n_jobs=self.n_jobs,
        )
        grids = landscape.fit_transform(diagrams)
        dims_present = list(landscape.homology_dimensions_)

        out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for d in self.homology_dims:
            if d not in dims_present:
                continue
            grid = _slice_landscape(grids, landscape, d, self.landscape_layers)
            sampling = landscape.samplings_[d]
            bin_width = (
                float(sampling[1] - sampling[0]) if len(sampling) > 1 else 1.0
            )
            l1 = np.sum(np.abs(grid), axis=(1, 2)) * bin_width
            l2 = np.sqrt(np.sum(grid * grid, axis=(1, 2)) * bin_width)
            out[d] = (l1, l2)
        return out

    def _persistence_entropy_h1(
        self, diagrams: np.ndarray, n_windows: int
    ) -> np.ndarray:
        if 1 not in self.homology_dims:
            return np.full(n_windows, np.nan)
        pe = PersistenceEntropy(n_jobs=self.n_jobs)
        entropy = pe.fit_transform(diagrams)  # (n_windows, n_homology_dims)
        h1_col = self.homology_dims.index(1)
        return entropy[:, h1_col]

    # ------- public API -------------------------------------------------

    def fit_transform(self, returns: np.ndarray) -> dict:
        """Run the full pipeline on ``returns`` and return feature arrays.

        Parameters
        ----------
        returns :
            Shape ``(T,)`` for ``mode='takens'``, or ``(T, d)`` (or ``(T,)``)
            for ``mode='multivariate'``.

        Returns
        -------
        dict with keys:
            ``L1_H0``, ``L1_H1``, ``L2_H0``, ``L2_H1`` :
                1-D arrays of length ``n_windows`` (one per sliding window).
                Filled with NaN if the corresponding homology dim was not
                requested.
            ``persistence_entropy_H1`` :
                1-D array of length ``n_windows``.
            ``diagrams`` :
                List of per-window persistence diagrams (each shape
                ``(n_points, 3)`` — birth, death, hom-dim).
        """
        returns = np.asarray(returns)
        cache_key = self._cache_key(returns)
        windows = self._build_windows(returns)
        diagrams = self._compute_diagrams(windows, cache_key)
        n_windows = int(diagrams.shape[0])

        norms = self._landscape_norms_by_dim(diagrams)
        nan_vec = lambda: np.full(n_windows, np.nan)  # noqa: E731

        result: dict = {"diagrams": [diagrams[i] for i in range(n_windows)]}
        for d in (0, 1):
            l1, l2 = norms.get(d, (nan_vec(), nan_vec()))
            result[f"L1_H{d}"] = l1
            result[f"L2_H{d}"] = l2
        result["persistence_entropy_H1"] = self._persistence_entropy_h1(
            diagrams, n_windows
        )
        return result
