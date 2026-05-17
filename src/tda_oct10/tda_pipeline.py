"""TDA pipeline.

Constructs sliding-window point clouds (Takens embeddings) from a 1-D signal
and computes Vietoris-Rips persistence diagrams plus persistence-landscape
vectorisations (and their L^1 norms) via giotto-tda.

Functions here are intentionally small and composable so they can be reused
from notebooks, the experiment driver, and the Lorenz sanity test without
duplicating logic.
"""

from __future__ import annotations

import numpy as np
from gtda.diagrams import PersistenceLandscape
from gtda.homology import VietorisRipsPersistence
from gtda.time_series import SingleTakensEmbedding, SlidingWindow


def takens_embed(series, dimension: int = 4, time_delay: int = 1) -> np.ndarray:
    """Delay-embed a 1-D series into R^dimension.

    Output shape: (len(series) - (dimension - 1) * time_delay, dimension).
    """
    series = np.asarray(series).reshape(-1)
    embedder = SingleTakensEmbedding(
        parameters_type="fixed",
        time_delay=time_delay,
        dimension=dimension,
    )
    return embedder.fit_transform(series)


def sliding_windows(point_cloud: np.ndarray, window_size: int = 100, stride: int = 1) -> np.ndarray:
    """Sliding windows over a point cloud of shape (n_points, n_features).

    Output shape: (n_windows, window_size, n_features).
    """
    sw = SlidingWindow(size=window_size, stride=stride)
    return sw.fit_transform(point_cloud)


def vietoris_rips_diagrams(
    windows: np.ndarray,
    homology_dimensions=(1,),
    n_jobs: int = -1,
) -> np.ndarray:
    """Vietoris-Rips persistence diagrams for each window.

    Input shape: (n_windows, window_size, n_features).
    Output: giotto-tda persistence diagram array of shape (n_windows, n_points, 3),
    where the last axis is (birth, death, homology_dimension).
    """
    vr = VietorisRipsPersistence(
        homology_dimensions=list(homology_dimensions),
        n_jobs=n_jobs,
    )
    return vr.fit_transform(windows)


def landscape_l1_norms(
    diagrams: np.ndarray,
    homology_dimension: int = 1,
    n_layers: int = 5,
    n_bins: int = 100,
) -> np.ndarray:
    """L^1 norm of the persistence landscape for the selected homology dim, per window.

    The landscape is fit jointly on the batch so the sampling grid is consistent
    across windows; the L^1 norm is then summed over (layers, bins) and rescaled
    by the bin width to approximate the integral.

    Returns a 1-D array of length n_windows.
    """
    landscape = PersistenceLandscape(n_layers=n_layers, n_bins=n_bins)
    grids = landscape.fit_transform(diagrams)
    # giotto-tda 0.6 returns (n_windows, n_layers, n_bins) when only one homology
    # dimension is present, or (n_windows, n_homology_dims, n_layers, n_bins) otherwise.
    if grids.ndim == 4:
        hom_dims_present = sorted(landscape.homology_dimensions_)
        try:
            h_idx = hom_dims_present.index(homology_dimension)
        except ValueError as e:
            raise ValueError(
                f"Requested homology_dimension={homology_dimension} not in diagrams "
                f"(found {hom_dims_present})."
            ) from e
        per_window = grids[:, h_idx]  # (n_windows, n_layers, n_bins)
    elif grids.ndim == 3:
        if homology_dimension not in landscape.samplings_:
            raise ValueError(
                f"Requested homology_dimension={homology_dimension} not present "
                f"(found {list(landscape.samplings_.keys())})."
            )
        per_window = grids  # (n_windows, n_layers, n_bins)
    else:
        raise RuntimeError(f"Unexpected landscape grid shape: {grids.shape}")

    samplings = landscape.samplings_[homology_dimension]
    bin_width = float(samplings[1] - samplings[0]) if len(samplings) > 1 else 1.0
    return np.sum(np.abs(per_window), axis=(1, 2)) * bin_width
