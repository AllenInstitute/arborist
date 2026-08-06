"""
Created on Sat Nov 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Geometry utilities for 3D space curves.

"""

import numpy as np
import torch
from scipy.interpolate import UnivariateSpline
from sklearn.decomposition import PCA


# --- Spline fitting ---

def fit_spline_1d(pts, k=3, s=None):
    """
    Fits a spline to a 1D array of points.

    Parameters
    ----------
    pts : numpy.ndarray
        Points to fit.
    k : int, optional
        Spline degree. Default is 3.
    s : float, optional
        Smoothness parameter. Default is None (auto).

    Returns
    -------
    UnivariateSpline
    """
    t = np.linspace(0, 1, len(pts))
    s = len(pts) / s if s else len(pts) / 15
    return UnivariateSpline(t, pts, k=k, s=s)


def fit_spline_3d(pts, k=3, s=None):
    """
    Fits a cubic spline independently to each spatial dimension.

    Parameters
    ----------
    pts : numpy.ndarray
        Array of shape (N, 3).
    k : int, optional
        Spline degree. Default is 3.
    s : float, optional
        Smoothness parameter. Default is None (auto).

    Returns
    -------
    Tuple[UnivariateSpline, UnivariateSpline, UnivariateSpline]
        Splines for x, y, z.
    """
    return (
        fit_spline_1d(pts[:, 0], k=k, s=s),
        fit_spline_1d(pts[:, 1], k=k, s=s),
        fit_spline_1d(pts[:, 2], k=k, s=s),
    )


# --- Curve resampling ---

def resample_curve_1d(pts, n_pts=None, s=None):
    """
    Smooths a 1D curve by fitting a spline and resampling it.

    Parameters
    ----------
    pts : numpy.ndarray
        Points to resample.
    n_pts : int, optional
        Number of output points. Default is None (same as input).
    s : float, optional
        Smoothness parameter. Default is None (auto).

    Returns
    -------
    numpy.ndarray
    """
    dt = max(n_pts or len(pts), 5)
    k = min(3, len(pts) - 1)
    if k == 0:
        return np.repeat(pts, n_pts, axis=0)
    t = np.linspace(0, 1, dt)
    spline = fit_spline_1d(pts, k=k, s=s)
    return spline(t)


def resample_curve_3d(pts, n_pts=None, s=None):
    """
    Smooths an (N, 3) curve by fitting a spline and resampling it.

    Parameters
    ----------
    pts : numpy.ndarray
        Array of shape (N, 3).
    n_pts : int, optional
        Number of output points. Default is None (same as input).
    s : float, optional
        Smoothness parameter. Default is None (auto).

    Returns
    -------
    numpy.ndarray
        Resampled curve of shape (n_pts, 3).
    """
    dt = max(n_pts or len(pts), 5)
    k = min(3, len(pts) - 1)
    if k == 0:
        return np.repeat(pts, n_pts, axis=0)
    spline_x, spline_y, spline_z = fit_spline_3d(pts, k=k, s=s)
    t = np.linspace(0, 1, dt)
    return np.column_stack((
        spline_x(t).astype(np.float32),
        spline_y(t).astype(np.float32),
        spline_z(t).astype(np.float32),
    ))


def curves_pca_projection(curve1, curve2):
    """
    Projects two 3D curves into a shared PCA coordinate system.

    Parameters
    ----------
    curve1 : numpy.ndarray
        First curve with shape ``(N, 3)``.
    curve2 : numpy.ndarray
        Second curve with shape ``(M, 3)``.

    Returns
    -------
    tuple
        Two projected curves with shape ``(N, 2)`` and ``(M, 2)``.
    """
    pts = np.vstack([curve1, curve2])
    center = pts.mean(axis=0)

    pca = PCA(n_components=3)
    pca.fit(pts - center)

    curve1_proj = pca.transform(curve1 - center)
    curve2_proj = pca.transform(curve2 - center)
    return curve1_proj[:, :2], curve2_proj[:, :2]


def curve_principal_direction(curve):
    """
    Computes the principal direction of a 3D curve using PCA.

    Parameters
    ----------
    curve : numpy.ndarray
        Array with shape (N, 3) containing the 3D coordinates of the curve.

    Returns
    -------
    numpy.ndarray
        Unit vector of shape (3,) representing the principal direction of the
        curve.
    """
    curve_pca = PCA(n_components=1)
    curve_pca.fit(curve)
    direction = curve_pca.components_[0]
    if direction[2] < 0:
        direction = -direction
    return direction / np.linalg.norm(direction)


def path_length(curve):
    """
    Computes the total arc length of a 3D curve.

    Parameters
    ----------
    curve : numpy.ndarray
        Array of shape (N, 3).

    Returns
    -------
    float
        Total arc length in the same units as the input coordinates.
    """
    return np.linalg.norm(curve[1:] - curve[:-1], axis=1).sum()


def max_l2_error(curve1, curve2):
    """
    Computes maximum pointwise L2 error between two curves.

    Parameters
    ----------
    curve1 : numpy.ndarray
        Ground truth curve.
    curve2 : numpy.ndarray
        Reconstruction curve.

    Returns
    -------
    float
        Maximum Euclidean error.
    """
    assert curve1.shape == curve2.shape, "Curves have different number of pts"
    return np.linalg.norm(curve1 - curve2, axis=1).max()


def reconstruct_diffs(diffs):
    """
    Reconstructs a curve from a sequence of offset vectors.

    Parameters
    ----------
    diffs : numpy.ndarray or torch.Tensor
        Array representing the differences between consecutive points.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        Reconstructed curve.
    """
    if isinstance(diffs, torch.Tensor):
        start = torch.zeros(1, 3, device=diffs.device, dtype=diffs.dtype)
        return torch.cat([start, start + torch.cumsum(diffs, dim=0)], dim=0)
    else:
        start = np.zeros((1, 3))
        return np.concatenate(
            [start, start + np.cumsum(diffs, axis=0)], axis=0
        )


def rmse(curve1, curve2):
    """
    Computes Root Mean Squared Error (RMSE) between two curves.

    Parameters
    ----------
    curve1 : numpy.ndarray
        Ground truth curve.
    curve2 : numpy.ndarray
        Reconstruction curve.

    Returns
    -------
    float
        RMSE between the two curves.
    """
    return np.sqrt(np.mean(np.sum((curve1 - curve2) ** 2, axis=1)))
