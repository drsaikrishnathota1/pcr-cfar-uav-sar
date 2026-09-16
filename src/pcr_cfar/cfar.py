"""Vectorized 2-D CFAR detectors for intensity SAR maps."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import percentile_filter, uniform_filter


def _ring_mean(img: np.ndarray, win: int, guard: int) -> tuple[np.ndarray, int]:
    n_win = win * win
    n_guard = guard * guard
    n_train = n_win - n_guard
    x = img.astype(np.float64)
    sum_win = uniform_filter(x, size=win, mode="nearest") * n_win
    sum_guard = uniform_filter(x, size=guard, mode="nearest") * n_guard
    return (sum_win - sum_guard) / n_train, n_train


def _ring_second_moment(img: np.ndarray, win: int, guard: int) -> np.ndarray:
    n_win = win * win
    n_guard = guard * guard
    sq = img.astype(np.float64) ** 2
    sum_win = uniform_filter(sq, size=win, mode="nearest") * n_win
    sum_guard = uniform_filter(sq, size=guard, mode="nearest") * n_guard
    return (sum_win - sum_guard) / n_guard if False else (sum_win - sum_guard) / (n_win - n_guard)


def local_cv(img: np.ndarray, win: int = 21, guard: int = 5) -> np.ndarray:
    mu, _ = _ring_mean(img, win, guard)
    m2 = _ring_second_moment(img, win, guard)
    var = np.clip(m2 - mu**2, 0.0, None)
    return np.sqrt(var) / np.clip(mu, 1e-12, None)


def exponential_scale_factor(n_train: int, pfa: float) -> float:
    """CA-CFAR scale for exponential (single-look) intensity clutter."""
    n_train = max(int(n_train), 1)
    return n_train * (pfa ** (-1.0 / n_train) - 1.0)


def ca_cfar_threshold(
    img: np.ndarray, win: int = 21, guard: int = 5, pfa: float = 1e-3
) -> np.ndarray:
    mu, n_train = _ring_mean(img, win, guard)
    return exponential_scale_factor(n_train, pfa) * np.clip(mu, 1e-12, None)


def go_cfar_threshold(
    img: np.ndarray, win: int = 21, guard: int = 5, pfa: float = 1e-3
) -> np.ndarray:
    """Greatest-of CFAR: max of left/right training rectangles (vectorized)."""
    pad = win // 2
    g = guard // 2
    lw = max(pad - g, 1)
    x = img.astype(np.float64)
    left = uniform_filter(x, size=(win, lw), mode="nearest")
    right = left
    # Shift so the kernel sits on each side of the CUT rather than being centered.
    shift = (lw // 2) + g + 1
    left = np.roll(left, -shift, axis=1)
    right = np.roll(right, shift, axis=1)
    mu = np.maximum(left, right)
    n_half = win * lw
    return exponential_scale_factor(n_half, pfa) * np.clip(mu, 1e-12, None)


def os_cfar_threshold(
    img: np.ndarray, win: int = 21, guard: int = 5, pfa: float = 1e-3, k_frac: float = 0.75
) -> np.ndarray:
    """Order-statistic CFAR using a local percentile of the window."""
    q = percentile_filter(
        img.astype(np.float64), percentile=100.0 * k_frac, size=win, mode="nearest"
    )
    n_train = win * win - guard * guard
    return exponential_scale_factor(n_train, pfa) * np.clip(q, 1e-12, None)


def detect(img: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    return img > threshold
