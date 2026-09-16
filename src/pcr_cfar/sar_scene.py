"""Synthetic UAV stripmap SAR with heterogeneous sea–land clutter and point ships."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import cfar as cfar_mod
from .cfar import ca_cfar_threshold, local_cv


@dataclass
class SceneConfig:
    height: int = 192
    width: int = 256
    looks: float = 2.0
    texture_shape: float = 1.6
    n_targets: int = 6
    target_scr_db: float = 8.0
    pfa_design: float = 1e-3
    win: int = 21
    guard: int = 5


def _coastline(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    yy = np.arange(h)[:, None]
    shore = (
        0.52 * w
        + 22.0 * np.sin(yy / 11.0)
        + 9.0 * np.sin(yy / 6.3)
        + rng.normal(0.0, 1.5, size=(h, 1))
    )
    xx = np.arange(w)[None, :]
    return (xx > shore).astype(np.float64)


def render_scene(cfg: SceneConfig, seed: int, inject_targets: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    h, w = cfg.height, cfg.width
    yy, xx = np.mgrid[0:h, 0:w]
    land = _coastline(h, w, rng)

    range_gain = (0.35 + 0.65 * (yy / max(h - 1, 1))) ** (-1.6)
    sea = 0.85 + 0.18 * np.sin(xx / 28.0) * np.sin(yy / 19.0)
    land_tex = 3.6 + 1.1 * np.sin(xx / 16.0) * np.cos(yy / 13.0)
    rcs = (1.0 - land) * sea + land * land_tex
    rcs *= range_gain
    rcs *= np.clip(1.0 + 0.08 * rng.standard_normal((h, w)), 0.25, None)

    texture = rng.gamma(cfg.texture_shape, 1.0 / cfg.texture_shape, size=(h, w))
    speckle = rng.gamma(cfg.looks, 1.0 / cfg.looks, size=(h, w))
    img = np.clip(rcs * texture * speckle, 1e-8, None)

    mu, _ = cfar_mod._ring_mean(img, cfg.win, cfg.guard)
    mask = np.zeros((h, w), dtype=np.float64)
    centers: list[tuple[int, int]] = []
    if inject_targets:
        pad = cfg.win
        for _ in range(cfg.n_targets):
            for _try in range(50):
                i = int(rng.integers(pad, h - pad))
                j = int(rng.integers(pad, max(pad + 1, int(0.62 * w))))
                if land[i, j] > 0.5:
                    continue
                if any(abs(i - ci) < 8 and abs(j - cj) < 8 for ci, cj in centers):
                    continue
                centers.append((i, j))
                scr = 10.0 ** (cfg.target_scr_db / 10.0)
                amp = float(mu[i, j] * scr)
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        wt = float(np.exp(-0.5 * (di * di + dj * dj) / 0.55))
                        img[i + di, j + dj] += amp * wt
                mask[i, j] = 1.0
                break

    log_img = np.log(img + 1e-8)
    t_ca = ca_cfar_threshold(img, cfg.win, cfg.guard, cfg.pfa_design)
    cv = local_cv(img, cfg.win, cfg.guard)
    return {
        "img": img.astype(np.float32),
        "log_img": log_img.astype(np.float32),
        "t_ca": t_ca.astype(np.float32),
        "cv": cv.astype(np.float32),
        "mask": mask.astype(np.float32),
        "land": land.astype(np.float32),
        "centers": centers,
        "seed": seed,
    }


def stack_channels(scene: dict) -> np.ndarray:
    log_t = np.log(scene["t_ca"] + 1e-8)
    cv = np.clip(scene["cv"], 0.0, 5.0) / 5.0
    return np.stack([scene["log_img"], log_t, cv], axis=0).astype(np.float32)
