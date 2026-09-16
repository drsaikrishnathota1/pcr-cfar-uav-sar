"""Manuscript figures — journal-grade atlas and operating-point flow."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects as pe
from matplotlib.colors import LightSource, Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle
from scipy.ndimage import gaussian_filter, maximum_filter, uniform_filter1d, zoom

from .cfar import detect

TEAL = "#0B6E63"
RED = "#C62828"
GOLD = "#F9A825"
INK = "#1C2331"


def _sar_display(img: np.ndarray) -> np.ndarray:
    x = np.log(img + 1e-8)
    lo, hi = np.percentile(x, [1.5, 99.6])
    return np.clip((x - lo) / (hi - lo + 1e-8), 0, 1)


def _mark_targets(ax, centers: list[tuple[int, int]], numbered: bool = False) -> None:
    for k, (i, j) in enumerate(centers):
        ax.add_patch(Circle((j, i), 5.2, fill=False, ec="white", lw=1.25, alpha=0.98, zorder=6))
        ax.add_patch(Circle((j, i), 5.2, fill=False, ec=INK, lw=0.4, alpha=0.75, zorder=6))
        if numbered:
            t = ax.text(
                j + 7.5, i - 7.5, f"T{k + 1}", color="white", fontsize=7.2, fontweight="bold", zorder=7
            )
            t.set_path_effects([pe.withStroke(linewidth=2.0, foreground=INK)])


def _add_cbar(fig, mappable, ax, label: str) -> None:
    cbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.028)
    cbar.set_label(label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    cbar.outline.set_linewidth(0.35)


def _coast(ax, land: np.ndarray | None) -> None:
    if land is None:
        return
    ax.contour(land, levels=[0.5], colors=GOLD, linewidths=0.85, origin="upper", zorder=5)


def _fa_blobs(det: np.ndarray) -> np.ndarray:
    return maximum_filter(det.astype(np.uint8), size=2).astype(bool)


def _hillshade(dens: np.ndarray, vmax: float) -> np.ndarray:
    ls = LightSource(azdeg=312, altdeg=42)
    normed = np.clip(dens / max(vmax, 1e-8), 0, 1)
    return ls.shade(
        normed,
        cmap=plt.cm.inferno,
        vert_exag=18.0,
        blend_mode="overlay",
        vmin=0.0,
        vmax=1.0,
    )


def draw_figure1(scene: dict, th: dict, path: Path) -> None:
    """Coastline false-alarm atlas with hillshaded density."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.linewidth": 0.55})
    img = scene["img"]
    disp = _sar_display(img)
    land = scene.get("land")
    det_ca = detect(img, th["CA-CFAR"])
    det_pcr = detect(img, th["PCR-CFAR"])
    for i, j in scene["centers"]:
        det_ca[max(i - 2, 0) : i + 3, max(j - 2, 0) : j + 3] = False
        det_pcr[max(i - 2, 0) : i + 3, max(j - 2, 0) : j + 3] = False
    dens_ca = gaussian_filter(det_ca.astype(np.float32), sigma=2.4)
    dens_pcr = gaussian_filter(det_pcr.astype(np.float32), sigma=2.4)
    vmax = max(float(np.percentile(dens_ca, 99.4)), 1e-4)
    ca_only = _fa_blobs(det_ca & ~det_pcr)
    pcr_only = _fa_blobs(det_pcr & ~det_ca)
    both = _fa_blobs(det_ca & det_pcr)

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.15), dpi=300)
    ax = axes[0, 0]
    ax.imshow(disp, cmap="gray", origin="upper", vmin=0, vmax=1, interpolation="nearest")
    _coast(ax, land)
    _mark_targets(ax, scene["centers"], numbered=True)
    ax.set_title("(a)  Two-look UAV stripmap  (gold = land–sea edge)", fontsize=10)

    ax = axes[0, 1]
    ax.imshow(disp, cmap="gray", origin="upper", vmin=0, vmax=1, interpolation="nearest")
    _coast(ax, land)
    yy, xx = np.where(ca_only)
    ax.scatter(xx, yy, s=7.5, c=RED, marker="s", linewidths=0, alpha=0.88, zorder=4, label="CA only")
    yy, xx = np.where(pcr_only)
    ax.scatter(xx, yy, s=7.5, c="#00E5FF", marker="s", linewidths=0, alpha=0.88, zorder=4, label="PCR only")
    yy, xx = np.where(both)
    ax.scatter(xx, yy, s=8.0, c="white", marker="s", linewidths=0, alpha=0.9, zorder=4, label="both")
    _mark_targets(ax, scene["centers"], numbered=True)
    ax.set_title("(b)  False-alarm disagreement on the coastline", fontsize=10)
    leg = ax.legend(
        loc="lower left",
        fontsize=7.0,
        framealpha=0.55,
        facecolor="black",
        edgecolor="none",
        labelcolor="white",
        markerscale=1.6,
        handletextpad=0.35,
        borderpad=0.35,
    )
    for txt in leg.get_texts():
        txt.set_color("white")

    ax = axes[1, 0]
    ax.imshow(_hillshade(dens_ca, vmax), origin="upper", interpolation="bilinear")
    _coast(ax, land)
    _mark_targets(ax, scene["centers"])
    ax.set_title("(c)  CA-CFAR false-alarm relief", fontsize=10)
    sm = plt.cm.ScalarMappable(norm=Normalize(0, vmax), cmap="inferno")
    sm.set_array([])
    _add_cbar(fig, sm, ax, "local FA density")

    ax = axes[1, 1]
    ax.imshow(_hillshade(dens_pcr, vmax), origin="upper", interpolation="bilinear")
    _coast(ax, land)
    _mark_targets(ax, scene["centers"])
    ax.set_title("(d)  PCR-CFAR false-alarm relief  (identical scale)", fontsize=10)
    sm = plt.cm.ScalarMappable(norm=Normalize(0, vmax), cmap="inferno")
    sm.set_array([])
    _add_cbar(fig, sm, ax, "local FA density")

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.75)
            spine.set_color("0.18")
    fig.text(
        0.5,
        0.012,
        "Gold contour: land–sea edge.  White rings: true small targets (held out of the FA maps).  "
        "Relief in (c)–(d) uses the same height scale: CA-CFAR lights the shoreline; PCR-CFAR extinguishes it.",
        ha="center",
        fontsize=8.1,
        color="0.18",
    )
    fig.subplots_adjust(left=0.03, right=0.97, top=0.955, bottom=0.055, wspace=0.14, hspace=0.16)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _upsample_chip(chip: np.ndarray, factor: int = 8) -> np.ndarray:
    return zoom(chip, factor, order=3, prefilter=True)


def draw_figure2(scene: dict, th: dict, path: Path, payload: dict | None = None) -> None:
    """Target chips, CA→PCR migration flow, A-scope, and range-azimuth zoom."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.linewidth": 0.75})
    styles = {
        "CA-CFAR": ("#607D8B", "o"),
        "GO-CFAR": ("#8D6E63", "s"),
        "OS-CFAR": ("#7E57C2", "D"),
        "CNN-CFAR": ("#EF6C00", "^"),
        "PCR-CFAR": (TEAL, "o"),
    }
    log_img = np.log(scene["img"] + 1e-8)
    centers = scene["centers"]
    n = max(len(centers), 1)
    det_pcr = detect(scene["img"], th["PCR-CFAR"])
    det_ca = detect(scene["img"], th["CA-CFAR"])

    fig = plt.figure(figsize=(12.2, 8.05), dpi=300)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[0.46, 1.18], hspace=0.28, wspace=0.24)
    gs_top = gs[0, :].subgridspec(1, n, wspace=0.08)

    for k, (i, j) in enumerate(centers):
        ax = fig.add_subplot(gs_top[0, k])
        r = 10
        sl = log_img[max(i - r, 0) : i + r + 1, max(j - r, 0) : j + r + 1]
        t_ca = np.log(th["CA-CFAR"][max(i - r, 0) : i + r + 1, max(j - r, 0) : j + r + 1] + 1e-8)
        t_pcr = np.log(th["PCR-CFAR"][max(i - r, 0) : i + r + 1, max(j - r, 0) : j + r + 1] + 1e-8)
        lo, hi = np.percentile(sl, [8, 99.4])
        hi = max(hi, float(log_img[i, j]))
        chip = np.clip((sl - lo) / (hi - lo + 1e-8), 0, 1)
        ax.imshow(_upsample_chip(chip), cmap="inferno", origin="upper", vmin=0, vmax=1, interpolation="bilinear")
        ax.contour(zoom(sl - t_ca, 8, order=1), levels=[0], colors="white", linewidths=1.05, alpha=0.85)
        ax.contour(zoom(sl - t_pcr, 8, order=1), levels=[0], colors="#B9F6CA", linewidths=1.45)
        ci, cj = (i - max(i - r, 0)) * 8 + 4, (j - max(j - r, 0)) * 8 + 4
        ax.plot(cj, ci, marker="+", markersize=8, markeredgewidth=1.25, color="white")
        hit = bool(det_pcr[max(i - 2, 0) : i + 3, max(j - 2, 0) : j + 3].any())
        for spine in ax.spines.values():
            spine.set_color("#00C853" if hit else "#FFD54F")
            spine.set_linewidth(1.85)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"T{k + 1}", fontsize=8.2, pad=2)
        if k == 0:
            ax.set_ylabel("21×21 · cubic", fontsize=7.6)

    ax = fig.add_subplot(gs[1, 0])
    ax.fill_between([1.8e-4, 1e-3], 0.9, 1.08, color=TEAL, alpha=0.10, zorder=0, linewidth=0)
    ax.axhline(0.9, color=TEAL, ls=":", lw=0.75, alpha=0.65, zorder=1)
    ax.axvline(1e-3, color=INK, ls="--", lw=1.05, zorder=1)
    clouds = (payload or {}).get("clouds", {})
    if "CA-CFAR" in clouds and "PCR-CFAR" in clouds:
        pfa_ca = np.clip(np.asarray(clouds["CA-CFAR"]["pfa"], dtype=float), 1e-6, 1.0)
        pd_ca = np.asarray(clouds["CA-CFAR"]["pd"], dtype=float)
        pfa_pcr = np.clip(np.asarray(clouds["PCR-CFAR"]["pfa"], dtype=float), 1e-6, 1.0)
        pd_pcr = np.asarray(clouds["PCR-CFAR"]["pd"], dtype=float)
        for xa, ya, xb, yb in zip(pfa_ca, pd_ca, pfa_pcr, pd_pcr):
            ax.annotate(
                "",
                xy=(xb, yb),
                xytext=(xa, ya),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="#90A4AE",
                    lw=0.55,
                    alpha=0.42,
                    mutation_scale=7,
                ),
                zorder=2,
            )
    means = {}
    for name, (color, marker) in styles.items():
        if name not in clouds:
            continue
        pfa = np.clip(np.asarray(clouds[name]["pfa"], dtype=float), 1e-6, 1.0)
        pd = np.asarray(clouds[name]["pd"], dtype=float)
        ax.scatter(
            pfa,
            pd,
            s=18 if name != "PCR-CFAR" else 28,
            c=color,
            marker=marker,
            alpha=0.28 if name != "PCR-CFAR" else 0.45,
            edgecolors="none",
            zorder=3,
        )
        mx, my = float(pfa.mean()), float(pd.mean())
        means[name] = (mx, my)
        ax.scatter(
            [mx],
            [my],
            s=175,
            c=color,
            marker=marker,
            edgecolors="white",
            linewidths=1.2,
            zorder=5,
            label=name,
        )
    if "CA-CFAR" in means and "PCR-CFAR" in means:
        ax.annotate(
            "",
            xy=means["PCR-CFAR"],
            xytext=means["CA-CFAR"],
            arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.85, mutation_scale=14),
            zorder=6,
        )
    ax.text(1.12e-3, 0.06, r"design $P_{fa}$", fontsize=7.4, color="0.2", rotation=90, va="bottom")
    ax.text(2.4e-4, 0.935, "desired", fontsize=7.6, color="#004D40")
    ax.set_xscale("log")
    ax.set_xlim(1.8e-4, 1.7e-2)
    ax.set_ylim(-0.04, 1.08)
    ax.set_xlabel(r"Empirical $P_{fa}$   (grey arrows: one scene, CA $\to$ PCR)")
    ax.set_ylabel(r"$P_d$")
    ax.set_title("(a)  Scene-wise migration at SCR $= 8$ dB", fontsize=10)
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(frameon=False, fontsize=7.3, loc="center right")

    gs_r = gs[1, 1].subgridspec(2, 1, height_ratios=[1.15, 0.62], hspace=0.08)
    ax = fig.add_subplot(gs_r[0])
    i, j = max(centers, key=lambda c: float(log_img[c[0], c[1]]))
    x = np.arange(scene["img"].shape[1])
    log_i = log_img[i]
    log_ca = uniform_filter1d(np.log(th["CA-CFAR"][i] + 1e-8), size=5, mode="nearest")
    log_pcr = uniform_filter1d(np.log(th["PCR-CFAR"][i] + 1e-8), size=5, mode="nearest")
    x_hi = min(scene["img"].shape[1] - 1, max(j + 55, int(0.72 * scene["img"].shape[1])))
    land = scene.get("land")
    if land is not None:
        land_row = land[i] > 0.5
        if land_row.any():
            xs = np.where(land_row)[0]
            ax.axvspan(xs.min(), xs.max(), color="#F6E7B4", alpha=0.38, zorder=0, label="land")
    y0 = float(np.percentile(np.concatenate([log_i, log_ca, log_pcr]), 6))
    y1 = float(np.percentile(np.concatenate([log_i, log_ca, log_pcr]), 99.6)) + 0.15
    log_i_plot = np.clip(log_i, y0, y1)
    ax.fill_between(
        x, log_ca, log_i_plot, where=log_i > log_ca, color=RED, alpha=0.32, linewidth=0, zorder=1, label="CA exceedance"
    )
    ax.fill_between(
        x, log_pcr, log_i_plot, where=log_i > log_pcr, color=TEAL, alpha=0.38, linewidth=0, zorder=2, label="PCR exceedance"
    )
    ax.plot(x, log_i_plot, color=INK, lw=0.8, zorder=3, label="log intensity")
    ax.plot(x, log_ca, color="#607D8B", lw=1.35, zorder=4, label="CA-CFAR $T$")
    ax.plot(x, log_pcr, color=TEAL, lw=1.85, zorder=5, label="PCR-CFAR $T$")
    ax.axvline(j, color=INK, ls=":", lw=0.9, zorder=6)
    ax.scatter([j], [min(max(log_i[j], y0), y1)], s=48, c=INK, zorder=7)
    ax.set_ylim(y0, y1)
    ax.set_xlim(0, x_hi)
    ax.set_ylabel("log intensity / $T$")
    ax.set_title("(b)  A-scope of the brightest target range gate", fontsize=10)
    ax.set_xticklabels([])
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, fontsize=6.6, loc="upper left", ncols=2)

    axz = fig.add_subplot(gs_r[1], sharex=ax)
    band = 7
    r0, r1 = max(i - band, 0), min(i + band + 1, log_img.shape[0])
    strip = log_img[r0:r1]
    lo, hi = np.percentile(strip, [5, 99.3])
    axz.imshow(
        np.clip((strip - lo) / (hi - lo + 1e-8), 0, 1),
        cmap="magma",
        origin="upper",
        aspect="auto",
        extent=[0, scene["img"].shape[1] - 1, r1 - 0.5, r0 - 0.5],
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )
    fa_ca = det_ca[r0:r1]
    fa_pcr = det_pcr[r0:r1]
    yy, xx = np.where(fa_ca)
    axz.scatter(xx, r0 + yy, s=14, c=RED, marker="s", linewidths=0, alpha=0.78, label="CA det.")
    yy, xx = np.where(fa_pcr)
    axz.scatter(
        xx, r0 + yy, s=22, facecolors="none", edgecolors="#69F0AE", linewidths=0.85, marker="o", label="PCR det."
    )
    axz.axhline(i, color="white", ls=":", lw=0.7, alpha=0.8)
    axz.axvline(j, color="white", ls=":", lw=0.7, alpha=0.8)
    axz.add_patch(plt.Rectangle((j - 8, i - 3.5), 16, 7, fill=False, ec="white", lw=0.9, ls="--"))
    axz.set_xlim(0, x_hi)
    axz.set_xlabel("azimuth cell")
    axz.set_ylabel("range")
    axz.legend(frameon=False, fontsize=6.4, loc="upper right", labelcolor="white")
    axz.tick_params(labelsize=7)

    fig.text(
        0.5,
        0.012,
        "Chips: cubic upsample of a 21×21 neighbourhood; grey/green contours are CA / PCR decision boundaries.  "
        "Grey arrows in (a) are per-scene CA→PCR moves.  Strip in (b) is ±7 range gates; red dots CA detections, green rings PCR.",
        ha="center",
        fontsize=7.8,
        color="0.18",
    )
    fig.subplots_adjust(left=0.07, right=0.985, top=0.94, bottom=0.085)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
