"""Evaluation, tables, and manuscript figures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .cfar import ca_cfar_threshold, detect, go_cfar_threshold, os_cfar_threshold
from .figures import draw_figure1 as _draw_figure1
from .figures import draw_figure2 as _draw_figure2
from .model import BlackBoxCfarNet, ResidualCfarNet
from .sar_scene import SceneConfig, render_scene, stack_channels


def _connected_hits(det: np.ndarray, centers: list[tuple[int, int]], radius: int = 2) -> float:
    if not centers:
        return 0.0
    hits = 0
    for i, j in centers:
        patch = det[max(i - radius, 0) : i + radius + 1, max(j - radius, 0) : j + radius + 1]
        hits += int(np.any(patch))
    return hits / len(centers)


def _false_alarms(det: np.ndarray, mask: np.ndarray, dilate: int = 2) -> tuple[int, float]:
    protect = np.zeros_like(det, dtype=bool)
    ys, xs = np.where(mask > 0.5)
    for i, j in zip(ys, xs):
        protect[max(i - dilate, 0) : i + dilate + 1, max(j - dilate, 0) : j + dilate + 1] = True
    fa = det & ~protect
    n_clutter = int((~protect).sum())
    return int(fa.sum()), float(fa.sum() / max(n_clutter, 1))


def load_models(ckpt_dir: Path, device: str = "cpu"):
    pcr = ResidualCfarNet()
    box = BlackBoxCfarNet()
    pcr.load_state_dict(torch.load(ckpt_dir / "pcr_cfar.pt", map_location=device, weights_only=True))
    box.load_state_dict(torch.load(ckpt_dir / "blackbox.pt", map_location=device, weights_only=True))
    pcr.eval()
    box.eval()
    return pcr.to(device), box.to(device)


@torch.no_grad()
def thresholds_for_scene(scene: dict, pcr: ResidualCfarNet, box: BlackBoxCfarNet, cfg: SceneConfig, device: str):
    img = scene["img"]
    x = torch.from_numpy(stack_channels(scene)[None]).to(device)
    log_img = torch.from_numpy(scene["log_img"][None, None]).to(device)
    log_t_pcr, gate, residual = pcr(x)
    log_t_box = box(log_img)
    t_pcr = np.exp(log_t_pcr.squeeze().cpu().numpy())
    t_box = np.exp(log_t_box.squeeze().cpu().numpy())
    t_ca = ca_cfar_threshold(img, cfg.win, cfg.guard, cfg.pfa_design)
    t_go = go_cfar_threshold(img, cfg.win, cfg.guard, cfg.pfa_design)
    t_os = os_cfar_threshold(img, cfg.win, cfg.guard, cfg.pfa_design)
    t_pcr_nocal = t_ca * np.exp(residual.squeeze().cpu().numpy())  # gate=1 ablation proxy
    return {
        "CA-CFAR": t_ca,
        "GO-CFAR": t_go,
        "OS-CFAR": t_os,
        "CNN-CFAR": t_box,
        "PCR-CFAR": t_pcr,
        "gate": gate.squeeze().cpu().numpy(),
        "residual": residual.squeeze().cpu().numpy(),
        "t_ungated": t_pcr_nocal,
    }


def evaluate(ckpt_dir: Path, out_dir: Path, device: str = "cpu") -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = SceneConfig()
    pcr, box = load_models(ckpt_dir, device)
    methods = ["CA-CFAR", "GO-CFAR", "OS-CFAR", "CNN-CFAR", "PCR-CFAR"]
    scr_grid = [4, 6, 8, 10, 12]
    pd_vs_scr = {m: [] for m in methods}
    summary = {m: {"pd": [], "pfa": [], "fa": []} for m in methods}

    # Table 1 operating point: SCR = 8 dB, 40 test scenes
    test_seeds = list(range(2000, 2040))
    for seed in test_seeds:
        cfg8 = SceneConfig(target_scr_db=8.0)
        scene = render_scene(cfg8, seed=seed)
        th = thresholds_for_scene(scene, pcr, box, cfg8, device)
        for m in methods:
            det = detect(scene["img"], th[m])
            pd = _connected_hits(det, scene["centers"])
            fa, pfa = _false_alarms(det, scene["mask"])
            summary[m]["pd"].append(pd)
            summary[m]["pfa"].append(pfa)
            summary[m]["fa"].append(fa)

    # Pd vs SCR at the same detectors (re-render intensity, fixed seeds)
    for scr in scr_grid:
        acc = {m: [] for m in methods}
        for seed in test_seeds[:20]:
            cfg_s = SceneConfig(target_scr_db=float(scr))
            scene = render_scene(cfg_s, seed=seed)
            th = thresholds_for_scene(scene, pcr, box, cfg_s, device)
            for m in methods:
                det = detect(scene["img"], th[m])
                acc[m].append(_connected_hits(det, scene["centers"]))
        for m in methods:
            pd_vs_scr[m].append(float(np.mean(acc[m])))

    # Ablation on same 8 dB set
    ablation = {"PCR-CFAR (full)": summary["PCR-CFAR"], "CA-CFAR prior only": summary["CA-CFAR"]}
    ab_pd = {"no Pfa loss (CNN-CFAR)": summary["CNN-CFAR"]}
    # ungated residual on 8 dB
    ung = {"pd": [], "pfa": []}
    for seed in test_seeds:
        scene = render_scene(SceneConfig(target_scr_db=8.0), seed=seed)
        th = thresholds_for_scene(scene, pcr, box, cfg, device)
        det = detect(scene["img"], th["t_ungated"])
        ung["pd"].append(_connected_hits(det, scene["centers"]))
        ung["pfa"].append(_false_alarms(det, scene["mask"])[1])

    table1 = []
    for m in methods:
        pd = float(np.mean(summary[m]["pd"]))
        pfa = float(np.mean(summary[m]["pfa"]))
        fa = float(np.mean(summary[m]["fa"]))
        rec =  pd
        prec = pd * 6 / max(pd * 6 + fa, 1e-9)  # 6 targets/scene nominal
        f1 = 2 * prec * rec / max(prec + rec, 1e-9) if rec + prec else 0.0
        table1.append(
            {
                "method": m,
                "Pd": round(pd, 3),
                "Pfa": float(f"{pfa:.3e}"),
                "FA_per_scene": round(fa, 2),
                "F1": round(f1, 3),
            }
        )

    table2 = [
        {
            "variant": "CA-CFAR (analytic prior only)",
            "Pd": round(float(np.mean(summary["CA-CFAR"]["pd"])), 3),
            "Pfa": float(f"{np.mean(summary['CA-CFAR']['pfa']):.3e}"),
        },
        {
            "variant": "Ungated residual (gate ≡ 1)",
            "Pd": round(float(np.mean(ung["pd"])), 3),
            "Pfa": float(f"{np.mean(ung['pfa']):.3e}"),
        },
        {
            "variant": "Black-box CNN-CFAR (no CA prior)",
            "Pd": round(float(np.mean(summary["CNN-CFAR"]["pd"])), 3),
            "Pfa": float(f"{np.mean(summary['CNN-CFAR']['pfa']):.3e}"),
        },
        {
            "variant": "PCR-CFAR (gated residual + Pfa loss)",
            "Pd": round(float(np.mean(summary["PCR-CFAR"]["pd"])), 3),
            "Pfa": float(f"{np.mean(summary['PCR-CFAR']['pfa']):.3e}"),
        },
    ]

    payload = {
        "table1": table1,
        "table2": table2,
        "pd_vs_scr": {"scr_db": scr_grid, **pd_vs_scr},
        "params": {"PCR-CFAR": ResidualCfarNet().count_parameters(), "CNN-CFAR": BlackBoxCfarNet().count_parameters()},
        "flops_per_pixel_pcr": 3 * (3 * 3 * 16 * 16) + 2 * (1 * 1 * 16),  # 3 conv 3x3 + 2x 1x1
        "note": "Synthetic UAV SAR, 40 scenes, 6 point targets, design Pfa=1e-3, 2-look K-speckle.",
        "clouds": {m: {"pd": summary[m]["pd"], "pfa": summary[m]["pfa"]} for m in methods},
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2))

    # Figure 1 — showcase scene
    show = render_scene(SceneConfig(target_scr_db=8.0, n_targets=5), seed=2026)
    th = thresholds_for_scene(show, pcr, box, cfg, device)
    fig1_path = out_dir / "fig1_detections.png"
    _draw_figure1(show, th, fig1_path)
    fig2_path = out_dir / "fig2_mechanism.png"
    _draw_figure2(show, th, fig2_path, payload)
    return payload
