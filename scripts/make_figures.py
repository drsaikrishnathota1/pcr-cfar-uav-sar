#!/usr/bin/env python3
"""Redraw the two manuscript figures from saved checkpoints and metrics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")

from pcr_cfar.evaluate import load_models, thresholds_for_scene
from pcr_cfar.figures import draw_figure1, draw_figure2
from pcr_cfar.sar_scene import SceneConfig, render_scene


def _publish(src: Path, *dests: Path) -> None:
    data = src.read_bytes()
    for dest in dests:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)


def main() -> None:
    out = ROOT / "results"
    fig_dir = ROOT / "figures"
    paper_fig = ROOT / "paper" / "figures"
    fig_dir.mkdir(exist_ok=True)
    paper_fig.mkdir(parents=True, exist_ok=True)

    metrics_path = out / "metrics.json"
    ckpt_dir = out / "checkpoints"
    if not metrics_path.is_file() or not (ckpt_dir / "pcr_cfar.pt").is_file():
        raise SystemExit("Missing results/metrics.json or results/checkpoints/*.pt. Run scripts/run_all.py first.")

    payload = json.loads(metrics_path.read_text())
    pcr, box = load_models(ckpt_dir, "cpu")
    cfg = SceneConfig(target_scr_db=8.0, n_targets=5)
    scene = render_scene(cfg, seed=2042)
    th = thresholds_for_scene(scene, pcr, box, cfg, "cpu")

    p1 = fig_dir / "fig1_detections.png"
    p2 = fig_dir / "fig2_operating.png"
    draw_figure1(scene, th, p1)
    draw_figure2(scene, th, p2, payload)
    _publish(p1, out / "fig1_detections.png", paper_fig / "fig1_detections.png")
    _publish(
        p2,
        out / "fig2_operating.png",
        out / "fig2_pd_pfa.png",
        fig_dir / "fig2_pd_pfa.png",
        paper_fig / "fig2_operating.png",
    )
    print("wrote", p1)
    print("wrote", p2)


if __name__ == "__main__":
    main()
