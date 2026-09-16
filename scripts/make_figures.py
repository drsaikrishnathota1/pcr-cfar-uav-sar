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

from pcr_cfar.cfar import detect
from pcr_cfar.evaluate import _connected_hits, _false_alarms, load_models, thresholds_for_scene
from pcr_cfar.figures import draw_figure1, draw_figure2
from pcr_cfar.sar_scene import SceneConfig, render_scene


def collect_clouds(pcr, box, n: int = 24) -> dict:
    methods = ["CA-CFAR", "GO-CFAR", "CNN-CFAR", "PCR-CFAR"]
    clouds = {m: {"pd": [], "pfa": []} for m in methods}
    cfg = SceneConfig(target_scr_db=8.0)
    for seed in range(2000, 2000 + n):
        scene = render_scene(cfg, seed=seed)
        th = thresholds_for_scene(scene, pcr, box, cfg, "cpu")
        for m in methods:
            det = detect(scene["img"], th[m])
            clouds[m]["pd"].append(_connected_hits(det, scene["centers"]))
            clouds[m]["pfa"].append(_false_alarms(det, scene["mask"])[1])
    return clouds


def main() -> None:
    out = ROOT / "results"
    fig_dir = ROOT / "figures"
    fig_dir.mkdir(exist_ok=True)
    payload = json.loads((out / "metrics.json").read_text())
    pcr, box = load_models(out / "checkpoints", "cpu")
    cfg = SceneConfig(target_scr_db=8.0, n_targets=5)
    scene = render_scene(cfg, seed=2042)
    th = thresholds_for_scene(scene, pcr, box, cfg, "cpu")
    if "clouds" not in payload:
        print("collecting scene-wise operating points...")
        payload["clouds"] = collect_clouds(pcr, box)
        (out / "metrics.json").write_text(json.dumps(payload, indent=2))
    p1 = fig_dir / "fig1_detections.png"
    p2 = fig_dir / "fig2_operating.png"
    draw_figure1(scene, th, p1)
    draw_figure2(scene, th, p2, payload)
    (out / "fig1_detections.png").write_bytes(p1.read_bytes())
    (out / "fig2_operating.png").write_bytes(p2.read_bytes())
    (fig_dir / "fig2_pd_pfa.png").write_bytes(p2.read_bytes())
    (out / "fig2_pd_pfa.png").write_bytes(p2.read_bytes())
    print("wrote", p1)
    print("wrote", p2)


if __name__ == "__main__":
    main()
