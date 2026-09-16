#!/usr/bin/env python3
"""Train, evaluate, and write figures/tables for the technical communication."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from pcr_cfar.evaluate import evaluate
from pcr_cfar.train import train_models


def main() -> None:
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    ckpt = out / "checkpoints"
    meta = train_models(ckpt, device=device, epochs=16)
    payload = evaluate(ckpt, out, device=device)
    payload["train"] = meta
    (out / "metrics.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["table1"], indent=2))
    print(json.dumps(payload["table2"], indent=2))
    print("wrote", out / "fig1_detections.png")
    print("wrote", out / "fig2_pd_pfa.png")


if __name__ == "__main__":
    main()
