#!/usr/bin/env python3
"""Train (if needed), evaluate, and write the two manuscript figures."""

from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
(ROOT / ".mplconfig").mkdir(exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import torch

from pcr_cfar.evaluate import evaluate
from pcr_cfar.train import train_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce PCR-CFAR metrics and figures.")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Retrain even if checkpoints already exist (overwrites weights).",
    )
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out = ROOT / "results"
    ckpt = out / "checkpoints"
    out.mkdir(exist_ok=True)
    have_weights = (ckpt / "pcr_cfar.pt").is_file() and (ckpt / "blackbox.pt").is_file()

    if args.train or not have_weights:
        print(f"training on {args.device} for {args.epochs} epochs")
        meta = train_models(ckpt, device=args.device, epochs=args.epochs)
    else:
        print(f"using existing checkpoints in {ckpt} (pass --train to retrain)")
        meta = {"skipped_train": True}

    print("evaluating...")
    payload = evaluate(ckpt, out, device=args.device)
    payload["train"] = meta
    (out / "metrics.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["table1"], indent=2))
    print(json.dumps(payload["table2"], indent=2))

    print("drawing manuscript figures...")
    runpy.run_path(str(ROOT / "scripts" / "make_figures.py"), run_name="__main__")


if __name__ == "__main__":
    main()
