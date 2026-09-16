# PCR-CFAR

Pfa-calibrated residual neural CFAR for small-target detection in cluttered UAV SAR. Companion code, weights, metrics, and figures for the *Computers & Electrical Engineering* technical communication.

## Manuscript

- Paper: [`paper/TECHNICAL_COMMUNICATION.md`](paper/TECHNICAL_COMMUNICATION.md)
- Highlights: [`paper/highlights.txt`](paper/highlights.txt)
- Figures: [`figures/fig1_detections.png`](figures/fig1_detections.png), [`figures/fig2_operating.png`](figures/fig2_operating.png)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

## Reproduce (one command)

Checkpoints are already in `results/checkpoints/`. The default path **does not retrain**; it evaluates and redraws the two paper figures.

```bash
python scripts/run_all.py
```

That writes:

- `results/metrics.json` (Tables 1–2)
- `figures/fig1_detections.png`, `figures/fig2_operating.png`
- copies under `paper/figures/` and `results/`

To retrain from scratch (overwrites weights):

```bash
python scripts/run_all.py --train
```

Figures only, using saved weights and metrics:

```bash
python scripts/make_figures.py
```
