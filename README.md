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
PYTHONPATH=src pytest
```

## Reproduce

```bash
PYTHONPATH=src python scripts/run_all.py      # train, evaluate, write metrics
PYTHONPATH=src python scripts/make_figures.py # redraw Fig. 1 and Fig. 2
```

Checkpoints: `results/checkpoints/pcr_cfar.pt`, `results/checkpoints/blackbox.pt`.  
Metrics: `results/metrics.json`.
