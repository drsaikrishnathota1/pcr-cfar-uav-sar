# PCR-CFAR

Pfa-calibrated residual neural CFAR for small-target detection in cluttered UAV SAR. Companion code, weights, metrics, and figures for a *Computers & Electrical Engineering* research paper.

## Manuscript

- Editorial Manager pack (upload these files): [`paper/CAEE_Editorial_Manager/`](paper/CAEE_Editorial_Manager/)
- Working draft: [`paper/TECHNICAL_COMMUNICATION.md`](paper/TECHNICAL_COMMUNICATION.md)
- Highlights: [`paper/highlights.txt`](paper/highlights.txt)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

## Reproduce

Default: evaluate existing checkpoints and redraw all seven figures.

```bash
python scripts/run_all.py
```

Retrain (overwrites weights):

```bash
python scripts/run_all.py --train
```

Figures and tables only:

```bash
python scripts/make_full_paper.py
python scripts/build_submission.py
```
