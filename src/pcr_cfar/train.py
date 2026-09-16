"""Train PCR-CFAR and a matched-capacity black-box CNN."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .model import BlackBoxCfarNet, ResidualCfarNet
from .sar_scene import SceneConfig, render_scene, stack_channels


class SarDataset(Dataset):
    def __init__(self, cfg: SceneConfig, seeds: list[int], random_scr: bool = True):
        self.cfg = cfg
        self.seeds = seeds
        self.random_scr = random_scr

    def __len__(self) -> int:
        return len(self.seeds)

    def __getitem__(self, idx: int):
        cfg = SceneConfig(**self.cfg.__dict__)
        if self.random_scr:
            rng = np.random.default_rng(self.seeds[idx] + 17)
            cfg.target_scr_db = float(rng.uniform(5.0, 12.0))
        scene = render_scene(cfg, seed=self.seeds[idx], inject_targets=True)
        x = stack_channels(scene)
        return {
            "x": torch.from_numpy(x),
            "log_img": torch.from_numpy(scene["log_img"][None]),
            "img": torch.from_numpy(scene["img"][None]),
            "mask": torch.from_numpy(scene["mask"][None]),
            "t_ca": torch.from_numpy(scene["t_ca"][None]),
        }


def pcr_loss(
    log_t: torch.Tensor,
    log_img: torch.Tensor,
    mask: torch.Tensor,
    pfa_design: float,
    tau: float = 0.25,
) -> torch.Tensor:
    score = (log_img - log_t) / tau
    pos = mask > 0.5
    neg = mask < 0.5
    bce = nn.functional.binary_cross_entropy_with_logits
    # Heavy positive weight: targets occupy ~6 pixels in 192x256.
    w_pos = 80.0
    loss_det = 0.0
    if pos.any():
        loss_det = loss_det + w_pos * bce(score[pos], torch.ones_like(score[pos]))
    if neg.any():
        loss_det = loss_det + bce(score[neg], torch.zeros_like(score[neg]))
    soft_pfa = torch.sigmoid(score[neg]).mean() if neg.any() else score.new_tensor(pfa_design)
    loss_pfa = (torch.log(soft_pfa + 1e-8) - np.log(pfa_design)) ** 2
    return loss_det + 4.0 * loss_pfa


def train_models(out_dir: Path, device: str = "cpu", epochs: int = 18) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = SceneConfig()
    train_set = SarDataset(cfg, list(range(80)))
    val_set = SarDataset(cfg, list(range(800, 820)), random_scr=False)
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=4, shuffle=False)

    pcr = ResidualCfarNet().to(device)
    box = BlackBoxCfarNet().to(device)
    opt_pcr = torch.optim.Adam(pcr.parameters(), lr=1.5e-3)
    opt_box = torch.optim.Adam(box.parameters(), lr=1.5e-3)

    history = []
    for epoch in range(1, epochs + 1):
        pcr.train()
        box.train()
        running = {"pcr": 0.0, "box": 0.0, "n": 0}
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False):
            x = batch["x"].to(device)
            log_img = batch["log_img"].to(device)
            mask = batch["mask"].to(device)

            opt_pcr.zero_grad()
            log_t, _, _ = pcr(x)
            lp = pcr_loss(log_t, log_img, mask, cfg.pfa_design)
            lp.backward()
            opt_pcr.step()

            opt_box.zero_grad()
            log_tb = box(log_img)
            lb = pcr_loss(log_tb, log_img, mask, cfg.pfa_design)
            lb.backward()
            opt_box.step()

            running["pcr"] += float(lp.item())
            running["box"] += float(lb.item())
            running["n"] += 1

        pcr.eval()
        val_pfa = []
        val_pd = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                log_img = batch["log_img"].to(device)
                mask = batch["mask"].to(device)
                log_t, _, _ = pcr(x)
                det = log_img > log_t
                pos = mask > 0.5
                neg = mask < 0.5
                val_pd.append(float(det[pos].float().mean().cpu()) if pos.any() else 0.0)
                val_pfa.append(float(det[neg].float().mean().cpu()) if neg.any() else 0.0)
        rec = {
            "epoch": epoch,
            "train_pcr": running["pcr"] / max(running["n"], 1),
            "train_box": running["box"] / max(running["n"], 1),
            "val_pd": float(np.mean(val_pd)),
            "val_pfa": float(np.mean(val_pfa)),
        }
        history.append(rec)
        print(
            f"epoch {epoch:02d}  loss_pcr={rec['train_pcr']:.3f}  "
            f"val_pd={rec['val_pd']:.3f}  val_pfa={rec['val_pfa']:.4f}"
        )

    torch.save(pcr.state_dict(), out_dir / "pcr_cfar.pt")
    torch.save(box.state_dict(), out_dir / "blackbox.pt")
    return {"history": history, "pcr_params": pcr.count_parameters(), "box_params": box.count_parameters()}
