"""Training loop — designed to run inside a Kaggle notebook cell.

Implements:
  FR-3.1 — 3D U-Net baseline, Dice + Cross-Entropy loss
  FR-3.2 — 5-fold patient-level cross-validation, configurable
  FR-3.3 — checkpoint on every val-Dice improvement, keep top-K
  FR-3.4 — log loss/Dice/LR to experiment tracker every run
  FR-3.5 — AMP + configurable gradient accumulation
  FR-3.7 — resume from any checkpoint without losing optimizer state

KAGGLE SESSION-CAP NOTE (Section 2.5, Risk table Section 13):
  checkpoint_every_epoch must actually run BEFORE the 12-hr wall clock,
  not just be configured true. Structure this as:
    for epoch in range(start_epoch, max_epochs):
        train_one_epoch(...)
        if val_dice_improved:
            save_checkpoint(model, optimizer, epoch, path=...)   # every time
  and provide a resume_from_checkpoint(path) entry point that restores
  model + optimizer + epoch counter, callable at the top of a fresh
  notebook session.
"""

from pathlib import Path


def save_checkpoint(model, optimizer, epoch: int, path: Path) -> None:
    """FR-3.3, FR-3.7. Must persist model state, optimizer state, and
    epoch/step counter — all three, or resume will silently diverge from
    the true training state.
    """
    raise NotImplementedError


def resume_from_checkpoint(path: Path):
    """FR-3.7. Returns (model, optimizer, start_epoch)."""
    raise NotImplementedError


def train_fold(fold_idx: int, cfg) -> None:
    """FR-3.1, FR-3.2, FR-3.4, FR-3.5. One cross-validation fold.

    Call resume_from_checkpoint() first if a checkpoint for this fold
    already exists in cfg.checkpoint_dir — do not assume a fresh start
    at the top of every Kaggle session.
    """
    raise NotImplementedError
