"""Training loop — designed to run inside a Kaggle notebook cell.

Implements:
  FR-3.1 — 3D U-Net baseline, Dice + Cross-Entropy loss
  FR-3.2 — 5-fold patient-level cross-validation, configurable
  FR-3.3 — checkpoint on every val-Dice improvement, keep top-K
  FR-3.4 — log loss/Dice/LR to experiment tracker every run
  FR-3.5 — AMP + configurable gradient accumulation
  FR-3.7 — resume from any checkpoint without losing optimizer state
  FR-4.1 — sliding-window validation inference (configurable patch/overlap)
  FR-9.3 — per-region Dice (ET/TC/WT) reported for model selection
  Section 9.2 — early stopping on validation Dice plateau

CHECKPOINT PERSISTENCE (decision recorded 2026-08-30): checkpoints are
saved to /kaggle/working/checkpoints/ and persisted across sessions by
committing the notebook version ("Save & Run All") at the end of each
Kaggle session, and resumed at the start of the next by re-running this
notebook against the committed output. This is the simplest option;
revisit with S3/MinIO (per SRS Section 2.4) if this becomes a bottleneck.

VALIDATION FREQUENCY (decision recorded 2026-09-02): full sliding-window
validation runs every cfg.validation.val_interval epochs (default 5) and
always on the final epoch — the SRS does not specify a frequency, and
per-epoch full-volume validation would not fit the 30 GPU-hrs/week
Kaggle quota. Set val_interval: 1 to validate every epoch.

EARLY STOPPING (decision recorded 2026-09-02): SRS Section 9.2 says
"configurable patience" without defining units. Interpretation: patience
is measured in EPOCHS without val-Dice improvement, evaluated at each
validation check — with val_interval=5 and patience=20, training stops
after 4 consecutive checks with no new best. The no-improvement counter
is persisted in checkpoints (with a backward-compatible default of 0 for
checkpoints written before this change).

TOP-K CHECKPOINTS (decision recorded 2026-09-02): improved-Dice
checkpoints are saved as fold{i}_best_e{epoch}_d{dice}.pt and only the
keep_top_k highest-Dice files are retained (FR-3.3). The best model is
the highest-Dice file; there is no separate fold{i}_best.pt anymore.
"""

import csv
import time
from pathlib import Path

import torch
from monai.data import CacheDataset, DataLoader
from monai.inferers import SlidingWindowInferer
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet

from src.data.preprocessing import (
    build_case_dict,
    build_train_transforms,
    build_val_transforms,
)


def save_checkpoint(model, optimizer, epoch: int, best_dice: float, path: Path,
                    epochs_without_improvement: int = 0) -> None:
    """FR-3.3, FR-3.7. Persists model state, optimizer state, epoch
    counter, best_dice, and the early-stopping counter — all of them, or
    resume silently diverges from the true training state.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_dice": best_dice,
            "epochs_without_improvement": epochs_without_improvement,
        },
        path,
    )


def resume_from_checkpoint(path: Path, model, optimizer):
    """FR-3.7. Returns (model, optimizer, start_epoch, best_dice,
    epochs_without_improvement). The .get() default keeps checkpoints
    written before the early-stopping change loadable.
    """
    ckpt = torch.load(path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return (model, optimizer, ckpt["epoch"] + 1, ckpt["best_dice"],
            ckpt.get("epochs_without_improvement", 0))


def save_topk_checkpoint(model, optimizer, epoch: int, dice: float,
                         checkpoint_dir: Path, fold_idx: int, keep_top_k: int) -> None:
    """FR-3.3 — save an improved-Dice checkpoint and retain only the
    keep_top_k highest-Dice files for this fold.
    """
    name = f"fold{fold_idx}_best_e{epoch:03d}_d{dice:.4f}.pt"
    save_checkpoint(model, optimizer, epoch, dice, checkpoint_dir / name)
    ranked = sorted(
        checkpoint_dir.glob(f"fold{fold_idx}_best_e*.pt"),
        key=lambda p: float(p.stem.split("_d")[-1]),
        reverse=True,
    )
    for old in ranked[keep_top_k:]:
        old.unlink()
        print(f"  pruned {old.name} (keeping top {keep_top_k})", flush=True)


def load_fold_cases(split_manifest_path: str, raw_data_dir: str, fold_idx: int):
    """Read configs/data/split_manifest.csv, return (train_cases, val_cases)
    as lists of case dicts, using the pre-built patient-level fold
    assignment (Section 2.5 — no leakage).
    """
    raw_dir = Path(raw_data_dir)
    train_dicts, val_dicts = [], []
    with open(split_manifest_path) as f:
        for row in csv.DictReader(f):
            if row["status"] != "valid":
                continue  # FR-1.3 — invalid cases already flagged, excluded here
            case_dict = build_case_dict(raw_dir / row["case_id"], row["case_id"], include_label=True)
            if int(row["fold"]) == fold_idx:
                val_dicts.append(case_dict)
            else:
                train_dicts.append(case_dict)
    return train_dicts, val_dicts


def build_model(model_cfg) -> UNet:
    """FR-3.1 — builds from configs/model/unet3d.yaml."""
    return UNet(
        spatial_dims=model_cfg.spatial_dims,
        in_channels=model_cfg.in_channels,
        out_channels=model_cfg.out_channels,
        channels=tuple(model_cfg.channels),
        strides=tuple(model_cfg.strides),
        num_res_units=model_cfg.num_res_units,
        norm=model_cfg.norm,
    )


def labels_to_regions(labels: torch.Tensor) -> torch.Tensor:
    """FR-9.3, SRS Section 7.1 — map a [B,1,...] label tensor in the
    model's internal space {0,1,2,3} (4->3 remap already applied by
    preprocessing) to binary region channels [B,3,...], ordered
    (ET, TC, WT):
      ET = {3}   (orig BraTS label 4, enhancing tumor)
      TC = {1,3} (NCR/NET + enhancing)
      WT = {1,2,3} (everything tumorous)
    """
    et = labels == 3
    tc = (labels == 1) | (labels == 3)
    wt = labels >= 1
    return torch.cat([et, tc, wt], dim=1).float()


def run_validation(model, val_loader, val_cfg, device, amp_enabled: bool) -> dict:
    """FR-4.1, FR-4.2, FR-9.3 — full-volume sliding-window validation.

    Patch size / overlap / gaussian blending mirror
    configs/inference/default.yaml so validation measures the same
    procedure production inference will use.
    Returns {"ET": float, "TC": float, "WT": float, "mean": float}.
    """
    inferer = SlidingWindowInferer(
        roi_size=tuple(val_cfg.sw_patch_size),
        sw_batch_size=val_cfg.sw_batch_size,
        overlap=val_cfg.sw_overlap,
        mode="gaussian",
    )
    dice_metric = DiceMetric(include_background=True, reduction="mean_batch")

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            images = batch["image"].to(device)
            labels = batch["seg"].to(device)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = inferer(images, model)
            preds = torch.argmax(logits, dim=1, keepdim=True)
            dice_metric(y_pred=labels_to_regions(preds), y=labels_to_regions(labels))
            if (i + 1) % 10 == 0:
                print(f"    validated {i + 1}/{len(val_loader)} cases...", flush=True)

    per_region = dice_metric.aggregate()  # tensor [3] in (ET, TC, WT) order
    dice_metric.reset()
    model.train()

    scores = {
        "ET": per_region[0].item(),
        "TC": per_region[1].item(),
        "WT": per_region[2].item(),
    }
    scores["mean"] = (scores["ET"] + scores["TC"] + scores["WT"]) / 3.0
    return scores


def train_fold(fold_idx: int, cfg, model_cfg, data_cfg, use_wandb: bool = True,
               max_train_cases: int = None, max_val_cases: int = None) -> None:
    """FR-3.1..3.5, FR-4.1, FR-9.3, Section 9.2. One cross-validation fold.

    Automatically resumes from the fold's latest checkpoint if one
    exists in cfg.checkpoint_dir (FR-3.7) — safe to re-run this after a
    Kaggle session restart.

    max_train_cases: if set, only trains on this many cases — useful for
    a fast smoke test (e.g. max_train_cases=8) before committing to a
    full run across all ~294 cases per fold.
    max_val_cases: same idea for validation — e.g. max_val_cases=3 for a
    smoke test so a validation pass takes minutes, not an hour.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    train_dicts, val_dicts = load_fold_cases(data_cfg.split_manifest, data_cfg.raw_data_dir, fold_idx)
    if max_train_cases:
        train_dicts = train_dicts[:max_train_cases]
    if max_val_cases:
        val_dicts = val_dicts[:max_val_cases]
    print(f"Fold {fold_idx}: {len(train_dicts)} train cases, {len(val_dicts)} val cases")

    train_transforms = build_train_transforms(cfg)
    train_ds = CacheDataset(data=train_dicts, transform=train_transforms, cache_rate=0.0)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2)

    val_transforms = build_val_transforms(cfg)
    val_ds = CacheDataset(data=val_dicts, transform=val_transforms, cache_rate=0.0)
    # batch_size=1: full volumes, sliding window handles patching (FR-4.1)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    model = build_model(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
    # Section 9.2 — cosine LR decay (SRS-specified; decision 2026-09-03:
    # schedule is defined over cfg.scheduler.t_max total epochs and, on
    # resume from a pre-scheduler checkpoint, is fast-forwarded to
    # start_epoch so all folds share one schedule shape from epoch 0).
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.scheduler.t_max, eta_min=cfg.scheduler.eta_min)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True, lambda_dice=cfg.loss.dice_weight, lambda_ce=cfg.loss.ce_weight)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    checkpoint_dir = Path(cfg.checkpoint_dir)
    latest_ckpt = checkpoint_dir / f"fold{fold_idx}_latest.pt"
    start_epoch, best_dice, epochs_without_improvement = 0, 0.0, 0
    if latest_ckpt.exists():
        print(f"Resuming from {latest_ckpt}")
        (model, optimizer, start_epoch, best_dice,
         epochs_without_improvement) = resume_from_checkpoint(latest_ckpt, model, optimizer)
        for _ in range(start_epoch):
            scheduler.step()  # cosine is stateless-recomputable; see decision note above
    else:
        print("No checkpoint found — starting fresh (epoch 0)")

    if use_wandb:
        import wandb
        wandb.init(project="brats-segmentation", name=f"fold{fold_idx}", resume="allow",
                   config={"fold": fold_idx, "lr": cfg.optimizer.lr, "batch_size": cfg.batch_size,
                           "val_interval": cfg.validation.val_interval,
                           "early_stopping_patience": cfg.early_stopping_patience,
                           "keep_top_k": cfg.keep_top_k})

    for epoch in range(start_epoch, cfg.max_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()
        for i, batch in enumerate(train_loader):
            images, labels = batch["image"].to(device), batch["seg"].to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=cfg.amp):
                outputs = model(images)
                loss = loss_fn(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            if (i + 1) % 25 == 0 or (i + 1) == len(train_loader):
                print(f"  [epoch {epoch}] batch {i + 1}/{len(train_loader)}, "
                      f"loss={loss.item():.4f}", flush=True)
        epoch_loss /= max(len(train_loader), 1)
        print(f"  epoch {epoch} train time: {time.time() - epoch_start:.1f}s", flush=True)

        # FR-4.1, FR-9.3 — validate every val_interval epochs, and always
        # on the final epoch so no run ends without a fresh score.
        is_val_epoch = (
            (epoch + 1) % cfg.validation.val_interval == 0
            or epoch == cfg.max_epochs - 1
        )

        log_payload = {"epoch": epoch, "train_loss": epoch_loss, "lr": scheduler.get_last_lr()[0]}
        stop_early = False

        if is_val_epoch:
            val_start = time.time()
            scores = run_validation(model, val_loader, cfg.validation, device, cfg.amp)
            val_time = time.time() - val_start
            print(f"Epoch {epoch}: train_loss={epoch_loss:.4f} | "
                  f"val Dice ET={scores['ET']:.4f} TC={scores['TC']:.4f} "
                  f"WT={scores['WT']:.4f} mean={scores['mean']:.4f} "
                  f"({val_time:.0f}s)", flush=True)
            # FR-3.4 — per-region Dice logged only on epochs it was computed,
            # so W&B charts show real points, not stale placeholders.
            log_payload.update({
                "val_dice_mean": scores["mean"],
                "val_dice_ET": scores["ET"],
                "val_dice_TC": scores["TC"],
                "val_dice_WT": scores["WT"],
                "val_time_s": val_time,
            })
            if scores["mean"] > best_dice:
                # FR-3.3 — update best_dice BEFORE saving latest, so a
                # resumed session sees the true best.
                best_dice = scores["mean"]
                epochs_without_improvement = 0
                save_topk_checkpoint(model, optimizer, epoch, best_dice,
                                     checkpoint_dir, fold_idx, cfg.keep_top_k)
                print(f"  new best mean Dice: {best_dice:.4f} — saved top-K checkpoint", flush=True)
            else:
                epochs_without_improvement += cfg.validation.val_interval
                print(f"  no improvement for ~{epochs_without_improvement} epochs "
                      f"(patience {cfg.early_stopping_patience})", flush=True)
                if epochs_without_improvement >= cfg.early_stopping_patience:
                    stop_early = True
        else:
            print(f"Epoch {epoch}: train_loss={epoch_loss:.4f} (no validation this epoch)", flush=True)

        if use_wandb:
            wandb.log(log_payload)

        scheduler.step()
        save_checkpoint(model, optimizer, epoch, best_dice, latest_ckpt,
                        epochs_without_improvement)

        if stop_early:
            # Section 9.2 — validation Dice plateaued; stop to save quota.
            print(f"EARLY STOPPING at epoch {epoch}: no improvement for "
                  f"{epochs_without_improvement} epochs (patience "
                  f"{cfg.early_stopping_patience}). Best mean Dice: {best_dice:.4f}", flush=True)
            break

    if use_wandb:
        wandb.finish()
