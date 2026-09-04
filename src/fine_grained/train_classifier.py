"""train_classifier.py — fine-tune a small torchvision backbone for the
RMN-vs-Foreign 2nd-stage classifier (bonus track).

Trains on the flat per-class crop folders produced by `src.fine_grained.prep`
(`processed_dir/{train,val}/{malaysian_rmn,foreign}/`). Entirely separate from
the main YOLO detector — this never touches configs/schema.yaml or the
detector's data/models.

    python -m src.fine_grained.train_classifier --config configs/fine_grained.yaml
    python -m src.fine_grained.train_classifier --config configs/fine_grained.yaml --smoke
    python -m src.fine_grained.train_classifier --config configs/fine_grained.yaml --dry-run

torch/torchvision are imported lazily (inside functions), so --dry-run works
without them, matching src/train/train.py's convention.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# src/fine_grained/train_classifier.py -> parents[2] is the repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "fine_grained.yaml"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _repo_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    if not isinstance(config, dict):
        raise SystemExit(f"config {path} did not parse to a mapping")
    return config


_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def _count_images(split_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not split_dir.is_dir():
        return counts
    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        counts[class_dir.name] = sum(
            1 for f in class_dir.iterdir() if f.suffix.lower() in _IMAGE_EXTS
        )
    return counts


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(backbone: str, num_classes: int):
    import torch.nn as nn
    from torchvision import models

    if backbone == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(f"unknown backbone {backbone!r}; expected resnet18 | efficientnet_b0")
    return model


def build_transforms(imgsz: int):
    from torchvision import transforms

    # Small dataset (a few hundred RMN images) -> lean heavily on augmentation.
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(imgsz, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.25),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(imgsz * 1.15)),
        transforms.CenterCrop(imgsz),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, val_tf


def class_weights(targets: list[int], num_classes: int):
    import torch

    counts = [max(1, targets.count(c)) for c in range(num_classes)]
    total = sum(counts)
    weights = [total / (num_classes * c) for c in counts]
    return torch.tensor(weights, dtype=torch.float32)


def subset_smoke(dataset, per_class: int = 8):
    """Stratified small subset of an ImageFolder dataset, for --smoke runs."""
    from collections import defaultdict

    from torch.utils.data import Subset

    by_class: dict[int, list[int]] = defaultdict(list)
    for idx, (_, target) in enumerate(dataset.samples):
        by_class[target].append(idx)
    indices = [i for idxs in by_class.values() for i in idxs[:per_class]]
    return Subset(dataset, indices)


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------
def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    import torch

    model.train(mode=train)
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_targets = [], []

    with torch.set_grad_enabled(train):
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            preds = outputs.argmax(dim=1)
            total_loss += loss.item() * images.size(0)
            correct += (preds == targets).sum().item()
            total += images.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())

    return {
        "loss": total_loss / max(1, total),
        "acc": correct / max(1, total),
        "preds": all_preds,
        "targets": all_targets,
    }


def confusion_and_per_class(targets: list[int], preds: list[int], classes: list[str]) -> str:
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(targets, preds, labels=list(range(len(classes))))
    lines = ["confusion matrix (rows=true, cols=pred):", "        " + "  ".join(f"{c[:10]:>10}" for c in classes)]
    for i, row in enumerate(cm):
        lines.append(f"{classes[i][:8]:>8} " + "  ".join(f"{v:>10}" for v in row))
    lines.append("per-class accuracy:")
    for i, c in enumerate(classes):
        n = cm[i].sum()
        acc = cm[i, i] / n if n else float("nan")
        lines.append(f"  {c:<16} {acc:.3f}  (n={n})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run(config: dict, smoke: bool = False, dry_run: bool = False) -> int:
    processed_dir = _repo_path(config["processed_dir"])
    train_dir = processed_dir / "train"
    val_dir = processed_dir / "val"

    train_counts = _count_images(train_dir)
    val_counts = _count_images(val_dir)
    logger.info("=== crop counts ===")
    logger.info("  train: %s (total %d)", train_counts, sum(train_counts.values()))
    logger.info("  val:   %s (total %d)", val_counts, sum(val_counts.values()))
    if not train_counts or not val_counts:
        raise SystemExit(
            f"no crops found under {processed_dir} — run "
            f"`python -m src.fine_grained.prep --config <this config>` first"
        )

    train_cfg = config.get("train", {})
    backbone = train_cfg.get("backbone", "resnet18")
    imgsz = train_cfg.get("imgsz", 224)
    epochs = 1 if smoke else train_cfg.get("epochs", 25)
    batch_size = train_cfg.get("batch_size", 32)
    lr = train_cfg.get("lr", 3e-4)
    weight_decay = train_cfg.get("weight_decay", 1e-4)
    num_workers = 0 if smoke else train_cfg.get("num_workers", 4)
    patience = train_cfg.get("patience", 7)
    weights_out = _repo_path(train_cfg.get("weights_out", "models/fine_grained_rmn_classifier.pt"))

    logger.info("=== resolved train config ===")
    for k, v in {
        "backbone": backbone, "imgsz": imgsz, "epochs": epochs, "batch_size": batch_size,
        "lr": lr, "weight_decay": weight_decay, "patience": patience,
        "weights_out": str(weights_out), "smoke": smoke,
    }.items():
        logger.info("  %-14s %s", k, v)

    if dry_run:
        logger.info("--dry-run: config and crop counts validated; not training.")
        return 0

    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device: %s", device)

    train_tf, val_tf = build_transforms(imgsz)
    train_ds = datasets.ImageFolder(str(train_dir), transform=train_tf)
    val_ds = datasets.ImageFolder(str(val_dir), transform=val_tf)
    classes = train_ds.classes  # alphabetical: ['foreign', 'malaysian_rmn']
    if val_ds.classes != classes:
        raise SystemExit(f"train/val class mismatch: {classes} vs {val_ds.classes}")
    logger.info("classes: %s", classes)

    if smoke:
        train_ds_used = subset_smoke(train_ds, per_class=8)
        val_ds_used = subset_smoke(val_ds, per_class=4)
    else:
        train_ds_used, val_ds_used = train_ds, val_ds

    train_loader = DataLoader(train_ds_used, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds_used, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=(device.type == "cuda"))

    targets = [t for _, t in train_ds.samples]
    weights = class_weights(targets, len(classes)).to(device)
    logger.info("class weights (inverse-frequency, index order=%s): %s", classes, weights.tolist())

    model = build_model(backbone, len(classes)).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    best_val_acc = -1.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        train_result = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_result = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()
        logger.info(
            "epoch %3d/%d  train_loss=%.4f train_acc=%.3f  val_loss=%.4f val_acc=%.3f",
            epoch, epochs, train_result["loss"], train_result["acc"],
            val_result["loss"], val_result["acc"],
        )

        if val_result["acc"] > best_val_acc:
            best_val_acc = val_result["acc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if not smoke and epochs_no_improve >= patience:
                logger.info("early stopping: no val improvement for %d epochs", patience)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final confusion matrix / per-class accuracy on val, at the best checkpoint.
    final_val = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
    logger.info("=== FINAL val_acc=%.3f (best during training: %.3f) ===",
                final_val["acc"], best_val_acc)
    logger.info("\n%s", confusion_and_per_class(final_val["targets"], final_val["preds"], classes))

    weights_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "backbone": backbone,
        "classes": classes,
        "imgsz": imgsz,
    }, weights_out)

    md5 = hashlib.md5(weights_out.read_bytes()).hexdigest()
    logger.info("saved weights: %s", weights_out)
    logger.info("MD5: %s", md5)
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(
        prog="python -m src.fine_grained.train_classifier",
        description="Fine-tune a torchvision backbone for RMN-vs-Foreign classification.",
    )
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    p.add_argument("--smoke", action="store_true",
                   help="tiny stratified subset, 1 epoch — verify the pipeline runs end-to-end")
    p.add_argument("--dry-run", action="store_true",
                   help="validate config + crop counts and exit without training")
    args = p.parse_args(argv)

    config = load_config(args.config)
    return run(config, smoke=args.smoke, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
