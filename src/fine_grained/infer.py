"""infer.py — RMN-vs-Foreign inference on a single military-vessel crop.

Meant to run downstream of the main detector: given a crop already classified
as `military_vessel` (class 7) by the frozen `predict()` interface, this
2nd-stage classifier labels it {malaysian_rmn, foreign, unknown}. "unknown" is
returned whenever the softmax confidence falls below `infer.conf_floor` —
better an honest "don't know" than a confident wrong nationality call.

    python -m src.fine_grained.infer --weights models/fine_grained_rmn_classifier.pt \\
        --image path/to/crop.jpg
    python -m src.fine_grained.infer --weights models/fine_grained_rmn_classifier.pt \\
        --image path/to/crop_dir/ --conf-floor 0.6

torch/torchvision are imported lazily so importing this module for wiring
(e.g. from app.py) doesn't require the heavy stack until inference is invoked.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

UNKNOWN = "unknown"


@dataclass(frozen=True)
class ClassificationResult:
    label: str
    confidence: float


def load_checkpoint(weights_path: Path | str):
    import torch

    from src.fine_grained.train_classifier import build_model

    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["backbone"], len(ckpt["classes"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["classes"], ckpt["imgsz"]


def build_transform(imgsz: int):
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize(int(imgsz * 1.15)),
        transforms.CenterCrop(imgsz),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def classify(image, model, transform, classes: list[str], conf_floor: float) -> ClassificationResult:
    """Classify one PIL RGB `image`. Returns "unknown" below `conf_floor`."""
    import torch

    with torch.no_grad():
        x = transform(image.convert("RGB")).unsqueeze(0)
        probs = torch.softmax(model(x), dim=1)[0]
        conf, idx = probs.max(dim=0)
        conf = float(conf)
        label = classes[int(idx)] if conf >= conf_floor else UNKNOWN
    return ClassificationResult(label=label, confidence=conf)


def _iter_images(path: Path):
    if path.is_dir():
        yield from sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    else:
        yield path


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(
        prog="python -m src.fine_grained.infer",
        description="RMN-vs-Foreign nationality classification on a military-vessel crop.",
    )
    p.add_argument("--weights", required=True, type=Path)
    p.add_argument("--image", required=True, type=Path,
                   help="a single crop image, or a directory of crops")
    p.add_argument("--conf-floor", type=float, default=0.55,
                   help="softmax confidence below this -> unknown")
    args = p.parse_args(argv)

    from PIL import Image

    model, classes, imgsz = load_checkpoint(args.weights)
    transform = build_transform(imgsz)

    for img_path in _iter_images(args.image):
        with Image.open(img_path) as img:
            result = classify(img, model, transform, classes, args.conf_floor)
        print(json.dumps({
            "image": str(img_path),
            "label": result.label,
            "confidence": round(result.confidence, 4),
        }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
