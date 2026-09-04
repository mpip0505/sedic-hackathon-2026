"""Create per-class flat crops for the fine-grained RMN vs foreign classifier.

Usage (module):
  python -m src.fine_grained.prep_crops --out data/fg_crops --limit 100

This script will:
- Copy all images under `data/raw/fine_grained/malaysian_rmn/**` into
  OUT/malaysian_rmn/ (flattened)
- For the `foreign` Roboflow YOLO detection format (train/valid/test), read
  corresponding label `.txt` files and crop each box out into OUT/foreign/.
"""
from pathlib import Path
import shutil
from PIL import Image
import argparse
import math


def yolo_to_bbox(yolo_vals, img_w, img_h):
    # yolo_vals: [x_center, y_center, w, h] normalized
    x_ctr, y_ctr, w, h = map(float, yolo_vals)
    bw = w * img_w
    bh = h * img_h
    x1 = int(max(0, x_ctr * img_w - bw / 2))
    y1 = int(max(0, y_ctr * img_h - bh / 2))
    x2 = int(min(img_w, x_ctr * img_w + bw / 2))
    y2 = int(min(img_h, y_ctr * img_h + bh / 2))
    return x1, y1, x2, y2


def copy_malaysian(src_root: Path, out_root: Path, limit: int | None = None):
    src_root = Path(src_root)
    dest = out_root / "malaysian_rmn"
    dest.mkdir(parents=True, exist_ok=True)
    copied = 0
    for p in src_root.rglob('*'):
        if p.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
            continue
        shutil.copy(p, dest / p.name)
        copied += 1
        if limit and copied >= limit:
            break
    return copied


def crop_foreign(foreign_root: Path, out_root: Path, limit_per_split: int | None = None):
    foreign_root = Path(foreign_root)
    dest = out_root / "foreign"
    dest.mkdir(parents=True, exist_ok=True)
    copied = 0
    # Expect splits: train, valid, test
    for split in ['train', 'valid', 'test']:
        images_dir = foreign_root / split / 'images'
        labels_dir = foreign_root / split / 'labels'
        if not images_dir.exists():
            continue
        for img_path in images_dir.glob('*'):
            if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                continue
            label_path = labels_dir / (img_path.stem + '.txt')
            if not label_path.exists():
                # no boxes for this image
                continue
            try:
                img = Image.open(img_path).convert('RGB')
            except Exception:
                continue
            w, h = img.size
            with open(label_path, 'r') as fh:
                for i, line in enumerate(fh):
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    # ignore class id (parts[0]) and take box
                    bbox = yolo_to_bbox(parts[1:5], w, h)
                    x1, y1, x2, y2 = bbox
                    if x2 <= x1 or y2 <= y1:
                        continue
                    crop = img.crop((x1, y1, x2, y2))
                    out_name = f"{img_path.stem}_{i}{img_path.suffix}"
                    crop.save(dest / out_name)
                    copied += 1
                    if limit_per_split and copied >= limit_per_split:
                        return copied
    return copied


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=Path('data/fg_crops'))
    p.add_argument('--raw', type=Path, default=Path('data/raw/fine_grained'))
    p.add_argument('--limit', type=int, default=0, help='Max images to copy per class (0=all)')
    args = p.parse_args()

    out = args.out
    if out.exists():
        # do not erase by default; create subfolders as needed
        out.mkdir(parents=True, exist_ok=True)
    else:
        out.mkdir(parents=True, exist_ok=True)

    mal_src = args.raw / 'malaysian_rmn'
    foreign_src = args.raw / 'foreign'

    lim = args.limit if args.limit > 0 else None
    print('Copying malaysian_rmn...')
    m = copy_malaysian(mal_src, out, lim)
    print(f'Copied {m} malaysian_rmn images into {out / "malaysian_rmn"}')

    print('Cropping foreign detections...')
    f = crop_foreign(foreign_src, out, lim)
    print(f'Created {f} foreign crops into {out / "foreign"}')


if __name__ == '__main__':
    main()
