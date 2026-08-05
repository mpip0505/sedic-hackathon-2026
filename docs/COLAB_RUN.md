# Running the baseline on Google Colab

Mac-primary workflow: everything below runs **inside a Colab notebook**, nothing on
your local machine. We **regenerate `data/processed/` in Colab** (reproducible, no
large uploads from your Mac) and **save weights back to Google Drive** so they survive
disconnect.

> ⚠️ **Colab wipes all local disk on disconnect.** The only things that persist are
> what you copy to Google Drive. Do the Drive-save step (§6) — it is not optional.

Nothing here restructures the repo. It drives the existing
`src/train/train.py` + `configs/train_baseline.yaml` and the existing data pipeline.

---

## 0. Set the runtime to GPU

**Runtime → Change runtime type → Hardware accelerator → GPU (T4)** → Save.
Do this *before* running anything, or torch installs CPU-only.

## 1. Confirm the GPU

```python
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE — set Runtime > Change runtime type > GPU (T4)")
!nvidia-smi
```

Expect `CUDA available: True` and `Tesla T4` (~15 GB). If it says NONE, fix the runtime
type (§0) and re-run — do not continue on CPU.

## 2. Clone the repo

```python
!git clone https://github.com/mpip0505/sedic-hackathon-2026.git
%cd sedic-hackathon-2026
```

## 3. Install dependencies

`requirements.txt` pins `torch==2.5.1` / `ultralytics==8.3.40`. Colab already ships a
CUDA torch; installing the pinned set keeps us reproducible and CUDA-enabled on T4.

```python
!pip install -q -r requirements.txt
# sanity: torch still sees the GPU after the install
import torch; print("CUDA:", torch.cuda.is_available(), "|", torch.__version__)
```

If `torch.cuda.is_available()` flips to `False` after this, restart the runtime
(**Runtime → Restart session**), re-run §1–§3, and it will pick up the CUDA build.

## 4. Get `data/processed/` onto Colab

### Recommended — regenerate it in Colab (reproducible, no big uploads)

Put your Roboflow key in a **Colab secret** (🔑 left sidebar → *Secrets* → add
`ROBOFLOW_API_KEY`, toggle notebook access), then:

> ⚠️ **The cells below reproduce the older three-dataset build.** The build behind
> `models/baseline_best.pt` also includes `military_surface` (add
> `scripts/download_military_surface.py` + its `yolo2yolo --dataset military_surface`)
> and runs step 4d with `--no-cross-domain`, since real surface-military data made the
> `surface_synth` stopgap unnecessary. `civilian_gapfill` (downloaded 2026-08-05) is
> **not** in any build — it has no `schema.yaml` mapping yet. See `data/DATASETS.md`.

```python
import os
from google.colab import userdata
os.environ["ROBOFLOW_API_KEY"] = userdata.get("ROBOFLOW_API_KEY")

# 4a. download the three Roboflow YOLO sets → data/raw/
!python scripts/download_military_ships.py
!python scripts/download_seaships.py
!python scripts/download_shiprsimagenet.py

# 4b. remap each set → data/interim/ (schema class IDs, polygons → HBB)
!python -m src.data.converters.yolo2yolo --dataset seaships
!python -m src.data.converters.yolo2yolo --dataset military_ships
!python -m src.data.converters.yolo2yolo --dataset shiprsimagenet

# 4c. merge → greedy dedup → stratified class×domain split → data/processed/
#     (also regenerates configs/data.yaml)
!python -m src.data.merge

# 4d. surface-military stopgap: cross-domain copy-paste, TRAIN split only
!python -m src.data.balance --clean

# 4e. sanity-check labels + train/val/test leakage (exits nonzero on failure)
!python -m src.data.validate
```

This reproduces the current build: **~10,572 real images kept**, split
train ≈ 7,403 / val ≈ 2,114 / test ≈ 1,055, plus ~380 `surface_synth` images added to
**train only** by step 4d (→ ~7,783 train images). Seed 42 makes it deterministic.

### Fallback — upload a zipped `data/processed/` via Google Drive

If you'd rather not re-download (e.g. Roboflow rate limits), zip `data/processed/`
locally, drop it in Drive, then in Colab:

```python
from google.colab import drive
drive.mount("/content/drive")
!unzip -q "/content/drive/MyDrive/guardian/data_processed.zip" -d data/
# expect data/processed/{images,labels}/{train,val,test} + manifest.csv + domains.json
!python -m src.data.validate           # re-verify after unzip
```

`configs/data.yaml` points at `data/processed`, so once either path is done the trainer
finds the data with no edits.

## 5. Dry-run, then launch the real run

Always dry-run first — it validates schema/data consistency and confirms the
zero-military abort won't fire (prints per-split + per-class counts):

```python
!python -m src.train.train --config configs/train_baseline.yaml --dry-run
```

You should see `military instances in train split: ~19,531` and
`--dry-run: config and data validated; not training.` Then the real run:

```python
!python -m src.train.train --config configs/train_baseline.yaml
```

This trains `yolo11m` for 100 epochs @ 640, copies best weights to
`models/baseline_best.pt`, and scores the **military-recall gate on the held-out
TEST split** (`conf_military = 0.10`). Outputs land in `outputs/runs/baseline/`.

**Resuming** after a disconnect (see §6 for keeping checkpoints on Drive):

```python
!python -m src.train.train --config configs/train_baseline.yaml --resume
```

## 6. Save outputs to Google Drive (critical — do this or you lose everything)

Mount Drive once, then copy the weights and the whole run folder out. The run folder
holds `weights/last.pt` + `best.pt`, `results.csv`, and the PR/confusion plots you'll
want for the technical brief.

```python
from google.colab import drive
drive.mount("/content/drive")

import shutil, pathlib
dest = pathlib.Path("/content/drive/MyDrive/guardian/runs")
dest.mkdir(parents=True, exist_ok=True)

# the full run folder (checkpoints, results.csv, plots)
shutil.copytree("outputs/runs/baseline", dest / "baseline", dirs_exist_ok=True)
# the promoted best weights the eval/inference path expects
shutil.copy2("models/baseline_best.pt", dest / "baseline_best.pt")
print("saved to", dest)
```

**Best practice for long runs:** mount Drive *before* training and point the run
straight at it so checkpoints stream to Drive as they're written — then a disconnect
loses nothing and `--resume` can pick up from Drive:

```python
!python -m src.train.train --config configs/train_baseline.yaml \
    --set project=/content/drive/MyDrive/guardian/runs
```

To pull the trained model back to your Mac later: download
`baseline_best.pt` from Drive → drop into `models/` locally →
`python -m src.inference.predict --source <img> --weights models/baseline_best.pt`.

---

## Expected runtime & VRAM on a T4

- **Config as-shipped:** `yolo11m`, `epochs: 100`, `imgsz: 640`, `batch: 16`,
  ~7,783 train images (≈ 487 iters/epoch).
- **Runtime:** roughly **~2.5–4 min/epoch** on a T4 → **~5–7 hours** for the full 100
  epochs. That can exceed a single **free** Colab session (idle/12 h limits), so either:
  - stream checkpoints to Drive (§6) and use `--resume` across sessions, **or**
  - shorten for a first pass: `--set epochs=50` (Ultralytics early-stops on `patience`
    anyway), **or**
  - use Colab Pro for an uninterrupted run.
- **VRAM — `batch: 16` is fine on a T4.** `yolo11m` @ 640, batch 16 trains in
  **~9–11 GB**, comfortably under the T4's ~15 GB. **No change needed.**
  Only if you hit a CUDA out-of-memory error, drop it without editing the config:

  ```python
  !python -m src.train.train --config configs/train_baseline.yaml --set batch=8
  ```

  (or `--set batch=-1` to let Ultralytics auto-pick a batch for the available VRAM).

> The gate number (military recall on TEST) is what matters, not wall-clock. Don't
> raise `conf_military` to make the run look better — it's deliberately low (see
> `configs/train_baseline.yaml`).
