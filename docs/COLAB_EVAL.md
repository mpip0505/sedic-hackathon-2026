# Baseline evaluation on Google Colab (EVAL ONLY — no training)

Runs the two analyses from `src/eval/detail.py` against the finished baseline
weights, on a GPU (fast — the full test split is ~1–2 min instead of ~30 min on a
Mac CPU):

- **(a)** the `conf_military` **threshold sweep** for `military_vessel`
  (recall / precision / gate pass-fail at 0.05 … 0.30), and
- **(b)** the **per-domain** recall breakdown (aerial vs. surface) + military
  recall per domain.

Nothing here trains. It only loads existing weights and scores the test split.

---

## Prerequisite (run ONCE, locally) — push the eval module

`src/eval/detail.py` is new; the Colab clone only sees what's on GitHub. From your
Mac, in the repo:

```bash
git add src/eval/detail.py
git commit -m "eval: conf_military sweep + per-domain recall (detail.py)"
git push origin main
```

> If you skip this, **Cell 5** below stops with a clear error.

Then: **Runtime → Change runtime type → GPU (T4)** before running the cells.

---

## Cell 1 — confirm GPU + mount Drive

```python
import torch
print("CUDA:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU — set Runtime > GPU (T4)")

from google.colab import drive
drive.mount("/content/drive")
```

## Cell 2 — clone this repo

```python
%cd /content
!rm -rf sedic-hackathon-2026
!git clone https://github.com/mpip0505/sedic-hackathon-2026.git
%cd /content/sedic-hackathon-2026
REPO = "/content/sedic-hackathon-2026"
```

## Cell 3 — install dependencies

```python
!pip install -q -r requirements.txt
# sanity: GPU still visible after the pinned torch install
import torch; print("CUDA after install:", torch.cuda.is_available(), "| torch", torch.__version__)
```

> If `CUDA after install` prints `False`, do **Runtime → Restart session**, then
> re-run Cells 1–3 (the restart picks up the CUDA build).

## Cell 4 — copy weights from Drive → `models/baseline_best.pt`

```python
import shutil, pathlib
pathlib.Path("models").mkdir(exist_ok=True)
src = "/content/drive/MyDrive/sedic/runs/baseline/weights/best.pt"
dst = "models/baseline_best.pt"
shutil.copy2(src, dst)
print("weights:", dst, pathlib.Path(dst).stat().st_size // (1024*1024), "MB")
```

## Cell 5 — guard: the eval module must be present

```python
import pathlib, sys
if not pathlib.Path("src/eval/detail.py").is_file():
    sys.exit("src/eval/detail.py missing — push it first (see the Prerequisite "
             "section), then re-run Cell 2 to re-clone.")
print("eval module present ✅")
```

## Cell 6 — unzip the processed dataset → `data/processed/`

Robust to whether the zip holds `data/processed/…`, `processed/…`, or the split
folders at its root:

```python
import pathlib, shutil, subprocess

REPO = pathlib.Path("/content/sedic-hackathon-2026")
zip_path = "/content/drive/MyDrive/sedic/data_processed.zip"
stage = pathlib.Path("/content/_unzip")
if stage.exists():
    shutil.rmtree(stage)
stage.mkdir(parents=True)

subprocess.run(["unzip", "-q", "-o", zip_path, "-d", str(stage)], check=True)

# find the folder that actually contains the split (images/ + labels/)
src_root = None
for cand in [stage, *stage.rglob("*")]:
    if cand.is_dir() and (cand / "images").is_dir() and (cand / "labels").is_dir():
        src_root = cand
        break
assert src_root is not None, f"no images/labels found under {stage}"

dst_root = REPO / "data" / "processed"
if dst_root.exists():
    shutil.rmtree(dst_root)
dst_root.parent.mkdir(parents=True, exist_ok=True)
shutil.move(str(src_root), str(dst_root))

print("processed at:", dst_root)
for s in ("train", "val", "test"):
    n = len(list((dst_root / "images" / s).glob("*")))
    print(f"  {s}: {n} images")
print("  domains.json:", (dst_root / "domains.json").is_file())
```

## Cell 7 — data.yaml absolute-path fix

Ultralytics resolves a relative `path:` against its `datasets/` dir, so point it at
the absolute processed root:

```python
import re, pathlib
p = pathlib.Path("configs/data.yaml")
txt = p.read_text()
abs_path = "/content/sedic-hackathon-2026/data/processed"
txt = re.sub(r"(?m)^path:.*$", f"path: {abs_path}", txt)
p.write_text(txt)
print(txt)
```

## Cell 8 — validate: confirm the split loaded (labels sane, no leakage)

```python
!python -m src.data.validate
```

Expect a `PASS` with nonzero checked images/boxes. (This reads `data/processed`
directly, so it works regardless of the path fix — it's the load sanity check.)

## Cell 9 — run the eval (sweep + per-domain) on GPU

```python
!python -m src.eval.detail \
    --weights models/baseline_best.pt \
    --data configs/data.yaml \
    --split test \
    --device 0 --batch 16 --imgsz 640 \
    --md-out outputs/eval/test_eval.md
```

This does ONE inference pass and prints, then writes to
`outputs/eval/test_eval.md`:

- **(a)** the `military_vessel` conf threshold sweep — recall / precision / TP / FP /
  FN and gate PASS·FAIL at each of 0.05, 0.10, 0.15, 0.20, 0.25, 0.30;
- **(b)** per-domain × per-class recall + **military recall per domain**
  (aerial vs. surface) and the overall >0.90 gate result.

The process exit code is `0` if the gate passes, `1` if it fails.

> `military` is scored at `conf 0.10`, civilian classes at `conf 0.25`, IoU 0.50 —
> so the military rows reconcile with the gate. Sweep the threshold with
> `--thresholds 0.05,0.075,0.10,0.125,0.15` if you want a finer grid.

## Cell 10 — show the tables + save results back to Drive

```python
import pathlib
print(pathlib.Path("outputs/eval/test_eval.md").read_text())

import shutil
out = pathlib.Path("/content/drive/MyDrive/sedic/eval")
out.mkdir(parents=True, exist_ok=True)
shutil.copy2("outputs/eval/test_eval.md", out / "test_eval.md")
print("saved →", out / "test_eval.md")
```

Paste the markdown from `outputs/eval/test_eval.md` straight into the README /
technical brief.

---

### Notes

- **Why GPU:** the eval collects predictions at a low conf floor (0.01) so precision
  denominators are complete; that's cheap on a GPU but slow on a Mac CPU. `--device 0`
  uses the Colab GPU; `--batch 16` is comfortable for a T4 on inference.
- **VAL instead of TEST:** add `--split val` in Cell 9 for a sanity comparison, but the
  reported gate number must come from `test` (the held-out split).
- **Reproducible:** everything is regenerated from Drive each session, so a Colab
  disconnect costs nothing but a re-run.
