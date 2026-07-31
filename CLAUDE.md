# CLAUDE.md — Project Guardian

Repo brief for future Claude Code sessions. Read this before touching anything.

## What this is
**Project Guardian** is a maritime object-detection system for the **SEDIC 2026
Visual Track** competition. It detects vessels across two visual domains:

- **frontal / surface** camera views (e.g. shore/ship cameras), and
- **aerial / satellite** views.

Built on **Ultralytics YOLO**. Python only — no npm/Node. The GUI is Streamlit.

## The one hard requirement
> **Recall > 90% on military / threat classes.**

Overall mAP is secondary. **When a tradeoff appears, favour military recall.**
The lower military confidence threshold (`conf_military: 0.10`) exists solely to
serve this gate.

## Reference docs
Three reference docs live in `docs/`. Read them before making decisions:

| File | What it is |
|------|-----------|
| `docs/SEDIC2026-track2.pdf` | **The official competition brief — the AUTHORITY on requirements and deliverables.** |
| `docs/DATA_SOURCING_MAP.md` | Which datasets we use, their sources/licences, and the merge workflow. |
| `docs/PROJECT_PLAYBOOK.md` | Repo structure, step-by-step approach, sequencing. (Its §3/§5 division of labour is superseded — see below.) |
| `docs/TEAM_TASKS.md` | **The live 5-person work split**: per-person task packets, the fine-grained data handoff contract, acceptance criteria, dependency map. |

**`docs/SEDIC2026-track2.pdf` is the source of truth.** If anything in this repo
— code, config, this file, the README, or either of the other two docs —
contradicts the PDF, **the PDF wins.** Update the repo to match it, not the
other way around.

## Non-negotiables
1. **`configs/schema.yaml` is THE CONTRACT.** Never hardcode class names or IDs
   anywhere. Converters, trainer, and eval all read classes from it via
   `load_classes()`.
2. **The `predict()` / `Detection` interface is FROZEN.** `--stub` must ALWAYS
   work with no weights and no torch. Everything downstream depends on it.
3. **`military_vessel` stays ONE coarse class in the detector.** Do not split it
   into ship types — that fragments positives and collapses recall.
   Fine-grained work (RMN vs foreign) lives in `src/fine_grained/` as a separate
   2nd-stage classifier.
4. **Never commit data or weights.** No images, no `.pt`/`.onnx`. See
   `.gitignore`.
5. **Every dataset gets a row in `data/DATASETS.md`** with its licence before
   its data is used.

## Path → purpose
| Path | Purpose |
|------|---------|
| `configs/schema.yaml` | The label-taxonomy contract + source→schema mappings |
| `configs/data.yaml` | YOLO dataset descriptor (mirrors schema classes) |
| `configs/train_baseline.yaml` | Config-driven training params |
| `data/{raw,interim,processed}/` | Datasets by pipeline stage (gitignored) |
| `data/DATASETS.md` | Provenance + licence log |
| `src/data/converters/` | Per-source label converters → unified schema |
| `src/data/` | Dataset assembly, splitting, dedup |
| `src/train/` | Training entry points (read train_baseline.yaml) |
| `src/inference/predict.py` | **Frozen** detection interface + `--stub` |
| `src/eval/` | Metrics, especially the military recall gate |
| `src/fine_grained/` | Bonus 2nd-stage RMN-vs-foreign classifier |
| `app/app.py` | Streamlit GUI, wired to `predict()` |
| `models/` | Trained weights (gitignored) |
| `outputs/{detections,runs}/` | Detection JSON + training runs (gitignored) |
| `deliverables/` | Technical brief, video, poster |
| `scripts/` | `setup.sh`, `download_datasets.sh` |

## Conventions
- **Python 3.10+.** Use modern typing (`X | None`, built-in generics).
- **Run modules, not scripts:** `python -m src.inference.predict ...`.
- Use **`pathlib`**, not `os.path`.
- **Type hints** on public functions.
- **`logging`, not `print`** (except deliberate CLI stdout output like JSON).
- **Config-driven:** parameters live in `configs/*.yaml`, not in code.
- Lint with **ruff** before every PR.

## Domain gotchas
- **Cross-dataset duplicates cause train/test leakage.** Known overlaps:
  - `military_ships` and `shiprsimagenet` share the ShipRSImageNet 50-class
    taxonomy and many identical images (exact dups at hamming 0).
  - FGSCR-42 derives from **DOTA + HRSC**; **ShipRSImageNet** overlaps HRSC/FGSD.
  Dedup across sources before splitting. `dota` maps to `null` (dropped) for this.
- **Dedup is GREEDY (leader), not single-linkage — do not revert.** An image is
  dropped only if within `schema.dedup.threshold` of an already-**kept**
  representative; candidates are never pairwise-chained. Single-linkage chained
  A~B~C~D and collapsed ~2,961 distinct SeaShips frames into one cluster (46%
  drop). Greedy keeps distinct frames (25% drop) and guarantees no two kept
  images are within threshold. Method + threshold live in `configs/schema.yaml`
  (`dedup:`), default `greedy` / **3** (conservative — err toward keeping; we
  cannot over-prune `military_vessel`). `--dedup-method cluster` is legacy only.
- **One duplicate threshold everywhere.** `merge` dedup, `validate`'s leakage
  check, and `balance`'s synth guard ALL read `schema.dedup.threshold`. If you
  change one, they must stay equal — otherwise validate false-alarms on pairs
  dedup deliberately kept. Don't hardcode a separate leak threshold.
- **`surface_synth` is a real domain value — keep it.** `military_vessel` has 0
  real surface instances, so `balance.py` pastes aerial military onto surface
  (SeaShips) backgrounds (cross-domain copy-paste), tagged `surface_synth` in
  `domains.json`, **TRAIN split only**. It's a reversible stopgap; don't fold it
  into `surface`, and never generate it for val/test.
- **Synthetic images must not leak.** A small pasted object barely changes the
  background hash, so every synthetic image is hashed and **dropped if within
  threshold of any val/test image**. Keep that guard — it's what keeps the gate
  measured on real, unseen data.
- **The val/test sets stay UN-augmented** (no `augcp_`/`augxd_` there) and the
  split is stratified per domain so per-domain performance is measurable.
- **Horizontal boxes before oriented boxes.** Get HBB working first; OBB is a
  later upgrade for arbitrarily-rotated aerial ships (`degrees: 10` augments
  toward this). `yolo2yolo` already envelopes polygon labels to HBB.
- **The low military threshold is DELIBERATE.** `conf_military: 0.10 < conf:
  0.25`. Do not "fix" it — it is the recall gate.
