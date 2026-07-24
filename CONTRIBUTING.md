# Contributing to Project Guardian

Four people, all using Claude Code, working against shared contracts. Keep the
contracts stable and the branches small.

## Ownership map
| Person | Area | Primary paths |
|--------|------|---------------|
| **P1** | Data pipeline | `src/data/`, `src/data/converters/`, `data/DATASETS.md` |
| **P2** | Model training | `src/train/`, `configs/train_baseline.yaml` |
| **P3** | Integration + GUI | `src/inference/`, `app/`, `src/eval/` wiring |
| **P4** | Deliverables + bonus | `deliverables/`, `src/fine_grained/`, OBB |

Changes to `configs/schema.yaml` or the `predict()`/`Detection` interface are
**shared contracts** — get a second person's sign-off before merging.

## Branch naming
```
<person>/<area>-<short-desc>
```
Examples: `p1/converters-seaships`, `p3/gui-box-drawing`, `p2/train-baseline`.

## Before every PR
```bash
ruff check .                                   # lint must pass
python -m src.inference.predict --source none --stub   # stub must still work
pytest -q                                      # tests must pass
```
Also confirm:
- No data or weights staged (`git status` — nothing under `data/`, `models/`,
  `outputs/`, no `*.pt`/`*.onnx`).
- Class names are read from `configs/schema.yaml`, not hardcoded.
- If you added a dataset, it has a row in `data/DATASETS.md` with its licence.

## Conventions
- Python 3.10+, run modules (`python -m ...`), `pathlib`, type hints, `logging`.
- Parameters live in `configs/*.yaml`, not in code.
- Keep `military_vessel` a single coarse detector class (see `CLAUDE.md`).
