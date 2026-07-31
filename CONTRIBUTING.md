# Contributing to Project Guardian

Five people, all using Claude Code, working against shared contracts. Keep the
contracts stable and the branches small.

## Ownership map
| Handle | Area | Primary paths |
|--------|------|---------------|
| **LEAD** | ML core — data, training, eval, inference, bonus classifier | `src/data/`, `src/train/`, `src/eval/`, `src/inference/`, `src/fine_grained/`, `configs/` |
| **GUI** | Landing page + navigation | `app/pages/`, `app/assets/` |
| **DATA-RMN** | Malaysian (TLDM) half of the bonus set | `data/raw/fine_grained/malaysian_rmn/` |
| **DATA-FOR** | Foreign-navy half + surface gap-fill | `data/raw/fine_grained/foreign/` |
| **DELIV** | Submission package | `deliverables/` |

Per-person task packets and the data handoff contract: **`docs/TEAM_TASKS.md`**.

Changes to `configs/schema.yaml` or the `predict()`/`Detection` interface are
**shared contracts** — get a second person's sign-off before merging.

## Branch naming
```
<handle>/<area>-<short-desc>
```
Examples: `lead/finegrained-classifier`, `gui/landing-page`, `deliv/brief-draft`.

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
