# Technical Brief — outline

_Last updated: 2026-09-04_

This is the **skeleton for `deliverables/technical_brief/`** (the mandatory PDF),
not the brief itself. It exists so whoever writes the PDF pulls each section
from the right source and doesn't drop a caveat that's already been recorded
elsewhere in the repo. Section order matches `docs/TEAM_TASKS.md`'s DELIV
packet — the competition brief names exactly three required contents (**the
dataset used, the model architecture, and the logic used for military
classification**); cover all three explicitly and in that order.

| # | Section | Pull content from |
|---|---|---|
| 1 | Problem + approach | `docs/PROJECT_PLAYBOOK.md` §1 |
| 2 | **Dataset** — 4 sources, licences, counts, both domains | `docs/PROGRESS.md` §2.1, `data/DATASETS.md` |
| 3 | Data pipeline — convert → dedup → stratified split | `docs/PROGRESS.md` §2.3–2.4 |
| 4 | **Architecture** — YOLO11m, 100 epochs, input size, HBB | `configs/train_baseline.yaml`, `docs/PROGRESS.md` §3 |
| 5 | **Military-classification logic** — the 50→8 collapse and *why* `military_vessel` stays one coarse class; the dual-threshold rule (`conf_military` 0.10 < `conf` 0.25) as the recall mechanism | `docs/PROGRESS.md` §2.2, `CLAUDE.md` non-negotiable #3 |
| 6 | Results — the gate, per-domain, per-class, conf sweep | `outputs/eval/test_eval.md` |
| 7 | **Bonus — RMN vs Foreign 2nd stage** | see below — **do not paste a bare accuracy number** |
| 8 | Reproducibility — repo layout, one-command rebuild | `README.md` |

**Non-negotiable framing rules for whoever drafts this:**
- Quote the gate at `conf_military = 0.10`, on the **TEST** split. The GUI demo
  runs at 0.25 for a cleaner picture; the number in the brief is the 0.10 one.
  Say which is which.
- Per-domain recall (aerial vs surface, both independently >0.90) is the
  multi-angle evidence — lead with that table, not the overall number.
- The dedup de-chaining story (`docs/dedup_audit.md`) is a credibility asset:
  it shows the numbers are honest, not just high.

---

## §7 — Bonus: RMN-vs-Foreign 2nd stage (honest framing, mandatory)

The classifier (`src/fine_grained/`) is built and its results are recorded in
`docs/PROGRESS.md` §4 and `data/DATASETS.md`. **Do not write "98% accuracy" as
a headline number without the caveat in the same breath** — the caveat is not
a weakness to bury, it's evidence the team tested for and understands domain
shift, which reads well to a technical judge.

**What to say, in this order:**

1. **What it is:** a 2nd-stage ResNet18 classifier that runs on crops the main
   detector already labelled `military_vessel`, and calls each one
   `malaysian_rmn` / `foreign` / `unknown` (confidence floor 0.55 for
   abstention — an honest "don't know" beats a confident wrong nationality
   call).
2. **The headline number, with its caveat attached in the same sentence:**
   *"0.98 validation accuracy — but this is inflated by a domain gap between
   the two data sources (curated RMN press photography, median crop area 691K
   px, vs. low-resolution Roboflow foreign-navy detection crops, median 58K
   px), so the model partly separates on image-source style rather than pure
   vessel identity."* Do not quote 0.98 alone in a table or slide without that
   clause attached.
3. **What was done to test the caveat, not just assert it:** a
   resolution/blur-degradation ablation (RMN validation images downsampled to
   foreign-like resolution before inference) did **not** flip predictions,
   ruling out that one specific shortcut — but train and validation for each
   class still come from the same disjoint source pool, so other domain cues
   (colour grading, framing/angle distribution, compression) can't be fully
   ruled out without independently-sourced RMN data at realistic field-camera
   quality.
4. **The honest claim, stated explicitly:** this classifier reliably
   distinguishes "RMN press-photo domain" from "this one foreign Roboflow
   dataset" — narrower than "distinguishes RMN vessels from foreign vessels in
   general." On real detector output (small, low-resolution field crops), it
   may be systematically biased toward predicting `foreign` regardless of true
   nationality.
5. **Forward-looking deployment note (this is the scalability/rigor payoff —
   use it in the jury pitch too):** the real fix is domain-matched RMN
   training data at field-camera resolution (not press photography), and/or
   fusing this classifier's call with AIS/IFF signals rather than relying on
   vision alone for a nationality determination.
6. **Decision record:** the team weighed retraining with domain-matched/
   degraded data against presenting the current result honestly, and chose the
   latter (Path A, 2026-09-04) — the RMN set is thin (346 images, 4 hull
   types) and a second data-collection pass wasn't justified for an optional
   bonus track on the submission timeline. Full decision-log entry:
   `docs/PROGRESS.md` §6, dated 2026-09-04.

**Numbers to cite verbatim** (source: `docs/PROGRESS.md` §4):
- Crop counts — train: `malaysian_rmn` 294 / `foreign` 882 (capped from 4,290
  at 3× the RMN count); val: `malaysian_rmn` 52 / `foreign` 760.
- Confusion matrix (val): `malaysian_rmn` 51/52 correct, `foreign` 760/760
  correct.
- Weights: `models/fine_grained_rmn_classifier.pt` (not committed — gitignored
  per `CLAUDE.md` non-negotiable #4; MD5 recorded in the training run log for
  manual backup).
