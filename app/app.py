"""app.py — Project Guardian: Maritime Domain Awareness GUI.

A single-file Streamlit front end for the frozen detection interface in
`src.inference.predict`. Upload an image or video, get colour-coded boxes, a
live summary panel, and (for video) BoT-SORT track IDs plus a downloadable
detection log.

Everything model-related goes through `predict()` / `track_video()` — this file
never imports ultralytics or torch directly, so `--stub` mode keeps working
with no weights installed.

Run:
    streamlit run app/app.py

Tweaking: colours live in `GROUP_COLOURS` below, theme in .streamlit/config.toml,
copy in the constants at the top. Class names/groups come from
configs/schema.yaml — never hardcode them here.
"""

from __future__ import annotations

import inspect
import re
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import yaml
from PIL import Image

# Make the repo root importable when Streamlit runs this file directly.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Imported after the sys.path setup above, deliberately. E402 is ignored for this
# file in pyproject.toml rather than with an inline suppression comment, because
# current ruff already understands the sys.path idiom and would then flag that
# comment itself as an unused directive (RUF100).
from src.analysis.trajectory_anomaly import (
    DEFAULT_COURSE_CHANGE_DEG,
    DEFAULT_LOITER_MIN_DURATION_S,
    DEFAULT_LOITER_RADIUS_PX,
    DEFAULT_MIN_MOVE_PX,
    detect_anomalies,
)
from src.analysis.trajectory_anomaly import render_markdown as render_anomaly_markdown
from src.fine_grained import infer as fg
from src.inference import predict as gp
from src.inference.predict import Detection
from src.reports.incident_report import ReportSummary, summarize_detections
from src.reports.incident_report import render_markdown as render_incident_markdown

# `fg` (like `gp`) keeps torch/torchvision as lazy imports inside its
# functions, so importing it here doesn't break stub mode's no-torch promise.

# --- Copy -------------------------------------------------------------------
APP_TITLE = "Project Guardian"
APP_SUBTITLE = "Maritime Domain Awareness"
TAGLINE = (
    "Real-time vessel detection and classification across surface and aerial "
    "views · SEDIC 2026 Visual Track"
)

# --- Operating point --------------------------------------------------------
# Validated recall/precision sweet spot for the demo. The evaluation gate still
# runs at the lower conf_military; this is the presentation default.
DEFAULT_CONF = 0.25
DEFAULT_CONF_MILITARY = 0.25

# --- Baseline picker ----------------------------------------------------------
# `models/` is gitignored — weights may not exist on a fresh clone or in CI;
# the existing weights-file-missing check below degrades that gracefully to an
# error message, not a crash.
#
# Shipped model (2026-09-04): `baseline2_best.pt` — the 2026-08-07 retrain on
# the civilian_gapfill-merged data build, ~27% fewer civilian-as-military false
# positives than the original `baseline_best.pt`. It was briefly logged as
# FAILing the canonical gate (metrics.py, 0.892) and marked "not shipped", but
# that was a bug in metrics.py's recall reading (it read Ultralytics' box.r at
# a shared cross-class max-F1 index, not at the actual operating `conf`) — with
# that fixed, the canonical gate reads 0.936, PASS, consistent with the
# per-domain diagnostic (detail.py: aerial 0.932 / surface 0.977 / overall
# 0.938). See docs/PROGRESS.md decision log (2026-09-04) and data/DATASETS.md.
# The original `baseline_best.pt` is kept in `models/` for reference but is no
# longer a preset here — use "Custom path…" to point at it if ever needed.
BASELINE_CHOICES: dict[str, str] = {
    "Baseline (shipped)": str(gp.DEFAULT_WEIGHTS),
}
CUSTOM_BASELINE_LABEL = "Custom path…"

# --- Evaluation report --------------------------------------------------------
# The landing page never hardcodes gate numbers — a stale figure in front of a
# judge is a real cost. Parsed live from the report `src.eval.detail` writes.
EVAL_REPORT_PATH = _REPO_ROOT / "outputs" / "eval" / "test_eval.md"
EVAL_GATE = 0.90
# The gate's actual operating threshold, read from the frozen predict() signature
# rather than retyped here.
CONF_MILITARY_GATE = inspect.signature(gp.predict).parameters["conf_military"].default

# --- Bonus: RMN-vs-Foreign nationality classifier (optional, 2nd-stage) -----
# Read from configs/fine_grained.yaml rather than retyped here, so the demo
# can't drift from what training actually used. Val accuracy 0.98, but that
# figure is inflated by an image-source domain gap between the two training
# sets, not pure vessel-identity recognition — see docs/PROGRESS.md §4. The
# sidebar surfaces that caveat; do not present the number without it.
_FG_CONFIG_PATH = _REPO_ROOT / "configs" / "fine_grained.yaml"
_FG_CONFIG: dict = {}
if _FG_CONFIG_PATH.exists():
    with open(_FG_CONFIG_PATH, "r", encoding="utf-8") as _fh:
        _FG_CONFIG = yaml.safe_load(_fh) or {}
FG_DEFAULT_WEIGHTS = _REPO_ROOT / _FG_CONFIG.get("train", {}).get(
    "weights_out", "models/fine_grained_rmn_classifier.pt"
)
FG_DEFAULT_CONF_FLOOR = _FG_CONFIG.get("infer", {}).get("conf_floor", 0.55)
FG_PAD_FRAC = _FG_CONFIG.get("crop", {}).get("pad_frac", 0.12)
# Side-by-side checkpoint picker, same pattern as BASELINE_CHOICES above. Add
# a new entry here (never hardcode a path elsewhere) when a new classifier
# checkpoint is trained and worth comparing against the current one.
FG_CUSTOM_LABEL = "Custom path…"
FG_BASELINE_CHOICES: dict[str, str] = {
    "RMN-vs-Foreign classifier (trained)": str(FG_DEFAULT_WEIGHTS),
}
FG_LABEL_DISPLAY = {
    "malaysian_rmn": "Malaysian RMN",
    "foreign": "Foreign navy",
    fg.UNKNOWN: "Unknown",
}
# Colours for the nationality split panel: own navy green, foreign amber
# (caution — red stays reserved for the military-contact alert), below-floor
# calls muted. Theme variables, so they follow light/dark like everything else.
FG_LABEL_COLOURS = {
    "malaysian_rmn": "var(--g-green)",
    "foreign": "var(--g-amber)",
    fg.UNKNOWN: "var(--g-muted)",
}
# Fixed display order, so a label sits in the same place from image to image
# instead of jumping around as counts change.
FG_LABEL_ORDER = ["malaysian_rmn", "foreign", fg.UNKNOWN]

# --- Colour coding by schema group (BGR for OpenCV, hex for HTML) -----------
GROUP_COLOURS: dict[str, tuple[tuple[int, int, int], str, str]] = {
    # group        BGR              hex        display label
    "military": ((38, 38, 220), "#dc2626", "Military"),
    "small_craft": ((11, 158, 245), "#f59e0b", "Small craft"),
    "civilian": ((200, 160, 45), "#2dd4bf", "Civilian"),
    "other": ((150, 150, 150), "#94a3b8", "Other"),
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}

st.set_page_config(
    page_title=f"{APP_TITLE} — {APP_SUBTITLE}",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Styling — small and targeted; the base theme lives in .streamlit/config.toml
# ---------------------------------------------------------------------------

def render_header() -> None:
    st.markdown(
        f"""
        <div class="guardian-header">
          <h1>{APP_TITLE} <span>— {APP_SUBTITLE}</span></h1>
          <p>{TAGLINE}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def inject_css(light_mode: bool = False) -> None:
    """Inject the landing/dashboard styles for dark and light mode."""
    if light_mode:
        palette = {
            "--g-navy": "#D5E0E7",
            "--g-panel": "#E2E9EE",
            "--g-panel-2": "#D9E3E9",
            "--g-line": "#B7C8D3",
            "--g-text": "#17324A",
            "--g-muted": "#4F687A",
            "--g-cyan": "#087EA4",
            "--g-green": "#17795F",
            "--g-amber": "#A76508",
            "--g-red": "#C62828",
            # Native st.info() alert. Its own dedicated trio (not reused from
            # the general panel/line/text vars above) because its current
            # colours — a muted slate-blue tint distinct from --g-panel/
            # --g-text — don't correspond 1:1 to any of them; giving it its
            # own variables lets it move to the always-present stylesheet
            # (see inject_css's css block) without changing how it looks.
            "--g-info-bg": "#C6DCE7",
            "--g-info-border": "#AFC9D5",
            "--g-info-text": "#31566B",
            # Military Recall performance card: its own dedicated trio (not
            # var(--g-panel)/var(--g-red)) so it reads as a highlighted
            # metric rather than an alert, in both themes. Values match what
            # the modal-scoped light-mode override already forces exactly,
            # so switching the base .mil-recall-card/.mil-recall-label
            # rules (below) to these produces zero visual change here.
            "--g-mil-bg": "#DCEAF0",
            "--g-mil-accent": "#0B8FB3",
            "--g-mil-label": "#087EA4",
        }
    else:
        palette = {
            "--g-navy": "#081522",
            "--g-panel": "#102333",
            "--g-panel-2": "#132b3d",
            "--g-line": "#28465a",
            "--g-text": "#e8f0f4",
            "--g-muted": "#99adbb",
            "--g-cyan": "#62b9cc",
            "--g-green": "#78b69f",
            "--g-amber": "#d7a756",
            "--g-red": "#d96666",
            # Matches the native (previously unstyled) dark-mode st.info()
            # alert exactly, read live from its computed style: translucent
            # blue fill, light blue text, and no visible border — border is
            # "transparent" rather than omitted so the shared always-present
            # rule below can apply `border:1px solid var(--g-info-border)`
            # in both themes without a light_mode branch; the alert's
            # box-sizing:border-box means that 1px doesn't change its
            # outer size, and transparent keeps it invisible here.
            "--g-info-bg": "rgba(61,157,243,0.2)",
            "--g-info-border": "transparent",
            "--g-info-text": "#C7EBFF",
            # Military Recall performance card: deep blue/cyan highlight,
            # matching the light-mode treatment's design philosophy instead
            # of the plain var(--g-panel)/var(--g-red) it used before.
            "--g-mil-bg": "#102A3A",
            "--g-mil-accent": "#19A7C9",
            "--g-mil-label": "#39B9D8",
        }

    # Light mode is a deliberately separate design, not an inverted dark theme:
    # layered card surfaces with shadows, a tinted hero, a filled Military
    # Recall accent, neutral status typography (colour lives in the dot only),
    # and a solid-navy primary button. Folded into the single stylesheet below
    # (rather than a second st.markdown call) so it doesn't add an extra
    # element-container gap that would shift the whole page down vs dark mode.
    light_only_css = ""
    if light_mode:
        light_only_css = """
              .info-panel, .brief-panel, .eval-pending, .taxonomy-card,
              .provenance-wrap, .guardian-header, .mil-recall-card {
                border-radius:8px;
                box-shadow:0 1px 2px rgba(16,42,67,.06), 0 1px 3px rgba(16,42,67,.05);
              }
              .chip-row, .metric-grid.secondary {
                border-radius:8px; overflow:hidden;
                box-shadow:0 1px 2px rgba(16,42,67,.06), 0 1px 3px rgba(16,42,67,.05);
              }
              .table-wrap {
                border-radius:8px;
                box-shadow:0 1px 2px rgba(16,42,67,.06), 0 1px 3px rgba(16,42,67,.05);
              }
              .entry-intro {
                background:#DCE7EC; border:1px solid var(--g-line);
                border-left:3px solid var(--g-cyan);
                border-radius:8px; padding:1.3rem 1.5rem;
                box-shadow:0 1px 2px rgba(16,42,67,.06), 0 1px 3px rgba(16,42,67,.05);
              }
              .mil-recall-card {
                background:#E8D6D7; border-left-width:4px;
              }
              .status-value.ok, .status-value.warn, .status-value.neutral {
                color:var(--g-muted);
              }
              .sedic-badge {
                background:#DCE7EC; border:1px solid var(--g-line);
              }
              .sedic-badge::before {
                content:""; display:inline-block; width:6px; height:6px;
                border-radius:50%; background:var(--g-cyan); margin-right:.45rem;
                vertical-align:middle;
              }
              [class*="st-key-entry_launch"] button[kind="primary"] {
                background:#176B55 !important; border:1px solid #3F8B76 !important;
                border-radius:10px !important; box-shadow:0 1px 2px rgba(16,42,67,.12);
              }
              [class*="st-key-entry_launch"] button[kind="primary"]::after {
                color:#fff;
              }
              [class*="st-key-entry_launch"] button[kind="primary"]:hover {
                background:#1E8267 !important; border-color:#4FA189 !important;
                filter:none !important; box-shadow:0 4px 14px rgba(16,42,67,.18);
              }
              /* Operational Briefing modal: a soft blue-grey rather than the
                 near-white dashboard panel colour, with cards a touch
                 lighter than the modal itself (var(--g-panel), already
                 lighter than this) so they stay visually separated, and a
                 gentle navy-tinted backdrop instead of the dark theme's
                 near-black dim/blur (which would otherwise turn muddy-grey
                 over light content). */
              body:has([data-testid="stDialog"]) [data-testid="stAppViewContainer"] {
                filter:blur(5px) brightness(.94) !important;
              }
              [data-testid="stDialog"] {
                background:rgba(16,42,67,.16) !important;
              }
              [data-testid="stDialog"] [role="dialog"] {
                background:#DDE7EC !important;
                box-shadow:0 10px 32px rgba(16,42,67,.16) !important;
              }
              /* Cards inside the modal a touch lighter than the modal itself
                 so they stay visually separated (modal #DDE7EC, cards
                 #E5ECEF) — these classes reuse var(--g-panel) via the base
                 stylesheet, so give them their own explicit tone here.
                 Military Recall is deliberately NOT in this shared group —
                 it gets its own distinct cool-cyan tint below so it reads as
                 a highlighted metric rather than a plain neutral card like
                 its siblings. */
              [data-testid="stDialog"] .brief-panel,
              [data-testid="stDialog"] .info-panel,
              [data-testid="stDialog"] .taxonomy-card,
              [data-testid="stDialog"] .eval-pending {
                background:#E5ECEF !important; border-color:#B7C8D3 !important;
              }
              /* Military Recall card: a highlighted metric, not an alert —
                 no red/pink/orange/yellow/purple/green background. Subtle
                 cool-blue tint (#DCEAF0) instead of the neutral #E5ECEF its
                 siblings use above, with a deeper cyan left accent
                 (#0B8FB3) distinct from --g-cyan so it reads as its own
                 premium accent rather than reusing the standard cyan used
                 elsewhere. The base .mil-recall-card/.mil-recall-label
                 rules still use var(--g-red) for the left border and label
                 (dark mode is untouched by this light-mode-only block). */
              [data-testid="stDialog"] .mil-recall-card {
                background:#DCEAF0 !important; border-color:#B7C8D3 !important;
                border-left-color:#0B8FB3 !important;
              }
              [data-testid="stDialog"] .mil-recall-label {
                color:#087EA4 !important;
              }

              /* --------------------------------------------------------------
                 Native Streamlit / BaseWeb component coverage.
                 .streamlit/config.toml pins base="dark" with hardcoded hex
                 colours (backgroundColor/secondaryBackgroundColor/textColor) —
                 that's a static, server-start-time theme with no per-session
                 switch, and it seeds these components directly rather than
                 through our --g-* variables. So toggling light_mode alone
                 never touched the sidebar chrome, selects, inputs, slider,
                 uploader or secondary buttons — this block hardcodes light
                 equivalents for each, scoped to light_mode so dark mode
                 (still fully served by config.toml) is untouched. Selectors
                 use data-testid/data-baseweb, not emotion-cache hash classes,
                 since only the former are stable across Streamlit builds.
                 -------------------------------------------------------------- */
              [data-testid="stSidebar"] {
                background:#C8D5DF !important; color:#17324A !important;
                border-right:1px solid #B7C8D3 !important;
              }
              [data-testid="stSidebar"] * {
                color:#17324A !important;
              }
              [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
              [data-testid="stSidebar"] .sb-label, [data-testid="stSidebar"] .sb-group-label,
              [data-testid="stSidebar"] .sb-desc, [data-testid="stSidebar"] .sb-value {
                color:var(--g-muted) !important;
              }
              /* Sidebar's own bordered cards (Models / Thresholds / Video)
                 read var(--g-panel) from the shared stylesheet — give them
                 their own slightly-darker-than-main-panel tone here so the
                 sidebar → cards step stays visible even though the sidebar
                 itself is now darker than the page background. */
              [data-testid="stSidebar"] [class*="st-key-sb_card_"] {
                background:#E0E8ED !important; border-color:#B7C8D3 !important;
              }
              [class*="st-key-sidebar_briefing"] button[kind="secondary"] {
                background:#E0E8ED !important; border-color:#B7C8D3 !important;
              }
              /* Selectbox fill/border/text and the file-uploader dropzone's
                 fill/border moved to the always-present stylesheet (see
                 near the top of the f-string css block below) so they're
                 driven by --g-* variables instead of being conditionally
                 injected — see that block's comment for why. */
              /* Text input / number input */
              [data-testid="stSidebar"] [data-baseweb="input"] {
                background:#E6EDF1 !important; border-color:#B7C8D3 !important;
              }
              [data-testid="stSidebar"] [data-baseweb="input"] input {
                background:#E6EDF1 !important; color:#17324A !important;
                -webkit-text-fill-color:#17324A !important;
              }
              [data-testid="stSidebar"] [data-testid="stNumberInput"] button {
                background:#E6EDF1 !important; color:#17324A !important;
                border-color:#B7C8D3 !important;
                transition:background-color 120ms ease, box-shadow 120ms ease, color 120ms ease;
              }
              [data-testid="stSidebar"] [data-testid="stNumberInputStepUp"]:hover,
              [data-testid="stSidebar"] [data-testid="stNumberInputStepDown"]:hover {
                background:#D0DEE6 !important; color:#17324A !important;
                box-shadow:0 1px 4px rgba(16,42,67,.12) !important;
              }
              [data-testid="stSidebar"] [data-testid="stNumberInputStepUp"]:active,
              [data-testid="stSidebar"] [data-testid="stNumberInputStepDown"]:active {
                background:#C3D4DE !important;
                box-shadow:inset 0 1px 2px rgba(16,42,67,.18) !important;
              }
              /* Slider: unfilled rail, filled rail, thumb. :not([data-testid])
                 excludes the tick-bar min/max labels and the value bubble,
                 which sit at this same ">div>div" depth and were otherwise
                 getting painted with the rail's slate background too. */
              [data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"] {
                background:#087EA4 !important;
              }
              [data-testid="stSidebar"] [data-baseweb="slider"] > div > div:not([data-testid]) {
                background:#B7C8D3 !important;
              }
              [data-testid="stSidebar"] [data-baseweb="slider"] > div > div > div:not([data-testid]) {
                background:#087EA4 !important;
              }
              [data-testid="stSidebar"] [data-testid="stSliderThumbValue"],
              [data-testid="stSidebar"] [data-testid="stSliderTickBarMin"],
              [data-testid="stSidebar"] [data-testid="stSliderTickBarMax"] {
                background:transparent !important; box-shadow:none !important;
                border:none !important; color:#17324A !important;
              }
              [data-testid="stSidebar"] [data-testid="stSliderThumbValue"] {
                color:#087EA4 !important;
              }
              /* Toggle / checkbox track */
              [data-testid="stSidebar"] [data-baseweb="checkbox"] > div:first-child {
                background:#B7C8D3 !important;
              }
              [data-testid="stSidebar"] [data-baseweb="checkbox"]:has(input:checked) > div:first-child {
                background:#087EA4 !important;
              }
              /* Main-content toggle (e.g. "Show original (before / after)").
                 Same track colours as the sidebar toggle above, just scoped
                 to stMain instead of stSidebar so the sidebar's own toggle
                 (and its rule above) is untouched — without a stMain scope,
                 the unstyled OFF track defaults to a near-white BaseWeb
                 grey that reads as too bright against the light panel. */
              [data-testid="stMain"] [data-baseweb="checkbox"] > div:first-child {
                background:#B7C8D3 !important;
              }
              [data-testid="stMain"] [data-baseweb="checkbox"]:has(input:checked) > div:first-child {
                background:#087EA4 !important;
              }
              /* Disabled controls keep readable contrast instead of fading
                 to near-invisible grey-on-grey. */
              [data-testid="stSidebar"] input:disabled,
              [data-testid="stSidebar"] [aria-disabled="true"] {
                background:#D9E3E9 !important; color:#4F687A !important;
                border-color:#B7C8D3 !important; -webkit-text-fill-color:#4F687A !important;
              }
              /* File uploader dropzone (main content). Fill/border moved to
                 the always-present stylesheet (see near the top of the
                 f-string css block below) — this bespoke icon/text tint has
                 no clean existing --g-* match (it's deliberately a touch
                 darker than --g-muted for legibility on the light panel),
                 and unlike a plain var() swap, hardcoding it unconditionally
                 would leave near-invisible dark teal text on the DARK theme's
                 own dark panel, so it stays conditional here. */
              [data-testid="stFileUploaderDropzone"] {
                color:#31566B !important;
              }
              [data-testid="stFileUploaderDropzone"] svg {
                fill:#31566B !important; color:#31566B !important;
              }
              [data-testid="stFileUploaderDropzone"] small,
              [data-testid="stFileUploaderDropzone"] span,
              [data-testid="stFileUploaderDropzone"] div {
                color:#31566B !important;
              }
              [data-testid="stFileUploaderDropzone"] button {
                background:#E6EDF1 !important; border:1px solid #AFC1CC !important;
                color:#17324A !important;
              }
              /* Native st.info() banner ("Upload an image or video to
                 begin...") moved to the always-present stylesheet (see near
                 the top of the f-string css block below) so it's driven by
                 --g-info-* variables instead of being conditionally
                 injected. */
              /* Secondary buttons app-wide (Browse files, Run detection,
                 Download CSV, ...) — the primary CTA keeps its own dark
                 green rule above and is untouched by this. */
              button[kind="secondary"] {
                background:#E6EDF1 !important; border:1px solid #AFC1CC !important;
                color:#17324A !important;
              }
              button[kind="secondary"] p, button[kind="secondary"] span {
                color:#17324A !important;
              }
              button[kind="secondary"]:hover {
                background:#D9E3E9 !important; border-color:#087EA4 !important;
                color:#087EA4 !important;
              }
              [data-testid="stTooltipIcon"] {
                color:#526B7A !important;
              }
              /* The icon's <svg> carries a hardcoded stroke colour from the
                 dark base theme (not currentColor), so overriding just the
                 parent's `color` above never reached it — stroke needs its
                 own explicit override. */
              [data-testid="stTooltipIcon"] svg {
                stroke:#526B7A !important; color:#526B7A !important;
              }
              /* li covers captions whose text starts with "*" (e.g. the
                 dataset-provenance attribution note) — Streamlit's markdown
                 parser renders those as a <ul><li>, not a <p>, so the rule
                 above alone never matched them and they stayed on the
                 dark theme's near-white text colour. */
              [data-testid="stCaptionContainer"] p,
              [data-testid="stCaptionContainer"] li {
                color:var(--g-muted) !important;
              }
              /* Resolved-path display in the sidebar (Weights (.pt) path).
                 It's an app-authored <code> inside .sb-value, not a
                 stTextInput, but it still inherits Streamlit markdown's
                 dark-theme <code> background, so it needs its own rule. */
              [data-testid="stSidebar"] .sb-value code {
                background:#E6EDF1 !important; color:#17324A !important;
                border:1px solid #B7C8D3 !important;
              }
              /* st.metric() and native headings/subheaders (st.subheader())
                 inherit Streamlit's dark base theme text colour directly,
                 same as the other native components above — not covered by
                 the app's own .metric-* / var(--g-text) rules since those
                 only style the hand-rolled HTML metric cards, not st.metric
                 itself. */
              [data-testid="stMetricLabel"] {
                color:var(--g-muted) !important;
              }
              [data-testid="stMetricValue"] {
                color:var(--g-text) !important;
              }
              h1, h2, h3, h4, h5, h6,
              [data-testid="stHeading"] {
                color:var(--g-text) !important;
              }
              /* Uploaded-file row (filename + size) and the st.image()
                 caption — same native-component gap as their dark_only_css
                 counterparts (stFileUploaderDropzone only covers the empty
                 dropzone, not this row; .stFileUploaderFileData is the
                 stable, literal Streamlit class name wrapping filename+size,
                 verified against Streamlit's own frontend source). Light
                 mode had no override for these at all, so they were
                 inheriting the dark base theme's near-white text directly. */
              [data-testid="stFileUploaderFileName"] {
                color:#17324A !important;
                -webkit-text-fill-color:#17324A !important;
                opacity:1 !important;
              }
              .stFileUploaderFileData > *:not([data-testid="stFileUploaderFileName"]) {
                color:#4F687A !important;
                -webkit-text-fill-color:#4F687A !important;
                opacity:1 !important;
              }
              [data-testid="stImageCaption"],
              [data-testid="stImageCaption"] p,
              [data-testid="stImageCaption"] span,
              [data-testid="stImageCaption"] div {
                color:#4F687A !important;
                -webkit-text-fill-color:#4F687A !important;
                opacity:1 !important;
              }
        """

    # Dark-mode-only counterpart to light_only_css, for the rare case where
    # dark mode (not light) needs a targeted fix and light mode's existing
    # native/default rendering must stay untouched — i.e. there's no
    # existing light-mode override here to safely fall back on, unlike the
    # --g-mil-*/--g-info-* variables above.
    dark_only_css = ""
    if not light_mode:
        dark_only_css = """
              /* st.image()'s caption ("synthetic scene — detections")
                 inherits Streamlit's dark base theme's caption styling
                 (small, light text at reduced opacity) directly from
                 config.toml, same root cause as the other native-component
                 fixes above — it's just that light mode already renders
                 acceptably off Streamlit's own default here (confirmed via
                 no existing override), so this stays dark-only rather than
                 an always-present rule that would also repaint light mode. */
              [data-testid="stImageCaption"] {
                color:#8FA9BA !important; opacity:1 !important;
                -webkit-text-fill-color:#8FA9BA !important;
              }
              /* Uploaded-file row (filename + size) is a distinct element
                 from the dropzone — [data-testid="stFileUploaderDropzone"]
                 only covers the empty/idle drop area, not this row, so it
                 never reached these two. Verified against Streamlit's own
                 frontend source (streamlit/static/static/js/5281.*.chunk.js):
                 the filename has a real, stable testid; the size text next
                 to it (e.g. "1.0MB") does not — it's rendered by an
                 internal component with only an auto-generated emotion
                 class, so the only stable hook for it is structural: it's
                 the sibling of the filename inside .stFileUploaderFileData,
                 which IS a stable, literal Streamlit class name (not an
                 emotion-cache hash). */
              [data-testid="stFileUploaderFileName"] {
                color:#8FA9BA !important; opacity:1 !important;
                -webkit-text-fill-color:#8FA9BA !important;
              }
              .stFileUploaderFileData > *:not([data-testid="stFileUploaderFileName"]) {
                color:#8FA9BA !important; opacity:1 !important;
                -webkit-text-fill-color:#8FA9BA !important;
              }
              /* "Max frames" number input. Its own light_only_css
                 counterpart (above, "Text input / number input") never
                 touched [data-testid="stNumberInputContainer"] — only the
                 input fill and step buttons — so dark mode fell all the way
                 through to Streamlit's native dark theme here, which is
                 what's broken: verified live that the container's own
                 native border colour (rgb(18,33,53)) is nearly identical to
                 the dark sidebar background behind it, reading as "no
                 border", while the step-button wrapper (a sibling of
                 [data-baseweb="input"] inside the same container) keeps its
                 own separate native background — the "floating black box".
                 overflow:hidden on the container itself, rather than
                 fighting each child's own corner radii, is what makes the
                 whole thing read as one unified rounded rectangle. */
              [data-testid="stSidebar"] [data-testid="stNumberInputContainer"] {
                border:1px solid #29465C !important; border-radius:10px !important;
                overflow:hidden !important; background:#0B1828 !important;
              }
              [data-testid="stSidebar"] [data-testid="stNumberInputContainer"] [data-baseweb="input"] {
                background:#0B1828 !important; border:none !important;
              }
              [data-testid="stSidebar"] [data-testid="stNumberInputField"] {
                color:#E6EEF5 !important; -webkit-text-fill-color:#E6EEF5 !important;
              }
              [data-testid="stSidebar"] [data-testid="stNumberInputContainer"] > div:not([data-baseweb="input"]) {
                background:#0B1828 !important;
              }
              [data-testid="stSidebar"] [data-testid="stNumberInputStepUp"],
              [data-testid="stSidebar"] [data-testid="stNumberInputStepDown"] {
                background:#0B1828 !important; border:none !important; color:#E6EEF5 !important;
                transition:background-color 120ms ease, box-shadow 120ms ease, color 120ms ease;
              }
              [data-testid="stSidebar"] [data-testid="stNumberInputStepUp"] svg,
              [data-testid="stSidebar"] [data-testid="stNumberInputStepDown"] svg {
                fill:currentColor !important;
              }
              [data-testid="stSidebar"] [data-testid="stNumberInputStepUp"]:hover,
              [data-testid="stSidebar"] [data-testid="stNumberInputStepDown"]:hover {
                background:#1B3852 !important; color:#F0F6FA !important;
                box-shadow:0 0 0 1px rgba(74,180,220,.15), 0 2px 6px rgba(0,0,0,.25) !important;
              }
              [data-testid="stSidebar"] [data-testid="stNumberInputStepUp"]:active,
              [data-testid="stSidebar"] [data-testid="stNumberInputStepDown"]:active {
                background:#142A3E !important;
                box-shadow:inset 0 1px 2px rgba(0,0,0,.25) !important;
              }
        """

    css = f"""
    <style>
      :root {{
        --g-navy:{palette['--g-navy']};
        --g-panel:{palette['--g-panel']};
        --g-panel-2:{palette['--g-panel-2']};
        --g-line:{palette['--g-line']};
        --g-text:{palette['--g-text']};
        --g-muted:{palette['--g-muted']};
        --g-cyan:{palette['--g-cyan']};
        --g-green:{palette['--g-green']};
        --g-amber:{palette['--g-amber']};
        --g-red:{palette['--g-red']};
        --g-info-bg:{palette['--g-info-bg']};
        --g-info-border:{palette['--g-info-border']};
        --g-info-text:{palette['--g-info-text']};
        --g-mil-bg:{palette['--g-mil-bg']};
        --g-mil-accent:{palette['--g-mil-accent']};
        --g-mil-label:{palette['--g-mil-label']};
      }}
      .stApp {{ background:var(--g-navy); color:var(--g-text); }}
      /* Native Streamlit/BaseWeb selectbox + file-uploader dropzone fill.
         Always present (NOT inside light_only_css / the light_mode branch)
         and driven entirely by the --g-* custom properties above: when
         :root's values change on a theme toggle, the browser recomputes
         every var() reference here immediately as part of the same style
         recalc — there's no gap where this markup is absent from the page
         waiting on a script rerun to inject it, which is what caused the
         brief stale-colour flash before. transition:none stops BaseWeb's
         own colour-change animation from adding a visible cross-fade on
         top of that swap. */
      [data-testid="stSidebar"] [data-baseweb="select"][data-baseweb="select"],
      [data-testid="stSidebar"] [data-baseweb="select"][data-baseweb="select"] * {{
        background-color:var(--g-panel) !important; border-color:var(--g-line) !important;
        color:var(--g-text) !important; transition:none !important;
      }}
      [data-testid="stSidebar"] .st-bc {{
        background-color:var(--g-panel) !important; transition:none !important;
      }}
      [data-testid="stSidebar"] [data-baseweb="select"] svg {{
        fill:var(--g-muted) !important;
      }}
      [data-testid="stSidebar"] [data-baseweb="popover"] {{
        background:var(--g-panel) !important;
      }}
      [data-baseweb="menu"] {{
        background:var(--g-panel) !important;
      }}
      [data-baseweb="menu"] li, [data-baseweb="menu"] li * {{
        color:var(--g-text) !important;
      }}
      [data-baseweb="menu"] li:hover {{
        background:var(--g-panel-2) !important;
      }}
      /* Selectbox dropdown menu (Baseline, Tracker, ...). BaseWeb renders
         this list in a portal appended to <body>, OUTSIDE
         [data-testid="stSidebar"] entirely, so it needs its own unscoped
         rule rather than inheriting from the sidebar-scoped ones above.
         stSelectboxVirtualDropdown is a generic testid shared by every
         selectbox on the page, so this covers all of them, not just one. */
      [data-testid="stSelectboxVirtualDropdown"] {{
        background:var(--g-panel) !important; border:1px solid var(--g-line) !important;
      }}
      [data-testid="stSelectboxVirtualDropdown"] li[role="option"] {{
        background:var(--g-panel) !important; color:var(--g-text) !important;
      }}
      [data-testid="stSelectboxVirtualDropdown"] li[role="option"] * {{
        color:var(--g-text) !important;
      }}
      [data-testid="stSelectboxVirtualDropdown"] li[role="option"]:hover,
      [data-testid="stSelectboxVirtualDropdown"] li[aria-selected="true"] {{
        background:var(--g-panel-2) !important;
      }}
      [data-testid="stFileUploaderDropzone"] {{
        background:var(--g-panel) !important; border:1px solid var(--g-line) !important;
        transition:none !important;
      }}
      /* Native st.info() alert ("Upload an image or video to begin...").
         --g-info-bg/border/text are its own dedicated variables (defined
         alongside the rest of the palette in inject_css), not reused from
         --g-panel/--g-line/--g-text, since its current colours don't match
         those 1:1 in either theme. border-width stays 1px in both themes —
         dark's --g-info-border is "transparent" rather than the rule
         omitting a border, so this stays a single always-present
         declaration with no light_mode branch; the alert's own
         box-sizing:border-box means that 1px doesn't shift its size. */
      [data-testid="stAlertContainer"][data-testid="stAlertContainer"] {{
        background-color:var(--g-info-bg) !important;
        border:1px solid var(--g-info-border) !important;
        color:var(--g-info-text) !important;
        transition:none !important;
      }}
      [data-testid="stAlertContainer"][data-testid="stAlertContainer"] * {{
        color:var(--g-info-text) !important;
        transition:none !important;
      }}
      [data-testid="stAlertContainer"][data-testid="stAlertContainer"] svg {{
        fill:var(--g-info-text) !important;
        transition:none !important;
      }}
      [data-testid="stWidgetLabel"] p, [data-testid="stCheckbox"] p,
      [data-testid="stToggle"] p, [data-testid="stCaptionContainer"] p {{
        color:var(--g-text) !important;
      }}
      [data-testid="stHeader"] {{ background:transparent; pointer-events:none; }}
      [data-testid="stHeader"] * {{ pointer-events:auto; }}
      #MainMenu, footer {{ visibility:hidden; }}
      .block-container {{ max-width:1320px; padding-top:3.25rem; padding-bottom:2.5rem; }}
      .guardian-entry {{ max-width:1200px; margin:0 auto; }}
      /* Blur/dim the dashboard's actual pixels while a dialog is open, rather
         than relying only on a translucent overlay: backdrop-filter blends
         with whatever colour sits underneath it, and Streamlit's sidebar
         (#122135) vs main area (#0a1220) are different base colours — even a
         strong overlay alpha left a visible seam between the two. Filtering
         the source content itself is colour-independent, so both regions end
         up genuinely uniform. */
      body:has([data-testid="stDialog"]) [data-testid="stAppViewContainer"] {{
        filter:blur(5px) brightness(.4);
      }}
      [data-testid="stDialog"] {{
        position:fixed !important; inset:0 !important;
        width:100vw !important; height:100vh !important;
        background:rgba(4,10,17,.35) !important;
        box-shadow:none !important;
        padding:2.5rem 1rem;
      }}
      /* Streamlit's own internal dialog wrapper (an unlabelled direct child
         of [data-testid="stDialog"]) ships a built-in rgba(0,0,0,.5)
         background sized to the dialog's scrollable content height, not the
         viewport — a second, static-positioned dim layer stacked under our
         fixed one. It scrolls with the modal's content instead of staying
         viewport-pinned, which is the "second dark rectangle that follows
         the modal while scrolling". Neutralised here; our own overlay above
         already provides the single fullscreen dim/blur. */
      [data-testid="stDialog"] > div {{
        background:transparent !important;
      }}
      [data-testid="stDialog"] [role="dialog"] {{
        max-width:1200px; width:92vw; border-radius:10px;
        margin-top:-38px;
        padding:0.75rem 3.5rem 2.75rem;
        background:var(--g-navy);
        box-shadow:0 8px 24px rgba(0,0,0,.28);
      }}
      [data-testid="stDialog"] button[aria-label="Close"] {{
        display:none;
      }}
      [data-testid="stHeaderActionElements"] {{ display:none; }}
      [class*="st-key-topbar"] {{ border-bottom:1px solid var(--g-line); padding:0 0 .4rem; margin-bottom:.6rem; }}
      [class*="st-key-topbar"] [data-testid="stHorizontalBlock"] {{ align-items:center; }}
      [class*="st-key-topbar"] [data-testid="stToggle"] {{ display:flex; justify-content:flex-end; }}
      .entry-brand {{ color:var(--g-text); font-weight:700; font-size:.86rem; letter-spacing:.13em; }}
      .entry-brand span {{ color:var(--g-cyan); font-weight:500; }}
      .sedic-badge {{
        display:flex; align-items:center;
        border:1px solid var(--g-line); color:var(--g-text); padding:.32rem .55rem;
        font-size:.64rem; letter-spacing:.12em; text-transform:uppercase; white-space:nowrap;
        position:relative; top:-8px;
      }}
      /* Status dot before "SEDIC 2026 · Visual Track". display:flex +
         align-items:center on .sedic-badge above makes the dot and the text
         two flex items sharing one vertical center — not vertical-align
         guesswork against text-line metrics, which is what left the dot
         visibly off-center from the text before. Structure (size, gap)
         lives here so both themes share it; the
         colour is dark mode's own — light_only_css's own .sedic-badge::before
         rule (same selector, later in the stylesheet so it wins the
         cascade tie) still fully overrides this back to var(--g-cyan) in
         light mode, unchanged. Dark mode has no such override, so this
         rule is what actually paints its dot. */
      .sedic-badge::before {{
        content:""; display:inline-block; width:6px; height:6px;
        border-radius:50%; background:#19A7C9; margin-right:.45rem;
        vertical-align:middle;
      }}
      .entry-intro {{ padding:0 0 .85rem; border-bottom:1px solid var(--g-line); }}
      .section-label {{ color:var(--g-cyan); font-size:.67rem; letter-spacing:.14em; text-transform:uppercase; font-weight:700; }}
      .entry-intro h1 {{ color:var(--g-text); margin:.35rem 0 .55rem; font-size:2.75rem; font-weight:600; letter-spacing:0; line-height:1.15; }}
      .entry-subtitle {{ color:var(--g-muted); font-size:1.15rem; font-weight:500; margin:0; letter-spacing:.01em; }}
      .entry-summary {{ color:var(--g-muted); line-height:1.6; max-width:640px; margin:.75rem 0 0; font-size:1rem; }}
      .info-panel {{ border:1px solid var(--g-line); background:var(--g-panel); padding:.85rem 1rem; }}
      .info-panel-title {{ color:var(--g-muted); font-size:.63rem; letter-spacing:.12em; text-transform:uppercase; font-weight:700; margin-bottom:.6rem; }}
      .status-row {{ display:flex; justify-content:space-between; align-items:center; padding:.42rem 0; border-bottom:1px solid var(--g-line); font-size:.82rem; }}
      .status-row:last-child, .meta-row:last-child {{ border-bottom:none; }}
      .status-row .label {{ color:var(--g-muted); }}
      .status-value {{ display:flex; align-items:center; font-weight:600; letter-spacing:.05em; font-size:.72rem; text-transform:uppercase; }}
      .status-dot {{ display:inline-block; width:7px; height:7px; border-radius:50%; margin-right:.45rem; flex:none; }}
      .status-value.ok {{ color:var(--g-green); }}
      .status-value.ok .status-dot {{ background:var(--g-green); }}
      .status-value.warn {{ color:var(--g-amber); }}
      .status-value.warn .status-dot {{ background:var(--g-amber); }}
      .status-value.neutral {{ color:var(--g-cyan); }}
      .status-value.neutral .status-dot {{ background:var(--g-cyan); }}
      .brief-panel {{ border:1px solid var(--g-line); background:var(--g-panel); padding:.9rem 1.25rem; height:100%; }}
      .brief-panel p {{ color:var(--g-muted); line-height:1.6; font-size:.85rem; margin:0; }}
      .meta-row {{ display:flex; justify-content:space-between; padding:.34rem 0; border-bottom:1px solid var(--g-line); font-size:.82rem; }}
      .meta-row .k {{ color:var(--g-muted); }}
      .meta-row .v {{ color:var(--g-text); font-weight:600; }}
      .performance-heading {{ display:flex; justify-content:space-between; gap:1rem; align-items:baseline; margin-bottom:.6rem; }}
      .performance-heading span {{ color:var(--g-muted); font-size:.65rem; letter-spacing:.1em; text-transform:uppercase; }}
      /* Evaluation panel scaled ~18% so it carries visual weight closer to
         the hero column beside it. .section-label / entry-section h2 are
         shared with the pipeline/taxonomy/provenance sections further down
         the page, so those two are scoped to .eval-section specifically;
         everything prefixed mil-recall- or metric- below is exclusive to
         this panel already and safe to scale directly. */
      .eval-section .section-label {{ font-size:.8rem; }}
      .entry-section.eval-section h2 {{ font-size:1.3rem; }}
      .mil-recall-card {{ border:1px solid var(--g-line); border-left:3px solid var(--g-mil-accent); background:var(--g-mil-bg); padding:1.05rem 1.25rem; }}
      .mil-recall-label {{ color:var(--g-mil-label); font-size:.78rem; letter-spacing:.1em; text-transform:uppercase; font-weight:700; }}
      .mil-recall-value {{ color:var(--g-text); font-size:2.85rem; font-weight:700; line-height:1.05; margin:.35rem 0 .45rem; }}
      .mil-recall-status {{ display:flex; align-items:center; gap:.8rem; flex-wrap:wrap; font-size:.92rem; }}
      .mil-recall-check {{ color:var(--g-green); font-weight:600; }}
      .mil-recall-check.fail {{ color:var(--g-red); }}
      .mil-recall-target {{ color:var(--g-muted); }}
      .metric-grid.secondary {{ display:grid; grid-template-columns:1fr 1fr; gap:1px; background:var(--g-line); border:1px solid var(--g-line); border-top:none; }}
      .metric-cell {{ background:var(--g-panel); padding:.6rem .95rem; }}
      .eval-pending {{ border:1px solid var(--g-line); background:var(--g-panel); padding:.75rem .9rem; color:var(--g-muted); font-size:.82rem; line-height:1.55; }}
      .eval-pending code {{ background:var(--g-panel-2); padding:.05rem .3rem; font-size:.78rem; }}
      .metric-label {{ color:var(--g-muted); font-size:.7rem; letter-spacing:.08em; text-transform:uppercase; }}
      .metric-value {{ color:var(--g-muted); font-size:1.25rem; font-weight:600; margin:.25rem 0 .05rem; }}
      .metric-note {{ color:var(--g-muted); font-size:.78rem; }}
      .metric-footer {{ display:flex; gap:1.3rem; flex-wrap:wrap; color:var(--g-muted); font-size:.85rem; margin-top:.7rem; }}
      .metric-footer strong {{ color:var(--g-text); font-weight:600; }}
      .entry-section {{ margin-top:2.75rem; }}
      .entry-section h2 {{ color:var(--g-text); font-size:1.1rem; font-weight:600; margin:.3rem 0 .45rem; }}
      .entry-section > p {{ color:var(--g-muted); margin:0 0 .85rem; line-height:1.5; font-size:.87rem; }}
      .pipeline-row {{ display:flex; align-items:stretch; gap:0; overflow-x:auto; border-top:1px solid var(--g-line); }}
      .pipeline-step {{ flex:1; min-width:150px; padding:1.3rem 1rem .7rem 1rem; border-right:1px solid var(--g-line); }}
      .pipeline-step:first-child {{ padding-left:0; }}
      .pipeline-step:last-child {{ border-right:none; }}
      .pipeline-no {{ color:var(--g-muted); font:600 .68rem monospace; letter-spacing:.05em; }}
      .pipeline-step h3 {{ color:var(--g-text); font-size:.85rem; font-weight:600; margin:.4rem 0 .22rem; }}
      .pipeline-step p {{ color:var(--g-muted); line-height:1.35; font-size:.74rem; margin:0; }}
      .taxonomy-card {{ background:var(--g-panel); border:1px solid var(--g-line); padding:.75rem .85rem; }}
      .taxonomy-card h3 {{ color:var(--g-text); font-size:.78rem; font-weight:600; margin:0 0 .55rem; text-transform:uppercase; letter-spacing:.06em; }}
      .taxonomy-card.civilian {{ border-top:2px solid var(--g-amber); }}
      .taxonomy-card.small-craft {{ border-top:2px solid var(--g-cyan); }}
      .taxonomy-card.military {{ border-top:2px solid var(--g-red); }}
      .tax-chip {{ display:inline-block; border:1px solid var(--g-line); color:var(--g-text); padding:.18rem .38rem; margin:0 .3rem .3rem 0; font-size:.68rem; background:var(--g-panel-2); }}
      .provenance-wrap {{ border:1px solid var(--g-line); overflow-x:auto; }}
      .provenance-table {{ width:100%; border-collapse:collapse; min-width:650px; font-size:.78rem; }}
      .provenance-table th {{ background:var(--g-panel-2); color:var(--g-muted); text-align:left; padding:.62rem .72rem; letter-spacing:.08em; font-size:.61rem; text-transform:uppercase; }}
      .provenance-table td {{ color:var(--g-text); padding:.62rem .72rem; border-top:1px solid var(--g-line); }}
      .provenance-table td.muted {{ color:var(--g-muted); }}
      .ack-panel {{ border-top:1px solid var(--g-line); border-bottom:1px solid var(--g-line); padding:1.1rem 0 .9rem; margin-bottom:1.4rem; text-align:center; }}
      .ack-panel h2 {{ color:var(--g-text); font-size:1rem; margin:.4rem 0 .35rem; }}
      .ack-panel p {{ color:var(--g-muted); font-size:.78rem; margin:0 0 1rem; }}
      .ack-label {{ color:var(--g-muted); font-size:.61rem; letter-spacing:.12em; text-transform:uppercase; margin:.85rem 0 .35rem; }}
      [class*="st-key-ack_logo"] [data-testid="StyledFullScreenButton"] {{ display:none; }}
      .entry-footer {{ color:var(--g-muted); font-size:.64rem; letter-spacing:.08em; text-transform:uppercase; padding:1rem 0; }}
      [data-testid="stButton"] button[kind="primary"] {{ background:var(--g-cyan); color:#f2f7f8; border:1px solid var(--g-cyan); border-radius:2px; font-weight:600; }}
      [data-testid="stButton"] button[kind="primary"]:hover {{ background:var(--g-cyan); opacity:.92; }}
      /* Download buttons (detection log CSV, incident report Markdown):
         a subtle cyan-accented variant of the standard secondary button,
         not the plain default — makes it read as its own actionable
         control instead of blending into the panel, in both themes, via
         the same --g-* tokens already used everywhere else. margin above
         AND below covers both call sites: a gap below the table it
         follows, and a gap on both sides for the incident report button. */
      [data-testid="stDownloadButton"] {{ margin:20px 0; }}
      [data-testid="stDownloadButton"] button {{
        background:var(--g-panel); border:1px solid var(--g-cyan);
        color:var(--g-cyan); border-radius:4px; font-weight:600;
      }}
      [data-testid="stDownloadButton"] button p,
      [data-testid="stDownloadButton"] button span {{ color:var(--g-cyan); }}
      [data-testid="stDownloadButton"] button svg {{ fill:var(--g-cyan); }}
      [data-testid="stDownloadButton"] button:hover {{
        background:var(--g-panel-2); border-color:var(--g-cyan); color:var(--g-cyan);
      }}
      /* Centering is handled by the [1,1.3,1] column layout around the
         button in landing_page(), not CSS — this block only styles it.
         Streamlit's own flex gap between element-containers floors at
         16px regardless of a spacer element's own height, so pulling the
         CTA's column row closer to the brief card needs a negative margin
         on the row itself (a real flex item), not a spacer div. */
      [data-testid="stHorizontalBlock"]:has([class*="st-key-entry_launch"]) {{
        margin-top:22px;
      }}
      /* font-size uses clamp() so the label shrinks just enough to stay on
         one line as the briefing dialog narrows below ~1280px (it's
         width:92vw under a 1200px cap, so its content — including this
         button — scales down with the viewport at smaller sizes). Fixed
         at 19px would wrap to two lines around 1024px wide. */
      [class*="st-key-entry_launch"] button[kind="primary"] {{
        background:#1f4d3a; color:#ffffff; border:1px solid #3f7a61;
        border-radius:10px;
        padding:.85rem 1.1rem; font-weight:700; font-size:clamp(13px, 2.34vw - 11px, 19px); letter-spacing:.01em;
        display:flex; align-items:center; justify-content:flex-start;
        transition:background .2s ease, border-color .2s ease, box-shadow .2s ease, transform .2s ease;
        box-shadow:0 1px 2px rgba(0,0,0,.2);
      }}
      /* Streamlit's own rule setting font-size:1rem on p/ol/ul/dl elements
         applies directly to this <p>, so it does NOT inherit the clamp()
         above from the button by default — without this override
         the text silently stays at 16px regardless of the button's own
         font-size (the bug behind the two-line wrap at ~1024px). */
      [class*="st-key-entry_launch"] button[kind="primary"] p {{ text-align:left; margin:0; font-size:inherit; }}
      [class*="st-key-entry_launch"] button[kind="primary"]::after {{
        content:"→"; margin-left:auto; padding-left:1.6rem; color:#ffffff; font-weight:700;
        transition:transform .2s ease;
      }}
      [class*="st-key-entry_launch"] button[kind="primary"]:hover {{
        background:#2c6e50; border-color:#4f8f70; filter:none;
        box-shadow:0 4px 14px rgba(0,0,0,.28); transform:translateY(-1px);
      }}
      [class*="st-key-entry_launch"] button[kind="primary"]:hover::after {{
        transform:translateX(4px);
      }}
      [data-testid="stSidebar"] {{ border-right:1px solid var(--g-line); }}
      [data-testid="stSidebar"] h3 {{ color:var(--g-text); font-size:.9rem; letter-spacing:.04em; }}
      /* Containment: Streamlit's widget wrappers are flex children, which
         default to min-width:auto — long content (e.g. the weights path)
         then forces them to their intrinsic width regardless of how much
         room the card actually has, pushing widgets/icons past the card
         edge. Resetting box-sizing/min-width/max-width on every descendant
         is what actually stops that, not just padding tweaks. */
      [data-testid="stSidebar"], [data-testid="stSidebar"] * {{ box-sizing:border-box; }}
      [data-testid="stSidebar"] [class*="st-key-sb_card_"] {{
        background:var(--g-panel); border:1px solid var(--g-line);
        border-radius:8px; padding:1rem 1.1rem;
        width:100%; max-width:100%; overflow:hidden;
      }}
      /* Extra breathing room below the Models card specifically — it's
         followed by a sibling group (AI Operating Thresholds) within the
         same section, not a new section, so it needs the ~20-24px group
         rhythm rather than the ~32px used between sections. Set directly
         on the card (a real flex item) rather than via a spacer element:
         an empty spacer element-container floors at 32px regardless of
         its own height, since Streamlit's own inter-element gap is what
         dominates then, not this element's height/margin. */
      [data-testid="stSidebar"] [class*="st-key-sb_card_model"] {{
        margin-bottom:16px; padding:16px 24px 30px;
      }}
      [data-testid="stSidebar"] [class*="st-key-sb_card_"] * {{
        min-width:0; max-width:100%;
      }}
      [data-testid="stSidebar"] [data-testid="stTextInput"] input,
      [data-testid="stSidebar"] [data-testid="stNumberInput"] input,
      [data-testid="stSidebar"] [data-baseweb="select"] {{
        width:100%; max-width:100%;
      }}
      [data-testid="stSidebar"] [data-testid="stTextInput"] input {{
        overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
      }}
      /* BaseWeb reserves ~9px of padding-top on every slider so the
         floating value bubble (e.g. "0.25") doesn't collide with the
         label above it. Tightened, not removed — zeroing it would clip
         the bubble into the label text. */
      [data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] {{
        padding-top:7px;
      }}
      .sb-label {{ color:var(--g-muted); font-size:.72rem; letter-spacing:.12em; text-transform:uppercase; font-weight:700; margin:0 0 1rem; }}
      .sb-title {{ color:var(--g-text); font-size:1.5rem; font-weight:700; margin:0 0 1rem; line-height:1.15; }}
      .sb-group-label {{ color:var(--g-muted); font-size:.8rem; letter-spacing:.06em; text-transform:uppercase; font-weight:600; margin:0 0 .5rem; }}
      .sb-value {{ font-size:.85rem; color:var(--g-muted); margin:-2px 0 0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
      .sb-desc {{
        color:var(--g-muted); font-size:.8rem; line-height:1.45; text-align:left;
        white-space:normal; overflow-wrap:break-word; word-break:break-word;
        width:100%; max-width:100%; margin:0 0 10px;
      }}
      /* One shared rule for every widget label row (text input, slider,
         select, toggle, ...): label text on the left, help icon pinned to
         the far right — so every row in the sidebar uses the identical
         layout mechanism instead of one-off nudges per widget type. */
      [data-testid="stSidebar"] [data-testid="stWidgetLabel"] {{
        display:flex; align-items:center; justify-content:space-between; width:100%;
      }}
      /* Toggle/checkbox rows put the switch, a hidden input and the label
         wrapper as siblings of one outer flex row. Giving that row
         align-items:center + a gap groups switch+text on the left, and
         making the label-wrapper flex:1 lets the shared rule above push
         its own help icon out to the same far-right edge the text-input
         rows already use — the same mechanism, not a separate one. */
      [data-testid="stSidebar"] [data-baseweb="checkbox"] {{
        display:flex; align-items:center; gap:10px; width:100%; padding-right:0;
      }}
      [data-testid="stSidebar"] [data-baseweb="checkbox"] > div:has([data-testid="stWidgetLabel"]) {{
        flex:1;
      }}
      .sb-value b {{ color:var(--g-text); font-weight:600; }}
      [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{ font-size:.98rem; font-weight:500; }}
      [class*="st-key-sidebar_briefing"] button {{ display:flex; align-items:center; justify-content:space-between; text-align:left; font-weight:600; font-size:.85rem; background:var(--g-panel); border:1px solid var(--g-line); border-left:3px solid var(--g-cyan); border-radius:8px; padding:.8rem 1.1rem; width:100%; max-width:100%; }}
      [class*="st-key-sidebar_briefing"] button::after {{ content:"→"; color:var(--g-cyan); margin-left:.6rem; font-weight:700; transition:transform .15s ease; flex:none; }}
      [class*="st-key-sidebar_briefing"] button:hover {{ background:var(--g-panel-2); border-color:var(--g-cyan); }}
      [class*="st-key-sidebar_briefing"] button:hover::after {{ transform:translateX(3px); }}
      .sb-legend {{ display:flex; flex-wrap:wrap; gap:.45rem; margin-top:.5rem; max-width:100%; }}
      .sb-legend-pill {{ display:inline-flex; align-items:center; gap:.4rem; padding:.42rem .85rem; border-radius:999px; font-size:.76rem; font-weight:600; line-height:1; }}
      .sb-legend-dot {{ width:7px; height:7px; border-radius:50%; flex:none; }}
      .guardian-header {{ background:var(--g-panel); border:1px solid var(--g-line); border-left:3px solid var(--g-cyan); padding:.9rem 1.15rem; margin-bottom:1rem; }}
      .guardian-header h1 {{ margin:0; font-size:1.35rem; font-weight:600; letter-spacing:0; color:var(--g-text); }}
      .guardian-header h1 span {{ color:var(--g-muted); font-weight:500; }}
      .guardian-header p {{ margin:.3rem 0 0; color:var(--g-muted); font-size:.85rem; }}
      .alert-military {{ background:rgba(217,102,102,.14); border:1px solid rgba(217,102,102,.5); border-left:3px solid var(--g-red); color:var(--g-text); padding:.7rem 1rem; margin:.3rem 0 1rem; font-size:.92rem; font-weight:600; letter-spacing:.2px; }}
      .alert-military small {{ display:block; font-weight:400; letter-spacing:0; font-size:.8rem; opacity:.85; margin-top:.15rem; }}
      .alert-clear {{ background:rgba(120,182,159,.14); border:1px solid rgba(120,182,159,.4); border-left:3px solid var(--g-green); color:var(--g-text); padding:.65rem 1rem; margin:.3rem 0 1rem; font-weight:500; font-size:.88rem; }}
      .chip-row {{ display:flex; gap:1px; flex-wrap:wrap; background:var(--g-line); border:1px solid var(--g-line); margin:.2rem 0 .6rem; }}
      .chip {{ padding:.55rem .85rem; min-width:116px; flex:1; background:var(--g-panel); }}
      .chip .k {{ font-size:.68rem; text-transform:uppercase; letter-spacing:.06em; color:var(--g-muted); }}
      .chip .v {{ font-size:1.3rem; font-weight:600; line-height:1.15; color:var(--g-text); }}
      .chip .s {{ font-size:.7rem; color:var(--g-muted); margin-top:.15rem; }}
      /* Nationality split: one stacked bar under the percentage chips, so the
         RMN-vs-foreign balance reads at a glance without parsing the table. */
      /* box-sizing + max-width matter: this is the only element here with an
         explicit width AND a border, and under content-box that is a 2px
         horizontal overflow of the block container — which Streamlit's
         resize observer turns into a re-measure/re-render loop (React #185,
         "maximum update depth"). Keep it strictly inside its container. */
      .nat-bar {{ display:flex; box-sizing:border-box; width:100%; max-width:100%; height:12px; border:1px solid var(--g-line); background:var(--g-panel-2); overflow:hidden; margin:.1rem 0 .45rem; }}
      .nat-seg {{ height:100%; flex:none; min-width:0; }}
      .nat-note {{ color:var(--g-muted); font-size:.78rem; margin:0 0 .9rem; }}
      /* Incident report panel heading — deliberately NOT st.subheader()
         (Streamlit's default ~1.5-1.75rem), which is what made this look
         like an oversized AI-report title. Same restrained scale as the
         rest of the dashboard's section headings. */
      .incident-head {{ display:flex; align-items:baseline; justify-content:space-between; flex-wrap:wrap; gap:.4rem .8rem; margin:1.6rem 0 .6rem; }}
      .incident-title {{ color:var(--g-text); font-size:1rem; font-weight:700; letter-spacing:.03em; text-transform:uppercase; margin:0; }}
      .incident-sub {{ color:var(--g-muted); font-size:.78rem; }}
      .legend-swatch {{ display:inline-block; width:9px; height:9px; margin-right:.35rem; vertical-align:middle; border:1px solid var(--g-line); }}
      .table-wrap {{ overflow-x:auto; overflow-y:auto; border:1px solid var(--g-line); max-height:none; }}
      .table-wrap.scrollable {{ max-height:460px; }}
      /* Streamlit's own markdown-container stylesheet applies a
         margin-bottom:1rem to any <table> rendered via unsafe_allow_html
         (selector specificity: class + type, e.g. ".st-emotion-cache-xxx
         table"), which beats a plain ".det-table" class selector despite
         coming first in the cascade — that 16px was the "empty space
         below the table" bug, not table-wrap or any parent height rule.
         !important because that generated class name isn't something we
         can target directly (it changes across Streamlit builds). */
      .det-table {{ width:100%; border-collapse:separate; border-spacing:0; font-size:.87rem; margin:0 !important; }}
      .det-table th {{ position:sticky; top:0; text-align:left; padding:.5rem .7rem; font-weight:600; text-transform:uppercase; font-size:.72rem; letter-spacing:.6px; background:var(--g-panel-2); color:var(--g-muted); border-bottom:1px solid var(--g-line); }}
      .det-table td {{ padding:.42rem .7rem; color:var(--g-text); border-bottom:1px solid var(--g-line); }}
      /* Tracking timeline accordion (the app's only st.expander, so this
         is safe unscoped): stExpanderDetails only exists in the DOM while
         expanded — Streamlit unmounts it entirely when collapsed — so this
         padding costs nothing in the collapsed state. No extra width/margin
         inset on .table-wrap itself: it fills 100% of this padded content
         area and is centred purely by that padding, not by its own sizing. */
      [data-testid="stExpanderDetails"] {{ padding:24px 24px 26px; }}
      {light_only_css}
      {dark_only_css}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Schema + model access (cached so repeat inference is instant)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_class_groups() -> dict[str, str]:
    """class name -> group name, straight from configs/schema.yaml."""
    return gp.class_groups()


@st.cache_data(show_spinner=False)
def get_military_classes() -> set[str]:
    return gp._military_class_names()


@st.cache_resource(show_spinner="Loading detection model…")
def get_model(weights: str):
    """Warm the YOLO weights once per session; later runs reuse this."""
    return gp.load_model(weights)


@st.cache_resource(show_spinner="Loading nationality classifier…")
def get_fg_model(weights: str):
    """Warm the bonus RMN-vs-Foreign classifier once per session."""
    return fg.load_checkpoint(weights)


@st.cache_data(show_spinner=False)
def load_eval_summary(path_str: str, mtime: float) -> dict | None:
    """Parse the military recall gate out of `outputs/eval/test_eval.md`.

    `mtime` is only there to bust the cache when `src.eval.detail` rewrites the
    report — it's a plain (non-underscore-prefixed) arg deliberately, so it's
    part of Streamlit's cache key; a leading underscore would exclude it from
    hashing and freeze this on whatever the first call returned.
    Returns None if the report hasn't been generated yet.
    """
    path = Path(path_str)
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")

    domain_recall: dict[str, float] = {}
    _, _, after = text.partition("Military recall per domain")
    for m in re.finditer(
        r"\|\s*\*{0,2}(aerial|surface|overall)\*{0,2}\s*\|\s*([\d.]+)\s*\|", after
    ):
        domain_recall[m.group(1)] = float(m.group(2))
    if "overall" not in domain_recall:
        return None

    split_m = re.search(r"—\s*(\w+)\s*split", text)
    iou_m = re.search(r"IoU\s*([\d.]+)", text)

    return {
        "split": split_m.group(1).upper() if split_m else path.stem.split("_")[0].upper(),
        "iou": float(iou_m.group(1)) if iou_m else None,
        "military_recall_overall": domain_recall.get("overall"),
        "military_recall_aerial": domain_recall.get("aerial"),
        "military_recall_surface": domain_recall.get("surface"),
    }


def group_of(class_name: str) -> str:
    return get_class_groups().get(class_name, "other")


def colour_of(class_name: str) -> tuple[tuple[int, int, int], str, str]:
    return GROUP_COLOURS.get(group_of(class_name), GROUP_COLOURS["other"])


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def annotate(
    image_bgr: np.ndarray,
    items: list[tuple[Detection, int | None]],
) -> np.ndarray:
    """Draw colour-coded boxes + labels. Returns a new BGR image."""
    out = image_bgr.copy()
    h, w = out.shape[:2]
    thickness = max(2, round(min(h, w) / 400))
    font_scale = max(0.45, min(h, w) / 1100)

    for det, track_id in items:
        bgr, _, _ = colour_of(det.class_name)
        x1, y1, x2, y2 = (round(v) for v in det.bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), bgr, thickness)

        label = f"{det.class_name} {det.confidence:.2f}"
        if track_id is not None:
            label = f"#{track_id} {label}"
        (tw, th), base = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(1, thickness - 1)
        )
        # Keep the label inside the frame when the box hugs the top edge.
        ty = y1 - base if y1 - th - base > 0 else y1 + th + base
        cv2.rectangle(
            out,
            (x1, ty - th - base),
            (min(x1 + tw + 6, w), ty + base),
            bgr,
            -1,
        )
        cv2.putText(
            out, label, (x1 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
            (255, 255, 255), max(1, thickness - 1), cv2.LINE_AA,
        )
    return out


def to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def show_image(image_rgb: np.ndarray, caption: str, slot=None) -> None:
    """Full-width image, tolerant of the Streamlit version's kwarg name.

    `use_container_width` only exists on st.image from 1.41; older builds want
    `use_column_width`. The demo must not care which one is installed.
    """
    target = slot if slot is not None else st
    try:
        target.image(image_rgb, caption=caption, use_container_width=True)
    except TypeError:
        target.image(image_rgb, caption=caption, use_column_width=True)


def placeholder_canvas(width: int = 1000, height: int = 560) -> np.ndarray:
    """A navy 'no input' canvas so stub mode has something to draw on."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):  # simple vertical sea-to-sky gradient
        t = y / height
        canvas[y, :] = (int(40 + 45 * t), int(24 + 30 * t), int(12 + 16 * t))
    cv2.putText(
        canvas, "STUB MODE - synthetic scene", (28, height - 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (170, 190, 210), 1, cv2.LINE_AA,
    )
    return canvas


# ---------------------------------------------------------------------------
# Summary / metrics panels
# ---------------------------------------------------------------------------
def render_alert(dets: list[Detection]) -> None:
    military = get_military_classes()
    flagged = [d for d in dets if d.class_name in military]
    if flagged:
        top = max(d.confidence for d in flagged)
        st.markdown(
            f"""
            <div class="alert-military">
              MILITARY VESSEL DETECTED — {len(flagged)} CONTACT(S)
              <small>Highest confidence {top:.2f} · review required</small>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="alert-clear">No military contacts at the current '
            "threshold</div>",
            unsafe_allow_html=True,
        )


def render_summary(dets: list[Detection], track_ids: set[int] | None = None) -> None:
    """Total vessels + per-group counts as colour-coded chips."""
    counts: dict[str, int] = {g: 0 for g in GROUP_COLOURS}
    for d in dets:
        counts[group_of(d.class_name)] += 1

    total_label = "Tracked vessels" if track_ids is not None else "Total vessels"
    total_value = len(track_ids) if track_ids is not None else len(dets)

    chips = [
        (
            '<div class="chip" style="border-top:2px solid var(--g-cyan)">'
            f'<div class="k">{total_label}</div>'
            f'<div class="v">{total_value}</div></div>'
        )
    ]
    for group, (_, hexcol, label) in GROUP_COLOURS.items():
        if group == "other" and not counts[group]:
            continue
        chips.append(
            f'<div class="chip" style="border-top:2px solid {hexcol}">'
            f'<div class="k">{label}</div>'
            f'<div class="v" style="color:{hexcol}">{counts[group]}</div></div>'
        )
    st.markdown(f'<div class="chip-row">{"".join(chips)}</div>',
                unsafe_allow_html=True)


def render_metrics(dets: list[Detection], elapsed_s: float, frames: int = 1) -> None:
    """Detection count / mean confidence / inference speed."""
    avg_conf = float(np.mean([d.confidence for d in dets])) if dets else 0.0
    fps = frames / elapsed_s if elapsed_s > 0 else 0.0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Detections", len(dets))
    c2.metric("Avg confidence", f"{avg_conf:.2f}" if dets else "—")
    c3.metric("Inference", f"{elapsed_s * 1000 / max(1, frames):.0f} ms/frame")
    c4.metric("Throughput", f"{fps:.1f} FPS")


def detections_dataframe(
    items: list[tuple[Detection, int | None]], with_track: bool,
    nationality: dict[int, fg.ClassificationResult] | None = None,
) -> pd.DataFrame:
    rows = []
    for idx, (det, track_id) in enumerate(items):
        x1, y1, x2, y2 = (round(v, 1) for v in det.bbox) if det.bbox else (0, 0, 0, 0)
        row = {
            "class": det.class_name,
            "group": GROUP_COLOURS[group_of(det.class_name)][2],
            "confidence": round(det.confidence, 3),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        }
        if nationality is not None:
            result = nationality.get(idx)
            row["nationality"] = FG_LABEL_DISPLAY.get(result.label, result.label) if result else None
            row["nationality_conf"] = round(result.confidence, 3) if result else None
        if with_track:
            row = {"frame": det.frame, "timestamp_s": det.timestamp,
                   "track_id": track_id, **row}
        rows.append(row)
    df = pd.DataFrame(rows)
    if with_track and not df.empty:
        # Nullable int, so an untracked box reads as blank rather than "nan".
        df["track_id"] = df["track_id"].astype("Int64")
    return df


# ---------------------------------------------------------------------------
# Bonus: RMN-vs-Foreign nationality classification (optional 2nd stage)
# ---------------------------------------------------------------------------
def classify_military_crops(
    image_bgr: np.ndarray,
    dets: list[Detection],
    weights: str,
    conf_floor: float,
) -> dict[int, fg.ClassificationResult]:
    """Run the bonus classifier on every military-group detection box.

    Returns `{index into dets: ClassificationResult}`, index-keyed rather than
    keyed on `Detection` itself since it isn't hashable. Best-effort: a
    crop/model failure is skipped rather than raised — this is an optional
    overlay on the frozen detection path, not part of it, so it must never
    take the main detection view down with it.
    """
    military = get_military_classes()
    idxs = [i for i, d in enumerate(dets) if d.class_name in military]
    if not idxs:
        return {}
    try:
        model, classes, imgsz = get_fg_model(weights)
    except Exception as exc:  # noqa: BLE001 — bonus overlay must never crash the view
        st.warning(f"Nationality classifier unavailable: {type(exc).__name__}: {exc}")
        return {}
    transform = fg.build_transform(imgsz)
    image_rgb = to_rgb(image_bgr)
    h, w = image_rgb.shape[:2]

    results: dict[int, fg.ClassificationResult] = {}
    for i in idxs:
        x1, y1, x2, y2 = dets[i].bbox
        bw, bh = x2 - x1, y2 - y1
        pad_x, pad_y = bw * FG_PAD_FRAC, bh * FG_PAD_FRAC
        cx1, cy1 = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
        cx2, cy2 = min(w, int(x2 + pad_x)), min(h, int(y2 + pad_y))
        if cx2 - cx1 < 8 or cy2 - cy1 < 8:
            continue
        try:
            crop = Image.fromarray(image_rgb[cy1:cy2, cx1:cx2])
            results[i] = fg.classify(crop, model, transform, classes, conf_floor)
        except Exception:  # noqa: BLE001, S112 — one bad crop must not skip the rest
            continue
    return results


def render_nationality_split(
    dets: list[Detection],
    nationality: dict[int, fg.ClassificationResult],
) -> None:
    """Headline percentage split of the RMN-vs-Foreign calls.

    The per-contact answer already lives in the results table; this is the
    at-a-glance answer — what share of the classified military contacts came
    back Malaysian RMN, foreign, or below the confidence floor. Percentages
    are of the contacts actually classified, and the note spells that
    denominator out so a skipped crop can't silently inflate a share.
    """
    if not nationality:
        return
    total = len(nationality)
    counts: dict[str, int] = {}
    confs: dict[str, list[float]] = {}
    for result in nationality.values():
        counts[result.label] = counts.get(result.label, 0) + 1
        confs.setdefault(result.label, []).append(result.confidence)

    ordered = [name for name in FG_LABEL_ORDER if name in counts]
    # Any label the classifier gains later still shows up, just after these.
    ordered += [name for name in counts if name not in FG_LABEL_ORDER]

    chips, segments = [], []
    for name in ordered:
        pct = 100.0 * counts[name] / total
        colour = FG_LABEL_COLOURS.get(name, "var(--g-cyan)")
        display = FG_LABEL_DISPLAY.get(name, name)
        avg = sum(confs[name]) / len(confs[name])
        chips.append(
            f'<div class="chip" style="border-top:2px solid {colour}">'
            f'<div class="k">{display}</div>'
            f'<div class="v" style="color:{colour}">{pct:.0f}%</div>'
            f'<div class="s">{counts[name]} of {total} · avg conf {avg:.0%}</div>'
            "</div>"
        )
        segments.append(
            f'<div class="nat-seg" style="width:{pct:.4f}%;background:{colour}" '
            f'title="{display} — {pct:.0f}%"></div>'
        )

    military_total = sum(1 for d in dets if d.class_name in get_military_classes())
    skipped = military_total - total
    note = f"Share of the {total} military contact(s) classified"
    note += f"; {skipped} skipped (crop too small)." if skipped > 0 else "."
    st.markdown(
        f'<div class="chip-row">{"".join(chips)}</div>'
        f'<div class="nat-bar">{"".join(segments)}</div>'
        f'<div class="nat-note">{note}</div>',
        unsafe_allow_html=True,
    )


def render_nationality_results(
    image_bgr: np.ndarray,
    dets: list[Detection],
    nationality: dict[int, fg.ClassificationResult],
) -> None:
    """Bonus panel: a crop thumbnail + predicted nationality per military contact."""
    if not nationality:
        return
    st.markdown('<div class="sb-label">Bonus: nationality classification</div>',
                unsafe_allow_html=True)
    st.caption(
        "RMN-vs-Foreign 2nd stage. Val accuracy 0.98, but that figure is "
        "inflated by an image-source domain gap between the training sets, "
        "not pure vessel-identity recognition — treat these as indicative, "
        "not certain. See docs/PROGRESS.md §4."
    )
    render_nationality_split(dets, nationality)
    image_rgb = to_rgb(image_bgr)
    items = list(nationality.items())
    cols = st.columns(min(4, len(items)))
    for n, (i, result) in enumerate(items):
        x1, y1, x2, y2 = (round(v) for v in dets[i].bbox)
        crop = image_rgb[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        col = cols[n % len(cols)]
        if crop.size:
            label = FG_LABEL_DISPLAY.get(result.label, result.label)
            show_image(crop, f"{label} · {result.confidence:.0%}", slot=col)


def render_table(df: pd.DataFrame) -> None:
    """Results table, rows tinted by group.

    Rendered as plain HTML on purpose: st.dataframe serializes through pyarrow,
    which segfaults the whole server on some numpy/pyarrow combinations. A demo
    that can hard-crash on the results table is not worth the sortable columns.
    """
    if df.empty:
        st.info("No detections above the current thresholds.")
        return

    tint = {label: hexcol for _, hexcol, label in GROUP_COLOURS.values()}
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    body = []
    for row in df.to_dict("records"):
        colour = tint.get(row.get("group", ""), "#94a3b8")
        cells = "".join(
            f'<td>{"—" if v is None or pd.isna(v) else v}</td>'
            for v in row.values()
        )
        body.append(
            f'<tr style="background:{colour}1f;border-left:3px solid {colour}">'
            f"{cells}</tr>"
        )
    # Only cap the height (and start scrolling) once there are enough rows
    # that the table would otherwise run long — small tables hug their
    # content instead of reserving a tall, mostly-empty box.
    wrap_class = "table-wrap scrollable" if len(df) > 10 else "table-wrap"
    st.markdown(
        f"""
        <div class="{wrap_class}">
          <table class="det-table">
            <thead><tr>{head}</tr></thead>
            <tbody>{"".join(body)}</tbody>
          </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Inference wrappers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=16)
def run_image(
    image_bytes: bytes,
    suffix: str,
    weights: str,
    conf: float,
    conf_military: float,
    stub: bool,
) -> tuple[list[Detection], float]:
    """Detect on one image. Cached on content + settings, so slider tweaks that
    revisit an earlier value are instant and re-renders never re-infer."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(image_bytes)
        path = tmp.name
    try:
        start = time.perf_counter()
        dets = gp.predict(
            source=path if image_bytes else "none",
            weights=weights or None,
            conf=conf,
            conf_military=conf_military,
            stub=stub,
        )
        return dets, time.perf_counter() - start
    finally:
        Path(path).unlink(missing_ok=True)


def decode_image(image_bytes: bytes) -> np.ndarray | None:
    """Bytes -> BGR array, or None when the file isn't a readable image."""
    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------
def image_view(
    image_bytes: bytes | None, suffix: str, settings: dict, filename: str
) -> None:
    """Still-image flow: auto-runs on upload and on every slider change."""
    if image_bytes:
        original = decode_image(image_bytes)
        if original is None:
            st.error(
                f"**{filename}** could not be read as an image. Try a JPG or PNG "
                "export of the same file."
            )
            return
    else:
        original = placeholder_canvas()

    try:
        dets, elapsed = run_image(
            image_bytes or b"",
            suffix or ".jpg",
            settings["weights"],
            settings["conf"],
            settings["conf_military"],
            settings["stub"],
        )
    except FileNotFoundError as exc:
        st.error(f"**Model unavailable.** {exc}")
        return
    except Exception as exc:  # noqa: BLE001 — a live demo must never show a traceback
        st.error(f"**Detection failed on this file.** {type(exc).__name__}: {exc}")
        return

    items = [(d, None) for d in dets]
    annotated = annotate(original, items)

    render_alert(dets)

    nationality: dict[int, fg.ClassificationResult] = {}
    if settings.get("fg_enabled"):
        try:
            nationality = classify_military_crops(
                original, dets, settings["fg_weights"], settings["fg_conf_floor"]
            )
        except Exception as exc:  # noqa: BLE001 — bonus overlay must never crash the view
            st.warning(f"Nationality classifier failed: {type(exc).__name__}: {exc}")
        render_nationality_results(original, dets, nationality)

    render_summary(dets)
    render_metrics(dets, elapsed)

    show_original = st.toggle(
        "Show original (before / after)", value=False,
        help="Flip between the raw input and the annotated detections.",
    )
    show_image(
        to_rgb(original if show_original else annotated),
        f"{filename} — {'original' if show_original else 'detections'}",
    )

    st.subheader(f"Detections ({len(dets)})")
    render_table(detections_dataframe(
        items, with_track=False,
        nationality=nationality if settings.get("fg_enabled") else None,
    ))


def render_incident_summary(summary: ReportSummary, source_name: str) -> None:
    """Compact SOC-style incident summary: a KPI strip + class-breakdown
    badges + a collapsed timeline table, in place of the verbose written
    report on screen. Pure presentation over ReportSummary's own fields —
    summarize_detections() is untouched, and the downloadable .md report
    (render_incident_markdown, called separately) stays byte-identical to
    the CLI output; only how this data looks on screen changes.
    """
    st.markdown(
        f'<div class="incident-head">'
        f'<div class="incident-title">Incident Report</div>'
        f'<div class="incident-sub">{source_name}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if summary.total_rows == 0:
        st.info("No detections were logged for this clip.")
        return

    # No frame-count tile here on purpose — total_rows is a per-frame number
    # (e.g. 180 for a 6s clip at ~30fps of one ship) that reads as "180
    # ships" at a glance; Tracked is the number that actually answers "how
    # many vessels". The raw count still lives in the Detection log table
    # below and its CSV, just not surfaced as a headline KPI here.
    military_accent = "var(--g-red)" if summary.n_military_vessels else None
    kpis = [
        ("Duration", f"{summary.clip_duration_s:.2f}s", None, None),
        ("Tracked", str(summary.n_tracked_vessels), None, "Distinct tracked vessels in this clip."),
        ("Military", str(summary.n_military_vessels), military_accent, "Distinct military-class tracks."),
        ("Confidence", f"{summary.overall_mean_confidence * 100:.0f}%", None, None),
    ]

    def _chip(label: str, value: str, accent: str | None, tooltip: str | None) -> str:
        border = f' style="border-top:2px solid {accent}"' if accent else ""
        value_style = f' style="color:{accent}"' if accent else ""
        title = f' title="{tooltip}"' if tooltip else ""
        return (
            f'<div class="chip"{border}{title}>'
            f'<div class="k">{label}</div>'
            f'<div class="v"{value_style}>{value}</div>'
            f"</div>"
        )

    chips_html = "".join(_chip(label, value, accent, tooltip) for label, value, accent, tooltip in kpis)
    st.markdown(f'<div class="chip-row">{chips_html}</div>', unsafe_allow_html=True)

    if summary.class_counts:
        # No frame-count per badge here either — class_counts tallies rows
        # (frame-level), same trap as the removed "Frame detections" KPI
        # (e.g. "military_vessel · 180" reading as 180 ships). Sort order
        # still uses the count (most-frequent class first), just doesn't
        # display it.
        badges = "".join(
            f'<span class="tax-chip">{cls}</span>'
            for cls, _count in sorted(summary.class_counts.items(), key=lambda kv: -kv[1])
        )
        st.markdown(
            f'<div class="incident-sub" style="margin:.6rem 0 .2rem">Detection class</div>'
            f"<div>{badges}</div>",
            unsafe_allow_html=True,
        )

    if summary.tracks:
        with st.expander(f"Tracking timeline ({len(summary.tracks)} track(s))"):
            timeline_df = pd.DataFrame([
                {
                    "track": t.track_id,
                    "class": t.class_name,
                    "first_seen_s": t.first_seen_s,
                    "duration_s": t.duration_s,
                    "detections": t.n_detections,
                    "max_conf": t.max_confidence,
                    "mean_conf": t.mean_confidence,
                }
                for t in summary.tracks
            ])
            render_table(timeline_df)


def _render_detection_log_panel(video_id: str) -> None:
    """Detection log table + CSV download for the last processed video.

    Same session-state read-back as _render_video_analysis_panels, and for
    the same reason: st.download_button click is itself a rerun, and that
    rerun reads "Run detection on video" back as False, so without this the
    table (and its own download button) would vanish the instant you used
    the CSV download button — the fresh-run code path that renders them
    never executes again on that rerun.
    """
    if st.session_state.get("guardian_last_video_key") != video_id:
        return
    df = st.session_state.get("guardian_last_video_df")
    source_name = st.session_state.get("guardian_last_video_source", "video")
    if df is None:
        return

    st.subheader(f"Detection log ({len(df)} rows)")
    render_table(df.head(500))
    if len(df) > 500:
        st.caption("Showing the first 500 rows — the CSV contains all of them.")
    st.download_button(
        "Download detection log (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"{Path(source_name).stem}_detections.csv",
        mime="text/csv",
        icon=":material/download:",
        key=f"dl_csv_{video_id}",
    )


def _render_video_analysis_panels(settings: dict, video_id: str) -> None:
    """Incident report + behaviour-anomaly panels for the last processed video.

    Reads the detection log back from st.session_state rather than taking a
    `df` argument, so it renders identically whether called right after a
    fresh run (this rerun) or from video_view()'s "button not clicked" early
    return on a later rerun (e.g. a sidebar slider changed) — in the latter
    case `df` was never recomputed locally, only what's stashed in session
    state survives. `video_id` guards against showing a previous video's
    stale report after a new file is uploaded: if it doesn't match what's
    stored, nothing renders (falls back gracefully — no error, no stale
    data), rather than assuming the stored df is still relevant.
    """
    if st.session_state.get("guardian_last_video_key") != video_id:
        return
    df = st.session_state.get("guardian_last_video_df")
    source_name = st.session_state.get("guardian_last_video_source", "video")
    if df is None:
        return
    stem = Path(source_name).stem

    summary = summarize_detections(df)
    render_incident_summary(summary, source_name=stem)
    report_md = render_incident_markdown(summary, source_name=stem)
    st.download_button(
        "Download incident report (Markdown)",
        data=report_md.encode("utf-8"),
        file_name=f"{stem}_incident_report.md",
        mime="text/markdown",
        icon=":material/download:",
        key=f"dl_incident_{video_id}",
    )

    if not settings.get("behavior_enabled"):
        return

    st.subheader("Behaviour anomaly flags")
    flags = detect_anomalies(
        df,
        restricted_zone=settings.get("restricted_zone"),
        loiter_radius_px=settings["loiter_radius_px"],
        loiter_min_duration_s=settings["loiter_min_duration_s"],
        course_change_deg=settings["course_change_deg"],
        min_move_px=settings["min_move_px"],
    )
    if not flags:
        st.info("No anomalies flagged for this clip.")
    else:
        flags_df = pd.DataFrame([asdict(f) for f in flags]).rename(
            columns={"class_name": "class", "at_timestamp_s": "timestamp"}
        )[["track_id", "class", "reason", "timestamp", "detail"]]
        render_table(flags_df)

    anomaly_md = render_anomaly_markdown(flags, source_name=stem)
    st.download_button(
        "Download anomaly report (Markdown)",
        data=anomaly_md.encode("utf-8"),
        file_name=f"{stem}_anomaly_report.md",
        mime="text/markdown",
        key=f"dl_anomaly_{video_id}",
    )


def video_view(video_bytes: bytes, suffix: str, settings: dict, filename: str) -> None:
    """Video flow: BoT-SORT tracking, live playback, downloadable log."""
    video_id = f"{filename}:{len(video_bytes)}"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(video_bytes)
        path = tmp.name

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        Path(path).unlink(missing_ok=True)
        st.error(
            f"**{filename}** could not be opened as a video. MP4 (H.264) is the "
            "safest format for the demo."
        )
        return
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    st.caption(
        f"{filename} · {total_frames or '?'} frames · tracker "
        f"{settings['tracker']} · stride {settings['vid_stride']}"
    )
    if not st.button("Run detection on video", type="primary",
                     use_container_width=True):
        st.video(video_bytes)
        Path(path).unlink(missing_ok=True)
        # A prior run's table/report/flags for THIS SAME video (video_id)
        # survive a rerun triggered by, say, an anomaly-threshold slider —
        # or by clicking either download button itself, which is a rerun
        # too — re-read from session_state rather than needing "Run
        # detection" clicked again.
        _render_detection_log_panel(video_id)
        _render_video_analysis_panels(settings, video_id)
        return

    if settings["stub"]:
        st.warning(
            "Stub mode has no video tracker — switch to the trained model in the "
            "sidebar to process video."
        )
        Path(path).unlink(missing_ok=True)
        return

    alert_slot = st.empty()
    summary_slot = st.empty()
    metrics_slot = st.empty()
    frame_slot = st.empty()
    progress = st.progress(0.0, text="Processing…")

    all_items: list[tuple[Detection, int | None]] = []
    seen_tracks: set[int] = set()
    military = get_military_classes()
    military_tracks: set[int] = set()
    processed = 0
    start = time.perf_counter()
    max_frames = settings["max_frames"]

    try:
        for tf in gp.track_video(
            source=path,
            weights=settings["weights"] or None,
            conf=settings["conf"],
            conf_military=settings["conf_military"],
            tracker=settings["tracker"],
            vid_stride=settings["vid_stride"],
        ):
            items = [(t.detection, t.track_id) for t in tf.detections]
            all_items.extend(items)
            for det, tid in items:
                if tid is not None:
                    seen_tracks.add(tid)
                    if det.class_name in military:
                        military_tracks.add(tid)

            caption = f"frame {tf.index}"
            if tf.timestamp:
                caption += f" · t={tf.timestamp:.2f}s"
            show_image(to_rgb(annotate(tf.image, items)), caption, slot=frame_slot)
            processed += 1

            # Refresh the panels a few times a second, not every frame.
            if processed % 5 == 1:
                with alert_slot.container():
                    render_alert([d for d, _ in all_items])
                with summary_slot.container():
                    render_summary([d for d, _ in all_items], track_ids=seen_tracks)
                with metrics_slot.container():
                    render_metrics(
                        [d for d, _ in all_items],
                        time.perf_counter() - start,
                        frames=processed,
                    )
            if total_frames:
                progress.progress(
                    min(1.0, tf.index / total_frames),
                    text=f"Frame {tf.index} / {total_frames}",
                )
            if processed >= max_frames:
                st.info(f"Stopped at the {max_frames}-frame limit (sidebar).")
                break
    except FileNotFoundError as exc:
        st.error(f"**Model unavailable.** {exc}")
        return
    except Exception as exc:  # noqa: BLE001 — a live demo must never show a traceback
        st.error(f"**Video processing failed.** {type(exc).__name__}: {exc}")
        return
    finally:
        Path(path).unlink(missing_ok=True)

    elapsed = time.perf_counter() - start
    progress.empty()

    dets_only = [d for d, _ in all_items]
    with alert_slot.container():
        render_alert(dets_only)
    with summary_slot.container():
        render_summary(dets_only, track_ids=seen_tracks)
    with metrics_slot.container():
        render_metrics(dets_only, elapsed, frames=max(1, processed))

    # Vessel count leads the sentence (not the frame count) so the number
    # that actually answers "how many ships" is the first thing read, not
    # buried after the much larger frame-count figure.
    st.success(
        f"Detected {len(seen_tracks)} vessel(s) ({len(military_tracks)} military) · "
        f"{processed} frame(s) processed in {elapsed:.1f}s."
    )

    df = detections_dataframe(all_items, with_track=True)
    # Stashed under video_id so a later rerun (an anomaly-threshold slider,
    # or clicking either download button below — that's a rerun too, and
    # reads this same "Run detection" button back as False) can still show
    # the table/incident report/anomaly flags without re-running inference
    # — see _render_detection_log_panel / _render_video_analysis_panels —
    # and so a freshly uploaded, different video doesn't show this one's
    # stale results.
    st.session_state["guardian_last_video_df"] = df
    st.session_state["guardian_last_video_source"] = filename
    st.session_state["guardian_last_video_key"] = video_id

    _render_detection_log_panel(video_id)
    _render_video_analysis_panels(settings, video_id)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def _sb_break(px: int = 8) -> None:
    """Fixed-height break with a faint centred rule, between major sections
    (Mission / Model Configuration / Video Processing / Classification) —
    replaces st.divider() so the gap is an exact, consistent value rather
    than whatever Streamlit's own widget-to-widget margin happens to add
    up to."""
    st.markdown(
        f'<div style="height:{px}px;display:flex;align-items:center;">'
        f'<div style="height:1px;width:100%;background:var(--g-line);opacity:.5;"></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _sb_gap(px: int = 40) -> None:
    st.markdown(f'<div style="height:{px}px"></div>', unsafe_allow_html=True)


def sidebar() -> dict:
    with st.sidebar:
        # The briefing modal (when shown) renders its own separate copy of
        # this toggle — see landing_page(). Both write through to the single
        # canonical st.session_state["guardian_light"]; see _theme_from_sidebar
        # / _theme_from_modal for why they can't just share one widget key.
        if st.session_state["guardian_briefed"]:
            st.session_state["guardian_light_sidebar"] = st.session_state["guardian_light"]
            st.toggle(
                "Light Interface",
                key="guardian_light_sidebar",
                on_change=_theme_from_sidebar,
            )
            st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)

        st.markdown('<div class="sb-label">Mission</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sb-desc">Access the operational brief, mission pipeline, detection taxonomy and dataset provenance.</div>',
            unsafe_allow_html=True,
        )
        if st.button("Operational Briefing", key="sidebar_briefing", use_container_width=True):
            st.session_state["guardian_briefed"] = False
            st.rerun()

        _sb_break()

        # Section: Model Configuration — one main title ("Detection
        # Controls") covering two grouped cards (Models, AI Operating
        # Thresholds), matching a settings-panel hierarchy: section > title
        # > group > control.
        st.markdown('<div class="sb-label">Model Configuration</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="sb-title">Detection Controls</div>',
                    unsafe_allow_html=True)

        st.markdown('<div class="sb-group-label">Models</div>', unsafe_allow_html=True)
        with st.container(key="sb_card_model"):
            weights_default = str(gp.DEFAULT_WEIGHTS)
            have_weights = gp.DEFAULT_WEIGHTS.exists()
            stub = st.toggle(
                "Stub mode (no model)",
                value=not have_weights,
                help="Synthetic detections — a safe fallback if weights are missing.",
            )
            baseline_label = st.selectbox(
                "Baseline",
                options=list(BASELINE_CHOICES) + [CUSTOM_BASELINE_LABEL],
                disabled=stub,
                help="The shipped model, or point at another checkpoint via "
                     "Custom path…",
            )
            if baseline_label == CUSTOM_BASELINE_LABEL:
                weights = st.text_input(
                    "Weights (.pt)", value=weights_default, disabled=stub,
                    help="Ignored in stub mode.",
                )
            else:
                weights = BASELINE_CHOICES[baseline_label]
                st.markdown(
                    f'<div class="sb-value" style="margin-bottom:.5rem"><code>{weights}</code></div>',
                    unsafe_allow_html=True,
                )
            if not stub:
                if Path(weights).exists():
                    st.markdown(
                        f'<div class="sb-value">Model <b>{Path(weights).name}</b></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.error("Weights file not found — enable stub mode, pick "
                              "another baseline, or fix the custom path.")

        st.markdown('<div class="sb-group-label">AI Operating Thresholds</div>',
                    unsafe_allow_html=True)
        with st.container(key="sb_card_thresholds"):
            conf = st.slider(
                "Confidence — civilian", 0.0, 0.95, DEFAULT_CONF, 0.01,
                help="Threshold for non-military classes.",
            )
            conf_military = st.slider(
                "Confidence — military", 0.0, 0.95, DEFAULT_CONF_MILITARY, 0.01,
                help="Threshold for the military group. Lower it to trade precision "
                     "for recall — the competition gate is recall > 90%.",
            )
            if conf_military > conf:
                st.warning("Military threshold above the civilian one weakens the "
                           "recall gate.")

        _sb_break()

        st.markdown('<div class="sb-label">Video Tracking</div>', unsafe_allow_html=True)
        with st.container(key="sb_card_video"):
            tracker = st.selectbox(
                "Tracker", ["botsort.yaml", "bytetrack.yaml"], index=0,
                help="BoT-SORT gives more stable IDs; ByteTrack is faster.",
            )
            vid_stride = st.slider(
                "Frame stride", 1, 5, 1,
                help="Process every Nth frame. Raise it for faster playback.",
            )
            max_frames = st.number_input(
                "Max frames", min_value=30, max_value=5000, value=600, step=30,
                help="Safety stop so a long clip can't stall a live demo.",
            )

        _sb_break()

        st.markdown('<div class="sb-label">Bonus: Nationality Classifier</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="sb-desc">Classifies each military contact as '
            'Malaysian RMN or foreign navy.</div>',
            unsafe_allow_html=True,
        )
        with st.container(key="sb_card_fg"):
            fg_enabled = st.toggle(
                "Classify military contacts",
                value=False,
                help="Runs the RMN-vs-Foreign classifier on every military-group "
                     "detection box (image mode only).",
            )
            fg_baseline_label = st.selectbox(
                "Classifier weights",
                options=list(FG_BASELINE_CHOICES) + [FG_CUSTOM_LABEL],
                disabled=not fg_enabled,
                help="Pick a trained checkpoint, or point at a custom path.",
            )
            if fg_baseline_label == FG_CUSTOM_LABEL:
                fg_weights = st.text_input(
                    "Weights (.pt)", value=str(FG_DEFAULT_WEIGHTS),
                    disabled=not fg_enabled,
                    help="Ignored while the toggle above is off.",
                )
            else:
                fg_weights = FG_BASELINE_CHOICES[fg_baseline_label]
                st.markdown(
                    f'<div class="sb-value" style="margin-bottom:.5rem"><code>{fg_weights}</code></div>',
                    unsafe_allow_html=True,
                )
            if fg_enabled and not Path(fg_weights).exists():
                st.error(
                    f"Classifier weights not found at `{fg_weights}` — train "
                    "first (`python -m src.fine_grained.train_classifier`) or "
                    "fix the path."
                )
            fg_conf_floor = st.slider(
                "Confidence floor (below this → unknown)",
                0.0, 0.99, float(FG_DEFAULT_CONF_FLOOR), 0.01,
                disabled=not fg_enabled,
                help="Softmax confidence below this returns 'unknown' rather "
                     "than a guess.",
            )

        _sb_break()

        st.markdown('<div class="sb-label">Behaviour Analysis</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="sb-desc">Flags loitering, sudden course changes, '
            'and restricted-zone entries in tracked video footage.</div>',
            unsafe_allow_html=True,
        )
        with st.container(key="sb_card_behavior"):
            behavior_enabled = st.toggle(
                "Enable behaviour analysis",
                value=False,
                help="Runs the loitering / course-change / restricted-zone "
                     "checks over the video detection log.",
            )
            loiter_radius_px = st.slider(
                "Loiter radius (px)", 5.0, 200.0, DEFAULT_LOITER_RADIUS_PX, 5.0,
                disabled=not behavior_enabled,
                help="A track that never leaves this radius for the minimum "
                     "duration below counts as loitering.",
            )
            loiter_min_duration_s = st.slider(
                "Loiter min duration (s)", 1.0, 60.0, DEFAULT_LOITER_MIN_DURATION_S, 1.0,
                disabled=not behavior_enabled,
                help="Minimum time spent within the loiter radius before "
                     "it's flagged.",
            )
            course_change_deg = st.slider(
                "Course change threshold (°)", 10.0, 180.0, DEFAULT_COURSE_CHANGE_DEG, 5.0,
                disabled=not behavior_enabled,
                help="Heading change between consecutive points that counts "
                     "as a sudden course change.",
            )
            min_move_px = st.slider(
                "Min movement (px)", 1.0, 50.0, DEFAULT_MIN_MOVE_PX, 1.0,
                disabled=not behavior_enabled,
                help="Movement below this is treated as jitter and ignored "
                     "when computing heading.",
            )
            zone_enabled = st.toggle(
                "Restrict to a zone",
                value=False,
                disabled=not behavior_enabled,
                help="Also flag a track when its path enters this pixel box "
                     "in the source video frame.",
            )
            zcol1, zcol2 = st.columns(2)
            with zcol1:
                zone_x1 = st.number_input(
                    "Zone x1", value=0, step=10, disabled=not zone_enabled,
                )
                zone_y1 = st.number_input(
                    "Zone y1", value=0, step=10, disabled=not zone_enabled,
                )
            with zcol2:
                zone_x2 = st.number_input(
                    "Zone x2", value=100, step=10, disabled=not zone_enabled,
                )
                zone_y2 = st.number_input(
                    "Zone y2", value=100, step=10, disabled=not zone_enabled,
                )
            restricted_zone = (
                (float(zone_x1), float(zone_y1), float(zone_x2), float(zone_y2))
                if behavior_enabled and zone_enabled else None
            )

        _sb_break()

        st.markdown('<div class="sb-label">Classification</div>', unsafe_allow_html=True)
        st.markdown('<div class="sb-group-label">Legend</div>', unsafe_allow_html=True)
        pills = "".join(
            f'<span class="sb-legend-pill" style="background:{hexcol}22;'
            f'border:1px solid {hexcol}55;color:var(--g-text)">'
            f'<span class="sb-legend-dot" style="background:{hexcol}"></span>{label}</span>'
            for group, (_, hexcol, label) in GROUP_COLOURS.items()
            if group != "other"
        )
        st.markdown(f'<div class="sb-legend">{pills}</div>', unsafe_allow_html=True)

    return {
        "stub": stub,
        "weights": "" if stub else weights,
        "conf": conf,
        "conf_military": conf_military,
        "tracker": tracker,
        "vid_stride": int(vid_stride),
        "max_frames": int(max_frames),
        "fg_enabled": fg_enabled,
        "fg_weights": fg_weights,
        "fg_conf_floor": fg_conf_floor,
        "behavior_enabled": behavior_enabled,
        "loiter_radius_px": loiter_radius_px,
        "loiter_min_duration_s": loiter_min_duration_s,
        "course_change_deg": course_change_deg,
        "min_move_px": min_move_px,
        "restricted_zone": restricted_zone,
    }


# ---------------------------------------------------------------------------
# Mission entry presentation
# ---------------------------------------------------------------------------
def _dismiss_briefing() -> None:
    st.session_state["guardian_briefed"] = True
    st.rerun()


def _theme_from_sidebar() -> None:
    """on_change callback for the sidebar's copy of the theme toggle.

    Writes the widget's own key (guardian_light_sidebar) through to the
    single canonical guardian_light — nothing outside this function ever
    reads guardian_light_sidebar for anything but driving this one
    widget. No explicit st.rerun(): this toggle is never rendered inside
    a fragment (the sidebar isn't one), so Streamlit's normal post-
    callback rerun is already a full app rerun. Calling st.rerun() here
    anyway would do nothing but render Streamlit's own "Calling
    st.rerun() within a callback is a no-op." warning banner — see
    _theme_from_modal for why that happens.
    """
    st.session_state["guardian_light"] = st.session_state["guardian_light_sidebar"]


def _theme_from_modal() -> None:
    """on_change callback for the Operational Briefing modal's copy of
    the theme toggle.

    Writes guardian_light_modal through to the canonical guardian_light,
    same as _theme_from_sidebar — but this toggle lives inside
    st.dialog, which wraps its body in a fragment (see
    dialog_decorator.py: non_optional_func runs inside _fragment(...)).
    So the rerun already queued for this widget's change is fragment-
    scoped, and that scope is decided by the frontend before this
    callback even runs server-side (Streamlit invokes on_change from
    SessionState.on_script_will_rerun(), which fires *before* the
    pending rerun starts). Calling st.rerun() in this callback would
    raise RerunException, which Streamlit silently swallows into a
    "Calling st.rerun() within a callback is a no-op." warning banner —
    verified live: the fragment-scoped rerun proceeds unchanged and
    inject_css() (called only from main(), outside the fragment) never
    re-runs, so the dashboard behind the modal stays on the old theme.
    Instead this only sets a flag; _flush_theme_rerun(), called from
    plain script flow at the top of the dialog's fragment body, issues
    the actual st.rerun(scope="app") from a place where it's a live
    request, not a no-op.
    """
    st.session_state["guardian_light"] = st.session_state["guardian_light_modal"]
    st.session_state["_guardian_theme_dirty"] = True


def _flush_theme_rerun() -> None:
    """Escalate a pending theme change (flagged by _theme_from_modal) to
    a full-app rerun.

    Must be called from inside the dialog's fragment body (not from a
    callback — see _theme_from_modal for why a callback can't do this).
    Called at the top of show_operational_briefing(), before any widget
    renders, so a stale fragment-scoped run is abandoned in favour of a
    full main() rerun that re-executes inject_css() and repaints the
    whole app — dashboard, sidebar and modal — in one shot.
    """
    if st.session_state.pop("_guardian_theme_dirty", False):
        st.rerun(scope="app")


def _show_logo(path: Path, slot, width: int) -> None:
    """Render acknowledgement assets at a deliberate, compact display size."""
    slot.container(key=f"ack_logo_{path.stem.replace('-', '_')}").image(str(path), width=width)


def landing_page() -> None:
    """Compact entry screen; it intentionally performs no model work."""
    st.markdown('<div class="guardian-entry">', unsafe_allow_html=True)
    with st.container(key="topbar"):
        top = st.columns((3.2, 1.3, 1))
        with top[0]:
            st.markdown(
                '<div class="entry-brand">PROJECT GUARDIAN '
                '<span>/ MARITIME DOMAIN AWARENESS</span></div>',
                unsafe_allow_html=True,
            )
        with top[1]:
            st.markdown('<div class="sedic-badge">SEDIC 2026 · Visual Track</div>',
                        unsafe_allow_html=True)
        with top[2]:
            st.session_state["guardian_light_modal"] = st.session_state["guardian_light"]
            st.toggle(
                "Light Interface",
                key="guardian_light_modal",
                on_change=_theme_from_modal,
            )
    hero_col, status_col = st.columns((1.7, 1), gap="large")
    with hero_col:
        st.markdown(
            '''<div class="entry-intro">
            <h1>Project Guardian</h1>
            <div class="entry-subtitle">AI-Powered Maritime Surveillance Platform</div>
            <div class="entry-summary">AI-assisted vessel detection and classification across surface and aerial imagery for maritime situational awareness.</div>
            </div>''',
            unsafe_allow_html=True,
        )
        st.markdown(
            '''<div class="brief-panel" style="margin:1.75rem 0 0">
            <div class="info-panel-title">Operational Brief</div>
            <p>Project Guardian is an AI-powered Maritime Domain Awareness platform developed for the
            Strategic Electronic Defence Innovation Challenge (SEDIC) 2026. The system assists maritime
            operators by automatically detecting, classifying and tracking vessels from surveillance
            imagery, improving situational awareness and supporting operational decision-making.</p></div>''',
            unsafe_allow_html=True,
        )

        _left, centre, _right = st.columns([1, 1.9, 1])
        with centre:
            if st.button(
                "Enter Command Centre", key="entry_launch", type="primary",
                use_container_width=True,
            ):
                _dismiss_briefing()
    with status_col:
        eval_mtime = EVAL_REPORT_PATH.stat().st_mtime if EVAL_REPORT_PATH.exists() else 0.0
        eval_summary = load_eval_summary(str(EVAL_REPORT_PATH), eval_mtime)
        if eval_summary and eval_summary["military_recall_overall"] is not None:
            overall = eval_summary["military_recall_overall"]
            aerial = eval_summary["military_recall_aerial"]
            surface = eval_summary["military_recall_surface"]
            gate_pass = overall >= EVAL_GATE
            status_line = (
                '<span class="mil-recall-check">✓ Competition Target Achieved</span>'
                if gate_pass else
                '<span class="mil-recall-check fail">Below competition target</span>'
            )
            secondary_cells = []
            if aerial is not None:
                secondary_cells.append(("Aerial Recall", aerial, "Military domain"))
            if surface is not None:
                secondary_cells.append(("Surface Recall", surface, "Military domain"))
            secondary_html = "".join(
                f'<div class="metric-cell"><div class="metric-label">{label}</div>'
                f'<div class="metric-value">{value * 100:.1f}%</div>'
                f'<div class="metric-note">{note}</div></div>'
                for label, value, note in secondary_cells
            )
            footer = (
                f'<div class="metric-footer"><span>Threshold '
                f'<strong>conf_military = {CONF_MILITARY_GATE:.2f}</strong></span>'
                f'<span>Split <strong>{eval_summary["split"]}</strong></span>'
                + (f'<span>IoU <strong>{eval_summary["iou"]:.2f}</strong></span>'
                   if eval_summary["iou"] is not None else "")
                + '</div>'
            )
            eval_body = (
                '<div class="mil-recall-card">'
                '<div class="mil-recall-label">Military Recall</div>'
                f'<div class="mil-recall-value">{overall * 100:.1f}%</div>'
                f'<div class="mil-recall-status">{status_line}'
                '<span class="mil-recall-target">Target &gt; 90%</span></div>'
                '</div>'
                f'<div class="metric-grid secondary">{secondary_html}</div>'
                f'{footer}'
            )
        else:
            eval_body = (
                '<div class="eval-pending">Evaluation report not yet generated for this '
                'checkout. Run <code>python -m src.eval.detail --weights '
                'models/baseline2_best.pt --split test</code> to produce '
                '<code>outputs/eval/test_eval.md</code>.</div>'
            )
        st.markdown(
            f'''<section class="entry-section eval-section" style="margin-top:0">
            <div class="section-label">Evaluation</div>
            <h2 style="margin:.2rem 0 .35rem">Held-out test performance</h2>
            {eval_body}
            </section>''',
            unsafe_allow_html=True,
        )

    st.markdown('<section class="entry-section"><div class="section-label">Detection pipeline</div><h2>Data preparation to tracked contacts</h2><p>A reproducible processing path for maritime imagery and video.</p>', unsafe_allow_html=True)
    pipeline = [
        ("01", "Raw datasets", "Maritime source collections."),
        ("02", "Dataset conversion", "Unified detection format."),
        ("03", "Duplicate removal", "Near-duplicate controls."),
        ("04", "Stratified split", "Train / validation / test."),
        ("05", "YOLO11m training", "Unified vessel detector."),
        ("06", "Detection", "Contact localisation."),
        ("07", "Multi-object tracking", "Persistent video tracks."),
    ]
    cells = [
        f'<div class="pipeline-step"><div class="pipeline-no">{number}</div><h3>{title}</h3><p>{copy}</p></div>'
        for number, title, copy in pipeline
    ]
    st.markdown(f'<div class="pipeline-row">{"".join(cells)}</div></section>', unsafe_allow_html=True)

    st.markdown('<section class="entry-section"><div class="section-label">Vessel taxonomy</div><h2>Unified detection classes</h2><p>Classes are read from the project schema and grouped for operational review.</p></section>', unsafe_allow_html=True)
    class_groups = get_class_groups()
    taxonomy_columns = st.columns(3)
    for column, group in zip(taxonomy_columns, ("civilian", "small_craft", "military")):
        class_names = sorted(name for name, mapped_group in class_groups.items()
                             if mapped_group == group)
        label = group.replace("_", " ").title()
        chips = "".join(
            f'<span class="tax-chip">{name.replace("_", " ")}</span>'
            for name in class_names
        ) or '<span class="tax-chip">No classes configured</span>'
        with column:
            css_group = group.replace("_", "-")
            st.markdown(f'<div class="taxonomy-card {css_group}"><h3>{label}</h3>{chips}</div>', unsafe_allow_html=True)

    st.markdown('<section class="entry-section"><div class="section-label">Dataset provenance</div><h2>Integrated source datasets</h2><p>Traceable inputs used in the current processed build.</p>', unsafe_allow_html=True)
    provenance = [
        ("military_ships", "Aerial", "CC BY 4.0", "2,746"),
        ("seaships", "Surface", "CC BY 4.0", "6,979"),
        ("shiprsimagenet", "Aerial", "CC BY 4.0*", "4,579"),
        ("military_surface", "Surface", "CC BY 4.0", "3,011"),
    ]
    rows = "".join(
        f'<tr><td>{name}</td><td class="muted">{domain}</td>'
        f'<td class="muted">{licence}</td><td>{images}</td></tr>'
        for name, domain, licence, images in provenance
    )
    st.markdown(f'<div class="provenance-wrap"><table class="provenance-table"><thead><tr><th>Dataset name</th><th>Domain</th><th>Licence</th><th>Image count</th></tr></thead><tbody>{rows}</tbody></table></div></section>', unsafe_allow_html=True)
    st.caption("* ShipRSImageNet derives from academic/research-use imagery; attribution is retained in the project provenance log.")
    st.markdown('<div class="ack-panel" style="margin-top:2.2rem"><div class="section-label">SEDIC 2026 acknowledgement</div></div>', unsafe_allow_html=True)
    asset_dir = _REPO_ROOT / "app" / "assets"
    acknowledgement = st.columns((.95, 1.55, 1.25, .85, .78, .92), gap="small")
    with acknowledgement[0]:
        st.markdown('<div class="ack-label" style="padding-top:1.7rem">Organised by</div>', unsafe_allow_html=True)
    with acknowledgement[1]:
        _show_logo(asset_dir / "upnm-logo.png", st, 205)
    with acknowledgement[2]:
        _show_logo(asset_dir / "ieee-logo-removebg-preview.png", st, 185)
    with acknowledgement[3]:
        st.markdown('<div class="ack-label" style="padding-top:1.7rem">Supported by</div>', unsafe_allow_html=True)
    with acknowledgement[4]:
        _show_logo(asset_dir / "bsep-logo.png", st, 82)
    with acknowledgement[5]:
        _show_logo(asset_dir / "stride-logo-removebg-preview.png", st, 108)
    st.markdown('<div class="entry-footer">Project Guardian · Maritime Domain Awareness Platform</div></div>', unsafe_allow_html=True)


@st.dialog(" ", width="large")
def show_operational_briefing() -> None:
    """Startup overlay: the landing page, unchanged, shown as a dismissable
    briefing modal over the dashboard instead of routed to as its own page.

    Uses Streamlit's native st.dialog rather than a hand-rolled fixed-position
    CSS overlay — the earlier custom backdrop broke under real browser window
    sizes (horizontal overflow left it not actually covering the viewport).
    st.dialog handles centring, backdrop and internal scrolling natively.
    Its own built-in close (X) is hidden via CSS: closing it that way doesn't
    set `guardian_briefed`, so the next rerun would immediately reopen it —
    dismissal only happens through the "Enter Command Centre" button below,
    which does set the flag.
    """
    # This function's body is the dialog's fragment (see
    # _dialog_decorator: it wraps non_optional_func in _fragment(...)).
    # Escalating here, before any widgets render, means a theme change
    # made via the toggle inside landing_page() below abandons this
    # fragment-scoped run in favour of a full main() rerun that repaints
    # the whole app — see _theme_from_modal / _flush_theme_rerun.
    _flush_theme_rerun()
    landing_page()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    if "guardian_briefed" not in st.session_state:
        st.session_state["guardian_briefed"] = False
    # guardian_light is the ONLY canonical theme value. guardian_light_sidebar
    # and guardian_light_modal are separate per-widget keys that exist solely
    # to back the sidebar's and modal's own st.toggle instances — each is
    # synced from guardian_light immediately before its widget renders (see
    # sidebar() / landing_page()), and each writes back to guardian_light via
    # its own on_change callback (_theme_from_sidebar / _theme_from_modal).
    # They must stay separate: the sidebar and modal toggles are two distinct
    # widgets rendered in mutually-exclusive branches (based on
    # guardian_briefed) and, for the modal, across a fragment boundary
    # (st.dialog) — sharing one widget key across two physically different
    # widget instances like that is what caused the toggle to silently reset
    # or desync.
    if "guardian_light" not in st.session_state:
        st.session_state["guardian_light"] = False
    if "guardian_light_sidebar" not in st.session_state:
        st.session_state["guardian_light_sidebar"] = False
    if "guardian_light_modal" not in st.session_state:
        st.session_state["guardian_light_modal"] = False
    inject_css(st.session_state["guardian_light"])

    # The upload dashboard is always the home screen now; the landing page
    # renders as a dismissable briefing overlay on top of it on first load.
    settings = sidebar()
    if not st.session_state["guardian_briefed"]:
        show_operational_briefing()

    # Warm the weights up front so the first real detection isn't the slow one.
    if not settings["stub"] and Path(settings["weights"]).exists():
        try:
            get_model(settings["weights"])
        except Exception as exc:  # noqa: BLE001 — degrade to stub, never crash
            st.error(f"**Could not load the model.** {type(exc).__name__}: {exc} — "
                     "enable stub mode in the sidebar to continue the demo.")
            return

    # Primary action first: the upload workflow is the hero and stays at the
    # top, uncontested. Operational summary (System Status/Information)
    # follows below as supporting context, set off with generous whitespace
    # rather than competing for the same visual weight as the upload card.
    uploaded = st.file_uploader(
        "Upload an image or video",
        type=sorted(s.lstrip(".") for s in IMAGE_SUFFIXES | VIDEO_SUFFIXES),
        help="Surface/frontal or aerial imagery. Video runs through the tracker.",
    )

    st.markdown('<div style="height:32px"></div>', unsafe_allow_html=True)

    status_col, info_col = st.columns(2, gap="medium")

    with status_col:
        model_ready = gp.DEFAULT_WEIGHTS.exists()
        model_state = (
            '<span class="status-value ok"><span class="status-dot"></span>LOADED</span>'
            if model_ready else
            '<span class="status-value warn"><span class="status-dot"></span>STUB MODE</span>'
        )
        status_rows = [
            ("Operational Status", '<span class="status-value ok"><span class="status-dot"></span>ONLINE</span>'),
            ("AI Inference", '<span class="status-value ok"><span class="status-dot"></span>READY</span>'),
            ("Detection Model", model_state),
            ("Multi-Object Tracking", '<span class="status-value ok"><span class="status-dot"></span>ENABLED</span>'),
            ("Deployment", '<span class="status-value neutral"><span class="status-dot"></span>ACTIVE</span>'),
        ]
        rows_html = "".join(
            f'<div class="status-row"><span class="label">{label}</span>{value}</div>'
            for label, value in status_rows
        )
        st.markdown(
            f'<div class="info-panel"><div class="info-panel-title">System Status</div>{rows_html}</div>',
            unsafe_allow_html=True,
        )

    with info_col:
        meta_rows = [
            ("Project", APP_TITLE),
            ("Deployment", "Maritime Operations Centre"),
            ("Platform", "Computer Vision"),
            ("Framework", "YOLO11m"),
            ("Version", "v1.0"),
        ]
        meta_html = "".join(
            f'<div class="meta-row"><span class="k">{k}</span><span class="v">{v}</span></div>'
            for k, v in meta_rows
        )
        st.markdown(
            f'<div class="info-panel"><div class="info-panel-title">System Information</div>{meta_html}</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div style="height:36px"></div>', unsafe_allow_html=True)

    if uploaded is None:
        if settings["stub"]:
            st.caption("No file yet — showing the stub scene.")
            image_view(None, ".jpg", settings, "synthetic scene")
        else:
            st.info("Upload an image or video to begin. Detection runs "
                    "automatically for images.")
    else:
        suffix = Path(uploaded.name).suffix.lower()
        payload = uploaded.getvalue()
        if not payload:
            st.error("That file came through empty. Try uploading it again.")
        elif suffix in IMAGE_SUFFIXES:
            image_view(payload, suffix, settings, uploaded.name)
        elif suffix in VIDEO_SUFFIXES:
            video_view(payload, suffix, settings, uploaded.name)
        else:
            st.error(f"**{suffix or 'That file type'}** isn't supported. Use JPG/PNG "
                     "images or MP4/MOV/AVI video.")


main()
