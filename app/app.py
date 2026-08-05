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
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# Make the repo root importable when Streamlit runs this file directly.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Imported after the sys.path setup above, deliberately. E402 is ignored for this
# file in pyproject.toml rather than with an inline suppression comment, because
# current ruff already understands the sys.path idiom and would then flag that
# comment itself as an unused directive (RUF100).
from src.inference import predict as gp
from src.inference.predict import Detection

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

# --- Evaluation report --------------------------------------------------------
# The landing page never hardcodes gate numbers — a stale figure in front of a
# judge is a real cost. Parsed live from the report `src.eval.detail` writes.
EVAL_REPORT_PATH = _REPO_ROOT / "outputs" / "eval" / "test_eval.md"
EVAL_GATE = 0.90
# The gate's actual operating threshold, read from the frozen predict() signature
# rather than retyped here.
CONF_MILITARY_GATE = inspect.signature(gp.predict).parameters["conf_military"].default

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
            "--g-navy": "#f4f7fa",
            "--g-panel": "#ffffff",
            "--g-panel-2": "#fafbfc",
            "--g-line": "#d6dce5",
            "--g-text": "#0f172a",
            "--g-muted": "#475569",
            "--g-cyan": "#2563eb",
            "--g-green": "#15803d",
            "--g-amber": "#b45309",
            "--g-red": "#c62828",
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
                box-shadow:0 1px 2px rgba(15,23,42,.06), 0 1px 3px rgba(15,23,42,.05);
              }
              .chip-row, .metric-grid.secondary, .table-wrap {
                border-radius:8px; overflow:hidden;
                box-shadow:0 1px 2px rgba(15,23,42,.06), 0 1px 3px rgba(15,23,42,.05);
              }
              .entry-intro {
                background:#eef3fb; border:1px solid var(--g-line);
                border-left:3px solid var(--g-cyan);
                border-radius:8px; padding:1.3rem 1.5rem;
                box-shadow:0 1px 2px rgba(15,23,42,.06), 0 1px 3px rgba(15,23,42,.05);
              }
              .mil-recall-card {
                background:#fef2f2; border-left-width:4px;
              }
              .status-value.ok, .status-value.warn, .status-value.neutral {
                color:var(--g-muted);
              }
              .sedic-badge {
                background:#eff6ff; border:1px solid #dce7f5; border-radius:4px;
              }
              .sedic-badge::before {
                content:""; display:inline-block; width:6px; height:6px;
                border-radius:50%; background:var(--g-cyan); margin-right:.45rem;
                vertical-align:middle;
              }
              [class*="st-key-entry_launch"] button[kind="primary"] {
                background:#1f4d3a !important; border:1px solid #3f7a61 !important;
                border-radius:8px !important; box-shadow:0 1px 2px rgba(15,23,42,.12);
              }
              [class*="st-key-entry_launch"] button[kind="primary"]::after {
                color:#fff; transition:transform .15s ease;
              }
              [class*="st-key-entry_launch"] button[kind="primary"]:hover {
                background:#29634b !important; border-color:#3f7a61 !important;
                filter:none !important; box-shadow:0 2px 6px rgba(15,23,42,.18);
              }
              [class*="st-key-entry_launch"] button[kind="primary"]:hover::after {
                transform:translateX(3px);
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
      }}
      .stApp {{ background:var(--g-navy); color:var(--g-text); }}
      [data-testid="stWidgetLabel"] p, [data-testid="stCheckbox"] p,
      [data-testid="stToggle"] p, [data-testid="stCaptionContainer"] p {{
        color:var(--g-text) !important;
      }}
      [data-testid="stHeader"] {{ background:transparent; pointer-events:none; }}
      [data-testid="stHeader"] * {{ pointer-events:auto; }}
      #MainMenu, footer {{ visibility:hidden; }}
      .block-container {{ max-width:1320px; padding-top:.2rem; padding-bottom:2.5rem; }}
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
      .sedic-badge {{ border:1px solid var(--g-line); color:var(--g-text); padding:.32rem .55rem; font-size:.64rem; letter-spacing:.12em; text-transform:uppercase; white-space:nowrap; position:relative; top:-8px; }}
      .entry-intro {{ padding:0 0 .7rem; border-bottom:1px solid var(--g-line); }}
      .section-label {{ color:var(--g-cyan); font-size:.67rem; letter-spacing:.14em; text-transform:uppercase; font-weight:700; }}
      .entry-intro h1 {{ color:var(--g-text); margin:.3rem 0 .3rem; font-size:2.05rem; font-weight:600; letter-spacing:0; line-height:1.15; }}
      .entry-subtitle {{ color:var(--g-muted); font-size:.92rem; font-weight:500; margin:0; }}
      .entry-summary {{ color:var(--g-muted); line-height:1.4; max-width:640px; margin:.4rem 0 0; font-size:.88rem; }}
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
      .brief-panel {{ border:1px solid var(--g-line); background:var(--g-panel); padding:.65rem 1rem; height:100%; }}
      .brief-panel p {{ color:var(--g-muted); line-height:1.6; font-size:.85rem; margin:0; }}
      .meta-row {{ display:flex; justify-content:space-between; padding:.34rem 0; border-bottom:1px solid var(--g-line); font-size:.82rem; }}
      .meta-row .k {{ color:var(--g-muted); }}
      .meta-row .v {{ color:var(--g-text); font-weight:600; }}
      .performance-heading {{ display:flex; justify-content:space-between; gap:1rem; align-items:baseline; margin-bottom:.6rem; }}
      .performance-heading span {{ color:var(--g-muted); font-size:.65rem; letter-spacing:.1em; text-transform:uppercase; }}
      .mil-recall-card {{ border:1px solid var(--g-line); border-left:3px solid var(--g-red); background:var(--g-panel); padding:.9rem 1.05rem; }}
      .mil-recall-label {{ color:var(--g-red); font-size:.66rem; letter-spacing:.1em; text-transform:uppercase; font-weight:700; }}
      .mil-recall-value {{ color:var(--g-text); font-size:2.4rem; font-weight:700; line-height:1.05; margin:.3rem 0 .4rem; }}
      .mil-recall-status {{ display:flex; align-items:center; gap:.7rem; flex-wrap:wrap; font-size:.78rem; }}
      .mil-recall-check {{ color:var(--g-green); font-weight:600; }}
      .mil-recall-check.fail {{ color:var(--g-red); }}
      .mil-recall-target {{ color:var(--g-muted); }}
      .metric-grid.secondary {{ display:grid; grid-template-columns:1fr 1fr; gap:1px; background:var(--g-line); border:1px solid var(--g-line); border-top:none; }}
      .metric-cell {{ background:var(--g-panel); padding:.5rem .8rem; }}
      .eval-pending {{ border:1px solid var(--g-line); background:var(--g-panel); padding:.75rem .9rem; color:var(--g-muted); font-size:.82rem; line-height:1.55; }}
      .eval-pending code {{ background:var(--g-panel-2); padding:.05rem .3rem; font-size:.78rem; }}
      .metric-label {{ color:var(--g-muted); font-size:.6rem; letter-spacing:.08em; text-transform:uppercase; }}
      .metric-value {{ color:var(--g-muted); font-size:1.05rem; font-weight:600; margin:.2rem 0 .05rem; }}
      .metric-note {{ color:var(--g-muted); font-size:.66rem; }}
      .metric-footer {{ display:flex; gap:1.2rem; flex-wrap:wrap; color:var(--g-muted); font-size:.72rem; margin-top:.6rem; }}
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
      [class*="st-key-entry_launch"] button[kind="primary"] {{
        background:#1f4d3a; color:#ffffff; border:1px solid #3f7a61;
        border-radius:5px; height:48px; min-width:300px; width:auto;
        padding:0 1.3rem; font-weight:600; font-size:.86rem; letter-spacing:.02em;
        display:flex; align-items:center; justify-content:flex-start;
      }}
      [class*="st-key-entry_launch"] button[kind="primary"] p {{ text-align:left; margin:0; }}
      [class*="st-key-entry_launch"] button[kind="primary"]::after {{
        content:"→"; margin-left:auto; padding-left:1.4rem; color:#ffffff; font-weight:600;
      }}
      [class*="st-key-entry_launch"] button[kind="primary"]:hover {{
        background:#29634b; border-color:#3f7a61; filter:none;
      }}
      [data-testid="stSidebar"] {{ border-right:1px solid var(--g-line); }}
      [data-testid="stSidebar"] h3 {{ color:var(--g-text); font-size:.9rem; letter-spacing:.04em; }}
      .sidebar-group {{ color:var(--g-muted); font-size:.63rem; letter-spacing:.14em; text-transform:uppercase; font-weight:700; margin:.75rem 0 .3rem; }}
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
      .legend-swatch {{ display:inline-block; width:9px; height:9px; margin-right:.35rem; vertical-align:middle; border:1px solid var(--g-line); }}
      .table-wrap {{ max-height:460px; overflow:auto; border:1px solid var(--g-line); }}
      .det-table {{ width:100%; border-collapse:collapse; font-size:.87rem; }}
      .det-table th {{ position:sticky; top:0; text-align:left; padding:.5rem .7rem; font-weight:600; text-transform:uppercase; font-size:.72rem; letter-spacing:.6px; background:var(--g-panel-2); color:var(--g-muted); border-bottom:1px solid var(--g-line); }}
      .det-table td {{ padding:.42rem .7rem; color:var(--g-text); border-bottom:1px solid var(--g-line); }}
      {light_only_css}
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
    items: list[tuple[Detection, int | None]], with_track: bool
) -> pd.DataFrame:
    rows = []
    for det, track_id in items:
        x1, y1, x2, y2 = (round(v, 1) for v in det.bbox) if det.bbox else (0, 0, 0, 0)
        row = {
            "class": det.class_name,
            "group": GROUP_COLOURS[group_of(det.class_name)][2],
            "confidence": round(det.confidence, 3),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        }
        if with_track:
            row = {"frame": det.frame, "timestamp_s": det.timestamp,
                   "track_id": track_id, **row}
        rows.append(row)
    df = pd.DataFrame(rows)
    if with_track and not df.empty:
        # Nullable int, so an untracked box reads as blank rather than "nan".
        df["track_id"] = df["track_id"].astype("Int64")
    return df


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
    st.markdown(
        f"""
        <div class="table-wrap">
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
    render_table(detections_dataframe(items, with_track=False))


def video_view(video_bytes: bytes, suffix: str, settings: dict, filename: str) -> None:
    """Video flow: BoT-SORT tracking, live playback, downloadable log."""
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

    st.success(
        f"Processed {processed} frame(s) in {elapsed:.1f}s · "
        f"{len(seen_tracks)} unique vessel track(s) · "
        f"{len(military_tracks)} military track(s)."
    )

    df = detections_dataframe(all_items, with_track=True)
    st.subheader(f"Detection log ({len(df)} rows)")
    render_table(df.head(500))
    if len(df) > 500:
        st.caption("Showing the first 500 rows — the CSV contains all of them.")
    st.download_button(
        "Download detection log (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"{Path(filename).stem}_detections.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def sidebar() -> dict:
    with st.sidebar:
        # The briefing modal (when shown) renders its own copy of this same
        # `key="guardian_light"` toggle — Streamlit forbids two widgets with
        # the same key in one run, so this one only appears once dismissed.
        if st.session_state["guardian_briefed"]:
            st.toggle("Light Interface", key="guardian_light")
        st.markdown("### Mission Configuration")
        if st.button("View Operational Briefing", key="sidebar_briefing",
                      use_container_width=True):
            st.session_state["guardian_briefed"] = False
            st.rerun()
        st.markdown('<div class="sidebar-group">Model configuration</div>',
                    unsafe_allow_html=True)
        st.markdown("### Detection controls")

        weights_default = str(gp.DEFAULT_WEIGHTS)
        have_weights = gp.DEFAULT_WEIGHTS.exists()
        stub = st.toggle(
            "Stub mode (no model)",
            value=not have_weights,
            help="Synthetic detections — a safe fallback if weights are missing.",
        )
        weights = st.text_input(
            "Weights (.pt)", value=weights_default, disabled=stub,
            help="Ignored in stub mode.",
        )
        if not stub:
            if Path(weights).exists():
                st.caption(f"Model: `{Path(weights).name}`")
            else:
                st.error("Weights file not found — enable stub mode or fix the path.")

        st.divider()
        st.markdown('<div class="sidebar-group">AI operating thresholds</div>',
                    unsafe_allow_html=True)
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

        st.divider()
        st.markdown("**Video**")
        st.markdown('<div class="sidebar-group">Video tracking</div>',
                    unsafe_allow_html=True)
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

        st.divider()
        st.markdown('<div class="sidebar-group">Classification legend</div>',
                    unsafe_allow_html=True)
        legend = "".join(
            f'<div><span class="legend-swatch" style="background:{hexcol}"></span>'
            f'<span style="font-size:.85rem">{label}</span></div>'
            for group, (_, hexcol, label) in GROUP_COLOURS.items()
            if group != "other"
        )
        st.markdown(f"**Legend**{legend}", unsafe_allow_html=True)

    return {
        "stub": stub,
        "weights": "" if stub else weights,
        "conf": conf,
        "conf_military": conf_military,
        "tracker": tracker,
        "vid_stride": int(vid_stride),
        "max_frames": int(max_frames),
    }


# ---------------------------------------------------------------------------
# Mission entry presentation
# ---------------------------------------------------------------------------
def _dismiss_briefing() -> None:
    st.session_state["guardian_briefed"] = True
    st.rerun()


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
            st.toggle("Light Interface", key="guardian_light")
    hero_col, status_col = st.columns((1.9, 1), gap="large")
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

        st.markdown('<div style="margin-top:1.75rem"></div>', unsafe_allow_html=True)
        if st.button("Enter Command Centre", key="entry_launch", type="primary",
                      use_container_width=False):
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
                'models/baseline_best.pt --split test</code> to produce '
                '<code>outputs/eval/test_eval.md</code>.</div>'
            )
        st.markdown(
            f'''<section class="entry-section" style="margin-top:0">
            <div class="section-label">Evaluation</div>
            <h2 style="margin:.15rem 0 .3rem">Held-out test performance</h2>
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
    landing_page()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    if "guardian_briefed" not in st.session_state:
        st.session_state["guardian_briefed"] = False
    if "guardian_light" not in st.session_state:
        st.session_state["guardian_light"] = False
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

    # Left: the operational workflow (upload, detections, preview, metrics).
    # Right: the operational summary column — live system status/info, moved
    # here from the briefing modal so they stay visible while using the app.
    left_col, right_col = st.columns((2, 1), gap="large")

    with right_col:
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
            f'<div class="info-panel" style="margin-top:.9rem">'
            f'<div class="info-panel-title">System Information</div>{meta_html}</div>',
            unsafe_allow_html=True,
        )

    with left_col:
        uploaded = st.file_uploader(
            "Upload an image or video",
            type=sorted(s.lstrip(".") for s in IMAGE_SUFFIXES | VIDEO_SUFFIXES),
            help="Surface/frontal or aerial imagery. Video runs through the tracker.",
        )

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
