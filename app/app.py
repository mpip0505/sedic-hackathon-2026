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

# Imported after the sys.path setup above, deliberately.
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
    page_icon="🛰️",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Styling — small and targeted; the base theme lives in .streamlit/config.toml
# ---------------------------------------------------------------------------
def inject_css() -> None:
    st.markdown(
        """
        <style>
          .guardian-header {
            background: linear-gradient(100deg, #0d2137 0%, #16324f 55%, #1c4568 100%);
            border: 1px solid #24405e; border-left: 5px solid #38bdf8;
            border-radius: 10px; padding: 1.1rem 1.4rem; margin-bottom: 1.1rem;
          }
          .guardian-header h1 {
            margin: 0; font-size: 1.75rem; font-weight: 700; letter-spacing: .2px;
            color: #f1f6fb;
          }
          .guardian-header h1 span { color: #7dd3fc; font-weight: 500; }
          .guardian-header p { margin: .35rem 0 0; color: #9fb6cc; font-size: .92rem; }
          .alert-military {
            background: linear-gradient(90deg, #7f1d1d 0%, #b91c1c 60%, #dc2626 100%);
            border: 1px solid #ef4444; border-radius: 8px; color: #fff;
            padding: .85rem 1.1rem; margin: .3rem 0 1rem;
            font-size: 1.12rem; font-weight: 700; letter-spacing: .6px;
            animation: pulse 1.6s ease-in-out infinite;
          }
          .alert-military small {
            display: block; font-weight: 400; letter-spacing: 0;
            font-size: .85rem; opacity: .9; margin-top: .15rem;
          }
          @keyframes pulse { 0%,100% {opacity:1} 50% {opacity:.78} }
          .alert-clear {
            background: #12261f; border: 1px solid #1f6f4a; border-radius: 8px;
            color: #86efac; padding: .7rem 1.1rem; margin: .3rem 0 1rem;
            font-weight: 600;
          }
          .chip-row { display: flex; gap: .6rem; flex-wrap: wrap; margin: .2rem 0 .6rem; }
          .chip {
            border-radius: 8px; padding: .55rem .9rem; min-width: 116px;
            background: #122134; border-left: 4px solid #38bdf8;
          }
          .chip .k { font-size: .72rem; text-transform: uppercase;
                     letter-spacing: .8px; color: #94a3b8; }
          .chip .v { font-size: 1.45rem; font-weight: 700; color: #f1f6fb;
                     line-height: 1.15; }
          .table-wrap { max-height: 460px; overflow: auto; border: 1px solid #23364f;
                        border-radius: 8px; }
          .det-table { width: 100%; border-collapse: collapse; font-size: .87rem; }
          .det-table th {
            position: sticky; top: 0; background: #16283e; color: #9fb6cc;
            text-align: left; padding: .5rem .7rem; font-weight: 600;
            text-transform: uppercase; font-size: .72rem; letter-spacing: .6px;
            border-bottom: 1px solid #23364f;
          }
          .det-table td { padding: .42rem .7rem; color: #e6edf6;
                          border-bottom: 1px solid #1b2c42; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        f"""
        <div class="guardian-header">
          <h1>🛰️ {APP_TITLE} <span>— {APP_SUBTITLE}</span></h1>
          <p>{TAGLINE}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
              ⚠ MILITARY VESSEL DETECTED — {len(flagged)} CONTACT(S)
              <small>Highest confidence {top:.2f} · review required</small>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="alert-clear">✔ No military contacts at the current '
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
            f'<div class="chip" style="border-left-color:#38bdf8">'
            f'<div class="k">{total_label}</div>'
            f'<div class="v">{total_value}</div></div>'
        )
    ]
    for group, (_, hexcol, label) in GROUP_COLOURS.items():
        if group == "other" and not counts[group]:
            continue
        chips.append(
            f'<div class="chip" style="border-left-color:{hexcol}">'
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
    if not st.button("▶ Run detection on video", type="primary",
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
        "⬇ Download detection log (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"{Path(filename).stem}_detections.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def sidebar() -> dict:
    with st.sidebar:
        st.markdown("### ⚙ Detection controls")

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
                st.caption(f"✔ Model: `{Path(weights).name}`")
            else:
                st.error("Weights file not found — enable stub mode or fix the path.")

        st.divider()
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
        legend = " ".join(
            f'<span style="color:{hexcol};font-weight:700">■</span> '
            f'<span style="font-size:.85rem">{label}</span>'
            for group, (_, hexcol, label) in GROUP_COLOURS.items()
            if group != "other"
        )
        st.markdown(f"**Legend**<br>{legend}", unsafe_allow_html=True)

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
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    inject_css()
    render_header()
    settings = sidebar()

    # Warm the weights up front so the first real detection isn't the slow one.
    if not settings["stub"] and Path(settings["weights"]).exists():
        try:
            get_model(settings["weights"])
        except Exception as exc:  # noqa: BLE001 — degrade to stub, never crash
            st.error(f"**Could not load the model.** {type(exc).__name__}: {exc} — "
                     "enable stub mode in the sidebar to continue the demo.")
            return

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
        return

    suffix = Path(uploaded.name).suffix.lower()
    payload = uploaded.getvalue()
    if not payload:
        st.error("That file came through empty. Try uploading it again.")
        return

    if suffix in IMAGE_SUFFIXES:
        image_view(payload, suffix, settings, uploaded.name)
    elif suffix in VIDEO_SUFFIXES:
        video_view(payload, suffix, settings, uploaded.name)
    else:
        st.error(f"**{suffix or 'That file type'}** isn't supported. Use JPG/PNG "
                 "images or MP4/MOV/AVI video.")


main()
