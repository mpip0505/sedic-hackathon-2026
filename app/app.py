"""app.py — Project Guardian Streamlit GUI (skeleton).

Wired to src.inference.predict.predict() and nothing else. No box-drawing yet;
this is the Phase 0 skeleton so integration can proceed against the frozen
interface. Run with:

    streamlit run app/app.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

# Make the repo root importable when Streamlit runs this file directly.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.inference.predict import Detection, _military_class_names, predict

st.set_page_config(page_title="Project Guardian", page_icon="🛥️", layout="wide")


def _run() -> None:
    st.title("🛥️ Project Guardian — Maritime Object Detection")
    st.caption("SEDIC 2026 Visual Track · military-vessel recall is the priority")

    # --- Sidebar controls ----------------------------------------------------
    with st.sidebar:
        st.header("Controls")
        stub = st.toggle(
            "Stub mode", value=True,
            help="Synthetic detections — works with no model/weights.",
        )
        weights = st.text_input(
            "Weights (.pt)", value="",
            help="Ignored in stub mode.", disabled=stub,
        )
        conf = st.slider("Confidence (general)", 0.0, 1.0, 0.25, 0.01)
        conf_military = st.slider(
            "Confidence (military)", 0.0, 1.0, 0.10, 0.01,
            help="Intentionally lower — the >90% military recall gate.",
        )
        if conf_military > conf:
            st.warning("Military threshold is above the general one — that "
                       "weakens the recall gate. Usually keep it lower.")

    # --- Input ---------------------------------------------------------------
    uploaded = st.file_uploader(
        "Upload an image or video",
        type=["jpg", "jpeg", "png", "bmp", "mp4", "mov", "avi", "mkv"],
    )

    source = "none"
    if uploaded is not None:
        suffix = Path(uploaded.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            source = tmp.name
        if suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            st.image(uploaded, caption=uploaded.name, use_container_width=True)
        else:
            st.video(uploaded)

    run = st.button("Run detection", type="primary")
    if not run:
        st.info("Configure the sidebar, optionally upload a file, then run. "
                "Stub mode needs no file — it uses a synthetic scene.")
        return

    # --- Inference -----------------------------------------------------------
    try:
        dets: list[Detection] = predict(
            source=source,
            weights=weights or None,
            conf=conf,
            conf_military=conf_military,
            stub=stub,
        )
    except NotImplementedError:
        st.error("Real inference isn't implemented yet. Turn on **Stub mode** "
                 "in the sidebar to preview the pipeline.")
        return

    # --- Military warning banner --------------------------------------------
    military = _military_class_names()
    flagged = [d for d in dets if d.class_name in military]
    if flagged:
        st.error(f"⚠️ {len(flagged)} MILITARY VESSEL(S) DETECTED — review required.")

    # --- Results table -------------------------------------------------------
    st.subheader(f"Detections ({len(dets)})")
    if dets:
        df = pd.DataFrame(d.to_dict() for d in dets)
        st.dataframe(df, use_container_width=True)
    else:
        st.write("No detections above the current thresholds.")


# Streamlit executes the module top-to-bottom on each rerun, so just run.
_run()
