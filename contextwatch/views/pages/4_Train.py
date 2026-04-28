from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from pages._analyze_helpers import parse_logs


logger = logging.getLogger(__name__)

API_URL = os.environ.get("API_URL", "http://localhost:8000")
BATCH_TIMEOUT = int(os.environ.get("BATCH_READ_TIMEOUT_SECONDS", "300"))
CHUNK_SIZE = int(os.environ.get("BATCH_CHUNK_SIZE", "10"))
TRAINING_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "training" / "training_train.jsonl"


def api_get(path: str, timeout: int = 10) -> Optional[dict]:
    try:
        response = requests.get(f"{API_URL}{path}", timeout=timeout)
        if response.status_code != 200:
            return None
        return response.json()
    except requests.RequestException:
        logger.error("GET request failed for path %s", path, exc_info=True)
        return None


def format_metric_value(value: Optional[float]) -> str:
    if not isinstance(value, (int, float)):
        return "N/A"
    if value == 0:
        return "0"
    if abs(value) < 1e-3:
        return f"{value:.2e}"
    return f"{value:.4f}"


@st.cache_data(show_spinner=False)
def load_training_records(path: str) -> list[dict]:
    data_path = Path(path)
    if not data_path.exists():
        return []
    rows: list[dict] = []
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip malformed rows, continue with remaining dataset
                continue
    return rows


# st.set_page_config(page_title="ContextWatch - Train", page_icon="CW", layout="wide")
left_title, right_reset = st.columns([4, 1])
with left_title:
    st.title("Train")
with right_reset:
    if st.button("Reset baseline", use_container_width=True):
        st.session_state.pop("baseline_radius", None)
        st.rerun()

model_info = api_get("/model/info")
if model_info is None:
    st.warning("API offline - cannot fetch model state.")
    st.stop()

if "baseline_radius" not in st.session_state:
    st.session_state["baseline_radius"] = model_info.get("vhm_radius", None)

st.caption("Controlled LogBERT finetuning for clean operational log sequences")

with st.container(border=True):
    c1, c2, c3, c4 = st.columns(4)
    current_radius = model_info.get("vhm_radius")
    decision_radius = model_info.get("decision_radius")
    baseline_radius = st.session_state.get("baseline_radius")
    c1.metric("VHM status", "Fitted" if model_info.get("vhm_fitted", False) else "Not Fitted")
    c2.metric("VHM radius", format_metric_value(current_radius))
    c3.metric("Decision radius", format_metric_value(decision_radius))
    delta = current_radius - baseline_radius if isinstance(current_radius, (int, float)) and isinstance(baseline_radius, (int, float)) else None
    if delta is None:
        c4.metric("Baseline delta", "N/A")
    elif delta == 0:
        c4.metric("Baseline delta", "0")
    elif abs(delta) < 1e-3:
        c4.metric("Baseline delta", f"{delta:+.2e}")
    else:
        c4.metric("Baseline delta", f"{delta:+.4f}")

st.warning(
    "Finetuning updates the LogBERT model weights, refits the normal hypersphere, "
    "and calibrates the anomaly boundary from labeled logs. This affects all future anomaly scoring."
)

if "finetune_input" not in st.session_state:
    st.session_state["finetune_input"] = ""

st.markdown("### Training input")
ex_col, clr_col = st.columns(2)
load_all_clicked = ex_col.button("Train on training data", use_container_width=True)
clear_clicked = clr_col.button("Clear", use_container_width=True)

if load_all_clicked:
    training_records = load_training_records(str(TRAINING_DATA_PATH))
    if not training_records:
        st.error(f"Training data file not found: {TRAINING_DATA_PATH}")
    else:
        st.session_state["auto_train_all_data"] = True

if clear_clicked:
    st.session_state["finetune_input"] = ""

st.text_area("Paste training logs (JSON array / JSONL / {\"logs\": [...]})", height=180, key="finetune_input")

st.markdown("### Training controls")
ft_col1, ft_col2 = st.columns(2)
with ft_col1:
    ft_epochs = st.slider("Epochs", min_value=1, max_value=20, value=5, step=1)
with ft_col2:
    ft_lr = st.select_slider(
        "Learning rate",
        options=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        value=1e-3,
        format_func=lambda x: f"{x:.0e}",
    )
use_system_data = st.checkbox(
    "Include labeled system datasets",
    value=True,
    help="Uses the repo's golden and synthetic normal/anomaly JSONL logs to calibrate a real anomaly boundary.",
)
run_finetune = st.button("Start finetuning", type="primary", use_container_width=True)
auto_train_all_data = bool(st.session_state.pop("auto_train_all_data", False))

with st.expander("How finetuning works", expanded=False):
    st.markdown("#### 1. Masked Log Key Prediction (MLKP)")
    st.markdown(
        "Randomly masks 15% of tokens per sequence. "
        "LogBERT forward pass produces a CLS embedding projected to vocabulary logits. "
        "Cross-entropy loss on masked positions back-propagates to the embedding table and MLKP head."
    )
    st.markdown("#### 2. Volume Hypersphere Minimization (VHM)")
    st.markdown(
        "Computes mean squared distance of embeddings to the hypersphere centre. "
        "Centre updates via exponential moving average. "
        "Radius is set to the 95th percentile of embedding distances."
    )
    st.markdown("#### 3. Combined loss")
    st.markdown("Total loss = `L_MLKP + λ × L_VHM`. After training, VHM is refit on all embeddings.")

if run_finetune or auto_train_all_data:
    ft_logs = []
    use_system_data_for_run = use_system_data
    request_timeout = (20, 600)

    if auto_train_all_data:
        ft_logs = load_training_records(str(TRAINING_DATA_PATH))
        use_system_data_for_run = False
        request_timeout = None  # Full dataset training can run >1 hour.
        if not ft_logs:
            st.error(f"No valid records found in {TRAINING_DATA_PATH}")
            st.stop()
        st.info(f"Starting full-dataset training on {len(ft_logs)} records. This may take a long time.")
    else:
        raw_input = st.session_state.get("finetune_input", "")
        if raw_input.strip():
            try:
                ft_logs = parse_logs(raw_input)
            except ValueError as exc:
                st.error(f"Invalid input: {exc}")
                st.stop()
        elif not use_system_data_for_run:
            st.error("Paste at least one labeled log or enable system datasets.")
            st.stop()

    if st.session_state.get("baseline_radius") is None:
        current_info = api_get("/model/info") or {}
        st.session_state["baseline_radius"] = current_info.get("vhm_radius")

    with st.spinner(f"Finetuning LogBERT on {len(ft_logs)} sequences for {ft_epochs} epochs..."):
        try:
            resp = requests.post(
                f"{API_URL}/finetune",
                json={"logs": ft_logs, "epochs": ft_epochs, "learning_rate": ft_lr, "use_system_data": use_system_data_for_run},
                timeout=request_timeout,
            )
        except requests.RequestException as exc:
            logger.error("Finetune request failed", exc_info=True)
            st.error(f"Request failed: {exc}")
            st.stop()

    if resp.status_code != 200:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        st.error(f"Finetuning failed: {detail}")
        st.stop()

    data = resp.json()
    st.session_state["finetune_result"] = data
    st.success(
        f"Finetuning complete. "
        f"{data['epochs_completed']} epochs on {data['n_sequences']} sequences. "
        f"VHM radius after: {data['vhm_radius_after']:.4f}. "
        f"Decision radius after: {data['decision_radius_after']:.4f}"
    )

result = st.session_state.get("finetune_result")
if result:
    calibration = result.get("calibration") or {}
    if calibration:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Calibration F1", f"{calibration.get('best_f1', 0.0):.3f}")
        m2.metric("Precision", f"{calibration.get('precision', 0.0):.3f}")
        m3.metric("Recall", f"{calibration.get('recall', 0.0):.3f}")
        m4.metric("Labeled samples", int(calibration.get("normal_samples", 0)) + int(calibration.get("anomaly_samples", 0)))
    metrics = result.get("metrics", [])
    if metrics:
        mdf = pd.DataFrame(metrics)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=mdf["epoch"], y=mdf["loss_mlkp"], name="MLKP loss", mode="lines+markers", line=dict(color="#3b82f6", width=2)))
        fig.add_trace(go.Scatter(x=mdf["epoch"], y=mdf["loss_vhm"], name="VHM loss", mode="lines+markers", line=dict(color="#10b981", width=2)))
        fig.add_trace(go.Scatter(x=mdf["epoch"], y=mdf["loss_total"], name="Total loss", mode="lines+markers", line=dict(color="#f59e0b", width=2, dash="dash")))
        fig.update_layout(title="Training loss per epoch", xaxis_title="Epoch", yaxis_title="Loss", height=350)
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=mdf["epoch"], y=mdf["vhm_radius"], name="VHM radius", mode="lines+markers", line=dict(color="#a78bfa", width=2)))
        baseline = st.session_state.get("baseline_radius")
        if baseline is not None:
            fig2.add_hline(
                y=baseline, line_dash="dash", line_color="#6b7280",
                annotation_text="Pre-finetune radius", annotation_position="bottom right",
            )
        fig2.update_layout(title="VHM hypersphere radius per epoch", xaxis_title="Epoch", yaxis_title="Radius", height=280)
        st.plotly_chart(fig2, use_container_width=True)
