from __future__ import annotations

import streamlit as st


def render_anomaly_card(item: dict) -> None:
    st.markdown("### Anomaly Detail")
    st.metric("Type", item.get("anomaly_type", "unknown"))
    st.metric("Score", f"{item.get('anomaly_score', 0):.3f}")
    st.metric("Confidence", f"{item.get('confidence', 0):.2f}")
    if item.get("explanation"):
        st.write(item["explanation"])
