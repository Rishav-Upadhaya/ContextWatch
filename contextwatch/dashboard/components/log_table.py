from __future__ import annotations

import pandas as pd
import streamlit as st


def render_log_table(rows: list[dict], title: str = "Logs") -> None:
    st.subheader(title)
    if not rows:
        st.info("No records available")
        return
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
