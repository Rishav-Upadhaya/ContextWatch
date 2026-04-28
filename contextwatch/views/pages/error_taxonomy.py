"""
ContextWatch Error Taxonomy Dashboard

Streamlit page for viewing classified anomalies and error distributions.

Tabs:
  1. Summary: Pie chart, line chart of error types over time
  2. Events: Sortable table of detected anomalies
  3. Search: Filter by error type, date range, confidence
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Optional, List

# Set page config
st.set_page_config(
    page_title="ContextWatch Error Taxonomy",
    page_icon="📊",
    layout="wide",
)

st.title("🔍 ContextWatch Error Taxonomy Dashboard")

# ─── Placeholder Data ───────────────────────────────────────────────────────

# For now, use mock data until PostgreSQL integration is complete
MOCK_ANOMALIES_DATA = [
    {
        "log_id": "cw-001",
        "anomaly_type": "TOOL_HALLUCINATION",
        "timestamp": datetime.now() - timedelta(hours=2),
        "confidence": 0.92,
        "score": 0.85,
        "explanation": "Model invoked non-existent tool 'sendEmail_v2'",
    },
    {
        "log_id": "cw-002",
        "anomaly_type": "CONTEXT_POISONING",
        "timestamp": datetime.now() - timedelta(hours=1),
        "confidence": 0.88,
        "score": 0.72,
        "explanation": "Malicious instruction detected in retrieved context",
    },
    {
        "log_id": "cw-003",
        "anomaly_type": "REGISTRY_OVERFLOW",
        "timestamp": datetime.now() - timedelta(minutes=30),
        "confidence": 0.75,
        "score": 0.65,
        "explanation": "Tool confusion among 50+ similar tools in registry",
    },
    {
        "log_id": "cw-004",
        "anomaly_type": "DELEGATION_CHAIN_FAILURE",
        "timestamp": datetime.now() - timedelta(minutes=15),
        "confidence": 0.82,
        "score": 0.78,
        "explanation": "Partial output from step 1 caused JSON parse failure in step 2",
    },
    {
        "log_id": "cw-005",
        "anomaly_type": "TOOL_HALLUCINATION",
        "timestamp": datetime.now() - timedelta(minutes=5),
        "confidence": 0.91,
        "score": 0.81,
        "explanation": "Schema validation failed: unexpected parameters",
    },
]

# ─── Tabs ──────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(["📈 Summary", "📋 Events", "🔎 Search", "🧠 Reasoning"])

# ─── TAB 1: Summary ─────────────────────────────────────────────────────────

with tab1:
    col1, col2, col3 = st.columns(3)
    
    total_anomalies = len(MOCK_ANOMALIES_DATA)
    with col1:
        st.metric("Total Anomalies", total_anomalies)
    
    avg_confidence = sum(a["confidence"] for a in MOCK_ANOMALIES_DATA) / len(MOCK_ANOMALIES_DATA)
    with col2:
        st.metric("Avg Confidence", f"{avg_confidence:.2%}")
    
    error_types = {}
    for anomaly in MOCK_ANOMALIES_DATA:
        etype = anomaly["anomaly_type"]
        error_types[etype] = error_types.get(etype, 0) + 1
    
    top_error = max(error_types.items(), key=lambda x: x[1])[0]
    with col3:
        st.metric("Top Error Type", top_error)
    
    # Distribution by error type
    st.subheader("Error Type Distribution")
    col_pie, col_bar = st.columns(2)
    
    with col_pie:
        df_types = pd.DataFrame(list(error_types.items()), columns=["Error Type", "Count"])
        fig_pie = px.pie(df_types, values="Count", names="Error Type", title="Anomaly Type Distribution")
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col_bar:
        df_types_sorted = df_types.sort_values("Count", ascending=True)
        fig_bar = px.bar(
            df_types_sorted,
            x="Count",
            y="Error Type",
            orientation="h",
            title="Anomalies per Type",
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Timeline (mock: hourly)
    st.subheader("Anomalies Over Time")
    hourly_data = []
    for i in range(24):
        timestamp = datetime.now() - timedelta(hours=24-i)
        count = len([a for a in MOCK_ANOMALIES_DATA if (datetime.now() - a["timestamp"]).total_seconds() / 3600 < 24-i])
        hourly_data.append({"Hour": timestamp.strftime("%H:%M"), "Anomalies": count})
    
    df_hourly = pd.DataFrame(hourly_data)
    fig_line = px.line(df_hourly, x="Hour", y="Anomalies", title="Hourly Anomaly Count", markers=True)
    st.plotly_chart(fig_line, use_container_width=True)

# ─── TAB 2: Events Table ────────────────────────────────────────────────────

with tab2:
    st.subheader("Detected Anomalies")
    
    # Convert to DataFrame
    df = pd.DataFrame(MOCK_ANOMALIES_DATA)
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df["confidence"] = df["confidence"].apply(lambda x: f"{x:.1%}")
    df["score"] = df["score"].apply(lambda x: f"{x:.3f}")
    
    # Display table
    st.dataframe(
        df[["log_id", "anomaly_type", "timestamp", "confidence", "score", "explanation"]],
        use_container_width=True,
        height=400,
    )
    
    st.caption(f"Showing {len(df)} anomalies. Click column headers to sort.")

# ─── TAB 3: Search & Filter ─────────────────────────────────────────────────

with tab3:
    st.subheader("Search & Filter")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_types = st.multiselect(
            "Error Types",
            options=["TOOL_HALLUCINATION", "CONTEXT_POISONING", "REGISTRY_OVERFLOW", "DELEGATION_CHAIN_FAILURE"],
            default=["TOOL_HALLUCINATION"],
            key="filter_types",
        )
    
    with col2:
        min_confidence = st.slider("Min Confidence", 0.0, 1.0, 0.7, key="filter_conf")
    
    with col3:
        date_range = st.date_input(
            "Date Range",
            value=(datetime.now().date() - timedelta(days=7), datetime.now().date()),
            key="filter_dates",
        )
    
    # Filter data
    df_filtered = pd.DataFrame(MOCK_ANOMALIES_DATA)
    df_filtered = df_filtered[df_filtered["anomaly_type"].isin(selected_types)]
    df_filtered = df_filtered[df_filtered["confidence"] >= min_confidence]
    if len(date_range) == 2:
        df_filtered = df_filtered[
            (df_filtered["timestamp"].dt.date >= date_range[0])
            & (df_filtered["timestamp"].dt.date <= date_range[1])
        ]
    
    df_filtered["timestamp"] = df_filtered["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df_filtered["confidence"] = df_filtered["confidence"].apply(lambda x: f"{x:.1%}")
    df_filtered["score"] = df_filtered["score"].apply(lambda x: f"{x:.3f}")
    
    st.dataframe(
        df_filtered[["log_id", "anomaly_type", "timestamp", "confidence", "score"]],
        use_container_width=True,
        height=400,
    )
    
    st.caption(f"Filtered: {len(df_filtered)} of {len(MOCK_ANOMALIES_DATA)} anomalies")
    
    # Export option
    if st.button("📥 Export as CSV"):
        csv = df_filtered.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"anomalies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

# ─── TAB 4: Agentic Reasoning ───────────────────────────────────────────────

with tab4:
    st.subheader("🧠 Agentic Error Explanation")
    
    st.markdown("""
    Use the multi-node LangGraph reasoning engine to generate structured explanations for detected anomalies.
    
    **Workflow:**
    1. **Context Gathering** - Fetch error details + similar historic errors
    2. **Signal Extraction** - Extract domain-specific signals from error patterns
    3. **Reasoning** - Claude-powered root cause analysis
    4. **Remediation** - Generate actionable fix suggestions
    """)
    
    st.markdown("---")
    
    # Select anomaly to explain
    anomaly_options = {f"{a['log_id']} ({a['anomaly_type']})": a for a in MOCK_ANOMALIES_DATA}
    selected_anomaly_str = st.selectbox(
        "Select an anomaly to explain:",
        options=list(anomaly_options.keys()),
        index=0,
    )
    
    selected_anomaly = anomaly_options[selected_anomaly_str]
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Anomaly ID**: {selected_anomaly['log_id']}")
    with col2:
        st.info(f"**Error Type**: {selected_anomaly['anomaly_type']}")
    
    # Explain button
    if st.button("🔍 Generate Explanation", key="btn_explain", use_container_width=True):
        with st.spinner("⏳ Generating explanation using LangGraph reasoning engine..."):
            try:
                import requests
                import json
                
                # Call reasoning API
                api_url = "http://localhost:8000/reasoning/explain_error"
                
                payload = {
                    "anomaly_id": selected_anomaly["log_id"],
                    "error_type": selected_anomaly["anomaly_type"],
                    "raw_log": {
                        "message": selected_anomaly["explanation"],
                        "timestamp": selected_anomaly["timestamp"].isoformat(),
                        "source": "anomaly_detector",
                        "pattern": {
                            "confidence": selected_anomaly["confidence"],
                            "score": selected_anomaly["score"],
                        }
                    }
                }
                
                response = requests.post(api_url, json=payload, timeout=30)
                response.raise_for_status()
                
                explanation = response.json()
                
                # Display results
                st.success("✅ Explanation Generated!")
                
                st.markdown("### 📊 Root Cause Analysis")
                st.markdown(explanation.get("root_cause_analysis", "No analysis available"))
                
                st.markdown("### 🛠️ Immediate Remediation Steps")
                cols = st.columns(min(len(explanation.get("remediation_steps", [])), 3))
                for idx, step in enumerate(explanation.get("remediation_steps", [])):
                    with cols[idx % 3]:
                        st.info(f"**{idx + 1}.** {step}")
                
                st.markdown("### 🛡️ Prevention Measures")
                for measure in explanation.get("prevention_measures", []):
                    st.markdown(f"- {measure}")
                
                st.markdown("### 📈 Confidence & Signals")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Confidence Level", explanation.get("confidence_level", "N/A"))
                with col2:
                    st.metric("Signal Confidence", f"{explanation.get('signal_confidence', 0):.2f}")
                with col3:
                    st.metric("Execution Steps", len(explanation.get("execution_trace", [])))
                
                st.markdown("### 📋 Execution Trace")
                with st.expander("View detailed execution trace"):
                    for trace_step in explanation.get("execution_trace", []):
                        st.code(trace_step, language="text")
                
            except requests.exceptions.ConnectionError:
                st.error("❌ Could not connect to API server. Ensure ContextWatch API is running on http://localhost:8000")
            except Exception as e:
                st.error(f"❌ Error generating explanation: {str(e)}")
    
    st.markdown("---")
    st.markdown("### 💡 About Agentic Reasoning")
    st.markdown("""
    The reasoning engine uses a **multi-node LangGraph workflow** to analyze errors:
    
    - **Node 1: Context Gathering** collects error metadata and retrieves similar past errors
    - **Node 2: Signal Extraction** identifies domain-specific patterns (tool registry size, schema violations, etc.)
    - **Node 3: Reasoning** calls Claude LLM for structured root cause analysis
    - **Node 4: Remediation** generates actionable remediation and prevention steps
    
    Each step is tracked in the **Execution Trace** for debugging and transparency.
    """)

# ─── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")
    
    refresh_interval = st.slider("Refresh Interval (sec)", 5, 300, 60)
    
    st.markdown("---")
    st.markdown("## Error Types")
    st.markdown("""
    - 🔧 **TOOL_HALLUCINATION**: Model invents tool capability
    - 🔓 **CONTEXT_POISONING**: Malicious instructions in context
    - 📦 **REGISTRY_OVERFLOW**: Confusion from too many tools
    - 🔗 **DELEGATION_CHAIN_FAILURE**: Multi-step workflow breaks
    """)
    
    st.markdown("---")
    st.info("**Note:** Currently using mock data. PostgreSQL integration coming soon.")
