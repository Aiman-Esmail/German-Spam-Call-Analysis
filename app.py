"""
Streamlit dashboard: German spam-call analytics (processed pipeline output).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

DATA_FILE = Path(__file__).resolve().parent / "processed_calls.csv"

ABOUT_SECTION = """
### Project Overview

This dashboard is part of a Data Engineering & MLOps Pipeline designed to monitor and analyze fraudulent phone calls in Germany.

### Key Features

- **Automated Ingestion:** Daily call logs are simulated and processed.
- **Legal Compliance:** The system calculates administrative fines (€0.20 per spam call) based on regulatory standards.
- **Real-time Analytics:** Provides insights into spam patterns and potential revenue recovery.
- **Tech Stack:** Python, Pandas, Streamlit, and Plotly.
"""


@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"Number": "string"})
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["fine_euro"] = df["fine_euro"].astype(float)
    return df


def duration_histogram(df: pd.DataFrame) -> pd.DataFrame:
    edges = [0, 30, 60, 120, 180, 300, 600, 1200, 10_000]
    labels = [
        "0-30 s",
        "31-60 s",
        "1-2 min",
        "2-3 min",
        "3-5 min",
        "5-10 min",
        "10-20 min",
        "> 20 min",
    ]
    s = pd.cut(df["Duration"], bins=edges, labels=labels, right=True, include_lowest=True)
    out = s.value_counts().reindex(labels, fill_value=0).reset_index()
    out.columns = ["duration_bucket", "calls"]
    return out


def main() -> None:
    st.set_page_config(
        page_title="Spam Call Analytics | MLOps Pipeline",
        page_icon="📞",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        div[data-testid="stMetricValue"] { font-size: 2.1rem; }
        .dashboard-title { font-size: 1.75rem; font-weight: 600; margin-bottom: 0.25rem; }
        .dashboard-sub { color: #6b7280; font-size: 0.95rem; margin-bottom: 1.25rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if not DATA_FILE.is_file():
        st.error(f"Data file not found: `{DATA_FILE}`. Run `python process_data.py` first.")
        st.stop()

    df = load_data(DATA_FILE)

    total_calls = len(df)
    spam_n = int((df["Label"] == "Spam").sum())
    total_revenue = float(df["fine_euro"].sum())

    st.markdown('<p class="dashboard-title">German Spam Call Monitoring</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="dashboard-sub">Operational view of processed call records & penalty revenue (MLOps data flow)</p>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total revenue", f"€{total_revenue:,.2f}")
    with c2:
        st.metric("Spam calls", f"{spam_n:,}")
    with c3:
        st.metric("Total calls", f"{total_calls:,}")

    st.divider()

    left, right = st.columns((1.15, 1.0), gap="large")

    with left:
        st.subheader("Call volume by duration")
        st.caption("Distribution of all calls across duration buckets (seconds).")
        hist = duration_histogram(df)
        fig = px.bar(
            hist,
            x="duration_bucket",
            y="calls",
            text="calls",
            color="calls",
            color_continuous_scale="Blues",
        )
        fig.update_layout(
            xaxis_title="Duration bucket",
            yaxis_title="Number of calls",
            showlegend=False,
            height=440,
            margin=dict(l=40, r=20, t=30, b=80),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(gridcolor="rgba(128,128,128,0.2)"),
            coloraxis_showscale=False,
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Spam call register")
        st.caption("Search by number, timestamp fragment, or duration.")
        q = st.text_input("Search", placeholder="e.g. +49151 or 2026-03", label_visibility="collapsed")
        spam_df = df[df["Label"] == "Spam"].copy()
        spam_df = spam_df.sort_values("Timestamp", ascending=False)
        if q.strip():
            needle = q.strip().lower()
            mask = (
                spam_df["Number"].str.lower().str.contains(needle, na=False)
                | spam_df["Timestamp"].astype(str).str.lower().str.contains(needle, na=False)
                | spam_df["Duration"].astype(str).str.contains(needle, na=False)
            )
            spam_df = spam_df[mask]
        st.dataframe(
            spam_df,
            use_container_width=True,
            height=440,
            column_config={
                "Timestamp": st.column_config.DatetimeColumn("Timestamp", format="YYYY-MM-DD HH:mm"),
                "Number": st.column_config.TextColumn("Number"),
                "Duration": st.column_config.NumberColumn("Duration (s)", format="%d"),
                "Label": st.column_config.TextColumn("Label"),
                "fine_euro": st.column_config.NumberColumn("Fine (€)", format="€%.2f"),
            },
            hide_index=True,
        )
        st.caption(f"Showing **{len(spam_df)}** spam row(s).")

    st.divider()
    with st.expander("About this dashboard"):
        st.markdown(ABOUT_SECTION)


if __name__ == "__main__":
    main()
