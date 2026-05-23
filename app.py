"""
Streamlit dashboard: German spam-call analytics (ML-powered pipeline output).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DATA_FILE = Path(__file__).resolve().parent / "processed_calls.csv"

ABOUT_SECTION = """
### Project Overview

This dashboard is part of a **Data Engineering & MLOps Pipeline** designed to monitor
and analyse fraudulent phone calls in Germany.

### Pipeline Steps

1. **`generate_data.py`** – Simulates realistic German phone-call records (E.164 format, 120-day window).
2. **`process_data.py`** – Engineers features and trains a **Random-Forest classifier** to detect spam.
3. **`app.py`** – This Streamlit dashboard visualises results in real time.

### Key Features

- **ML Classification:** Random-Forest model with 7 engineered features (temporal, number-type, duration).
- **Spam Probability:** Every call gets a confidence score from the model.
- **Legal Compliance:** Administrative fines (€0.20 per spam call) based on regulatory standards.
- **Real-time Analytics:** Time-series trends, peak-hour heatmaps, duration distributions.

### Tech Stack

Python · Pandas · scikit-learn · Streamlit · Plotly
"""


# ---------------------------------------------------------------------------
# Data loading & helpers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"Number": "string"})
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["fine_euro"] = df["fine_euro"].astype(float)
    df["spam_probability"] = df["spam_probability"].astype(float) if "spam_probability" in df.columns else 0.0
    df["predicted_label"] = df.get("predicted_label", df["Label"])
    df["date"] = df["Timestamp"].dt.date
    df["hour"] = df["Timestamp"].dt.hour
    df["day_of_week"] = df["Timestamp"].dt.day_name()
    return df


def duration_histogram(df: pd.DataFrame) -> pd.DataFrame:
    edges = [0, 30, 60, 120, 180, 300, 600, 1200, 10_000]
    labels = ["0-30 s", "31-60 s", "1-2 min", "2-3 min",
              "3-5 min", "5-10 min", "10-20 min", "> 20 min"]
    s = pd.cut(df["Duration"], bins=edges, labels=labels, right=True, include_lowest=True)
    out = s.value_counts().reindex(labels, fill_value=0).reset_index()
    out.columns = ["duration_bucket", "calls"]
    return out


def time_series_df(df: pd.DataFrame) -> pd.DataFrame:
    daily = (
        df.groupby(["date", "predicted_label"])
        .size()
        .reset_index(name="calls")
    )
    return daily


def heatmap_df(df: pd.DataFrame) -> pd.DataFrame:
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    spam = df[df["predicted_label"] == "Spam"].copy()
    pivot = (
        spam.groupby(["day_of_week", "hour"])
        .size()
        .reset_index(name="calls")
        .pivot(index="day_of_week", columns="hour", values="calls")
        .reindex(order)
        .fillna(0)
    )
    # fill missing hours
    for h in range(24):
        if h not in pivot.columns:
            pivot[h] = 0
    pivot = pivot[sorted(pivot.columns)]
    return pivot


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
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
        .dashboard-title { font-size: 1.75rem; font-weight: 700; margin-bottom: 0.15rem; }
        .dashboard-sub   { color: #6b7280; font-size: 0.95rem; margin-bottom: 1.25rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if not DATA_FILE.is_file():
        st.error(f"Data file not found: `{DATA_FILE}`. Run `python process_data.py` first.")
        st.stop()

    df = load_data(DATA_FILE)

    total_calls   = len(df)
    spam_n        = int((df["predicted_label"] == "Spam").sum())
    spam_pct      = spam_n / total_calls * 100 if total_calls else 0
    total_revenue = float(df["fine_euro"].sum())
    avg_prob      = float(df.loc[df["predicted_label"] == "Spam", "spam_probability"].mean()) if spam_n else 0.0

    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown('<p class="dashboard-title">📞 German Spam Call Monitoring</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="dashboard-sub">ML-powered pipeline · Random-Forest classifier · Regulatory fine tracker</p>',
        unsafe_allow_html=True,
    )

    # ── KPI row ─────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total calls",    f"{total_calls:,}")
    c2.metric("Spam detected",  f"{spam_n:,}",          f"{spam_pct:.1f} % of total")
    c3.metric("Total fines",    f"€{total_revenue:,.2f}")
    c4.metric("Avg spam confidence", f"{avg_prob:.0%}")

    st.divider()

    # ── Row 1 : Time-series  +  Duration histogram ───────────────────────
    col_ts, col_dur = st.columns(2, gap="large")

    with col_ts:
        st.subheader("📅 Daily call volume")
        st.caption("Spam vs Normal calls over time.")
        daily = time_series_df(df)
        fig_ts = px.line(
            daily,
            x="date",
            y="calls",
            color="predicted_label",
            color_discrete_map={"Spam": "#ef4444", "Normal": "#3b82f6"},
            markers=True,
        )
        fig_ts.update_layout(
            xaxis_title="Date",
            yaxis_title="Number of calls",
            legend_title="Label",
            height=380,
            margin=dict(l=40, r=20, t=20, b=60),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(gridcolor="rgba(128,128,128,0.15)"),
        )
        st.plotly_chart(fig_ts, use_container_width=True)

    with col_dur:
        st.subheader("⏱ Call volume by duration")
        st.caption("Distribution of all calls across duration buckets.")
        hist = duration_histogram(df)
        fig_dur = px.bar(
            hist,
            x="duration_bucket",
            y="calls",
            text="calls",
            color="calls",
            color_continuous_scale="Blues",
        )
        fig_dur.update_layout(
            xaxis_title="Duration bucket",
            yaxis_title="Number of calls",
            showlegend=False,
            height=380,
            margin=dict(l=40, r=20, t=20, b=80),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(gridcolor="rgba(128,128,128,0.15)"),
            coloraxis_showscale=False,
        )
        fig_dur.update_traces(textposition="outside", cliponaxis=False)
        st.plotly_chart(fig_dur, use_container_width=True)

    st.divider()

    # ── Row 2 : Heatmap  +  Spam/Normal donut ───────────────────────────
    col_heat, col_donut = st.columns((1.6, 1.0), gap="large")

    with col_heat:
        st.subheader("🕐 Spam peak hours heatmap")
        st.caption("When do spam calls happen? (day × hour)")
        pivot = heatmap_df(df)
        fig_heat = go.Figure(
            go.Heatmap(
                z=pivot.values,
                x=[f"{h:02d}:00" for h in pivot.columns],
                y=pivot.index.tolist(),
                colorscale="Reds",
                showscale=True,
                hoverongaps=False,
            )
        )
        fig_heat.update_layout(
            xaxis_title="Hour of day",
            yaxis_title="",
            height=340,
            margin=dict(l=10, r=10, t=20, b=60),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    with col_donut:
        st.subheader("🔵 Spam vs Normal")
        st.caption("Share of predicted labels across all calls.")
        label_counts = df["predicted_label"].value_counts().reset_index()
        label_counts.columns = ["label", "count"]
        fig_donut = px.pie(
            label_counts,
            names="label",
            values="count",
            hole=0.55,
            color="label",
            color_discrete_map={"Spam": "#ef4444", "Normal": "#3b82f6"},
        )
        fig_donut.update_traces(textposition="outside", textinfo="percent+label")
        fig_donut.update_layout(
            showlegend=False,
            height=340,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    st.divider()

    # ── Row 3 : Spam register ────────────────────────────────────────────
    st.subheader("🚨 Spam call register")
    st.caption("Search by number, timestamp fragment, or duration.")

    q = st.text_input("Search", placeholder="e.g. +49151 or 2026-03", label_visibility="collapsed")
    spam_df = df[df["predicted_label"] == "Spam"].copy()
    spam_df = spam_df.sort_values("spam_probability", ascending=False)

    if q.strip():
        needle = q.strip().lower()
        mask = (
            spam_df["Number"].str.lower().str.contains(needle, na=False, regex=False)
            | | spam_df["Timestamp"].astype(str).str.lower().str.contains(needle, na=False, regex=False)
            | spam_df["Duration"].astype(str).str.contains(needle, na=False)
        )
        spam_df = spam_df[mask]

    display_cols = ["Timestamp", "Number", "Duration", "spam_probability", "fine_euro"]
    st.dataframe(
        spam_df[display_cols],
        use_container_width=True,
        height=420,
        column_config={
            "Timestamp":        st.column_config.DatetimeColumn("Timestamp", format="YYYY-MM-DD HH:mm"),
            "Number":           st.column_config.TextColumn("Number"),
            "Duration":         st.column_config.NumberColumn("Duration (s)", format="%d"),
            "spam_probability": st.column_config.ProgressColumn("Spam confidence", min_value=0, max_value=1, format="%.0%%"),
            "fine_euro":        st.column_config.NumberColumn("Fine (€)", format="€%.2f"),
        },
        hide_index=True,
    )
    st.caption(f"Showing **{len(spam_df)}** spam row(s) · sorted by confidence ↓")

    st.divider()
    with st.expander("ℹ️ About this dashboard"):
        st.markdown(ABOUT_SECTION)


if __name__ == "__main__":
    main()
