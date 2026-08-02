import asyncio

import asyncpg
import polars as pl
import streamlit as st

from evaluator.config import config
from evaluator.drift_store import DriftStore

st.set_page_config(page_title="Post-RAG Drift Evaluator", page_icon="📊", layout="wide")
st.title("📊 Enterprise Post-RAG Latent Space Drift Telemetry")

st.sidebar.header("Configuration Profile")
threshold = st.sidebar.slider(
    "JS Divergence Threshold Alert Limit", 0.05, 0.50, 0.15, 0.01
)

if st.sidebar.button("Refresh", key="refresh_button"):
    st.rerun()


def _load_dashboard() -> tuple[dict | None, pl.DataFrame | None, bool, list]:
    """Run async store reads on a loop-local pool (safe for Streamlit reruns)."""

    async def _load() -> tuple[dict | None, pl.DataFrame | None, bool, list]:
        pool = await asyncpg.create_pool(
            dsn=config.DATABASE_URL, min_size=1, max_size=2
        )
        try:
            store = DriftStore(pool=pool)
            latest = await store.get_latest_drift()
            trend = await store.get_trend(window=30)
            anomaly = await store.detect_anomaly(window=30)
            frames = await store.get_recent_frames(limit=10)
            return latest, trend, anomaly, frames
        finally:
            await pool.close()

    try:
        return asyncio.run(_load())
    except Exception:
        return None, None, False, []


latest, trend, anomaly, frames = _load_dashboard()

if latest is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label="Jensen-Shannon Divergence Score",
            value=f"{latest['jsd_score']:.4f}",
        )
    with col2:
        if latest["is_drifted"]:
            st.error("🚨 CRITICAL STATE: SYSTEM EMBEDDING DRIFT DETECTED")
        else:
            st.success("🟢 STATUS NORMAL: RETRIEVAL MATRIX EMBEDDING STABLE")

    st.subheader("Drift History Trend")
    if trend is not None and trend.height > 0:
        trend_df = trend.to_pandas().set_index("timestamp")
        chart_cols = [c for c in ("jsd_score", "rolling_mean") if c in trend_df.columns]
        chart_df = trend_df[chart_cols].copy()
        if "rolling_std" in trend_df.columns:
            chart_df["upper_band"] = (
                trend_df["rolling_mean"] + 2 * trend_df["rolling_std"]
            )
            chart_df["lower_band"] = (
                trend_df["rolling_mean"] - 2 * trend_df["rolling_std"]
            )
        st.line_chart(chart_df)
    else:
        st.info(
            "No drift history recorded yet. Run a comprehensive drift check to populate history."
        )

    st.subheader("Anomaly Detection")
    if anomaly:
        st.warning("⚠️ Anomaly detected in recent drift history.")
    else:
        st.info("No anomalies detected in recent history.")

    if frames:
        st.subheader("Recent Telemetry Frames")
        st.dataframe(
            pl.DataFrame(
                {
                    "trace_id": [f.trace_id for f in frames],
                    "rag_type": [f.metadata.rag_type for f in frames],
                    "timestamp": [f.timestamp.isoformat() for f in frames],
                    "query": [f.query.text for f in frames],
                    "confidence": [f.output.confidence_score for f in frames],
                }
            ).to_pandas()
        )
else:
    st.info(
        "No drift data available. Run a comprehensive drift check to populate the store."
    )
