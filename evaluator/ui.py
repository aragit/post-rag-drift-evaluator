import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
import polars as pl
import numpy as np
from sklearn.decomposition import PCA
from evaluator.config import config
from evaluator.drift_monitor import DriftMonitor

st.set_page_config(page_title="Post-RAG Drift Evaluator", page_icon="📊", layout="wide")
st.title("📊 Enterprise Post-RAG Latent Space Drift Telemetry")

st.sidebar.header("Configuration Profile")
threshold = st.sidebar.slider("JS Divergence Threshold Alert Limit", 0.05, 0.50, 0.15, 0.01)

monitor = DriftMonitor(threshold=threshold)

np.random.seed(42)
base_arr = np.random.normal(0, 1, (100, 128))
curr_arr = np.random.normal(0.2, 1.1, (100, 128))

df_base = pl.DataFrame({"embedding": base_arr.tolist()})
df_curr = pl.DataFrame({"embedding": curr_arr.tolist()})

js_score, is_drifted = monitor.compute_jensen_shannon_drift(df_base, df_curr)

col1, col2 = st.columns(2)
with col1:
    st.metric(label="Jensen-Shannon Divergence Score", value=f"{js_score:.4f}")
with col2:
    if is_drifted:
        st.error("🚨 CRITICAL STATE: SYSTEM EMBEDDING DRIFT DETECTED")
    else:
        st.success("🟢 STATUS NORMAL: RETRIEVAL MATRIX EMBEDDING STABLE")

st.subheader("2D PCA Coordinate Projection: Baseline Traffic vs Current Traffic Space")

pca = PCA(n_components=2)
all_data = np.vstack([base_arr, curr_arr])
pca.fit(all_data)

base_2d = pca.transform(base_arr)
curr_2d = pca.transform(curr_arr)

chart_data = pl.DataFrame({
    "PCA Axis 1": np.concatenate([base_2d[:, 0], curr_2d[:, 0]]),
    "PCA Axis 2": np.concatenate([base_2d[:, 1], curr_2d[:, 1]]),
    "Dataset Group": ["Baseline Core Data"] * 100 + ["Live Current Ingestion"] * 100
})

st.scatter_chart(
    chart_data.to_pandas(),
    x="PCA Axis 1",
    y="PCA Axis 2",
    color="Dataset Group",
    use_container_width=True
)
