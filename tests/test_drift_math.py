import pytest
import polars as pl
import numpy as np
from evaluator.drift_monitor import DriftMonitor

def test_drift_monitor_zero_divergence():
    """Asserts that identical data distributions correctly result in a 0.0 drift score."""
    monitor = DriftMonitor(threshold=0.15)
    
    np.random.seed(42)
    base_embeddings = np.random.normal(0, 1, (100, 128)).tolist()
    
    df_base = pl.DataFrame({"embedding": base_embeddings})
    
    drift_score, is_drifted = monitor.compute_jensen_shannon_drift(df_base, df_base)
    
    assert drift_score == pytest.approx(0.0, abs=1e-5)
    assert is_drifted is False

def test_drift_monitor_detects_significant_shift():
    """Asserts that distinct distribution changes correctly trigger system alerts."""
    monitor = DriftMonitor(threshold=0.15)
    np.random.seed(42)
    
    base_embeddings = np.random.normal(0, 1, (100, 128)).tolist()
    current_embeddings = np.random.normal(5, 2, (100, 128)).tolist()
    
    df_base = pl.DataFrame({"embedding": base_embeddings})
    df_curr = pl.DataFrame({"embedding": current_embeddings})
    
    drift_score, is_drifted = monitor.compute_jensen_shannon_drift(df_base, df_curr)
    
    assert drift_score > 0.15
    assert is_drifted is True
