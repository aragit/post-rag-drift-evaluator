from __future__ import annotations

import pytest

from evaluator.latent_drift.adaptive import AdaptiveThresholdManager

# ── Core Threshold Tests ────────────────────────────────────────────────


def test_manager_initial_threshold():
    """Initial threshold should be the base threshold when no scores."""
    manager = AdaptiveThresholdManager(base_threshold=0.15)
    assert manager.get_threshold() == 0.15


def test_manager_updates_threshold():
    """After adding scores, threshold adapts via z-score."""
    manager = AdaptiveThresholdManager(base_threshold=0.15, sensitivity_z=2.0, min_threshold=0.0, max_threshold=1.0)

    # Low variance: threshold should be near mean + z*std
    for v in [0.04, 0.05, 0.06, 0.05, 0.04, 0.06, 0.05, 0.04, 0.06, 0.05]:
        manager.update(v)
    threshold_low_var = manager.get_threshold()
    assert threshold_low_var > 0.0  # non-trivial threshold

    # Check that threshold is computed as mean + z*std
    # variance = sum((x - mean)^2) / (n-1) = 2*(0.01)^2 / 9 ≈ 0.0000222
    # std ≈ 0.00471
    # threshold ≈ 0.05 + 2.0 * 0.00471 ≈ 0.0594
    assert threshold_low_var == pytest.approx(0.05 + 2.0 * 0.00471, abs=0.01)


def test_manager_identical_scores_threshold():
    """When all scores are identical, threshold = mean (std=0)."""
    manager = AdaptiveThresholdManager(base_threshold=0.15, sensitivity_z=2.0)

    for _ in range(10):
        manager.update(0.05)

    # mean=0.05, std=0, threshold = 0.05 + 2*0 = 0.05
    # But base_threshold=0.15 and no clamping... threshold becomes 0.05
    assert manager.get_threshold() == pytest.approx(0.05, abs=1e-6)


def test_manager_high_variance_increases_threshold():
    """High variance scores should produce a higher threshold."""
    manager = AdaptiveThresholdManager(base_threshold=0.05, sensitivity_z=2.0)

    # Low variance
    low_var_manager = AdaptiveThresholdManager(base_threshold=0.05, sensitivity_z=2.0)
    for v in [0.10, 0.10, 0.10, 0.10, 0.10]:
        low_var_manager.update(v)
    low_threshold = low_var_manager.get_threshold()

    # High variance
    for v in [0.01, 0.20, 0.05, 0.25, 0.10]:
        manager.update(v)
    high_threshold = manager.get_threshold()

    assert high_threshold >= low_threshold


def test_manager_threshold_clamped_to_bounds():
    """Threshold should be clamped to [min_threshold, max_threshold]."""
    manager = AdaptiveThresholdManager(
        base_threshold=0.15,
        sensitivity_z=10.0,  # very high z → would exceed max
        min_threshold=0.05,
        max_threshold=0.50,
    )

    for _ in range(20):
        manager.update(0.9)

    threshold = manager.get_threshold()
    assert threshold <= 0.50


def test_manager_threshold_clamped_to_min():
    """Threshold should not fall below min_threshold."""
    manager = AdaptiveThresholdManager(
        base_threshold=0.10,
        sensitivity_z=1.0,
        min_threshold=0.05,
        max_threshold=0.50,
    )

    for _ in range(20):
        manager.update(0.001)

    threshold = manager.get_threshold()
    assert threshold >= 0.05


def test_manager_reset_clears_window():
    """Reset should clear all historical scores."""
    manager = AdaptiveThresholdManager()

    for v in [0.1, 0.2, 0.3]:
        manager.update(v)

    assert len(manager.window) == 3
    manager.reset()
    assert len(manager.window) == 0
    assert manager.get_threshold() == 0.15  # base threshold


def test_manager_window_property():
    """Window property should return a list copy."""
    manager = AdaptiveThresholdManager()

    for v in [0.1, 0.2, 0.3]:
        manager.update(v)

    window = manager.window
    assert isinstance(window, list)
    assert len(window) == 3

    # Modifying returned list should not affect internal state
    window.clear()
    assert len(manager.window) == 3


def test_manager_respects_window_size():
    """Window should only retain the last ``window_size`` scores."""
    manager = AdaptiveThresholdManager(window_size=10)

    for i in range(20):
        manager.update(float(i) * 0.01)

    assert len(manager.window) == 10
    assert manager.window[-1] == 0.19
