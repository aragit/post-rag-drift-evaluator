from __future__ import annotations

import time

from evaluator.guardrails.policy import (
    PolicyDecision,
    PolicyEvaluator,
)
from evaluator.optimization.models import OptimizationAction


def _make_action(
    action_type: str = "rollback_config",
    target_run_id: str = "r1",
    change_id: str = "c1",
    params: dict | None = None,
    executed_at: float | None = None,
) -> OptimizationAction:
    metadata = {
        "source": "test",
    }
    if params is not None:
        metadata["params"] = params
    if executed_at is not None:
        metadata["executed_at"] = executed_at
    return OptimizationAction(
        action_type=action_type,
        target_run_id=target_run_id,
        change_id=change_id,
        description="test action",
        metadata=metadata,
    )


# ── Cooldown Period Tests ───────────────────────────────────────────────


def test_cooldown_allows_different_action_types():
    """Different action types targeting the same run should pass cooldown."""
    evaluator = PolicyEvaluator(cooldown_period_s=300.0)
    now = time.time()

    hist = [_make_action(action_type="adjust_temperature", target_run_id="r1", executed_at=now - 60)]
    action = _make_action(action_type="adjust_top_k", target_run_id="r1")

    result = evaluator.validate_action(action, hist)
    assert result.allowed
    assert result.rule_violated is None


def test_cooldown_blocks_identical_action_within_window():
    """Identical action within cooldown period should be blocked."""
    evaluator = PolicyEvaluator(cooldown_period_s=300.0)
    now = time.time()

    hist = [_make_action(action_type="rollback_config", target_run_id="r1", change_id="c1", executed_at=now - 10)]
    action = _make_action(action_type="rollback_config", target_run_id="r1", change_id="c1")

    result = evaluator.validate_action(action, hist)
    assert not result.allowed
    assert result.rule_violated == "cooldown_period"


def test_cooldown_allows_identical_action_after_window():
    """Identical action after cooldown period should be allowed."""
    evaluator = PolicyEvaluator(cooldown_period_s=300.0)
    now = time.time()

    hist = [_make_action(action_type="rollback_config", target_run_id="r1", change_id="c1", executed_at=now - 400)]
    action = _make_action(action_type="rollback_config", target_run_id="r1", change_id="c1")

    result = evaluator.validate_action(action, hist)
    assert result.allowed


def test_cooldown_no_history_allows():
    """No execution history means cooldown always passes."""
    evaluator = PolicyEvaluator(cooldown_period_s=300.0)
    action = _make_action()
    result = evaluator.validate_action(action, [])
    assert result.allowed


# ── Parameter Bounds Tests ──────────────────────────────────────────────


def test_parameter_bounds_blocks_high_top_k():
    """top_k=200 should violate bounds."""
    evaluator = PolicyEvaluator()
    action = _make_action(
        action_type="adjust_top_k",
        params={"top_k": 200},
    )
    result = evaluator.validate_action(action, [])
    assert not result.allowed
    assert result.rule_violated == "parameter_bounds"
    assert "top_k" in result.reason


def test_parameter_bounds_blocks_negative_temperature():
    """temperature=-0.5 should violate bounds."""
    evaluator = PolicyEvaluator()
    action = _make_action(
        action_type="adjust_temperature",
        params={"temperature": -0.5},
    )
    result = evaluator.validate_action(action, [])
    assert not result.allowed
    assert result.rule_violated == "parameter_bounds"


def test_parameter_bounds_allows_valid_temperature():
    """temperature=0.7 should pass."""
    evaluator = PolicyEvaluator()
    action = _make_action(
        action_type="adjust_temperature",
        params={"temperature": 0.7},
    )
    result = evaluator.validate_action(action, [])
    assert result.allowed


def test_parameter_bounds_allows_valid_top_k():
    """top_k=15 should pass."""
    evaluator = PolicyEvaluator()
    action = _make_action(
        action_type="adjust_top_k",
        params={"top_k": 15},
    )
    result = evaluator.validate_action(action, [])
    assert result.allowed


def test_parameter_bounds_custom_bounds():
    """Custom bounds should override defaults."""
    evaluator = PolicyEvaluator(
        custom_bounds={"top_k": (1.0, 10.0)}
    )
    action = _make_action(
        action_type="adjust_top_k",
        params={"top_k": 15},
    )
    result = evaluator.validate_action(action, [])
    assert not result.allowed


# ── Flapping Protection Tests ───────────────────────────────────────────


def test_flapping_blocks_excessive_toggles():
    """More than max_flapping_per_hour toggles on same run should be blocked."""
    evaluator = PolicyEvaluator(
        cooldown_period_s=0,
        max_flapping_per_hour=2,
        flapping_window_s=3600.0,
    )
    now = time.time()

    history = [
        _make_action(target_run_id="r1", executed_at=now - 100),
        _make_action(target_run_id="r1", executed_at=now - 50),
    ]
    action = _make_action(target_run_id="r1")

    result = evaluator.validate_action(action, history)
    assert not result.allowed
    assert result.rule_violated == "max_flapping"


def test_flapping_allows_within_limit():
    """Toggles within the limit should pass."""
    evaluator = PolicyEvaluator(
        cooldown_period_s=0,
        max_flapping_per_hour=5,
        flapping_window_s=3600.0,
    )
    now = time.time()

    history = [
        _make_action(target_run_id="r1", executed_at=now - 100),
        _make_action(target_run_id="r1", executed_at=now - 50),
    ]
    action = _make_action(target_run_id="r1")

    result = evaluator.validate_action(action, history)
    assert result.allowed


def test_flapping_ignores_old_history():
    """Actions older than the flapping window should not count."""
    evaluator = PolicyEvaluator(
        cooldown_period_s=0,
        max_flapping_per_hour=2,
        flapping_window_s=60.0,  # 1 minute window
    )
    now = time.time()

    history = [
        _make_action(target_run_id="r1", executed_at=now - 120),  # older than window
    ]
    action = _make_action(target_run_id="r1")

    result = evaluator.validate_action(action, history)
    assert result.allowed


# ── PolicyDecision Tests ──────────────────────────────────────────────────


def test_policy_decision_schema():
    """PolicyDecision should have the right fields."""
    decision = PolicyDecision(allowed=True, reason="ok")
    assert decision.allowed is True
    assert decision.reason == "ok"
    assert decision.rule_violated is None


def test_policy_decision_with_violation():
    """PolicyDecision with violation should set fields correctly."""
    decision = PolicyDecision(
        allowed=False,
        reason="too hot",
        rule_violated="parameter_bounds",
    )
    assert decision.allowed is False
    assert decision.rule_violated == "parameter_bounds"


# ── Integration Test ────────────────────────────────────────────────────


def test_policy_all_checks_pass():
    """A valid action with clean history should pass all checks."""
    evaluator = PolicyEvaluator()
    action = _make_action(
        action_type="adjust_temperature",
        params={"temperature": 0.5},
        target_run_id="r1",
    )
    result = evaluator.validate_action(action, [])
    assert result.allowed
    assert result.rule_violated is None
