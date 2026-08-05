import pytest
import requests

PROMETHEUS_URL = "http://localhost:9090"
GRAFANA_URL = "http://localhost:3000"
API_URL = "http://localhost:8000"


def _stack_reachable() -> bool:
    """True when the Sentrix control plane (API + Prometheus + Grafana) is up.

    These are live-stack integration tests. In CI, where ``docker compose up``
    is not run, they self-skip (rather than failing with ConnectionRefused).
    """
    try:
        return (
            requests.get(f"{API_URL}/metrics", timeout=2).status_code == 200
            and requests.get(f"{PROMETHEUS_URL}/-/healthy", timeout=2).status_code == 200
            and requests.get(f"{GRAFANA_URL}/api/health", timeout=2).status_code == 200
        )
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _stack_reachable(),
    reason="Sentrix control plane not running "
    "(start with: docker compose up -d && ./scripts/test_stack.sh)",
)


def test_api_metrics_format():
    """Verify the API exposes valid Prometheus text format headers."""
    response = requests.get(f"{API_URL}/metrics", timeout=5)
    assert response.status_code == 200
    assert "sentrix_up" in response.text
    assert "sentrix_drift_score" in response.text


def test_prometheus_target_health():
    """Verify Prometheus active scrape targets are healthy."""
    response = requests.get(f"{PROMETHEUS_URL}/api/v1/targets", timeout=5)
    assert response.status_code == 200
    targets = response.json()["data"]["activeTargets"]

    assert len(targets) > 0, "No active scrape targets found in Prometheus."
    target_states = [t["health"] for t in targets]
    assert "up" in target_states, "No active scrape target marked as UP."


def test_drift_score_bounds():
    """Verify Jensen-Shannon Divergence adheres to valid bounds [0, 1]."""
    query_url = f"{PROMETHEUS_URL}/api/v1/query?query=sentrix_drift_score{{metric_type=\"vector_jsd\"}}"
    response = requests.get(query_url, timeout=5)
    assert response.status_code == 200

    results = response.json()["data"]["result"]
    if results:
        for metric in results:
            score = float(metric["value"][1])
            assert 0.0 <= score <= 1.0, f"JSD score {score} out of bounds [0, 1]."


def test_grafana_authenticated_access():
    """Verify Grafana admin credentials work with the provisioned password."""
    response = requests.get(
        f"{GRAFANA_URL}/api/org",
        auth=("admin", "visdrift"),
        timeout=5
    )
    assert response.status_code == 200
    assert response.json()["name"] == "Main Org."
