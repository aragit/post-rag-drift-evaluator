from __future__ import annotations

from unittest.mock import patch

from api.app import create_app
from api.middleware.rate_limit import reset_rate_limits
from evaluator.config import config
from tests.test_api_ingestion import FakeRepo, _frame, _frame_json, api_client


# --- Auth tests ---


async def test_requests_pass_without_api_key_when_auth_disabled():
    repo = FakeRepo()
    app = create_app(store=repo)

    async with api_client(app) as client:
        response = await client.post(
            "/v1/telemetry/frames", json=_frame_json(_frame())
        )

    assert response.status_code == 202


async def test_401_when_api_key_required_and_missing():
    repo = FakeRepo()
    app = create_app(store=repo)

    with patch.object(config, "API_KEY_REQUIRED", True):
        with patch.object(config, "API_KEYS", ["secret-key"]):
            async with api_client(app) as client:
                response = await client.post(
                    "/v1/telemetry/frames", json=_frame_json(_frame())
                )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API key"


async def test_401_when_api_key_invalid():
    repo = FakeRepo()
    app = create_app(store=repo)

    with patch.object(config, "API_KEY_REQUIRED", True):
        with patch.object(config, "API_KEYS", ["secret-key"]):
            async with api_client(app) as client:
                response = await client.post(
                    "/v1/telemetry/frames",
                    json=_frame_json(_frame()),
                    headers={"X-API-Key": "wrong-key"},
                )

    assert response.status_code == 401


async def test_202_when_valid_api_key_provided():
    repo = FakeRepo()
    app = create_app(store=repo)

    with patch.object(config, "API_KEY_REQUIRED", True):
        with patch.object(config, "API_KEYS", ["secret-key"]):
            async with api_client(app) as client:
                response = await client.post(
                    "/v1/telemetry/frames",
                    json=_frame_json(_frame()),
                    headers={"X-API-Key": "secret-key"},
                )

    assert response.status_code == 202


async def test_valid_bearer_token_authentication():
    repo = FakeRepo()
    app = create_app(store=repo)

    with patch.object(config, "API_KEY_REQUIRED", True):
        with patch.object(config, "API_KEYS", ["secret-key"]):
            async with api_client(app) as client:
                response = await client.post(
                    "/v1/telemetry/frames",
                    json=_frame_json(_frame()),
                    headers={"Authorization": "Bearer secret-key"},
                )

    assert response.status_code == 202


async def test_health_endpoints_bypass_auth():
    repo = FakeRepo()
    app = create_app(store=repo)

    with patch.object(config, "API_KEY_REQUIRED", True):
        with patch.object(config, "API_KEYS", []):
            async with api_client(app) as client:
                healthz_resp = await client.get("/healthz")
                metrics_resp = await client.get("/metrics")

    assert healthz_resp.status_code == 200
    assert healthz_resp.json() == {"status": "alive"}
    assert metrics_resp.status_code == 200


# --- Rate limiting tests ---


async def test_rate_limit_returns_429_when_exceeded():
    repo = FakeRepo()
    app = create_app(store=repo)

    old_limit = config.RATE_LIMIT_PER_MINUTE
    config.RATE_LIMIT_PER_MINUTE = 2
    reset_rate_limits()

    try:
        async with api_client(app) as client:
            r1 = await client.post(
                "/v1/telemetry/frames", json=_frame_json(_frame())
            )
            r2 = await client.post(
                "/v1/telemetry/frames", json=_frame_json(_frame())
            )
            r3 = await client.post(
                "/v1/telemetry/frames", json=_frame_json(_frame())
            )

        assert r1.status_code == 202
        assert r2.status_code == 202
        assert r3.status_code == 429
        assert "Rate limit exceeded" in r3.json()["detail"]
    finally:
        config.RATE_LIMIT_PER_MINUTE = old_limit
        reset_rate_limits()


async def test_rate_limit_per_api_key():
    repo = FakeRepo()
    app = create_app(store=repo)

    old_limit = config.RATE_LIMIT_PER_MINUTE
    old_required = config.API_KEY_REQUIRED
    old_keys = config.API_KEYS

    config.RATE_LIMIT_PER_MINUTE = 1
    config.API_KEY_REQUIRED = True
    config.API_KEYS = ["key-a", "key-b"]
    reset_rate_limits()

    try:
        async with api_client(app) as client:
            # key-a: 1 request allowed
            r1 = await client.post(
                "/v1/telemetry/frames",
                json=_frame_json(_frame()),
                headers={"X-API-Key": "key-a"},
            )
            # key-a: exceeds limit
            r2 = await client.post(
                "/v1/telemetry/frames",
                json=_frame_json(_frame()),
                headers={"X-API-Key": "key-a"},
            )
            # key-b: separate bucket, should succeed
            r3 = await client.post(
                "/v1/telemetry/frames",
                json=_frame_json(_frame()),
                headers={"X-API-Key": "key-b"},
            )

        assert r1.status_code == 202
        assert r2.status_code == 429
        assert r3.status_code == 202
    finally:
        config.RATE_LIMIT_PER_MINUTE = old_limit
        config.API_KEY_REQUIRED = old_required
        config.API_KEYS = old_keys
        reset_rate_limits()
