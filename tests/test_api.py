"""Tests for src.api — the FastAPI wrapper.

Uses ``MockProvider`` so tests don't hit any real network. Bearer-token
auth is exercised by toggling ``API_AUTH_TOKEN`` via monkeypatch.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import src.api as api_module
from src.api import app


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """A TestClient with the lifespan hook bootstrapped from config.yaml.

    config.yaml ships with model_type=mock, so the fake provider is
    deterministic and offline.
    """
    monkeypatch.delenv("API_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("MODERATOR_CONFIG", str(Path("config.yaml").resolve()))
    with TestClient(app) as c:
        yield c


def test_root_returns_service_info(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["service"] == "strict-moderator"
    assert body["docs_url"] == "/docs"


def test_health_reports_loaded_model(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_type"] == "mock"
    assert body["auth_required"] is False


def test_moderate_classifies_clean_message(client: TestClient) -> None:
    resp = client.post("/moderate", json={"text": "Привет, как дела?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["verdict"] in {"ALLOW", "BLOCK"}
    assert body["category"] in {
        "ok",
        "spam",
        "gray_platform_switch",
        "hidden_aggression",
        "other",
    }
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["parse_ok"] is True
    # Mock provider reports zero tokens; the field must still be present.
    assert "usage" in body
    assert body["usage"]["prompt_tokens"] == 0
    assert body["usage"]["completion_tokens"] == 0
    assert body["usage"]["total_tokens"] == 0


def test_moderate_blocks_obvious_spam(client: TestClient) -> None:
    resp = client.post(
        "/moderate",
        json={"text": "Earn $5000/week guaranteed — DM me CRYPTO for free signals"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["verdict"] == "BLOCK"


def test_moderate_blocks_platform_switch(client: TestClient) -> None:
    resp = client.post(
        "/moderate",
        json={"text": "Hey, message me on whatsapp +380 67 123 45 67"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["verdict"] == "BLOCK"
    assert body["category"] == "gray_platform_switch"


def test_moderate_rejects_empty_text(client: TestClient) -> None:
    resp = client.post("/moderate", json={"text": ""})
    assert resp.status_code == 422  # FastAPI validation error


def test_moderate_rejects_oversized_text(client: TestClient) -> None:
    big = "x" * (api_module.MAX_TEXT_LENGTH + 1)
    resp = client.post("/moderate", json={"text": big})
    assert resp.status_code == 422


def test_batch_moderation_returns_one_result_per_input(client: TestClient) -> None:
    resp = client.post(
        "/moderate/batch",
        json={"texts": ["Hi there!", "Earn $5000 in crypto", "Привет :)"]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["results"]) == 3
    for item in body["results"]:
        assert item["verdict"] in {"ALLOW", "BLOCK"}
        assert "category" in item


def test_batch_rejects_empty_list(client: TestClient) -> None:
    resp = client.post("/moderate/batch", json={"texts": []})
    assert resp.status_code == 422


def test_batch_rejects_oversized_batch(client: TestClient) -> None:
    texts = ["msg"] * (api_module.MAX_BATCH_SIZE + 1)
    resp = client.post("/moderate/batch", json={"texts": texts})
    assert resp.status_code == 422


def test_batch_rejects_empty_string_item(client: TestClient) -> None:
    """Per-item min_length=1 mirrors the single endpoint so empty strings
    in a batch are rejected up-front instead of hitting the LLM."""
    resp = client.post("/moderate/batch", json={"texts": ["valid", "", "also valid"]})
    assert resp.status_code == 422


def test_batch_rejects_oversized_string_item(client: TestClient) -> None:
    """Per-item max_length=MAX_TEXT_LENGTH bounds resource use."""
    big = "x" * (api_module.MAX_TEXT_LENGTH + 1)
    resp = client.post("/moderate/batch", json={"texts": ["short", big]})
    assert resp.status_code == 422


# ---------- auth ----------
@pytest.fixture
def auth_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("API_AUTH_TOKEN", "supersecret")
    monkeypatch.setenv("MODERATOR_CONFIG", str(Path("config.yaml").resolve()))
    with TestClient(app) as c:
        yield c


def test_health_does_not_require_auth(auth_client: TestClient) -> None:
    """Liveness probes from k8s/Docker mustn't need a token."""
    resp = auth_client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["auth_required"] is True


def test_moderate_requires_auth_when_token_set(auth_client: TestClient) -> None:
    resp = auth_client.post("/moderate", json={"text": "Hi"})
    assert resp.status_code == 401


def test_moderate_rejects_wrong_bearer(auth_client: TestClient) -> None:
    resp = auth_client.post(
        "/moderate",
        json={"text": "Hi"},
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert resp.status_code == 401


def test_moderate_accepts_correct_bearer(auth_client: TestClient) -> None:
    resp = auth_client.post(
        "/moderate",
        json={"text": "Hi"},
        headers={"Authorization": "Bearer supersecret"},
    )
    assert resp.status_code == 200
    assert resp.json()["verdict"] in {"ALLOW", "BLOCK"}


# ---------- final_verdict / REVIEW tier ----------


def test_moderate_response_exposes_final_verdict_and_threshold(client: TestClient) -> None:
    """The response must carry the routing-tier verdict and the threshold it
    was derived from, so clients can explain REVIEW decisions to users."""
    resp = client.post("/moderate", json={"text": "Привет, как дела?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["final_verdict"] in {"ALLOW", "BLOCK", "REVIEW"}
    assert 0.0 <= body["confidence_threshold"] <= 1.0


def test_health_exposes_confidence_threshold(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "confidence_threshold" in body
    assert 0.0 <= body["confidence_threshold"] <= 1.0


def test_review_tier_triggered_when_threshold_above_mock_confidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raising the threshold above every MockProvider confidence level (≤0.9)
    must force final_verdict=REVIEW for every message."""
    from dataclasses import replace

    from src.utils import Config

    monkeypatch.delenv("API_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("MODERATOR_CONFIG", str(Path("config.yaml").resolve()))

    with TestClient(app) as raw_client:
        # Monkey-patch the loaded config in place to a very strict threshold.
        assert api_module._state.config is not None
        api_module._state.config = replace(
            api_module._state.config, confidence_threshold=0.999
        )
        try:
            resp = raw_client.post("/moderate", json={"text": "hello"})
            assert resp.status_code == 200
            body = resp.json()
            assert body["final_verdict"] == "REVIEW"
            assert body["verdict"] in {"ALLOW", "BLOCK"}  # model's raw call is untouched
        finally:
            # Restore for other tests.
            cfg = Config.from_file(Path("config.yaml").resolve())
            api_module._state.config = cfg


def test_batch_response_exposes_final_verdict_on_each_item(client: TestClient) -> None:
    resp = client.post(
        "/moderate/batch",
        json={"texts": ["hi", "hey", "hola"]},
    )
    assert resp.status_code == 200
    for item in resp.json()["results"]:
        assert item["final_verdict"] in {"ALLOW", "BLOCK", "REVIEW"}
