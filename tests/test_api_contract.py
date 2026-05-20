from __future__ import annotations

import importlib.util

import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("fastapi") is None,
    reason="FastAPI is not installed in this environment",
)


def test_api_health() -> None:
    from fastapi.testclient import TestClient

    from content_intel.api.main import app

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_api_search() -> None:
    from fastapi.testclient import TestClient

    from content_intel.api.main import app

    client = TestClient(app)
    response = client.post(
        "/search",
        json={"query": "guaranteed investment return contact", "top_k": 3},
    )
    assert response.status_code == 200
    assert len(response.json()) == 3

