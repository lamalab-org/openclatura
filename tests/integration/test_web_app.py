"""Tests for the FastAPI web surface added in PR5.

Skipped when the ``[web]`` extra is not installed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from bluenamer.web import create_app  # noqa: E402


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app())


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "version" in body


def test_name_endpoint(client):
    r = client.post("/name", json={"smiles": "CCO"})
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "ethanol"
    assert body["smiles"] == "CCO"
    assert body["ok"] is True


def test_name_endpoint_with_trace(client):
    r = client.post("/name", json={"smiles": "CC(=O)Nc1ccccc1", "include_trace": True})
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "N-phenylacetamide"
    assert body["rules_hit"]
    assert "trace_segments" in body


def test_batch_endpoint_preserves_order(client):
    r = client.post(
        "/batch",
        json={"smiles": ["CCO", "c1ccccc1", "CC(=O)O"], "processes": 1},
    )
    assert r.status_code == 200
    rows = r.json()
    assert [row["smiles"] for row in rows] == ["CCO", "c1ccccc1", "CC(=O)O"]
    assert [row["name"] for row in rows] == ["ethanol", "benzene", "acetic acid"]


def test_describe_endpoint(client):
    r = client.post("/describe", json={"smiles": "CCO"})
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "ethanol"
    assert isinstance(body["paragraphs"], list)
    assert body["paragraphs"]


def test_name_endpoint_validates_payload(client):
    r = client.post("/name", json={"smiles": 123})  # type: ignore[dict-item]
    assert r.status_code == 422


def test_name_endpoint_captures_errors_not_raises(client):
    r = client.post("/name", json={"smiles": "definitely-not-smiles"})
    # Even on a bad SMILES the endpoint must return 200; the engine
    # captures naming errors on the payload.
    assert r.status_code == 200
    body = r.json()
    assert "name" in body
    assert "error" in body
