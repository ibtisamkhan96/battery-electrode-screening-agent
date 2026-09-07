"""Exercises the FastAPI app directly, no live MP/LLM keys involved: api.main._build_graph
is monkeypatched to return a fake graph before any request runs, so these tests check the
API's own contract (endpoints, status codes, job polling, auth gate) rather than the
agent's reasoning, which test_graph_routing.py already covers separately.
"""
import time

import api.main as main_module
from fastapi.testclient import TestClient


class FakeGraph:
    def invoke(self, state):
        time.sleep(0.05)
        return {"log": [f"processed: {state['raw_query']}"], "final_report": "fake report"}


def _client_with_fake_graph():
    main_module._build_graph = lambda provider, api_key: FakeGraph()
    main_module._job_store = main_module.JobStore()
    return TestClient(main_module.app)


def test_health():
    client = _client_with_fake_graph()
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
    print("PASSED: health check")


def test_submit_and_poll_query():
    client = _client_with_fake_graph()
    r = client.post("/query", json={"query": "find a Li cathode without cobalt", "api_key": "test-key"})
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    assert r.json()["status"] == "pending"

    for _ in range(20):
        status = client.get(f"/query/{job_id}").json()
        if status["status"] == "completed":
            break
        time.sleep(0.05)
    else:
        raise AssertionError("job never completed")

    assert status["result"]["final_report"] == "fake report"
    print("PASSED: submit and poll query")


def test_unknown_job_returns_404():
    client = _client_with_fake_graph()
    r = client.get("/query/does-not-exist")
    assert r.status_code == 404
    print("PASSED: unknown job returns 404")


def test_query_without_api_key_is_rejected():
    client = _client_with_fake_graph()
    r = client.post("/query", json={"query": "find a Li cathode without cobalt"})
    assert r.status_code == 422
    print("PASSED: missing api_key rejected before a graph is ever built")


def test_auth_gate_blocks_without_token(monkeypatch):
    monkeypatch.setenv("API_AUTH_TOKEN", "secret123")
    client = _client_with_fake_graph()

    r = client.post("/query", json={"query": "find a Li cathode", "api_key": "test-key"})
    assert r.status_code == 401

    r = client.post("/query", json={"query": "find a Li cathode", "api_key": "test-key"},
                     headers={"Authorization": "Bearer secret123"})
    assert r.status_code == 200
    print("PASSED: auth gate blocks without token, allows with correct token")


if __name__ == "__main__":
    test_health()
    test_submit_and_poll_query()
    test_unknown_job_returns_404()
    test_query_without_api_key_is_rejected()
    print("ALL API TESTS PASSED (run test_auth_gate_blocks_without_token via pytest for the monkeypatch fixture)")
