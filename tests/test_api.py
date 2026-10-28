import httpx
from fastapi.testclient import TestClient

from backend.app import main

client = TestClient(main.app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_returns_answer_and_records_metric(monkeypatch):
    async def fake_call(function_name, payload):
        assert function_name == "answer"
        assert payload == {"question": "Refund policy?"}
        return {
            "answer": "See the refund policy. [S1]",
            "citations": [{"id": "S1", "source": "refund_policy.md"}],
            "evidence_count": 2,
            "model": "openai/gpt-oss-20b",
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        }

    recorded = []

    async def fake_record(metric):
        recorded.append(metric)

    monkeypatch.setattr(main, "call_edge_function", fake_call)
    monkeypatch.setattr(main, "record_metric", fake_record)

    response = client.post(
        "/api/chat",
        json={"question": "  Refund policy?  "},
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "See the refund policy. [S1]"
    assert len(recorded) == 1
    assert recorded[0]["request_type"] == "chat"
    assert recorded[0]["outcome"] == "success"
    assert recorded[0]["retrieved_count"] == 2
    assert recorded[0]["model_name"] == "openai/gpt-oss-20b"
    assert recorded[0]["prompt_tokens"] == 100
    assert recorded[0]["completion_tokens"] == 20


def test_search_forwards_match_count(monkeypatch):
    async def fake_call(function_name, payload):
        assert function_name == "search"
        assert payload == {
            "question": "Shipping policy?",
            "match_count": 3,
        }
        return {"result_count": 1, "results": [{"content": "Shipping"}]}

    async def fake_record(metric):
        pass

    monkeypatch.setattr(main, "call_edge_function", fake_call)
    monkeypatch.setattr(main, "record_metric", fake_record)

    response = client.post(
        "/api/search",
        json={"question": "Shipping policy?", "match_count": 3},
    )

    assert response.status_code == 200
    assert response.json()["result_count"] == 1


def test_rejects_empty_question():
    response = client.post("/api/chat", json={"question": "   "})
    assert response.status_code == 422


def test_rejects_invalid_match_count():
    response = client.post(
        "/api/search",
        json={"question": "Shipping?", "match_count": 100},
    )
    assert response.status_code == 422


def test_upstream_failure_returns_502(monkeypatch):
    async def fake_call(function_name, payload):
        request = httpx.Request("POST", "https://example.invalid")
        response = httpx.Response(503, request=request)
        raise httpx.HTTPStatusError(
            "Upstream unavailable",
            request=request,
            response=response,
        )

    recorded = []

    async def fake_record(metric):
        recorded.append(metric)

    monkeypatch.setattr(main, "call_edge_function", fake_call)
    monkeypatch.setattr(main, "record_metric", fake_record)

    response = client.post(
        "/api/chat",
        json={"question": "Refund policy?"},
    )

    assert response.status_code == 502
    assert recorded[0]["outcome"] == "failed"
    assert recorded[0]["error_code"] == "upstream_http_503"
