import json
from types import SimpleNamespace

import httpx
import pytest

from kwresearch.clients.dataforseo import BudgetExceeded, DataForSEO, DataForSEOError
from kwresearch.clients.llm import LLM, response_cost
from kwresearch.db import DB


def make(tmp_path, handler, max_cost=1.0):
    db = DB(tmp_path / "c.db")
    c = DataForSEO("login", "pw", db, run_id=1, max_cost_usd=max_cost)
    c.http = httpx.Client(base_url="https://api.dataforseo.com/v3", transport=httpx.MockTransport(handler))
    return c, db


def ok(items, cost=0.01):
    return httpx.Response(200, json={"status_code": 20000, "cost": cost, "tasks": [
        {"status_code": 20000, "result": [{"items": items}]}]})


def test_post_caches_and_logs_cost(tmp_path):
    calls = []

    def handler(req):
        calls.append(json.loads(req.content))
        return ok([{"keyword": "a", "keyword_difficulty": 42}])
    c, db = make(tmp_path, handler)
    assert c.bulk_keyword_difficulty(["a"]) == {"a": 42.0}
    assert c.bulk_keyword_difficulty(["a"]) == {"a": 42.0}
    assert len(calls) == 1                       # second call served from cache
    assert calls[0][0]["location_name"] == "India"
    assert db.run_cost(1) == {"dataforseo": 0.01}


def test_budget_cap(tmp_path):
    c, _ = make(tmp_path, lambda req: ok([], cost=0.6), max_cost=1.0)
    c.keyword_suggestions("a")
    c.keyword_suggestions("b")
    with pytest.raises(BudgetExceeded):
        c.keyword_suggestions("c")


def test_resume_budget_includes_previous_spend(tmp_path):
    c, db = make(tmp_path, lambda req: ok([], cost=0.3), max_cost=1.0)
    db.log_call(1, "dataforseo", "/previous", 0.8, False)
    resumed = DataForSEO("login", "pw", db, run_id=1, max_cost_usd=1.0)
    resumed.http = c.http
    assert resumed.spent == pytest.approx(0.8)
    resumed.keyword_suggestions("new")
    with pytest.raises(BudgetExceeded):
        resumed.keyword_suggestions("another")


def test_missing_dataforseo_price_is_visible(tmp_path):
    def handler(req):
        return httpx.Response(200, json={"status_code": 20000, "tasks": [
            {"status_code": 20000, "result": []}]})
    c, db = make(tmp_path, handler)
    with pytest.raises(DataForSEOError, match="valid cost"):
        c.keyword_suggestions("new")
    summary = db.run_cost_summary(1)
    assert summary["unpriced_calls"] == 1
    assert summary["total_usd"] == 0
    resumed = DataForSEO("login", "pw", db, run_id=1, max_cost_usd=1.0)
    assert resumed.spent >= resumed.max_cost


def test_dataforseo_error_response_still_records_returned_cost(tmp_path):
    c, db = make(tmp_path, lambda req: httpx.Response(200, json={
        "status_code": 50000, "status_message": "task rejected", "cost": 0.004, "tasks": []}))
    with pytest.raises(DataForSEOError, match="task rejected"):
        c.keyword_suggestions("new")
    assert db.run_cost_summary(1)["total_usd"] == pytest.approx(0.004)


def test_openrouter_uses_response_cost_without_a_lookup(tmp_path):
    db = DB(tmp_path / "llm.db")
    replies = iter([
        SimpleNamespace(usage={"cost": 0.00012345}, choices=[SimpleNamespace(
            message=SimpleNamespace(content='{"ok":true}'))]),
        SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10), choices=[SimpleNamespace(
            message=SimpleNamespace(content='{"ok":true}'))]),
    ])
    calls = []
    llm = object.__new__(LLM)
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs: (calls.append(kwargs), next(replies))[1])))
    llm.db, llm.run_id, llm.temperature, llm.use_cache = db, 7, 0.0, False
    assert llm.json("model", "system", "first") == {"ok": True}
    assert llm.json("model", "system", "second") == {"ok": True}
    assert all(call["extra_body"] == {"usage": {"include": True}} for call in calls)
    summary = db.run_cost_summary(7)
    assert summary["total_usd"] == pytest.approx(0.00012345)
    assert summary["unpriced_calls"] == 1
    assert response_cost({"cost": 0}) == 0
    assert response_cost(None) is None


def test_no_results_is_empty(tmp_path):
    c, _ = make(tmp_path, lambda req: httpx.Response(200, json={"status_code": 20000, "cost": 0, "tasks": [
        {"status_code": 40102, "status_message": "No Search Results."}]}))
    assert c.ranked_keywords("new-site.com", limit=100) == []


def test_lazy_embedder_falls_back_without_model(monkeypatch):
    import builtins
    from kwresearch.clients.embeddings import LazyEmbedder
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name.startswith("sentence_transformers"):
            raise ImportError("not installed")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    emb = LazyEmbedder()
    assert emb._impl is None          # nothing loaded at construction
    assert not emb                    # falls back -> lexical clustering


def test_base64_token_as_password_is_decoded(tmp_path):
    import base64
    from kwresearch.clients.dataforseo import normalize_credentials
    tok = base64.b64encode(b"me@x.com:realpass123").decode()
    assert normalize_credentials(" me@x.com ", tok) == ("me@x.com", "realpass123")
    assert normalize_credentials("me@x.com", '"plainpassword"') == ("me@x.com", "plainpassword")


def test_401_fails_fast_with_help(tmp_path):
    from kwresearch.clients.dataforseo import DataForSEOError
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(401, json={})
    c, _ = make(tmp_path, handler)
    with pytest.raises(DataForSEOError, match="API password"):
        c.keyword_suggestions("a")
    assert len(calls) == 1            # no retries on auth errors
