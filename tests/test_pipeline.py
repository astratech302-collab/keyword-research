import csv
import hashlib

import numpy as np
import pytest

from kwresearch.clients.dataforseo import parse_keyword_item, parse_ranked_item, trend_ratios
from kwresearch.clients.llm import extract_json
from kwresearch.config import RunConfig
from kwresearch.context import Ctx
from kwresearch.db import DB
from kwresearch.mock import FakeDataForSEO, FakeLLM, fake_fetch
from kwresearch.pipeline.candidates import normalize_keyword
from kwresearch.pipeline.cluster import lexical_units, signature
from kwresearch.pipeline.scoring import decide_action, rank_score
from kwresearch.runner import run_pipeline


# ---------------------------------------------------------------- units
def test_normalize():
    assert normalize_keyword("  Data  Governance Tools? ") == "data governance tools"
    assert normalize_keyword("“best CRM”") == "best crm"


def test_signature_merges_variants():
    assert signature("best data governance tools") == signature("data governance tool")
    assert signature("what is data lineage") == signature("data lineage")  # intent split happens elsewhere


def test_lexical_units_uses_core_keyword():
    kws = [{"keyword": "crm software"}, {"keyword": "customer relationship management tool", "core_keyword": "crm software"},
           {"keyword": "email marketing"}]
    units = sorted(sorted(u) for u in lexical_units(kws))
    assert units == [[0, 1], [2]]


def test_trend_ratios():
    monthly = [{"year": 2026, "month": 12 - i, "search_volume": 200 if i < 3 else 100} for i in range(12)] + \
              [{"year": 2025, "month": 12 - i, "search_volume": 50} for i in range(3)]
    t3, t12 = trend_ratios(monthly, None)
    assert t3 == pytest.approx(2.0)
    assert t12 == pytest.approx(4.0)
    assert trend_ratios([], {"quarterly": 10, "yearly": -20}) == (pytest.approx(1.1), pytest.approx(0.8))


def test_parse_items_flat_and_nested():
    flat = {"keyword": "x", "keyword_info": {"search_volume": 10, "cpc": 1.5},
            "keyword_properties": {"keyword_difficulty": 30, "core_keyword": "x"},
            "search_intent_info": {"main_intent": "commercial"}}
    assert parse_keyword_item(flat)["kd"] == 30
    nested = {"keyword_data": flat, "ranked_serp_element": {"serp_item": {"rank_group": 7, "url": "u", "etv": 3}}}
    r = parse_ranked_item(nested)
    assert (r["position"], r["url"], r["intent"]) == (7, "u", "commercial")


def test_extract_json():
    assert extract_json('```json\n{"a":1}\n```') == {"a": 1}
    assert extract_json('Sure! {"a": [1,2]} hope that helps') == {"a": [1, 2]}


def test_rank_score_prefers_page_two():
    assert rank_score(14) > rank_score(6) > rank_score(40) > rank_score(None)
    assert rank_score(2) < rank_score(6)


def _c(**kw):
    base = {"business_score": 80, "intent": "commercial", "best_position": None, "kd": 30, "own_urls": [],
            "mapping": {}, "cannibalization": False}
    base.update(kw)
    return base


def test_decide_action():
    cfg = RunConfig(domain="x.com")
    assert decide_action(_c(business_score=10), cfg)[0] == "IGNORE"
    assert decide_action(_c(best_position=14, mapping={"outcome": "EXISTING_WEAK_PAGE"}), cfg)[0] == "QUICK_WIN"
    assert decide_action(_c(mapping={"outcome": "CONTENT_GAP"}), cfg)[0] == "NEW_LANDING_PAGE"
    assert decide_action(_c(intent="informational", mapping={"outcome": "CONTENT_GAP"}), cfg)[0] == "SUPPORTING_CONTENT"
    assert decide_action(_c(kd=80, mapping={"outcome": "CONTENT_GAP"}), cfg)[0] == "STRATEGIC"
    assert decide_action(_c(best_position=2), cfg)[0] == "MAINTAIN"
    assert decide_action(_c(mapping={"outcome": "IRRELEVANT"}), cfg)[0] == "IGNORE"


# ---------------------------------------------------------------- end-to-end
class HashEmbedder:
    """Deterministic bag-of-words embedder so the embedding path runs without a model download."""
    def encode(self, texts):
        out = np.zeros((len(texts), 64), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in t.lower().split():
                out[i, int(hashlib.md5(tok.encode()).hexdigest(), 16) % 64] += 1
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(n == 0, 1, n)


def make_ctx(tmp_path, embedder=None):
    cfg = RunConfig(domain="acmelineage.io", output_dir=str(tmp_path))
    db = DB(tmp_path / "t.db")
    rid = db.create_run(cfg.domain, cfg.model_dump_json())
    ctx = Ctx(cfg, db, rid, FakeDataForSEO(), FakeLLM(), embedder)
    ctx.cache["fetch"] = fake_fetch(cfg.domain)
    return ctx


@pytest.mark.parametrize("embedder", [None, HashEmbedder()])
def test_full_pipeline(tmp_path, embedder):
    ctx = make_ctx(tmp_path, embedder)
    res = run_pipeline(ctx)
    assert res["site"]["pages"] >= 5
    assert res["filter"]["dropped"].get("negative", 0) > 0          # jobs / salary / certification removed
    clusters = ctx.db.clusters(ctx.run_id)
    assert clusters and all("action" in c and "seo_score" in c and "business_score" in c for c in clusters)
    actions = {c["action"] for c in clusters}
    assert {"QUICK_WIN", "NEW_LANDING_PAGE"} <= actions
    # informational and commercial keywords never share a cluster
    for c in clusters:
        intents = {r["intent"] for r in ctx.db.query(
            "SELECT intent FROM keywords WHERE run_id=? AND cluster_id=?", (ctx.run_id, c["cluster_id"]))}
        assert not ({"informational"} & intents and {"commercial", "transactional"} & intents)
    out = ctx.cfg.run_dir / f"run_{ctx.run_id}"
    html = (out / "report.html").read_text()
    assert "Top opportunities" in html and "<svg" in html
    rows = list(csv.DictReader(open(out / "clusters.csv")))
    assert len(rows) == len(clusters)


def test_resume_skips_done_phases(tmp_path):
    ctx = make_ctx(tmp_path)
    run_pipeline(ctx)
    ctx.cache.clear()
    res = run_pipeline(ctx)
    assert set(res) == {"report"}          # everything else checkpointed
    res = run_pipeline(ctx, force=["score"])
    assert "score" in res


def test_interrupted_phase_is_marked_and_resumes(tmp_path, monkeypatch):
    import kwresearch.runner as runner

    ctx = make_ctx(tmp_path)
    calls = []

    def first(_ctx):
        calls.append("first")
        return {"ok": True}

    def fail(_ctx):
        calls.append("fail")
        raise RuntimeError("OpenRouter credits exhausted")

    monkeypatch.setattr(runner, "PHASES", [("first", first), ("llm_work", fail)])
    with pytest.raises(RuntimeError, match="credits exhausted"):
        runner.run_pipeline(ctx)

    run = ctx.db.latest_run(ctx.cfg.domain)
    assert run["status"] == "interrupted"
    statuses = {r["name"]: r["status"] for r in ctx.db.query(
        "SELECT name, status FROM phases WHERE run_id=?", (ctx.run_id,))}
    assert statuses == {"first": "done", "llm_work": "interrupted"}

    def recovered(_ctx):
        calls.append("recovered")
        return {"ok": True}

    ctx.db.resume_run(ctx.run_id)
    monkeypatch.setattr(runner, "PHASES", [("first", first), ("llm_work", recovered)])
    result = runner.run_pipeline(ctx)

    assert calls == ["first", "fail", "recovered"]  # completed phase was not repeated
    assert set(result) == {"llm_work"}
    assert ctx.db.latest_run(ctx.cfg.domain)["status"] == "done"
