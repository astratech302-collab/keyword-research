"""Phase 11 - map every cluster to an existing page (or declare a gap).

Top-N clusters (by pre-score) get an LLM judgement with candidate pages + SERP evidence.
The long tail gets a deterministic heuristic so cost stays bounded."""
from __future__ import annotations

import json
import logging
import re

import numpy as np

from ..clients.llm import batched
from ..context import Ctx
from ..parallel import parallel_map
from ..gsc import page_key
from .cluster import tokens

log = logging.getLogger(__name__)

OUTCOMES = ["EXISTING_GOOD_PAGE", "EXISTING_WEAK_PAGE", "WRONG_PAGE_TYPE", "CONTENT_GAP",
            "CANNIBALIZATION", "IRRELEVANT"]
COMMERCIAL_PAGES = {"home", "product", "feature", "solution", "pricing", "comparison", "integration", "category"}
INFO_PAGES = {"blog", "guide", "glossary", "docs", "case_study"}

MAP_SYSTEM = f"""You are a senior SEO strategist mapping keyword clusters to a website's pages.
For each cluster decide ONE outcome:
- EXISTING_GOOD_PAGE: a page already matches the topic AND the search intent/page type; improve it.
- EXISTING_WEAK_PAGE: a page covers the topic with the right page type but thinly; expand it.
- WRONG_PAGE_TYPE: the closest page has the wrong format for the intent (e.g. a blog post for a
  commercial "X software" query where Google ranks product/listicle pages) -> a new page is needed.
- CONTENT_GAP: no page covers this; a new page is needed.
- CANNIBALIZATION: two or more of the site's pages split this topic; consolidate.
- IRRELEVANT: the cluster would not bring this business's buyers.
Use the SERP evidence (what page types Google ranks) when present. Recommend the page type that
matches what ranks. Suggest a short URL slug for new pages. Be concrete and brief."""


def _page_text(p: dict) -> str:
    topics = " ".join(json.loads(p.get("topics_json") or "[]"))
    return f"{p['title']} {p['h1']} {p['meta']} {topics}"


def _cluster_text(c: dict) -> str:
    return " | ".join([c["primary_keyword"]] + c["keywords"][:6])


def candidate_pages(ctx: Ctx, clusters: list[dict], pages: list[dict], k: int = 3) -> dict[int, list[tuple[str, float]]]:
    out: dict[int, list[tuple[str, float]]] = {}
    if not pages:
        return {c["cluster_id"]: [] for c in clusters}
    if ctx.embedder:
        P = ctx.embedder.encode([_page_text(p) for p in pages])
        C = ctx.embedder.encode([_cluster_text(c) for c in clusters])
        S = C @ P.T
        for i, c in enumerate(clusters):
            top = np.argsort(-S[i])[:k]
            out[c["cluster_id"]] = [(pages[j]["url"], float(S[i, j])) for j in top]
    else:
        ptoks = [set(tokens(_page_text(p))) for p in pages]
        for c in clusters:
            ct = set(tokens(_cluster_text(c)))
            sims = [len(ct & pt) / (len(ct) or 1) for pt in ptoks]
            top = np.argsort(-np.array(sims))[:k]
            out[c["cluster_id"]] = [(pages[j]["url"], float(sims[j])) for j in top]
    return out


def heuristic(c: dict, cands: list[tuple[str, float]], page_by_url: dict[str, dict]) -> dict:
    if c.get("cannibalization"):
        return {"outcome": "CANNIBALIZATION", "target_url": c["best_url"], "source": "heuristic",
                "rationale": "multiple URLs rank for this cluster"}
    if c.get("best_position") and c["best_position"] <= 30:
        return {"outcome": "EXISTING_WEAK_PAGE" if c["best_position"] > 3 else "EXISTING_GOOD_PAGE",
                "target_url": c["best_url"], "source": "heuristic",
                "rationale": f"already ranks #{c['best_position']}"}
    if cands:
        url, sim = cands[0]
        ptype = (page_by_url.get(url) or {}).get("page_type", "other")
        good_type = ptype in (COMMERCIAL_PAGES if c["intent"] in ("commercial", "transactional") else INFO_PAGES | COMMERCIAL_PAGES)
        if sim >= 0.55:
            return {"outcome": "EXISTING_WEAK_PAGE" if good_type else "WRONG_PAGE_TYPE",
                    "target_url": url if good_type else None, "source": "heuristic",
                    "rationale": f"closest page ({ptype}) similarity {sim:.2f}"}
    return {"outcome": "CONTENT_GAP", "target_url": None, "source": "heuristic",
            "recommended_page_type": "landing page" if c["intent"] in ("commercial", "transactional") else "article",
            "suggested_slug": "/" + re.sub(r"[^a-z0-9]+", "-", c["primary_keyword"]).strip("-") + "/",
            "rationale": "no page on the site covers this topic"}


def run(ctx: Ctx, batch_size: int = 8) -> dict:
    rid = ctx.run_id
    clusters = ctx.db.clusters(rid)
    pages = ctx.db.query("SELECT * FROM pages WHERE run_id=?", (rid,))
    page_by_url = {p["url"]: p for p in pages}
    gsc_pages = {r["key"]: r for r in ctx.db.query("SELECT * FROM gsc_pages WHERE run_id=?", (rid,))}
    cands = candidate_pages(ctx, clusters, pages)
    sm = ctx.site_model()
    business = {k: sm.get(k) for k in ("company", "one_liner", "products", "customers")}

    ranked = sorted([c for c in clusters if c["intent"] != "navigational"], key=lambda c: -c.get("priority", 0))
    llm_ids = {c["cluster_id"] for c in ranked[:ctx.cfg.limits.llm_mapping_clusters]}

    for c in clusters:
        c["mapping"] = heuristic(c, cands.get(c["cluster_id"], []), page_by_url)
        c["candidate_pages"] = [{"url": u, "similarity": round(s, 3)} for u, s in cands.get(c["cluster_id"], [])]

    todo = [c for c in clusters if c["cluster_id"] in llm_ids]
    batches = list(batched(todo, batch_size))
    def map_batch(batch: list[dict]) -> dict:
        payload = []
        for c in batch:
            page_ids = list(dict.fromkeys([u["url"] for u in c["own_urls"][:3]] +
                                          [p["url"] for p in c["candidate_pages"]]))
            payload.append({
                "cluster_id": c["cluster_id"], "primary_keyword": c["primary_keyword"], "intent": c["intent"],
                "keywords": c["keywords"][:10], "monthly_volume": c["volume"],
                "site_rankings": [{"url": u["url"], "best_position": u["best"], "keywords": u["keywords"]}
                                  for u in c["own_urls"][:3]],
                "serp": {k: c["serp"][k] for k in ("dominant_page_type", "page_types", "top_domains")}
                        if c.get("serp") else None,
                "candidate_pages": [{"url": u, "type": page_by_url.get(u, {}).get("page_type"),
                                     "title": page_by_url.get(u, {}).get("title"),
                                     "h1": page_by_url.get(u, {}).get("h1"),
                                     "search_console": {k: (gsc_pages.get(page_key(u)) or {}).get(k)
                                                        for k in ("clicks", "impressions", "ctr", "position")}
                                     if page_key(u) in gsc_pages else None} for u in page_ids],
            })
        user = (f"Business: {json.dumps(business)}\nClusters: {json.dumps(payload)}\n\n"
                'Return {"mappings":[{"cluster_id":1,"outcome":"' + "|".join(OUTCOMES) + '",'
                '"target_url":"existing url or null","recommended_page_type":"e.g. product landing page, '
                'comparison page, listicle, how-to guide, glossary entry","suggested_slug":"/path/ for new pages or null",'
                '"rationale":"one sentence","content_notes":["2-4 concrete things the page must include"]}]}')
        return ctx.llm.json(ctx.cfg.llm.smart, MAP_SYSTEM, user)

    results = parallel_map(map_batch, batches, ctx.cfg.limits.llm_workers,
                           lambda done, total: log.info("LLM page mapping: %d/%d batches complete", done, total))
    for batch, data in zip(batches, results):
        by_id = {c["cluster_id"]: c for c in batch}
        for m in data.get("mappings", []):
            try:
                c = by_id.get(int(m.get("cluster_id")))
            except (TypeError, ValueError):
                continue
            if not c or m.get("outcome") not in OUTCOMES:
                continue
            m["source"] = "llm"
            c["mapping"] = m
    ctx.db.save_clusters(rid, clusters)
    return {"llm_mapped": len(todo), "heuristic_mapped": len(clusters) - len(todo)}
