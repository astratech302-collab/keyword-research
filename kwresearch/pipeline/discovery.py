"""Phases 4-5 - build seeds from three sources, expand into the candidate pool."""
from __future__ import annotations

import logging
import json

from ..clients.dataforseo import parse_keyword_item
from ..context import Ctx
from ..parallel import parallel_map
from ..gsc import page_key
from . import candidates

log = logging.getLogger(__name__)


def build_seeds(ctx: Ctx) -> list[str]:
    sm = ctx.site_model()
    brand = [b.lower() for b in sm.get("brand_terms", [])]
    seeds: list[str] = [candidates.normalize_keyword(s) for s in sm.get("seed_topics", [])]
    # Source C: what we already rank for (positions 1-30, best traffic first), minus brand terms.
    own = ctx.db.query(
        "SELECT keyword FROM rankings WHERE run_id=? AND is_own=1 AND position<=30 ORDER BY etv DESC LIMIT 60",
        (ctx.run_id,))
    seeds += [r["keyword"] for r in own]
    observed = ctx.db.query(
        "SELECT key FROM gsc_queries WHERE run_id=? AND position<=30 "
        "ORDER BY clicks DESC, impressions DESC LIMIT 60", (ctx.run_id,))
    seeds += [r["key"] for r in observed]
    gsc_pages = {r["key"]: r for r in ctx.db.query(
        "SELECT key, impressions FROM gsc_pages WHERE run_id=?", (ctx.run_id,))}
    pages = ctx.db.query("SELECT url, topics_json FROM pages WHERE run_id=?", (ctx.run_id,))
    pages.sort(key=lambda p: -(gsc_pages.get(page_key(p["url"])) or {}).get("impressions", 0))
    for page in pages[:20]:
        if page_key(page["url"]) in gsc_pages:
            seeds.extend(json.loads(page["topics_json"] or "[]"))
    # Competitor keywords that several competitors share are strong category seeds.
    shared = ctx.db.query(
        "SELECT keyword, COUNT(*) n FROM rankings WHERE run_id=? AND is_own=0 AND position<=10 "
        "GROUP BY keyword HAVING n>=2 ORDER BY n DESC LIMIT 60", (ctx.run_id,))
    seeds += [r["keyword"] for r in shared]
    out = []
    for s in dict.fromkeys(seeds):
        if s and len(s.split()) <= 5 and not any(b and b in s for b in brand):
            out.append(s)
    return out[:200]


def run(ctx: Ctx) -> dict:
    lim = ctx.cfg.limits
    seeds = build_seeds(ctx)
    stats = {"seeds": len(seeds)}

    log.info("keyword discovery: fetching keywords for site")
    items = ctx.dfs.keywords_for_site(ctx.cfg.domain, limit=lim.keywords_for_site)
    stats["keywords_for_site"] = candidates.add(ctx, [r for r in map(parse_keyword_item, items) if r],
                                                "keywords_for_site")
    if seeds:
        log.info("keyword discovery: expanding %d seeds into ideas", len(seeds))
        items = ctx.dfs.keyword_ideas(seeds, limit=lim.keyword_ideas)
        stats["keyword_ideas"] = candidates.add(ctx, [r for r in map(parse_keyword_item, items) if r],
                                                "keyword_ideas")
        n = 0
        suggestion_seeds = seeds[:lim.suggestion_seeds]
        results = parallel_map(
            lambda s: ctx.dfs.keyword_suggestions(s, limit=lim.suggestions_per_seed),
            suggestion_seeds,
            lim.dataforseo_workers,
            lambda done, total: log.info("keyword suggestions: %d/%d seeds complete", done, total),
        )
        for items in results:
            n += candidates.add(ctx, [r for r in map(parse_keyword_item, items) if r], "keyword_suggestions")
        stats["keyword_suggestions"] = n
    candidates.flush(ctx)
    stats["pool_size"] = len(candidates.pool(ctx))
    ctx.cache["seeds"] = seeds
    return stats
