"""Phase 3 - discover SEO competitors, vet them against the business, pull their top-20 keywords."""
from __future__ import annotations

import json
import logging

from ..clients.dataforseo import parse_ranked_item
from ..context import Ctx
from ..parallel import parallel_map
from . import candidates
from .footprint import save_rankings
from .site import Fetch, default_fetch, parse_page

log = logging.getLogger(__name__)

VET_SYSTEM = """You decide which domains are true SEO competitors for a business: sites whose pages
compete for the same buyers searching for the same problems/products. Exclude publishers, marketplaces,
directories, generic giants and sites serving a different customer, unless they clearly target the
same buyers with the same kind of offering."""


def _excluded(domain: str, cfg) -> bool:
    d = domain.lower().removeprefix("www.")
    if d == cfg.domain or d.endswith("." + cfg.domain):
        return True
    return any(d == x or d.endswith("." + x) for x in cfg.exclude_domains)


def discover(ctx: Ctx) -> list[dict]:
    cfg = ctx.cfg
    found: dict[str, dict] = {}
    for it in ctx.dfs.competitors_domain(cfg.domain, limit=50):
        d = (it.get("domain") or "").lower()
        organic = ((it.get("full_domain_metrics") or {}).get("organic") or {})
        found[d] = {"domain": d, "source": "competitors_domain", "intersections": it.get("intersections") or 0,
                    "avg_position": it.get("avg_position"), "etv": organic.get("etv") or 0.0}

    if len([d for d in found if not _excluded(d, cfg)]) < 3:
        # New / small site: find who ranks for our seed topics instead.
        seeds = ctx.site_model().get("seed_topics", [])[:100]
        if seeds:
            for it in ctx.dfs.serp_competitors(seeds, limit=50):
                d = (it.get("domain") or "").lower()
                found.setdefault(d, {"domain": d, "source": "serp_competitors",
                                     "intersections": it.get("keywords_count") or 0,
                                     "avg_position": it.get("avg_position"), "etv": it.get("etv") or 0.0})

    for d in cfg.competitors:
        d = d.lower().removeprefix("https://").removeprefix("http://").removeprefix("www.").split("/")[0]
        found[d] = {**found.get(d, {"intersections": 0, "avg_position": None, "etv": 0.0}),
                    "domain": d, "source": "user"}
    return [c for c in found.values() if c["domain"] and not _excluded(c["domain"], cfg)]


def vet(ctx: Ctx, comps: list[dict], fetch: Fetch | None = None) -> list[dict]:
    fetch = fetch or default_fetch()
    auto = sorted([c for c in comps if c["source"] != "user"], key=lambda c: -c["intersections"])[:20]
    if auto:
        for c in auto:
            status, html = fetch(f"https://{c['domain']}/")
            p = parse_page(c["domain"], html) if status == 200 and html else {}
            c["_home"] = {"title": p.get("title", ""), "meta": p.get("meta", ""), "h1": p.get("h1", "")}
        sm = ctx.site_model()
        user = (f"Business: {json.dumps({k: sm.get(k) for k in ('company', 'one_liner', 'products', 'customers')})}\n"
                f"Candidates: {json.dumps([{'domain': c['domain'], 'shared_keywords': c['intersections'], **c['_home']} for c in auto])}\n"
                'Return {"competitors":[{"domain":"","relevant":true,"reason":"short"}]}')
        data = ctx.llm.json(ctx.cfg.llm.smart, VET_SYSTEM, user)
        verdict = {d.get("domain"): d for d in data.get("competitors", [])}
        for c in auto:
            v = verdict.get(c["domain"], {})
            c["relevant"] = bool(v.get("relevant"))
            c["reason"] = v.get("reason", "")
            c.pop("_home", None)
    for c in comps:
        if c["source"] == "user":
            c["relevant"], c["reason"] = True, "user-supplied"
        c.setdefault("relevant", False)
        c.setdefault("reason", "not in top-20 by overlap")

    n = ctx.cfg.limits.max_competitors
    users = [c for c in comps if c["source"] == "user"]
    auto_ok = sorted([c for c in comps if c["source"] != "user" and c["relevant"]],
                     key=lambda c: -c["intersections"])
    for c in (users + auto_ok)[:max(n, len(users))]:
        c["selected"] = True
    return comps


def run(ctx: Ctx, fetch: Fetch | None = None) -> dict:
    comps = vet(ctx, discover(ctx), fetch or ctx.cache.get("fetch"))
    db, rid = ctx.db, ctx.run_id
    db.execute("DELETE FROM competitors WHERE run_id=?", (rid,))
    db.executemany(
        "INSERT INTO competitors (run_id,domain,source,intersections,etv,avg_position,relevant,reason,selected)"
        " VALUES (?,?,?,?,?,?,?,?,?)",
        [(rid, c["domain"], c["source"], c["intersections"], c["etv"], c["avg_position"], int(c["relevant"]),
          c["reason"], int(c.get("selected", False))) for c in comps])
    db.commit()

    selected = [c["domain"] for c in comps if c.get("selected")]
    total = 0
    results = parallel_map(
        lambda d: ctx.dfs.ranked_keywords(
            d, limit=ctx.cfg.limits.competitor_keywords_per_domain, max_position=20),
        selected,
        ctx.cfg.limits.dataforseo_workers,
        lambda done, count: log.info("competitor keywords: %d/%d domains complete", done, count),
    )
    for d, items in zip(selected, results):
        rows = [r for r in (parse_ranked_item(i) for i in items) if r]
        save_rankings(ctx, d, rows, is_own=False)
        total += candidates.add(ctx, rows, f"competitor:{d}")
    candidates.flush(ctx)
    log.info("competitors selected: %s", selected)
    return {"discovered": len(comps), "selected": selected, "new_keywords": total}
