"""Phase 2 - what Google already associates with the domain."""
from __future__ import annotations

from ..clients.dataforseo import parse_ranked_item
from ..context import Ctx
from . import candidates


def save_rankings(ctx: Ctx, domain: str, rows: list[dict], is_own: bool) -> None:
    ctx.db.executemany(
        "INSERT OR REPLACE INTO rankings (run_id,domain,keyword,position,url,etv,is_own) VALUES (?,?,?,?,?,?,?)",
        [(ctx.run_id, domain, candidates.normalize_keyword(r["keyword"]), r.get("position"), r.get("url"),
          r.get("etv") or 0.0, int(is_own)) for r in rows if r.get("position")])
    ctx.db.commit()


def run(ctx: Ctx) -> dict:
    items = ctx.dfs.ranked_keywords(ctx.cfg.domain, limit=ctx.cfg.limits.own_ranked_keywords)
    rows = [r for r in (parse_ranked_item(i) for i in items) if r]
    save_rankings(ctx, ctx.cfg.domain, rows, is_own=True)
    candidates.add(ctx, rows, "own_ranked")
    candidates.flush(ctx)
    striking = sum(1 for r in rows if r.get("position") and 4 <= r["position"] <= 30)
    return {"ranked_keywords": len(rows), "striking_distance": striking}
