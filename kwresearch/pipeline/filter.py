"""Phase 6 - cheap rule filters first, then LLM business relevance on what survives."""
from __future__ import annotations

import json
import logging
import re

from ..clients.llm import batched
from ..context import Ctx
from ..parallel import parallel_map
from . import candidates

log = logging.getLogger(__name__)

NON_LATIN = re.compile(r"[^\x00-\x7FÀ-ɏ]")

RELEVANCE_SYSTEM = """You score search keywords for business relevance to ONE specific company.
Scale (r):
5 = searcher is looking for exactly what the company sells (product/category/solution/alternative)
4 = strongly related problem or use-case; a buyer could realistically convert
3 = relevant audience and topic; good for educational content that supports the product
2 = tangential; same industry but weak link to what is sold
1 = mostly unrelated, or wrong audience (students, job seekers, consumers when B2B, ...)
0 = unrelated or a different meaning of the word
Be strict: high search volume is NOT relevance. Judge the searcher, not the words."""


def rule_filter(ctx: Ctx) -> dict:
    cfg = ctx.cfg
    pool = candidates.pool(ctx)
    negs = [re.compile(r"(?<![a-z0-9])" + re.escape(t) + r"(?![a-z0-9])") for t in cfg.all_negative_terms]
    brand = [b.lower() for b in ctx.site_model().get("brand_terms", []) if len(b) >= 3]
    own_pos = {r["keyword"]: (r["position"], r["url"]) for r in ctx.db.query(
        "SELECT keyword, position, url FROM rankings WHERE run_id=? AND is_own=1", (ctx.run_id,))}
    latin_lang = cfg.language_code in ("en", "es", "fr", "de", "it", "pt", "nl", "sv", "da", "no", "pl")
    reasons: dict[str, int] = {}

    for kw, r in pool.items():
        r["kept"], r["drop_reason"] = 1, None
        if kw in own_pos:
            r["own_position"], r["own_url"] = own_pos[kw]
        why = None
        if len(kw) > 80 or len(kw.split()) > 10:
            why = "too_long"
        elif latin_lang and NON_LATIN.search(kw):
            why = "non_target_script"
        elif any(b in kw for b in brand):
            why = "own_brand"
        else:
            for rx, t in zip(negs, cfg.all_negative_terms):
                if rx.search(kw):
                    why = f"negative:{t}"
                    break
        if not why and (r.get("search_volume") or 0) < cfg.limits.min_search_volume and not r.get("own_position"):
            why = "low_volume"
        if why:
            r["kept"], r["drop_reason"] = 0, why
            reasons[why.split(":")[0]] = reasons.get(why.split(":")[0], 0) + 1

    kept = [r for r in pool.values() if r["kept"]]
    if len(kept) > cfg.limits.max_candidates:
        # Protect striking-distance rankings, then take the highest-demand keywords.
        kept.sort(key=lambda r: (0 if (r.get("own_position") or 999) <= 30 else 1, -(r.get("search_volume") or 0)))
        for r in kept[cfg.limits.max_candidates:]:
            r["kept"], r["drop_reason"] = 0, "over_cap"
        reasons["over_cap"] = len(kept) - cfg.limits.max_candidates
    return {"pool": len(pool), "after_rules": sum(r["kept"] for r in pool.values()), "dropped": reasons}


def llm_relevance(ctx: Ctx, batch_size: int = 150) -> dict:
    cfg = ctx.cfg
    sm = ctx.site_model()
    business = {k: sm.get(k) for k in ("company", "one_liner", "industry", "products", "customers",
                                         "problems", "not_relevant")}
    business["goal"] = cfg.goal
    business["country"] = cfg.country
    kept = sorted([r for r in candidates.pool(ctx).values() if r["kept"]], key=lambda r: r["keyword"])
    batches = list(batched(kept, batch_size))
    def score_batch(batch: list[dict]) -> dict:
        listing = "\n".join(f"{i}\t{r['keyword']}" for i, r in enumerate(batch))
        user = (f"Company: {json.dumps(business)}\n\nKeywords (index<TAB>keyword):\n{listing}\n\n"
                'Return {"scores":[{"i":0,"r":3}, ...]} with one entry per index.')
        return ctx.llm.json(cfg.llm.fast, RELEVANCE_SYSTEM, user)

    results = parallel_map(score_batch, batches, cfg.limits.llm_workers,
                           lambda done, total: log.info("keyword relevance: %d/%d batches complete", done, total))
    for batch, data in zip(batches, results):
        scores = {int(d["i"]): d.get("r") for d in data.get("scores", []) if "i" in d}
        for i, r in enumerate(batch):
            s = scores.get(i)
            r["relevance"] = float(s) if s is not None else None
    dropped = 0
    for r in kept:
        rel = r.get("relevance")
        if rel is not None and rel < cfg.thresholds.min_relevance_to_keep and not \
                ((r.get("own_position") or 999) <= 10 and rel >= 1):
            r["kept"], r["drop_reason"] = 0, "low_relevance"
            dropped += 1
    return {"scored": len(kept), "low_relevance_dropped": dropped,
            "kept": sum(1 for r in candidates.pool(ctx).values() if r["kept"])}


def run(ctx: Ctx) -> dict:
    stats = rule_filter(ctx)
    stats.update(llm_relevance(ctx))
    candidates.flush(ctx)
    return stats
