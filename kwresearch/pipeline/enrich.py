"""Phase 7 - fill metric gaps. Labs endpoints already return volume/CPC/KD/intent for most
keywords, so we only pay for the gaps (bulk KD, search intent) and never ask the LLM for metrics."""
from __future__ import annotations

import logging
import time

from ..context import Ctx
from ..parallel import parallel_map
from . import candidates

log = logging.getLogger(__name__)


def run(ctx: Ctx) -> dict:
    kept = [r for r in candidates.pool(ctx).values() if r["kept"]]
    need_kd = [r["keyword"] for r in kept if r.get("kd") is None]
    need_intent = [r["keyword"] for r in kept if not r.get("intent")]

    log.info("metric enrichment: %d keywords need KD; %d need intent", len(need_kd), len(need_intent))
    jobs = []
    if need_kd:
        jobs.append(("kd", lambda: ctx.dfs.bulk_keyword_difficulty(need_kd)))
    if need_intent:
        jobs.append(("intent", lambda: ctx.dfs.search_intent(need_intent)))
    completed = parallel_map(
        lambda job: job[1](), jobs, min(ctx.cfg.limits.dataforseo_workers, 2),
        lambda done, total: log.info("metric enrichment: %d/%d API groups complete", done, total),
    )
    by_name = {job[0]: result for job, result in zip(jobs, completed)}
    kd = by_name.get("kd", {})
    intents = by_name.get("intent", {})
    for r in kept:
        if r.get("kd") is None and r["keyword"] in kd:
            r["kd"] = kd[r["keyword"]]
        if not r.get("intent") and r["keyword"] in intents:
            r["intent"], r["intent_prob"] = intents[r["keyword"]]
        r["intent"] = r.get("intent") or "informational"

    candidates.flush(ctx)
    now = time.time()
    ctx.db.executemany(
        "INSERT INTO keyword_metrics_history VALUES (?,?,?,?,?,?,?,?)",
        [(r["keyword"], ctx.cfg.country, ctx.cfg.language, now, r.get("search_volume"), r.get("cpc"),
          r.get("kd"), r.get("intent")) for r in kept])
    ctx.db.commit()
    return {"kd_filled": len(kd), "kd_requested": len(need_kd),
            "intent_filled": len(intents), "intent_requested": len(need_intent)}
