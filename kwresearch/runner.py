"""Deterministic pipeline orchestration with per-phase checkpoints (resume after failure)."""
from __future__ import annotations

import json
import logging
import time
from typing import Callable

from .clients.dataforseo import BudgetExceeded
from .context import Ctx
from .pipeline import (cluster, competitors, discovery, enrich, filter as kwfilter, footprint,
                       mapping, scoring, serp, site)
from .report import render

log = logging.getLogger(__name__)

PHASES: list[tuple[str, Callable[[Ctx], dict]]] = [
    ("site", site.run),                  # 1  crawl + site model
    ("footprint", footprint.run),        # 2  existing rankings
    ("competitors", competitors.run),    # 3  discover + vet + competitor keywords
    ("discovery", discovery.run),        # 4-5 seeds + expansion
    ("filter", kwfilter.run),            # 6  rules + LLM relevance
    ("enrich", enrich.run),              # 7  KD / intent gaps
    ("cluster", cluster.run),            # 9-10 clusters + primary keyword
    ("prescore", scoring.prescore),      # 12-13 metrics + SEO/business scores
    ("serp", serp.run),                  # 8  SERP check for shortlist only
    ("mapping", mapping.run),            # 11 cluster -> page
    ("score", scoring.run),              # 14-15 effort + actions
    ("report", render.run),              # report + CSV
]


def run_pipeline(ctx: Ctx, only: list[str] | None = None, force: list[str] | None = None) -> dict:
    force = force or []
    results: dict[str, dict] = {}
    selected = [(name, fn) for name, fn in PHASES if not only or name in only]
    total = len(selected)
    log.info("Run #%s for %s: %d phase%s selected", ctx.run_id, ctx.cfg.domain, total,
             "" if total == 1 else "s")
    for phase_no, (name, fn) in enumerate(selected, 1):
        if ctx.db.phase_done(ctx.run_id, name) and name not in force and name != "report":
            log.info("[%d/%d %s] already done - skipping (use --force %s to redo)",
                     phase_no, total, name, name)
            continue
        t0 = time.time()
        ctx.db.start_phase(ctx.run_id, name)
        log.info("[%d/%d %s] running...", phase_no, total, name)
        try:
            meta = fn(ctx) or {}
            status = "done"
        except BudgetExceeded as e:
            log.warning("[%d/%d %s] %s - continuing with cached/partial data",
                        phase_no, total, name, e)
            meta, status = {"budget_exceeded": str(e)}, "partial"
            ctx.cache["budget_hit"] = True
        except (Exception, KeyboardInterrupt) as e:
            seconds = round(time.time() - t0, 1)
            ctx.db.mark_phase(ctx.run_id, name, "interrupted", {
                "error_type": type(e).__name__, "error": str(e), "seconds": seconds,
            })
            ctx.db.finish_run(ctx.run_id, "interrupted")
            log.error("[%d/%d %s] interrupted after %.1fs: %s", phase_no, total, name, seconds, e)
            log.error("Completed phases and API responses are saved; rerun with --resume after fixing the issue.")
            raise
        meta["seconds"] = round(time.time() - t0, 1)
        ctx.db.mark_phase(ctx.run_id, name, status, meta)
        results[name] = meta
        log.info("[%d/%d %s] %s %s", phase_no, total, name, status,
                 json.dumps(meta, default=str)[:400])
    ctx.db.finish_run(ctx.run_id, "partial" if ctx.cache.get("budget_hit") else "done")
    log.info("Run #%s finished. API spend: %s", ctx.run_id, ctx.db.run_cost(ctx.run_id))
    return results
