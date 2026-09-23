"""Candidate keyword pool shared by discovery phases. Normalises + merges on insert."""
from __future__ import annotations

import json
import re
import unicodedata

from ..context import Ctx

_WS = re.compile(r"\s+")
_STRIP = re.compile(r"[\"'“”‘’`´]|[?!,;:]+$")

METRIC_FIELDS = ["search_volume", "cpc", "competition", "kd", "core_keyword", "intent",
                 "monthly_json", "trend_3m", "trend_12m"]


def normalize_keyword(kw: str) -> str:
    kw = unicodedata.normalize("NFKC", kw or "").lower().strip()
    kw = _STRIP.sub("", kw)
    kw = _WS.sub(" ", kw).strip(" -_/|.")
    return kw


def _load(ctx: Ctx) -> dict[str, dict]:
    if "candidates" not in ctx.cache:
        pool = {}
        for r in ctx.db.keywords(ctx.run_id, kept_only=False):
            r["sources_json"] = json.loads(r["sources_json"] or "[]")
            r["monthly_json"] = json.loads(r["monthly_json"] or "[]")
            pool[r["keyword"]] = r
        ctx.cache["candidates"] = pool
    return ctx.cache["candidates"]


def pool(ctx: Ctx) -> dict[str, dict]:
    return _load(ctx)


def add(ctx: Ctx, items: list[dict], source: str) -> int:
    """Merge parsed keyword dicts into the pool. Returns number of new keywords."""
    p = _load(ctx)
    new = 0
    for it in items:
        kw = normalize_keyword(it.get("keyword", ""))
        if not kw:
            continue
        cur = p.get(kw)
        if cur is None:
            cur = {"keyword": kw, "sources_json": [], "kept": 1, "monthly_json": []}
            p[kw] = cur
            new += 1
        if source not in cur["sources_json"]:
            cur["sources_json"].append(source)
        for f in METRIC_FIELDS:
            v = it.get(f)
            if v not in (None, [], "") and cur.get(f) in (None, [], ""):
                cur[f] = v
    return new


def flush(ctx: Ctx) -> None:
    ctx.db.upsert_keywords(ctx.run_id, list(_load(ctx).values()))
