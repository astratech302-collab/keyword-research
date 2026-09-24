"""Phases 12-15 - cluster metrics, the two scores (SEO opportunity / business value), effort,
and the final action per cluster. Every sub-score is stored so the report can explain itself."""
from __future__ import annotations

import math
from collections import defaultdict

from ..context import Ctx
from . import candidates

COMMERCIAL = {"commercial", "transactional"}


def _wavg(pairs: list[tuple[float | None, float]], default: float | None = None) -> float | None:
    num = den = 0.0
    for v, w in pairs:
        if v is None:
            continue
        num += v * w
        den += w
    return num / den if den else default


def rank_score(pos: float | None) -> float:
    if not pos:
        return 0
    if pos <= 3:
        return 10       # already winning; little upside
    if pos <= 10:
        return 70
    if pos <= 20:
        return 100      # page 2 -> page 1 is the cheapest meaningful gain
    if pos <= 30:
        return 60
    if pos <= 50:
        return 40
    return 15


def cluster_metrics(ctx: Ctx) -> list[dict]:
    rid = ctx.run_id
    clusters = ctx.db.clusters(rid)
    kw = {r["keyword"]: r for r in candidates.pool(ctx).values()}
    selected = [r["domain"] for r in ctx.db.query(
        "SELECT domain FROM competitors WHERE run_id=? AND selected=1", (rid,))]
    ranks: dict[str, list[dict]] = defaultdict(list)
    for r in ctx.db.query("SELECT * FROM rankings WHERE run_id=?", (rid,)):
        ranks[r["keyword"]].append(r)
    gsc = {r["key"]: r for r in ctx.db.query("SELECT * FROM gsc_queries WHERE run_id=?", (rid,))}

    for c in clusters:
        members = [kw[k] for k in c["keywords"] if k in kw]
        vols = [(m.get("search_volume") or 0) for m in members]
        w = [max(v, 1) for v in vols]
        prim = kw.get(c["primary_keyword"], {})
        c["size"] = len(members)
        c["volume"] = int(sum(vols))
        c["primary_volume"] = int(prim.get("search_volume") or 0)
        kd = _wavg([(m.get("kd"), wi) for m, wi in zip(members, w)])
        c["kd"] = round(kd, 1) if kd is not None else None
        c["primary_kd"] = prim.get("kd")
        c["cpc"] = round(_wavg([(m.get("cpc"), wi) for m, wi in zip(members, w)], 0.0), 2)
        c["relevance"] = _wavg([(m.get("relevance"), wi) for m, wi in zip(members, w)], 2.5)
        c["trend"] = _wavg([(m.get("trend_12m") or m.get("trend_3m"), wi) for m, wi in zip(members, w)])
        observed = [gsc[m["keyword"]] for m in members if m["keyword"] in gsc]
        c["gsc_clicks"] = sum(r["clicks"] for r in observed)
        c["gsc_impressions"] = sum(r["impressions"] for r in observed)
        c["gsc_position"] = _wavg([(r["position"], r["impressions"]) for r in observed])

        own_urls: dict[str, dict] = {}
        comp_best: dict[str, int] = {}
        for m in members:
            for r in ranks.get(m["keyword"], []):
                if r["is_own"]:
                    u = own_urls.setdefault(r["url"], {"url": r["url"], "keywords": 0, "best": 999})
                    u["keywords"] += 1
                    u["best"] = min(u["best"], r["position"])
                elif r["domain"] in selected and r["position"] <= 20:
                    comp_best[r["domain"]] = min(comp_best.get(r["domain"], 99), r["position"])
        urls = sorted(own_urls.values(), key=lambda u: (u["best"], -u["keywords"]))
        c["own_urls"] = urls
        c["best_position"] = urls[0]["best"] if urls else None
        c["best_url"] = urls[0]["url"] if urls else None
        pos_best = kw.get(c["primary_keyword"], {}).get("own_position")
        c["primary_position"] = pos_best
        strong = [u for u in urls if u["best"] <= 30]
        total_ranked = sum(u["keywords"] for u in urls) or 1
        c["cannibalization"] = len(strong) >= 2 and strong[1]["keywords"] / total_ranked >= 0.25
        c["competitors_ranking"] = comp_best
        c["competitor_coverage"] = len(comp_best) / len(selected) if selected else 0.0
    return clusters


def seo_business_scores(ctx: Ctx, clusters: list[dict]) -> None:
    cfg = ctx.cfg
    W = cfg.weights
    maxv = max((c["volume"] for c in clusters), default=1) or 1
    cpcs = sorted(c["cpc"] or 0 for c in clusters)
    max_gsc = max((c.get("gsc_impressions") or 0 for c in clusters), default=0)

    def pct(v: float) -> float:
        if not cpcs:
            return 0
        import bisect
        return 100 * bisect.bisect_left(cpcs, v) / max(len(cpcs) - 1, 1)

    for c in clusters:
        s: dict[str, float] = {}
        s["demand"] = 100 * math.log1p(c["volume"]) / math.log1p(maxv)
        s["feasibility"] = 100 - (c["kd"] if c["kd"] is not None else 50)
        s["existing_rank"] = rank_score(c["best_position"] or c.get("gsc_position"))
        ranks_well = c["best_position"] is not None and c["best_position"] <= 20
        s["competitor_gap"] = 100 * c["competitor_coverage"] * (0.5 if ranks_well else 1.0)
        t = c["trend"]
        s["trend"] = 50 if t is None else max(0, min(100, 50 + 50 * (t - 1)))
        s["relevance"] = 100 * (c["relevance"] or 0) / 5
        s["intent"] = float(cfg.intent_value.get(c["intent"], 50))
        s["cpc"] = pct(c["cpc"] or 0)
        seo_total = (W.demand * s["demand"] + W.feasibility * s["feasibility"] + W.existing_rank * s["existing_rank"]
                     + W.competitor_gap * s["competitor_gap"] + W.trend * s["trend"])
        seo_weight = W.demand + W.feasibility + W.existing_rank + W.competitor_gap + W.trend
        if c.get("gsc_impressions") and max_gsc:
            # Search Console impressions are observed exposure, not monthly search volume.
            visibility = math.log1p(c["gsc_impressions"]) / math.log1p(max_gsc)
            s["gsc_opportunity"] = visibility * rank_score(c.get("gsc_position"))
            seo_total += W.gsc_opportunity * s["gsc_opportunity"]
            seo_weight += W.gsc_opportunity
        seo = seo_total / seo_weight
        biz = (W.relevance * s["relevance"] + W.intent * s["intent"] + W.cpc * s["cpc"]) / \
              (W.relevance + W.intent + W.cpc)
        c["subscores"] = {k: round(v, 1) for k, v in s.items()}
        c["seo_score"] = round(seo, 1)
        c["business_score"] = round(biz, 1)
        # Geometric mean: a cluster must be good on BOTH axes to rank high.
        c["priority"] = round(math.sqrt(max(seo, 0) * max(biz, 0)), 1)


ACTION_EFFORT = {"QUICK_WIN": 20, "OPTIMIZE_EXISTING": 35, "CONSOLIDATE": 40, "SUPPORTING_CONTENT": 45,
                 "NEW_LANDING_PAGE": 65, "STRATEGIC": 80, "MAINTAIN": 5, "IGNORE": 0}

ACTION_LABEL = {
    "QUICK_WIN": "Quick win - improve existing page",
    "OPTIMIZE_EXISTING": "Optimise existing page",
    "CONSOLIDATE": "Consolidate cannibalising pages",
    "NEW_LANDING_PAGE": "Create landing page",
    "SUPPORTING_CONTENT": "Create supporting article (topical authority)",
    "STRATEGIC": "Long-term strategic target",
    "MAINTAIN": "Already top 3 - maintain",
    "IGNORE": "Ignore",
}


def decide_action(c: dict, cfg) -> tuple[str, str]:
    th = cfg.thresholds
    m = c.get("mapping") or {}
    outcome = m.get("outcome")
    pos = c["best_position"]
    kd = c["kd"] if c["kd"] is not None else 50
    if outcome == "IRRELEVANT":
        return "IGNORE", m.get("rationale") or "judged irrelevant during page mapping"
    if c["business_score"] < th.ignore_below_business_value:
        return "IGNORE", f"low business value ({c['business_score']:.0f})"
    if c["intent"] == "navigational":
        return "IGNORE", "navigational (someone else's brand/site)"
    if outcome == "CANNIBALIZATION" or c.get("cannibalization"):
        return "CONSOLIDATE", f"{len([u for u in c['own_urls'] if u['best'] <= 30])} of your URLs compete"
    if pos and pos <= 3:
        return "MAINTAIN", f"already position {pos}"
    lo, hi = th.quick_win_positions
    has_page = outcome in ("EXISTING_GOOD_PAGE", "EXISTING_WEAK_PAGE") or (pos and outcome is None)
    if pos and lo <= pos <= hi and has_page and kd <= th.quick_win_max_kd:
        return "QUICK_WIN", f"ranking #{pos}, KD {kd:.0f}"
    if has_page and outcome != "WRONG_PAGE_TYPE":
        return "OPTIMIZE_EXISTING", f"relevant page exists{f' (rank #{pos})' if pos else ' but does not rank'}"
    if kd >= th.strategic_min_kd and c["business_score"] >= 70:
        return "STRATEGIC", f"high value but KD {kd:.0f}"
    if c["intent"] in COMMERCIAL:
        why = "existing page is the wrong type" if outcome == "WRONG_PAGE_TYPE" else "no page targets this"
        return "NEW_LANDING_PAGE", why
    return "SUPPORTING_CONTENT", "informational demand around your product"


def finalize(ctx: Ctx, clusters: list[dict]) -> None:
    for c in clusters:
        action, why = decide_action(c, ctx.cfg)
        kd = c["kd"] if c["kd"] is not None else 50
        effort = 0.5 * ACTION_EFFORT[action] + 0.5 * kd if action not in ("IGNORE", "MAINTAIN") else 0
        c["action"], c["action_label"], c["action_reason"] = action, ACTION_LABEL[action], why
        c["effort"] = round(effort, 1)
        hi_opp, lo_eff = c["priority"] >= 55, effort < 50
        c["quadrant"] = ("DO NOW" if hi_opp and lo_eff else "STRATEGIC BET" if hi_opp
                         else "OPTIONAL" if lo_eff else "DEPRIORITISE")
        if action in ("IGNORE", "MAINTAIN"):
            c["quadrant"] = action.title()
        # Final ordering score: priority, lightly discounted by effort.
        c["rank_score"] = round(c["priority"] * (1 - 0.3 * effort / 100), 1) if action not in ("IGNORE", "MAINTAIN") else 0
    order = sorted(clusters, key=lambda c: -c["rank_score"])
    for i, c in enumerate(order, 1):
        c["rank"] = i if c["rank_score"] > 0 else None


def prescore(ctx: Ctx) -> dict:
    clusters = cluster_metrics(ctx)
    seo_business_scores(ctx, clusters)
    ctx.db.save_clusters(ctx.run_id, clusters)
    return {"clusters": len(clusters)}


def run(ctx: Ctx) -> dict:
    clusters = ctx.db.clusters(ctx.run_id)
    finalize(ctx, clusters)
    ctx.db.save_clusters(ctx.run_id, clusters)
    counts: dict[str, int] = defaultdict(int)
    for c in clusters:
        counts[c["action"]] += 1
    return dict(counts)
