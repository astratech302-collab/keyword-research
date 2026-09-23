"""Report: HTML (action-oriented) + CSV exports."""
from __future__ import annotations

import csv
import datetime as dt
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..context import Ctx
from ..pipeline.scoring import ACTION_LABEL

TEMPLATES = Path(__file__).parent / "templates"
ACTION_COLOR = {"QUICK_WIN": "#1a9e5c", "OPTIMIZE_EXISTING": "#3b82c4", "NEW_LANDING_PAGE": "#d9822b",
                "SUPPORTING_CONTENT": "#8a63d2", "STRATEGIC": "#c2410c", "CONSOLIDATE": "#b8a200",
                "MAINTAIN": "#8a8f98", "IGNORE": "#c4c7cc"}
ACTIONABLE = ["QUICK_WIN", "OPTIMIZE_EXISTING", "NEW_LANDING_PAGE", "SUPPORTING_CONTENT", "STRATEGIC", "CONSOLIDATE"]


def recommendation_text(c: dict) -> str:
    m = c.get("mapping") or {}
    a = c["action"]
    if a in ("QUICK_WIN", "OPTIMIZE_EXISTING"):
        return f"Improve {_path(m.get('target_url') or c.get('best_url')) or 'existing page'}"
    if a == "CONSOLIDATE":
        return "Merge/redirect: " + ", ".join(_path(u["url"]) for u in c["own_urls"][:3])
    if a in ("NEW_LANDING_PAGE", "SUPPORTING_CONTENT", "STRATEGIC"):
        slug = m.get("suggested_slug") or ""
        ptype = m.get("recommended_page_type") or ("landing page" if a == "NEW_LANDING_PAGE" else "article")
        return f"Create {ptype} {slug}".strip()
    return c.get("action_reason", "")


def _path(url: str | None) -> str | None:
    if not url:
        return url
    return re.sub(r"^https?://[^/]+", "", url) or "/"


def scatter_svg(clusters: list[dict], w: int = 640, h: int = 420, pad: int = 44) -> str:
    pts = [c for c in clusters if c["action"] != "IGNORE" or c["volume"] > 0][:600]
    maxv = max((c["volume"] for c in pts), default=1) or 1
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Opportunity matrix" class="scatter">']
    x0, y0, x1, y1 = pad, h - pad, w - 12, 12
    X = lambda v: x0 + (x1 - x0) * v / 100
    Y = lambda v: y0 - (y0 - y1) * v / 100
    out.append(f'<rect x="{X(50)}" y="{Y(100)}" width="{X(100)-X(50)}" height="{Y(50)-Y(100)}" class="q-hot"/>')
    for v in (0, 25, 50, 75, 100):
        out.append(f'<line x1="{X(v)}" y1="{y0}" x2="{X(v)}" y2="{y1}" class="grid"/>'
                   f'<line x1="{x0}" y1="{Y(v)}" x2="{x1}" y2="{Y(v)}" class="grid"/>'
                   f'<text x="{X(v)}" y="{y0+16}" class="tick" text-anchor="middle">{v}</text>'
                   f'<text x="{x0-8}" y="{Y(v)+4}" class="tick" text-anchor="end">{v}</text>')
    out.append(f'<text x="{(x0+x1)/2}" y="{h-4}" class="axis" text-anchor="middle">SEO opportunity →</text>')
    out.append(f'<text x="12" y="{(y0+y1)/2}" class="axis" text-anchor="middle" '
               f'transform="rotate(-90 12 {(y0+y1)/2})">Business value →</text>')
    for c in sorted(pts, key=lambda c: -c["volume"]):
        r = 3 + 9 * math.sqrt(c["volume"] / maxv)
        col = ACTION_COLOR.get(c["action"], "#999")
        label = f'{c["primary_keyword"]} · vol {c["volume"]:,} · {ACTION_LABEL[c["action"]]}'
        out.append(f'<circle cx="{X(c["seo_score"]):.1f}" cy="{Y(c["business_score"]):.1f}" r="{r:.1f}" '
                   f'fill="{col}" fill-opacity="0.65" stroke="{col}"><title>{_esc(label)}</title></circle>')
    out.append("</svg>")
    return "".join(out)


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def build_context(ctx: Ctx) -> dict:
    rid, db, cfg = ctx.run_id, ctx.db, ctx.cfg
    clusters = db.clusters(rid)
    for c in clusters:
        c["recommendation"] = recommendation_text(c)
        c["color"] = ACTION_COLOR.get(c.get("action"), "#999")
    actionable = sorted([c for c in clusters if c.get("action") in ACTIONABLE], key=lambda c: -c["rank_score"])
    by_action: dict[str, list[dict]] = defaultdict(list)
    for c in actionable:
        by_action[c["action"]].append(c)

    # Existing-page view: group quick wins / optimisations / consolidations by target URL.
    pages: dict[str, dict] = {}
    for c in actionable:
        if c["action"] not in ("QUICK_WIN", "OPTIMIZE_EXISTING", "CONSOLIDATE"):
            continue
        url = (c.get("mapping") or {}).get("target_url") or c.get("best_url") or "(unmapped)"
        p = pages.setdefault(url, {"url": url, "clusters": [], "volume": 0, "score": 0.0})
        p["clusters"].append(c)
        p["volume"] += c["volume"]
        p["score"] = max(p["score"], c["rank_score"])
    existing_pages = sorted(pages.values(), key=lambda p: -p["score"])

    all_kw = db.keywords(rid, kept_only=False)
    kept = [k for k in all_kw if k["kept"]]
    drops = Counter((k["drop_reason"] or "").split(":")[0] for k in all_kw if not k["kept"])
    comps = db.query("SELECT * FROM competitors WHERE run_id=? ORDER BY selected DESC, intersections DESC", (rid,))
    ignored_notable = sorted([c for c in clusters if c.get("action") == "IGNORE"], key=lambda c: -c["volume"])[:15]
    run = db.query("SELECT * FROM runs WHERE id=?", (rid,))[0]
    sm = ctx.site_model()

    return {
        "cfg": cfg, "sm": sm, "run": run, "generated": dt.datetime.now().strftime("%d %b %Y %H:%M"),
        "stats": {
            "pool": len(all_kw), "kept": len(kept), "clusters": len(clusters), "actionable": len(actionable),
            "pages_to_improve": len(existing_pages),
            "new_landing": len(by_action["NEW_LANDING_PAGE"]),
            "new_articles": len(by_action["SUPPORTING_CONTENT"]),
            "consolidate": len(by_action["CONSOLIDATE"]), "strategic": len(by_action["STRATEGIC"]),
            "quick_wins": len(by_action["QUICK_WIN"]),
            "ranked": db.query("SELECT COUNT(*) n FROM rankings WHERE run_id=? AND is_own=1", (rid,))[0]["n"],
        },
        "top": actionable[:25], "by_action": by_action, "existing_pages": existing_pages[:30],
        "new_pages": (by_action["NEW_LANDING_PAGE"] + by_action["STRATEGIC"])[:25],
        "supporting": by_action["SUPPORTING_CONTENT"][:40], "consolidate": by_action["CONSOLIDATE"],
        "ignored": ignored_notable, "competitors": comps, "drops": dict(drops.most_common()),
        "phases": db.phase_meta(rid), "cost": db.run_cost(rid),
        "scatter": scatter_svg(clusters), "action_color": ACTION_COLOR, "action_label": ACTION_LABEL,
        "weights": cfg.weights.model_dump(), "budget_hit": ctx.cache.get("budget_hit", False),
    }


def write_csvs(ctx: Ctx, out: Path) -> None:
    rid = ctx.run_id
    clusters = ctx.db.clusters(rid)
    cmap = {c["cluster_id"]: c for c in clusters}
    with open(out / "keywords.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["keyword", "cluster_id", "cluster_primary", "search_volume", "cpc", "kd", "intent", "relevance_0_5",
                    "own_position", "own_url", "trend_12m", "sources", "kept", "drop_reason"])
        for k in ctx.db.keywords(rid, kept_only=False):
            c = cmap.get(k["cluster_id"] or -1, {})
            w.writerow([k["keyword"], k["cluster_id"], c.get("primary_keyword"), k["search_volume"], k["cpc"], k["kd"],
                        k["intent"], k["relevance"], k["own_position"], k["own_url"], k["trend_12m"],
                        "|".join(json.loads(k["sources_json"] or "[]")), k["kept"], k["drop_reason"]])
    cols = ["rank", "cluster_id", "primary_keyword", "action", "recommendation", "action_reason", "intent", "size",
            "volume", "primary_volume", "kd", "cpc", "best_position", "best_url", "seo_score", "business_score",
            "priority", "effort", "rank_score", "quadrant", "competitor_coverage"]
    with open(out / "clusters.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols + ["mapping_outcome", "target_url", "suggested_slug", "page_type", "rationale",
                           "serp_dominant_type", "keywords"])
        for c in sorted(clusters, key=lambda c: (c.get("rank") is None, c.get("rank") or 0)):
            m = c.get("mapping") or {}
            row = [c.get(k) if k != "recommendation" else recommendation_text(c) for k in cols]
            w.writerow(row + [m.get("outcome"), m.get("target_url"), m.get("suggested_slug"),
                              m.get("recommended_page_type"), m.get("rationale"),
                              (c.get("serp") or {}).get("dominant_page_type"), " | ".join(c["keywords"][:30])])


def run(ctx: Ctx) -> dict:
    out = ctx.cfg.run_dir / f"run_{ctx.run_id}"
    out.mkdir(parents=True, exist_ok=True)
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=select_autoescape(["html"]))
    env.filters["num"] = lambda v, d=0: "—" if v is None else f"{v:,.{d}f}"
    env.filters["path"] = lambda u: _path(u) or "/"
    html = env.get_template("report.html").render(**build_context(ctx))
    (out / "report.html").write_text(html, encoding="utf-8")
    write_csvs(ctx, out)
    return {"report": str(out / "report.html")}
