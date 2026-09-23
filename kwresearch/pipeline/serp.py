"""Phase 8 - live SERP check for the shortlisted clusters only (never for the whole pool).
Tells us what *kind* of page Google rewards for the query, which drives the page recommendation."""
from __future__ import annotations

import logging
import re
from collections import Counter
from urllib.parse import urlparse

from ..context import Ctx
from ..parallel import parallel_map

log = logging.getLogger(__name__)

DIRECTORIES = {"g2.com", "capterra.com", "gartner.com", "trustradius.com", "softwareadvice.com",
               "getapp.com", "clutch.co", "goodfirms.co", "sourceforge.net", "producthunt.com",
               "justdial.com", "indiamart.com", "yelp.com", "tripadvisor.com"}
FORUMS = {"reddit.com", "quora.com", "stackoverflow.com", "stackexchange.com"}
PUBLISHERS = {"wikipedia.org", "forbes.com", "techtarget.com", "investopedia.com", "medium.com",
              "ibm.com", "youtube.com"}


def classify_result(item: dict) -> str:
    url = item.get("url") or ""
    title = (item.get("title") or "").lower()
    host = (item.get("domain") or urlparse(url).netloc).lower().removeprefix("www.")
    path = urlparse(url).path.lower()
    base = ".".join(host.split(".")[-2:])
    if base in DIRECTORIES:
        return "directory/review site"
    if base in FORUMS:
        return "forum"
    if re.search(r"\b(vs\.?|versus|alternatives?|compare|comparison)\b", title) or re.search(r"(-vs-|alternative|compare)", path):
        return "comparison"
    if re.search(r"\b(best|top \d+|\d+ best)\b", title):
        return "listicle"
    if re.search(r"/(blog|guide|guides|learn|resources|articles?|glossary|what-is|academy)/", path + "/") \
            or re.search(r"^(what|how|why|when)\b", title) or base in PUBLISHERS:
        return "article/guide"
    if path.strip("/").count("/") <= 1 or re.search(r"/(product|products|platform|features?|solutions?|pricing|software|tools?)", path):
        return "product/landing page"
    return "other"


def run(ctx: Ctx) -> dict:
    n = ctx.cfg.limits.serp_checks
    if n <= 0:
        return {"checked": 0}
    clusters = ctx.db.clusters(ctx.run_id)
    shortlist = sorted([c for c in clusters if c["intent"] != "navigational"],
                       key=lambda c: -c.get("priority", 0))[:n]
    competitors = {r["domain"] for r in ctx.db.query(
        "SELECT domain FROM competitors WHERE run_id=? AND selected=1", (ctx.run_id,))}
    by_id = {c["cluster_id"]: c for c in clusters}
    results = parallel_map(
        lambda c: ctx.dfs.serp_organic(c["primary_keyword"], depth=10)[:10],
        shortlist,
        ctx.cfg.limits.dataforseo_workers,
        lambda done, total: log.info("SERP checks: %d/%d complete", done, total)
        if done == 1 or done % 5 == 0 or done == total else None,
    )
    for c, items in zip(shortlist, results):
        kinds = Counter(classify_result(i) for i in items)
        doms = [(i.get("domain") or "").removeprefix("www.") for i in items]
        by_id[c["cluster_id"]]["serp"] = {
            "keyword": c["primary_keyword"],
            "dominant_page_type": kinds.most_common(1)[0][0] if kinds else None,
            "page_types": dict(kinds),
            "top_domains": doms,
            "own_domain_present": any(d.endswith(ctx.cfg.domain) for d in doms),
            "competitors_in_top10": sorted({d for d in doms if d in competitors}),
            "results": [{"pos": i.get("rank_group"), "domain": i.get("domain"), "title": i.get("title"),
                         "url": i.get("url"), "kind": classify_result(i)} for i in items],
        }
    ctx.db.save_clusters(ctx.run_id, clusters)
    return {"checked": len(shortlist)}
