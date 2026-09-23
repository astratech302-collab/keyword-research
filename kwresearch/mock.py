"""Offline fakes for DataForSEO, the LLM and the crawler.

`kwresearch run --mock` exercises the whole pipeline with zero API spend so you can check the
report shape and tune weights before paying for real data. Also used by the test-suite.
The data is synthetic (a fictional data-governance SaaS) and NOT real SEO data."""
from __future__ import annotations

import hashlib
import json
import re

TOPICS = {
    "data governance": ["tools", "software", "platform", "framework", "best practices", "policy", "what is",
                        "roles", "maturity model", "jobs", "certification"],
    "data lineage": ["tool", "tools", "software", "what is", "vs data provenance", "automated", "open source",
                     "column level", "examples"],
    "data catalog": ["tools", "software", "what is", "vs data dictionary", "open source", "comparison", "features"],
    "metadata management": ["tools", "software", "what is", "best practices", "course"],
    "data quality": ["tools", "monitoring", "dimensions", "what is", "metrics", "salary"],
    "data observability": ["tools", "platform", "what is", "vs monitoring"],
}
COMPETITORS = ["collibra.com", "atlan.com", "alation.com", "montecarlodata.com", "wikipedia.org", "gartner.com"]


def _h(s: str, mod: int) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % mod


def _kw_universe() -> list[str]:
    out = []
    for head, mods in TOPICS.items():
        out.append(head)
        for m in mods:
            if m == "what is":
                out.append(f"what is {head}")
            elif m.startswith("vs "):
                out.append(f"{head} {m}")
            else:
                out.append(f"{head} {m}")
            out.append(f"best {head} {m}" if m in ("tools", "software", "platform") else f"{head} {m} guide")
    return list(dict.fromkeys(out))


def _intent(kw: str) -> str:
    if kw.startswith("what is") or "guide" in kw or "examples" in kw or "dimensions" in kw:
        return "informational"
    if any(w in kw for w in ("tools", "software", "platform", "best", "comparison", " vs ")):
        return "commercial"
    return "informational"


def _kw_item(kw: str) -> dict:
    vol = [10, 20, 50, 90, 170, 320, 590, 880, 1300, 2400, 4400][_h(kw, 11)]
    if len(kw.split()) <= 2:
        vol *= 3
    monthly = [{"year": 2026 - (i // 12), "month": 12 - (i % 12), "search_volume": int(vol * (1 + 0.02 * (12 - i)))}
               for i in range(15)]
    core = re.sub(r"^(what is|best) ", "", kw)
    core = re.sub(r" (guide)$", "", core)
    return {"keyword": kw, "keyword_info": {"search_volume": vol, "cpc": round(1 + _h(kw + "c", 250) / 10, 2),
                                           "competition": _h(kw + "k", 100) / 100, "monthly_searches": monthly},
            "keyword_properties": {"keyword_difficulty": 15 + _h(kw + "d", 70), "core_keyword": core},
            "search_intent_info": {"main_intent": _intent(kw)}}


class FakeDataForSEO:
    spent = 0.0

    def ranked_keywords(self, target: str, limit: int = 1000, max_position: int | None = None) -> list[dict]:
        kws = _kw_universe()
        out = []
        for kw in kws:
            if _h(target + kw, 3) == 0:
                continue
            pos = 1 + _h(target + kw + "p", 60)
            if max_position and pos > max_position:
                continue
            slug = re.sub(r"[^a-z0-9]+", "-", kw.replace("what is ", "").replace("best ", ""))
            url = f"https://{target}/{'blog/' if _intent(kw) == 'informational' else 'product/'}{slug}"
            out.append({"keyword_data": _kw_item(kw), "ranked_serp_element": {
                "serp_item": {"rank_group": pos, "url": url, "etv": 1000 / pos}}})
        return out[:limit]

    def competitors_domain(self, target: str, limit: int = 50) -> list[dict]:
        return [{"domain": d, "intersections": 400 - 50 * i, "avg_position": 8 + i,
                 "full_domain_metrics": {"organic": {"etv": 10000 / (i + 1)}}} for i, d in enumerate(COMPETITORS)]

    def serp_competitors(self, keywords, limit=50):
        return []

    def keywords_for_site(self, target, limit=1000):
        return [_kw_item(k) for k in _kw_universe()[::2]]

    def keyword_ideas(self, seeds, limit=1000):
        return [_kw_item(k) for k in _kw_universe()]

    def keyword_suggestions(self, seed, limit=200):
        return [_kw_item(k) for k in _kw_universe() if seed.split()[0] in k][:limit]

    def bulk_keyword_difficulty(self, keywords):
        return {k: 15 + _h(k + "d", 70) for k in keywords}

    def search_intent(self, keywords):
        return {k: (_intent(k), 0.8) for k in keywords}

    def serp_organic(self, keyword, depth=10):
        doms = ["g2.com", "atlan.com", "collibra.com", "techtarget.com", "alation.com", "ibm.com",
                "montecarlodata.com", "reddit.com", "gartner.com", "datacamp.com"]
        out = []
        for i in range(depth):
            d = doms[(i + _h(keyword, 10)) % len(doms)]
            title = f"{'10 Best ' if 'tool' in keyword else 'What is '}{keyword.title()}"
            out.append({"type": "organic", "rank_group": i + 1, "domain": d, "title": title,
                        "url": f"https://{d}/blog/{keyword.replace(' ', '-')}"})
        return out


class FakeLLM:
    """Rule-based stand-in that returns the same JSON shapes as the real prompts."""

    def json(self, model: str, system: str, user: str):
        if "classify web pages" in system:
            pages = json.loads(user.split("\n\n", 1)[1])
            return {"pages": [{"i": p["i"], "page_type": "blog" if "/blog/" in p["url"] else
                               ("home" if p["url"].rstrip("/").count("/") == 2 else "product"),
                               "topics": [p["h1"].lower()], "conversion_value": 0.3 if "/blog/" in p["url"] else 0.8}
                              for p in pages]}
        if "business model of a company" in system:
            return {"company": "Acme Lineage", "industry": "data governance software",
                    "one_liner": "Automated data lineage and catalog platform for data teams",
                    "products": ["data lineage", "data catalog", "data governance platform"],
                    "customers": ["data engineering teams", "data governance leads"],
                    "problems": ["tracking data lineage", "finding trusted data", "compliance"],
                    "brand_terms": ["acme lineage"], "not_relevant": ["jobs", "courses"],
                    "seed_topics": list(TOPICS.keys())}
        if "true SEO competitors" in system:
            cands = json.loads(user.split("Candidates: ", 1)[1].split("\n")[0])
            return {"competitors": [{"domain": c["domain"], "relevant": c["domain"] not in ("wikipedia.org", "gartner.com"),
                                     "reason": "same buyers" if c["domain"] not in ("wikipedia.org", "gartner.com")
                                     else "publisher/analyst"} for c in cands]}
        if "business relevance" in system:
            lines = user.split("index<TAB>keyword):\n", 1)[1].split("\n\n")[0].splitlines()
            out = []
            for ln in lines:
                i, kw = ln.split("\t", 1)
                r = 5 if any(w in kw for w in ("tool", "software", "platform", "catalog", "lineage")) else 3
                if "observability" in kw:
                    r = 2
                out.append({"i": int(i), "r": r})
            return {"scores": out}
        if "mapping keyword clusters" in system:
            clusters = json.loads(user.split("Clusters: ", 1)[1].split("\n\nReturn")[0])
            res = []
            for c in clusters:
                ranks = c.get("site_rankings") or []
                if ranks and ranks[0]["best_position"] <= 30:
                    oc, url = "EXISTING_WEAK_PAGE", ranks[0]["url"]
                elif c["intent"] == "commercial":
                    oc, url = "CONTENT_GAP", None
                else:
                    oc, url = "CONTENT_GAP", None
                slug = "/" + re.sub(r"[^a-z0-9]+", "-", c["primary_keyword"]).strip("-") + "/"
                res.append({"cluster_id": c["cluster_id"], "outcome": oc, "target_url": url,
                            "recommended_page_type": "listicle / comparison landing page" if c["intent"] == "commercial" else "how-to guide",
                            "suggested_slug": None if url else slug,
                            "rationale": "mock mapping", "content_notes": ["comparison table", "pricing FAQ"]})
            return {"mappings": res}
        return {}


def fake_fetch(domain: str):
    paths = ["", "/pricing", "/product/data-lineage", "/product/data-catalog", "/blog/what-is-data-governance",
             "/blog/data-lineage-examples", "/about"]

    def fetch(url: str):
        if url.endswith("robots.txt"):
            return 200, f"Sitemap: https://{domain}/sitemap.xml"
        if url.endswith("sitemap.xml"):
            return 200, "<urlset>" + "".join(f"<loc>https://{domain}{p}</loc>" for p in paths) + "</urlset>"
        path = re.sub(r"^https?://[^/]+", "", url).rstrip("/")
        if path in paths or url.rstrip("/").endswith(domain):
            name = path.split("/")[-1].replace("-", " ").title() or "Acme Lineage - automated data lineage"
            return 200, (f"<html><head><title>{name}</title><meta name='description' content='{name} for data teams'>"
                         f"</head><body><main><h1>{name}</h1><h2>Features</h2><p>{name} helps data teams.</p></main></body></html>")
        if not url.startswith(f"https://{domain}"):
            host = re.sub(r"^https?://", "", url).split("/")[0]
            return 200, f"<html><head><title>{host} data platform</title></head><body><h1>{host}</h1></body></html>"
        return 404, ""
    return fetch
