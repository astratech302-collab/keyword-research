"""Phase 1 - crawl the site and build a structured site model (the basis for business relevance)."""
from __future__ import annotations

import json
import logging
import re
from collections import deque
from typing import Callable
from urllib.parse import urljoin, urlparse, urldefrag

import httpx
from bs4 import BeautifulSoup

from ..clients.llm import batched
from ..context import Ctx
from ..parallel import parallel_map

log = logging.getLogger(__name__)

UA = "Mozilla/5.0 (compatible; kwresearch/0.1; +https://example.com/bot)"
SKIP_EXT = re.compile(r"\.(jpg|jpeg|png|gif|svg|webp|pdf|zip|mp4|mp3|css|js|xml|ico|woff2?)$", re.I)
SKIP_PATH = re.compile(r"/(tag|tags|author|page|feed|wp-json|cart|checkout|account|login|signup|search)(/|$)", re.I)
PRIORITY = [
    (re.compile(r"^/?$"), 0),
    (re.compile(r"/(pricing|plans)"), 1),
    (re.compile(r"/collections?/"), 2),                     # e-commerce category pages
    (re.compile(r"/products/[^/]+"), 3),                    # e-commerce product detail pages
    (re.compile(r"/(product|products|platform|features?|solutions?|services?|use-cases?|integrations?)"), 2),
    (re.compile(r"/(compare|vs|alternatives?|customers?|industries|industry)"), 3),
    (re.compile(r"/(blog|resources?|guides?|learn|articles?|docs|glossary)"), 5),
]

Fetch = Callable[[str], tuple[int, str]]


def default_fetch() -> Fetch:
    client = httpx.Client(headers={"User-Agent": UA}, follow_redirects=True, timeout=20)

    def fetch(url: str) -> tuple[int, str]:
        try:
            r = client.get(url)
            ctype = r.headers.get("content-type", "")
            if "html" not in ctype and "xml" not in ctype and "text" not in ctype:
                return r.status_code, ""
            return r.status_code, r.text
        except httpx.HTTPError as e:
            log.debug("fetch failed %s: %s", url, e)
            return 0, ""
    return fetch


def url_priority(url: str) -> int:
    path = urlparse(url).path.lower()
    for rx, p in PRIORITY:
        if rx.search(path):
            return p
    return 4


def _same_site(url: str, domain: str) -> bool:
    host = urlparse(url).netloc.lower()
    return host in (domain, "www." + domain)


def sitemap_urls(base: str, domain: str, fetch: Fetch, cap: int = 5000) -> list[str]:
    candidates = []
    status, robots = fetch(urljoin(base, "/robots.txt"))
    if status == 200:
        candidates += re.findall(r"(?im)^sitemap:\s*(\S+)", robots)
    candidates += [urljoin(base, "/sitemap.xml"), urljoin(base, "/sitemap_index.xml")]
    seen_maps, urls = set(), []
    queue = deque(candidates)
    while queue and len(urls) < cap and len(seen_maps) < 30:
        sm = queue.popleft()
        if sm in seen_maps:
            continue
        seen_maps.add(sm)
        status, xml = fetch(sm)
        if status != 200 or "<loc>" not in xml:
            continue
        locs = [l.strip() for l in re.findall(r"<loc>\s*(.*?)\s*</loc>", xml, re.S)]
        if "<sitemapindex" in xml:
            queue.extend(locs)
        else:
            urls += [u for u in locs if _same_site(u, domain)]
    return list(dict.fromkeys(urls))


def parse_page(url: str, html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    robots = soup.find("meta", attrs={"name": re.compile("^robots$", re.I)})
    noindex = bool(robots and "noindex" in (robots.get("content") or "").lower())
    canonical = soup.find("link", rel="canonical")
    for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "header", "form"]):
        tag.decompose()
    title = (soup.title.get_text(" ", strip=True) if soup.title else "")[:200]
    h1 = soup.find("h1")
    meta = soup.find("meta", attrs={"name": re.compile("^description$", re.I)})
    h2s = [h.get_text(" ", strip=True)[:120] for h in soup.find_all(["h2"])][:12]
    main = soup.find("main") or soup.find("article") or soup.body or soup
    text = re.sub(r"\s+", " ", main.get_text(" ", strip=True))[:1500]
    return {
        "url": url,
        "canonical": canonical.get("href") if canonical else None,
        "title": title,
        "h1": h1.get_text(" ", strip=True)[:200] if h1 else "",
        "meta": (meta.get("content") or "")[:300] if meta else "",
        "h2s": h2s,
        "text_excerpt": text,
        "noindex": noindex,
    }


def crawl(ctx: Ctx, fetch: Fetch | None = None) -> list[dict]:
    cfg = ctx.cfg
    fetch = fetch or default_fetch()
    base = f"https://{cfg.domain}"
    max_pages = cfg.limits.max_crawl_pages

    urls = sitemap_urls(base, cfg.domain, fetch)
    # Keep a balanced sample: all money pages first, then content.
    user_skip = [re.compile(p, re.I) for p in cfg.crawl_exclude]

    def skip(u: str) -> bool:
        path = urlparse(u).path
        return bool(SKIP_EXT.search(u) or SKIP_PATH.search(path) or any(r.search(path) for r in user_skip))

    urls = [u for u in urls if not skip(u)]
    urls.sort(key=lambda u: (url_priority(u), len(u)))
    queue = deque([base + "/"] + urls)
    seen: set[str] = set()
    pages: list[dict] = []
    discover_links = len(urls) < max_pages  # no/small sitemap -> BFS through links

    while queue and len(pages) < max_pages:
        url = urldefrag(queue.popleft())[0].rstrip("/") or base
        if url in seen:
            continue
        seen.add(url)
        status, html = fetch(url)
        if status != 200 or not html:
            continue
        page = parse_page(url, html)
        if page["noindex"]:
            continue
        pages.append(page)
        if len(pages) == 1 or len(pages) % 10 == 0 or len(pages) == max_pages:
            log.info("crawl progress: %d/%d pages", len(pages), max_pages)
        if discover_links:
            soup = BeautifulSoup(html, "html.parser")
            links = []
            for a in soup.find_all("a", href=True):
                u = urldefrag(urljoin(url, a["href"]))[0]
                if u.startswith("http") and _same_site(u, cfg.domain) and not skip(u) and "?" not in u:
                    links.append(u.rstrip("/"))
            links.sort(key=url_priority)
            queue.extend(l for l in links if l not in seen)
    log.info("crawled %d pages", len(pages))
    return pages


# ------------------------------------------------------------------ LLM steps
PAGE_SYSTEM = """You classify web pages of one company's website for SEO planning.
For each page return: page_type (one of: home, product, feature, solution, pricing, comparison,
integration, category, blog, guide, glossary, docs, case_study, about, contact, legal, other),
topics (1-4 short noun phrases a searcher might use), conversion_value (0.0-1.0: how directly the
page can convert a visitor into a lead/customer)."""


def classify_pages(ctx: Ctx, pages: list[dict]) -> None:
    batches = list(batched(pages, 20))
    def classify(batch: list[dict]) -> dict:
        payload = [{"i": i, "url": p["url"], "title": p["title"], "h1": p["h1"], "meta": p["meta"],
                    "h2s": p["h2s"][:6]} for i, p in enumerate(batch)]
        return ctx.llm.json(ctx.cfg.llm.fast, PAGE_SYSTEM,
                            'Return {"pages":[{"i":0,"page_type":"...","topics":["..."],'
                            '"conversion_value":0.5}]}\n\n' + json.dumps(payload))

    results = parallel_map(classify, batches, ctx.cfg.limits.llm_workers,
                           lambda done, total: log.info("page classification: %d/%d batches complete", done, total))
    for batch, data in zip(batches, results):
        by_i = {d.get("i"): d for d in (data.get("pages") or [])}
        for i, p in enumerate(batch):
            d = by_i.get(i, {})
            p["page_type"] = d.get("page_type") or "other"
            p["topics"] = d.get("topics") or []
            p["conversion_value"] = float(d.get("conversion_value") or 0.3)


SUMMARY_SYSTEM = """You are an SEO strategist. Build a factual business model of a company from its
own website pages. Do not invent products that the pages do not support."""


def build_site_model(ctx: Ctx, pages: list[dict]) -> dict:
    cfg = ctx.cfg
    key_pages = sorted(pages, key=lambda p: (url_priority(p["url"]), -p.get("conversion_value", 0)))[:25]
    digest = [{"url": p["url"], "type": p.get("page_type"), "title": p["title"], "h1": p["h1"],
               "meta": p["meta"], "topics": p.get("topics"), "excerpt": p["text_excerpt"][:500]}
              for p in key_pages]
    hints = {"business_type": cfg.business_type, "goal": cfg.goal, "target_customer": cfg.target_customer,
             "products": cfg.products, "country": cfg.country}
    user = f"""Domain: {cfg.domain}
User-provided hints (trust these): {json.dumps(hints)}
Pages: {json.dumps(digest)}

Return JSON:
{{"company": "", "industry": "", "one_liner": "",
  "products": ["core products/services, most important first"],
  "customers": ["who buys"], "problems": ["problems solved"],
  "brand_terms": ["brand/product names"],
  "not_relevant": ["adjacent topics that look related but would NOT bring buyers"],
  "seed_topics": ["20-40 short head terms (1-4 words) customers would search, mix of product,
                   problem and category terms; no brand names"]}}"""
    log.info("building site model from %d representative pages", len(key_pages))
    model = ctx.llm.json(cfg.llm.smart, SUMMARY_SYSTEM, user)
    if cfg.products:
        model["products"] = list(dict.fromkeys(cfg.products + model.get("products", [])))
    model["brand_terms"] = list(dict.fromkeys(cfg.brand_terms + model.get("brand_terms", [])))
    return model


def run(ctx: Ctx, fetch: Fetch | None = None) -> dict:
    pages = crawl(ctx, fetch or ctx.cache.get("fetch"))
    if not pages:
        raise RuntimeError(f"Could not crawl any pages from {ctx.cfg.domain}")
    classify_pages(ctx, pages)
    model = build_site_model(ctx, pages)
    db, rid = ctx.db, ctx.run_id
    db.execute("DELETE FROM pages WHERE run_id=?", (rid,))
    db.executemany(
        "INSERT OR REPLACE INTO pages (run_id,url,title,h1,meta,h2s_json,text_excerpt,page_type,"
        "topics_json,conversion_value,noindex) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        [(rid, p["url"], p["title"], p["h1"], p["meta"], json.dumps(p["h2s"]), p["text_excerpt"],
          p["page_type"], json.dumps(p["topics"]), p["conversion_value"], int(p["noindex"])) for p in pages])
    db.execute("INSERT OR REPLACE INTO site_model VALUES (?,?)", (rid, json.dumps(model)))
    db.commit()
    return {"pages": len(pages), "company": model.get("company"), "seed_topics": len(model.get("seed_topics", []))}
