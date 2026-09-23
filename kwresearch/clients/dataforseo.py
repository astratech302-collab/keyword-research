"""DataForSEO REST client: cached, cost-tracked, budget-capped.

Only the factual data layer lives here. Nothing in this module guesses a metric.
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
import threading
import time
from typing import Any

import httpx

from ..db import DB

log = logging.getLogger(__name__)

BASE_URL = "https://api.dataforseo.com/v3"
CACHE_TTL_SECONDS = 30 * 24 * 3600
NO_RESULTS_CODES = {40102}   # "No Search Results" -> empty, not an error
SAFE_REQUESTS_PER_MINUTE = 300  # steady-flow ceiling, well below DataForSEO's general 2,000 RPM


AUTH_HELP = (
    "DataForSEO rejected the credentials (HTTP {code}). DATAFORSEO_PASSWORD must be the *API password* "
    "from https://app.dataforseo.com/api-access - not your website login password and not the base64 "
    "'Authorization' token. Also check the account is activated and has a positive balance."
)


def normalize_credentials(login: str | None, password: str | None) -> tuple[str | None, str | None]:
    """Trim stray quotes/whitespace and accept a pasted base64 'login:password' token as the password."""
    import base64
    login = (login or "").strip().strip("'\"") or None
    password = (password or "").strip().strip("'\"") or None
    if login and password and ":" not in password and len(password) >= 16:
        try:
            decoded = base64.b64decode(password, validate=True).decode()
        except Exception:
            decoded = ""
        if decoded.startswith(login + ":"):
            log.warning("DATAFORSEO_PASSWORD looks like the base64 auth token - using the API password inside it")
            password = decoded[len(login) + 1:]
    return login, password


class DataForSEOError(RuntimeError):
    pass


class BudgetExceeded(RuntimeError):
    pass


class DataForSEO:
    def __init__(self, login: str, password: str, db: DB, run_id: int | None = None,
                 max_cost_usd: float = 10.0, location_name: str = "India",
                 language_name: str = "English", language_code: str = "en",
                 timeout: float = 120.0, use_cache: bool = True):
        login, password = normalize_credentials(login, password)
        if not login or not password:
            raise DataForSEOError("DATAFORSEO_LOGIN / DATAFORSEO_PASSWORD are not set")
        self.http = httpx.Client(base_url=BASE_URL, auth=(login, password), timeout=timeout)
        self.db = db
        self.run_id = run_id
        self.max_cost = max_cost_usd
        self.spent = 0.0
        self.location_name = location_name
        self.language_name = language_name
        self.language_code = language_code
        self.use_cache = use_cache
        self._budget_lock = threading.Lock()
        self._rate_lock = threading.Lock()
        self._next_request_at = 0.0

    # ------------------------------------------------------------------ core
    def post(self, endpoint: str, task: dict) -> list[dict]:
        """POST a single task; return task['result'] (list). Cached by payload."""
        key = hashlib.sha256((endpoint + json.dumps(task, sort_keys=True)).encode()).hexdigest()
        if self.use_cache:
            rows = self.db.query("SELECT response_json, created_at FROM api_cache WHERE key=?", (key,))
            if rows and time.time() - rows[0]["created_at"] < CACHE_TTL_SECONDS:
                self.db.log_call(self.run_id, "dataforseo", endpoint, 0.0, True)
                return json.loads(rows[0]["response_json"])

        with self._budget_lock:
            if self.spent >= self.max_cost:
                raise BudgetExceeded(f"DataForSEO budget ${self.max_cost:.2f} reached (spent ${self.spent:.2f})")

        self._pace_request()
        data = self._request(endpoint, [task])
        cost = float(data.get("cost") or 0.0)
        with self._budget_lock:
            self.spent += cost
        self.db.log_call(self.run_id, "dataforseo", endpoint, cost, False)

        t = (data.get("tasks") or [{}])[0]
        code = t.get("status_code")
        if code in NO_RESULTS_CODES:
            result: list[dict] = []
        elif code != 20000:
            raise DataForSEOError(f"{endpoint}: task error {code} {t.get('status_message')}")
        else:
            result = t.get("result") or []

        self.db.execute("INSERT OR REPLACE INTO api_cache VALUES (?,?,?,?)",
                        (key, endpoint, time.time(), json.dumps(result)))
        self.db.commit()
        log.debug("dataforseo %s cost=$%.4f total=$%.4f", endpoint, cost, self.spent)
        return result

    def _pace_request(self) -> None:
        """Spread concurrent requests into a steady provider-friendly flow."""
        interval = 60.0 / SAFE_REQUESTS_PER_MINUTE
        with self._rate_lock:
            now = time.monotonic()
            wait = max(0.0, self._next_request_at - now)
            self._next_request_at = max(now, self._next_request_at) + interval
        if wait:
            time.sleep(wait)

    def _request(self, endpoint: str, payload: list[dict], attempts: int = 4) -> dict:
        delay = 2.0
        for i in range(attempts):
            try:
                r = self.http.post(endpoint, json=payload)
                if r.status_code in (401, 403):
                    raise DataForSEOError(AUTH_HELP.format(code=r.status_code))
                if r.status_code in (429, 500, 502, 503, 504):
                    raise httpx.HTTPStatusError("retryable", request=r.request, response=r)
                r.raise_for_status()
                data = r.json()
                if data.get("status_code") != 20000:
                    raise DataForSEOError(f"{endpoint}: {data.get('status_code')} {data.get('status_message')}")
                return data
            except (httpx.TransportError, httpx.HTTPStatusError) as e:
                if i == attempts - 1:
                    raise DataForSEOError(f"{endpoint}: {e}") from e
                log.warning("dataforseo %s failed (%s), retry in %.0fs", endpoint, e, delay)
                time.sleep(delay + random.uniform(0, 0.5))
                delay *= 2
        raise AssertionError("unreachable")

    def _loc(self, with_language: bool = True) -> dict:
        d: dict[str, Any] = {"location_name": self.location_name}
        if with_language:
            d["language_name"] = self.language_name
        return d

    def _items(self, endpoint: str, task: dict) -> list[dict]:
        res = self.post(endpoint, task)
        return (res[0].get("items") or []) if res else []

    def _paged(self, endpoint: str, task: dict, total: int, page: int = 1000) -> list[dict]:
        out: list[dict] = []
        offset = 0
        while offset < total:
            t = dict(task, limit=min(page, total - offset), offset=offset)
            items = self._items(endpoint, t)
            out.extend(items)
            log.info("DataForSEO progress: %s %d/%d rows", endpoint.rsplit("/", 2)[-2],
                     min(len(out), total), total)
            if len(items) < t["limit"]:
                break
            offset += len(items)
        return out

    # ------------------------------------------------------------- endpoints
    L = "/dataforseo_labs/google"

    def ranked_keywords(self, target: str, limit: int = 1000, max_position: int | None = None) -> list[dict]:
        task: dict[str, Any] = {"target": target, **self._loc(), "item_types": ["organic"],
                                "order_by": ["ranked_serp_element.serp_item.etv,desc"]}
        if max_position:
            task["filters"] = [["ranked_serp_element.serp_item.rank_group", "<=", max_position]]
        return self._paged(f"{self.L}/ranked_keywords/live", task, limit)

    def competitors_domain(self, target: str, limit: int = 50) -> list[dict]:
        task = {"target": target, **self._loc(), "limit": limit, "item_types": ["organic"]}
        return self._items(f"{self.L}/competitors_domain/live", task)

    def serp_competitors(self, keywords: list[str], limit: int = 50) -> list[dict]:
        task = {"keywords": keywords[:200], **self._loc(), "limit": limit, "item_types": ["organic"]}
        return self._items(f"{self.L}/serp_competitors/live", task)

    def keywords_for_site(self, target: str, limit: int = 1000) -> list[dict]:
        task = {"target": target, **self._loc(), "order_by": ["keyword_info.search_volume,desc"]}
        return self._paged(f"{self.L}/keywords_for_site/live", task, limit)

    def keyword_ideas(self, seeds: list[str], limit: int = 1000) -> list[dict]:
        task = {"keywords": seeds[:200], **self._loc()}
        return self._paged(f"{self.L}/keyword_ideas/live", task, limit)

    def keyword_suggestions(self, seed: str, limit: int = 200) -> list[dict]:
        task = {"keyword": seed, **self._loc(), "limit": limit,
                "order_by": ["keyword_info.search_volume,desc"]}
        return self._items(f"{self.L}/keyword_suggestions/live", task)

    def bulk_keyword_difficulty(self, keywords: list[str]) -> dict[str, float]:
        out: dict[str, float] = {}
        for i in range(0, len(keywords), 1000):
            log.info("keyword difficulty: %d-%d of %d", i + 1, min(i + 1000, len(keywords)), len(keywords))
            for it in self._items(f"{self.L}/bulk_keyword_difficulty/live",
                                  {"keywords": keywords[i:i + 1000], **self._loc()}):
                if it.get("keyword_difficulty") is not None:
                    out[it["keyword"]] = float(it["keyword_difficulty"])
        return out

    def search_intent(self, keywords: list[str]) -> dict[str, tuple[str, float]]:
        out: dict[str, tuple[str, float]] = {}
        for i in range(0, len(keywords), 1000):
            log.info("search intent: %d-%d of %d", i + 1, min(i + 1000, len(keywords)), len(keywords))
            for it in self._items(f"{self.L}/search_intent/live",
                                  {"keywords": keywords[i:i + 1000], "language_code": self.language_code}):
                ki = it.get("keyword_intent") or {}
                if ki.get("label"):
                    out[it["keyword"]] = (ki["label"], float(ki.get("probability") or 0))
        return out

    def serp_organic(self, keyword: str, depth: int = 10) -> list[dict]:
        task = {"keyword": keyword, **self._loc(), "depth": depth}
        res = self.post("/serp/google/organic/live/advanced", task)
        items = (res[0].get("items") or []) if res else []
        return [i for i in items if i.get("type") == "organic"]


# ----------------------------------------------------------------- parsing
def parse_keyword_item(item: dict) -> dict | None:
    """Normalise any Labs keyword item (flat or nested under keyword_data) into our schema."""
    kd_obj = item.get("keyword_data") or item
    kw = kd_obj.get("keyword")
    if not kw:
        return None
    info = kd_obj.get("keyword_info") or {}
    props = kd_obj.get("keyword_properties") or {}
    intent = kd_obj.get("search_intent_info") or {}
    monthly = info.get("monthly_searches") or []
    t3, t12 = trend_ratios(monthly, info.get("search_volume_trend"))
    return {
        "keyword": kw,
        "search_volume": info.get("search_volume"),
        "cpc": info.get("cpc"),
        "competition": info.get("competition"),
        "kd": props.get("keyword_difficulty"),
        "core_keyword": props.get("core_keyword"),
        "intent": intent.get("main_intent"),
        "intent_prob": None,
        "monthly_json": [
            {"y": m.get("year"), "m": m.get("month"), "v": m.get("search_volume")} for m in monthly
        ][:24],
        "trend_3m": t3,
        "trend_12m": t12,
    }


def parse_ranked_item(item: dict) -> dict | None:
    base = parse_keyword_item(item)
    if not base:
        return None
    serp = ((item.get("ranked_serp_element") or {}).get("serp_item")) or {}
    base["position"] = serp.get("rank_group")
    base["url"] = serp.get("url")
    base["etv"] = serp.get("etv") or 0.0
    return base


def trend_ratios(monthly: list[dict], trend: dict | None) -> tuple[float | None, float | None]:
    """Return (last-3-months vs previous-3, last-3-months vs same-3-months-a-year-ago) ratios."""
    if trend and trend.get("quarterly") is not None and trend.get("yearly") is not None:
        return 1 + trend["quarterly"] / 100.0, 1 + trend["yearly"] / 100.0
    vals = [m.get("search_volume") for m in sorted(
        monthly, key=lambda m: (m.get("year") or 0, m.get("month") or 0), reverse=True)]
    vals = [v for v in vals if v is not None]
    if len(vals) < 6:
        return None, None
    last3, prev3 = sum(vals[:3]), sum(vals[3:6])
    t3 = last3 / prev3 if prev3 else None
    t12 = None
    if len(vals) >= 15:
        yago = sum(vals[12:15])
        t12 = last3 / yago if yago else None
    return t3, t12
