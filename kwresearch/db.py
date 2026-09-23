"""SQLite persistence. One file per project; every run is kept so metrics history accumulates."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    domain TEXT NOT NULL,
    config_json TEXT NOT NULL,
    started_at REAL NOT NULL,
    finished_at REAL,
    status TEXT NOT NULL DEFAULT 'running'
);
CREATE TABLE IF NOT EXISTS phases (
    run_id INTEGER, name TEXT, status TEXT, finished_at REAL, meta_json TEXT,
    PRIMARY KEY (run_id, name)
);
CREATE TABLE IF NOT EXISTS pages (
    run_id INTEGER, url TEXT, title TEXT, h1 TEXT, meta TEXT, h2s_json TEXT,
    text_excerpt TEXT, page_type TEXT, topics_json TEXT, conversion_value REAL,
    noindex INTEGER DEFAULT 0,
    PRIMARY KEY (run_id, url)
);
CREATE TABLE IF NOT EXISTS site_model (run_id INTEGER PRIMARY KEY, json TEXT);
CREATE TABLE IF NOT EXISTS competitors (
    run_id INTEGER, domain TEXT, source TEXT, intersections INTEGER, etv REAL,
    avg_position REAL, relevant INTEGER, reason TEXT, selected INTEGER DEFAULT 0,
    PRIMARY KEY (run_id, domain)
);
CREATE TABLE IF NOT EXISTS rankings (
    run_id INTEGER, domain TEXT, keyword TEXT, position INTEGER, url TEXT, etv REAL,
    is_own INTEGER,
    PRIMARY KEY (run_id, domain, keyword)
);
CREATE TABLE IF NOT EXISTS keywords (
    run_id INTEGER, keyword TEXT, sources_json TEXT,
    search_volume INTEGER, cpc REAL, competition REAL, kd REAL,
    intent TEXT, intent_prob REAL, core_keyword TEXT, monthly_json TEXT,
    trend_3m REAL, trend_12m REAL,
    relevance REAL, relevance_reason TEXT,
    kept INTEGER DEFAULT 1, drop_reason TEXT,
    cluster_id INTEGER, own_position INTEGER, own_url TEXT,
    PRIMARY KEY (run_id, keyword)
);
CREATE INDEX IF NOT EXISTS idx_kw_cluster ON keywords(run_id, cluster_id);
CREATE TABLE IF NOT EXISTS clusters (
    run_id INTEGER, cluster_id INTEGER, data_json TEXT,
    PRIMARY KEY (run_id, cluster_id)
);
CREATE TABLE IF NOT EXISTS keyword_metrics_history (
    keyword TEXT, location TEXT, language TEXT, captured_at REAL,
    search_volume INTEGER, cpc REAL, kd REAL, intent TEXT
);
CREATE TABLE IF NOT EXISTS api_cache (
    key TEXT PRIMARY KEY, endpoint TEXT, created_at REAL, response_json TEXT
);
CREATE TABLE IF NOT EXISTS api_calls (
    run_id INTEGER, provider TEXT, endpoint TEXT, cost REAL, cached INTEGER, at REAL
);
"""

KEYWORD_COLS = [
    "keyword", "sources_json", "search_volume", "cpc", "competition", "kd", "intent",
    "intent_prob", "core_keyword", "monthly_json", "trend_3m", "trend_12m", "relevance",
    "relevance_reason", "kept", "drop_reason", "cluster_id", "own_position", "own_url",
]


class DB:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        with self._lock:
            self.conn.executescript(SCHEMA)
            self.conn.commit()

    # ---------- generic ----------
    def execute(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            return self.conn.execute(sql, tuple(params))

    def executemany(self, sql: str, rows: Iterable[Iterable[Any]]) -> None:
        with self._lock:
            self.conn.executemany(sql, [tuple(r) for r in rows])

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self.conn.execute(sql, tuple(params)).fetchall()]

    def commit(self) -> None:
        with self._lock:
            self.conn.commit()

    # ---------- runs / phases ----------
    def create_run(self, domain: str, config_json: str) -> int:
        cur = self.execute(
            "INSERT INTO runs (domain, config_json, started_at) VALUES (?,?,?)",
            (domain, config_json, time.time()),
        )
        self.commit()
        return int(cur.lastrowid)

    def latest_run(self, domain: str) -> dict | None:
        rows = self.query("SELECT * FROM runs WHERE domain=? ORDER BY id DESC LIMIT 1", (domain,))
        return rows[0] if rows else None

    def finish_run(self, run_id: int, status: str = "done") -> None:
        self.execute("UPDATE runs SET finished_at=?, status=? WHERE id=?", (time.time(), status, run_id))
        self.commit()

    def resume_run(self, run_id: int) -> None:
        """Mark an existing run active again without discarding its checkpoints."""
        self.execute("UPDATE runs SET finished_at=NULL, status='running' WHERE id=?", (run_id,))
        self.commit()

    def phase_done(self, run_id: int, name: str) -> bool:
        rows = self.query("SELECT status FROM phases WHERE run_id=? AND name=?", (run_id, name))
        return bool(rows) and rows[0]["status"] == "done"

    def start_phase(self, run_id: int, name: str) -> None:
        self.execute(
            "INSERT OR REPLACE INTO phases (run_id, name, status, finished_at, meta_json) VALUES (?,?,?,?,?)",
            (run_id, name, "running", None, "{}"),
        )
        self.commit()

    def mark_phase(self, run_id: int, name: str, status: str = "done", meta: dict | None = None) -> None:
        self.execute(
            "INSERT OR REPLACE INTO phases (run_id, name, status, finished_at, meta_json) VALUES (?,?,?,?,?)",
            (run_id, name, status, time.time(), json.dumps(meta or {})),
        )
        self.commit()

    def phase_meta(self, run_id: int) -> dict[str, dict]:
        return {r["name"]: json.loads(r["meta_json"] or "{}") for r in
                self.query("SELECT name, meta_json FROM phases WHERE run_id=? ORDER BY finished_at", (run_id,))}

    # ---------- keywords ----------
    def upsert_keywords(self, run_id: int, rows: list[dict]) -> None:
        cols = ["run_id"] + KEYWORD_COLS
        sql = f"INSERT OR REPLACE INTO keywords ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})"
        self.executemany(sql, ([run_id] + [_enc(r.get(c)) for c in KEYWORD_COLS] for r in rows))
        self.commit()

    def keywords(self, run_id: int, kept_only: bool = True) -> list[dict]:
        sql = "SELECT * FROM keywords WHERE run_id=?" + (" AND kept=1" if kept_only else "")
        return self.query(sql, (run_id,))

    # ---------- clusters ----------
    def save_clusters(self, run_id: int, clusters: list[dict]) -> None:
        self.execute("DELETE FROM clusters WHERE run_id=?", (run_id,))
        self.executemany(
            "INSERT INTO clusters (run_id, cluster_id, data_json) VALUES (?,?,?)",
            ((run_id, c["cluster_id"], json.dumps(c)) for c in clusters),
        )
        self.commit()

    def clusters(self, run_id: int) -> list[dict]:
        return [json.loads(r["data_json"]) for r in
                self.query("SELECT data_json FROM clusters WHERE run_id=? ORDER BY cluster_id", (run_id,))]

    # ---------- api cost ----------
    def log_call(self, run_id: int | None, provider: str, endpoint: str, cost: float, cached: bool) -> None:
        self.execute("INSERT INTO api_calls VALUES (?,?,?,?,?,?)",
                     (run_id, provider, endpoint, cost, int(cached), time.time()))
        self.commit()

    def run_cost(self, run_id: int) -> dict[str, float]:
        rows = self.query(
            "SELECT provider, SUM(cost) AS c FROM api_calls WHERE run_id=? AND cached=0 GROUP BY provider",
            (run_id,))
        return {r["provider"]: round(r["c"] or 0.0, 4) for r in rows}


def _enc(v: Any) -> Any:
    if isinstance(v, (list, dict)):
        return json.dumps(v)
    return v
