"""Read Google Search Console's Queries and Pages CSV exports."""
from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path
from urllib.parse import urlparse

from .context import Ctx
from .pipeline.candidates import normalize_keyword


def page_key(url: str) -> str:
    parsed = urlparse(url)
    return (parsed.hostname or "").lower().removeprefix("www.") + parsed.path.rstrip("/").lower()


def input_hashes(paths: list[Path]) -> list[str]:
    return sorted(hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in paths)


def read_csv(path: str | Path, domain: str) -> tuple[str, list[dict]]:
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = [h.strip().lower() for h in (reader.fieldnames or [])]
        expected = {"clicks", "impressions", "ctr", "position"}
        kind = "queries" if "top queries" in headers else "pages" if "top pages" in headers else None
        if not kind or not expected.issubset(headers):
            raise ValueError(f"{path}: expected Top queries or Top pages,Clicks,Impressions,CTR,Position header")
        out = []
        for line, raw in enumerate(reader, 2):
            row = {k.strip().lower(): (v or "").strip() for k, v in raw.items() if k}
            label = row[f"top {kind}"]
            if not label:
                continue
            try:
                clicks = int(row["clicks"].replace(",", ""))
                impressions = int(row["impressions"].replace(",", ""))
                ctr = float(row["ctr"].rstrip("%")) / 100
                position = float(row["position"].replace(",", ""))
            except ValueError as exc:
                raise ValueError(f"{path}:{line}: invalid Search Console metric") from exc
            if (clicks < 0 or impressions < 0 or clicks > impressions or
                    not math.isfinite(ctr) or not 0 <= ctr <= 1 or
                    not math.isfinite(position) or position <= 0):
                raise ValueError(f"{path}:{line}: invalid Search Console metric")
            if kind == "pages":
                parsed = urlparse(label)
                host = (parsed.hostname or "").lower().removeprefix("www.")
                if parsed.scheme not in ("http", "https") or not (host == domain or host.endswith("." + domain)):
                    raise ValueError(f"{path}:{line}: page URL is outside {domain}")
                key = page_key(label)
            else:
                key = normalize_keyword(label)
            if key:
                out.append({"key": key, "label": label, "clicks": clicks, "impressions": impressions,
                            "ctr": ctr, "position": position})
    if not out:
        raise ValueError(f"{path}: no Search Console rows found")
    return kind, out


def run(ctx: Ctx) -> dict:
    paths = ctx.gsc_csv
    if not paths:
        return {"files": 0, "queries": 0, "pages": 0}
    data = {"queries": {}, "pages": {}}
    for path in paths:
        kind, rows = read_csv(path, ctx.cfg.domain)
        for row in rows:
            # Duplicate keys from multiple exports are aggregated; position uses impression weights.
            key = row["key"]
            cur = data[kind].get(key)
            if cur:
                total = cur["impressions"] + row["impressions"]
                if total:
                    cur["position"] = (cur["position"] * cur["impressions"] +
                                       row["position"] * row["impressions"]) / total
                cur["clicks"] += row["clicks"]
                cur["impressions"] = total
                cur["ctr"] = cur["clicks"] / total if total else 0
            else:
                data[kind][key] = row.copy()
    rid = ctx.run_id
    for kind in ("queries", "pages"):
        ctx.db.execute(f"DELETE FROM gsc_{kind} WHERE run_id=?", (rid,))
        ctx.db.executemany(
            f"INSERT INTO gsc_{kind} VALUES (?,?,?,?,?,?,?)",
            [(rid, r["key"], r["label"], r["clicks"], r["impressions"], r["ctr"], r["position"])
             for r in data[kind].values()],
        )
    ctx.db.commit()
    return {"files": len(paths), "queries": len(data["queries"]), "pages": len(data["pages"]),
            "_input_hashes": input_hashes(paths)}
