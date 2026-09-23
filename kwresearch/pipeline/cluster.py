"""Phases 9-10 - hybrid clustering and primary-keyword selection.

1. Split by intent bucket (informational vs commercial/transactional vs navigational):
   "what is data lineage" and "data lineage tools" need different pages, so never merge them.
2. Union-find on lexical signature + DataForSEO core_keyword -> "units" (near-duplicates).
3. If embeddings are available, agglomerative clustering (cosine) over unit centroids.
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter, defaultdict

import numpy as np

from ..context import Ctx
from . import candidates

log = logging.getLogger(__name__)

STOP = set("""a an the of for to in on at by with and or vs versus is are was be what how why when
where which who does do can best top free online near me my your 2023 2024 2025 2026 2027 list
examples example guide""".split())

BUCKET = {"informational": "info", "commercial": "commercial", "transactional": "commercial",
          "navigational": "nav"}


def stem(t: str) -> str:
    for suf in ("ies", "es", "s"):
        if len(t) > 4 and t.endswith(suf) and not t.endswith("ss"):
            return t[: -len(suf)] + ("y" if suf == "ies" else "")
    return t


def tokens(kw: str) -> list[str]:
    return [stem(t) for t in re.findall(r"[a-z0-9+#.]+", kw.lower()) if t not in STOP]


def signature(kw: str) -> str:
    toks = tokens(kw)
    return " ".join(sorted(set(toks))) or kw


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def lexical_units(kws: list[dict]) -> list[list[int]]:
    uf = UF(len(kws))
    first: dict[tuple, int] = {}
    for i, r in enumerate(kws):
        # A keyword is its own core, so "x" links with any keyword whose core_keyword is "x".
        keys = [("sig", signature(r["keyword"])), ("core", r["keyword"])]
        if r.get("core_keyword"):
            keys.append(("core", candidates.normalize_keyword(r["core_keyword"])))
        for k in keys:
            if k in first:
                uf.union(first[k], i)
            else:
                first[k] = i
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(kws)):
        groups[uf.find(i)].append(i)
    return list(groups.values())


def _agglomerate(vecs: np.ndarray, threshold: float) -> np.ndarray:
    from sklearn.cluster import AgglomerativeClustering
    if len(vecs) == 1:
        return np.zeros(1, dtype=int)
    model = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                    distance_threshold=threshold)
    return model.fit_predict(vecs)


def embed_units(vecs: np.ndarray, threshold: float, max_block: int = 3000) -> np.ndarray:
    """Agglomerative is O(n^2) memory; partition big inputs with k-means first."""
    n = len(vecs)
    if n <= max_block:
        return _agglomerate(vecs, threshold)
    from sklearn.cluster import MiniBatchKMeans
    k = math.ceil(n / (max_block * 0.6))
    coarse = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=3).fit_predict(vecs)
    labels = np.zeros(n, dtype=int)
    offset = 0
    for c in range(k):
        idx = np.where(coarse == c)[0]
        if len(idx) == 0:
            continue
        sub = _agglomerate(vecs[idx], threshold)
        labels[idx] = sub + offset
        offset += int(sub.max()) + 1
    return labels


def primary_keyword(members: list[dict], vecs: np.ndarray | None, dominant_intent: str) -> dict:
    """0.35 volume + 0.25 relevance + 0.20 intent alignment + 0.20 representativeness."""
    maxv = max((m.get("search_volume") or 0) for m in members) or 1
    if vecs is not None:
        centroid = vecs.mean(axis=0)
        centroid /= (np.linalg.norm(centroid) or 1)
        rep = vecs @ centroid
    else:
        tok_freq = Counter(t for m in members for t in set(tokens(m["keyword"])))
        top = {t for t, _ in tok_freq.most_common(3)}
        rep = np.array([len(top & set(tokens(m["keyword"]))) / (len(top) or 1) for m in members])
    best, best_s = members[0], -1.0
    for i, m in enumerate(members):
        vol = math.log1p(m.get("search_volume") or 0) / math.log1p(maxv)
        rel = (m.get("relevance") if m.get("relevance") is not None else 2.5) / 5
        align = 1.0 if m.get("intent") == dominant_intent else 0.0
        s = 0.35 * vol + 0.25 * rel + 0.20 * align + 0.20 * float(rep[i])
        # prefer shorter head terms on ties
        s -= 0.01 * max(0, len(m["keyword"].split()) - 3)
        if s > best_s:
            best, best_s = m, s
    return best


def run(ctx: Ctx, threshold: float = 0.28) -> dict:
    kws = [r for r in candidates.pool(ctx).values() if r["kept"]]
    if not kws:
        ctx.db.save_clusters(ctx.run_id, [])
        return {"clusters": 0}
    emb = ctx.embedder
    if emb:
        log.info("clustering: encoding %d keywords", len(kws))
    all_vecs = emb.encode([r["keyword"] for r in kws]) if emb else None
    if all_vecs is not None:
        log.info("clustering: embeddings ready; grouping keywords by intent")

    by_bucket: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(kws):
        by_bucket[BUCKET.get(r.get("intent") or "", "info")].append(i)

    groups: list[list[int]] = []
    for bucket, idx in by_bucket.items():
        log.info("clustering: processing %s bucket (%d keywords)", bucket, len(idx))
        sub = [kws[i] for i in idx]
        units = [[idx[j] for j in u] for u in lexical_units(sub)]
        if all_vecs is None or len(units) == 1:
            groups.extend(units)
            continue
        cents = []
        for u in units:
            w = np.array([max(kws[i].get("search_volume") or 0, 1) for i in u], dtype=np.float32)
            c = (all_vecs[u] * w[:, None]).sum(axis=0)
            cents.append(c / (np.linalg.norm(c) or 1))
        labels = embed_units(np.vstack(cents), threshold)
        merged: dict[int, list[int]] = defaultdict(list)
        for u, lab in zip(units, labels):
            merged[int(lab)].extend(u)
        groups.extend(merged.values())

    clusters = []
    for g in groups:
        members = [kws[i] for i in g]
        vol_by_intent: Counter = Counter()
        for m in members:
            vol_by_intent[m.get("intent") or "informational"] += (m.get("search_volume") or 0) + 1
        dom = vol_by_intent.most_common(1)[0][0]
        vecs = all_vecs[g] if all_vecs is not None else None
        prim = primary_keyword(members, vecs, dom)
        clusters.append({"members": members, "primary": prim["keyword"], "intent": dom})

    clusters.sort(key=lambda c: -sum((m.get("search_volume") or 0) for m in c["members"]))
    out = []
    for cid, c in enumerate(clusters, start=1):
        for m in c["members"]:
            m["cluster_id"] = cid
        out.append({"cluster_id": cid, "primary_keyword": c["primary"], "intent": c["intent"],
                    "keywords": [m["keyword"] for m in sorted(c["members"],
                                                             key=lambda m: -(m.get("search_volume") or 0))]})
    candidates.flush(ctx)
    ctx.db.save_clusters(ctx.run_id, out)
    return {"keywords": len(kws), "clusters": len(out), "embeddings": all_vecs is not None}
