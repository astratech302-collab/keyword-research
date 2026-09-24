from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import RunConfig
from .db import DB


@dataclass
class Ctx:
    cfg: RunConfig
    db: DB
    run_id: int
    dfs: Any            # DataForSEO (or a fake with the same methods)
    llm: Any            # LLM (or a fake)
    embedder: Any = None
    cache: dict = field(default_factory=dict)   # in-process memo (embeddings etc.)
    gsc_csv: list[Path] = field(default_factory=list)

    def site_model(self) -> dict:
        import json
        rows = self.db.query("SELECT json FROM site_model WHERE run_id=?", (self.run_id,))
        return json.loads(rows[0]["json"]) if rows else {}
