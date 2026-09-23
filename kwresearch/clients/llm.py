"""OpenRouter chat client (OpenAI-compatible) returning parsed JSON. Cached + cost-logged."""
from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import time
from typing import Any

from ..db import DB

log = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1"


class LLMError(RuntimeError):
    pass


class LLM:
    def __init__(self, api_key: str, db: DB, run_id: int | None = None, temperature: float = 0.0,
                 use_cache: bool = True):
        if not api_key:
            raise LLMError("OPENROUTER_API_KEY is not set")
        from openai import OpenAI
        self.client = OpenAI(base_url=OPENROUTER_URL, api_key=api_key, default_headers={
            "HTTP-Referer": "https://github.com/kwresearch", "X-Title": "kwresearch"})
        self.db = db
        self.run_id = run_id
        self.temperature = temperature
        self.use_cache = use_cache

    def json(self, model: str, system: str, user: str, attempts: int = 3) -> Any:
        key = "llm:" + hashlib.sha256(f"{model}\n{system}\n{user}".encode()).hexdigest()
        if self.use_cache:
            rows = self.db.query("SELECT response_json FROM api_cache WHERE key=?", (key,))
            if rows:
                self.db.log_call(self.run_id, "openrouter", model, 0.0, True)
                log.info("OpenRouter cache hit: %s", model)
                return json.loads(rows[0]["response_json"])

        last_err: Exception | None = None
        use_format = True
        for i in range(attempts):
            try:
                log.info("OpenRouter request: %s (attempt %d/%d)", model, i + 1, attempts)
                kwargs: dict[str, Any] = dict(
                    model=model, temperature=self.temperature,
                    messages=[{"role": "system", "content": system + "\nRespond with valid JSON only."},
                              {"role": "user", "content": user}],
                    extra_body={"usage": {"include": True}},
                )
                if use_format:
                    kwargs["response_format"] = {"type": "json_object"}
                resp = self.client.chat.completions.create(**kwargs)
                text = resp.choices[0].message.content or ""
                cost = float(getattr(resp.usage, "cost", 0.0) or 0.0) if resp.usage else 0.0
                self.db.log_call(self.run_id, "openrouter", model, cost, False)
                data = extract_json(text)
                self.db.execute("INSERT OR REPLACE INTO api_cache VALUES (?,?,?,?)",
                                (key, model, time.time(), json.dumps(data)))
                self.db.commit()
                log.info("OpenRouter response: %s cost=$%.4f", model, cost)
                return data
            except Exception as e:  # noqa: BLE001 - provider errors vary
                last_err = e
                if "response_format" in str(e):
                    use_format = False
                log.warning("LLM call failed (%s), attempt %d/%d", e, i + 1, attempts)
                time.sleep((2 ** (i + 1)) + random.uniform(0, 0.5))
        raise LLMError(f"LLM call failed after {attempts} attempts: {last_err}")


def extract_json(text: str) -> Any:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        s, e = text.find(open_c), text.rfind(close_c)
        if s != -1 and e > s:
            try:
                return json.loads(text[s:e + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"No JSON in LLM output: {text[:200]}")


def batched(seq: list, n: int):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]
