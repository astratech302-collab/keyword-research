"""Run configuration. Everything tunable lives here so scoring stays explainable."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class LLMModels(BaseModel):
    # Any OpenRouter model id (https://openrouter.ai/models).
    fast: str = "google/gemini-3.8-flash"      # bulk: relevance filter, page classification
    smart: str = "anthropic/claude-opus-5.5"   # low volume: site summary, page mapping, competitor vetting
    temperature: float = 0.0


class Limits(BaseModel):
    max_crawl_pages: int = 100
    max_competitors: int = 5
    competitor_keywords_per_domain: int = 1000   # ranked keywords pulled per competitor (top-20 only)
    own_ranked_keywords: int = 2000
    keywords_for_site: int = 1000
    keyword_ideas: int = 1000
    suggestion_seeds: int = 10                   # seeds that also get keyword_suggestions
    suggestions_per_seed: int = 200
    max_candidates: int = 5000                   # cap after normalisation, before LLM relevance
    min_search_volume: int = 10
    serp_checks: int = 50                        # live SERP lookups for top clusters (0 = off)
    llm_mapping_clusters: int = 150              # clusters that get an LLM page-mapping judgement
    max_cost_usd: float = 10.0                   # hard stop for DataForSEO spend per run
    llm_workers: int = Field(default=12, ge=1, le=32)        # bounded OpenRouter concurrency
    dataforseo_workers: int = Field(default=5, ge=1, le=20)  # DataForSEO supports up to 30 concurrent DB calls


class ScoreWeights(BaseModel):
    # SEO opportunity (how winnable / how much traffic)
    demand: float = 0.30
    feasibility: float = 0.30
    existing_rank: float = 0.20
    competitor_gap: float = 0.10
    trend: float = 0.10
    # Business value (is the traffic worth anything to *this* business)
    relevance: float = 0.55
    intent: float = 0.30
    cpc: float = 0.15


class Thresholds(BaseModel):
    ignore_below_business_value: float = 35
    quick_win_positions: tuple[int, int] = (4, 30)
    quick_win_max_kd: float = 60
    strategic_min_kd: float = 60
    min_relevance_to_keep: int = 2               # LLM relevance 0-5; below this the keyword is dropped


DEFAULT_INTENT_VALUE = {
    "transactional": 100,
    "commercial": 85,
    "informational": 50,
    "navigational": 20,
}

DEFAULT_NEGATIVE_TERMS = [
    "job", "jobs", "salary", "salaries", "career", "careers", "internship", "hiring",
    "course", "courses", "certification", "certificate", "tutorial pdf", "pdf", "ppt",
    "free download", "download free", "meaning in hindi", "meaning in urdu", "in hindi",
    "interview questions", "resume", "wikipedia", "reddit", "quizlet", "login", "sign in",
]


class RunConfig(BaseModel):
    domain: str
    country: str = "India"                  # DataForSEO location_name
    language: str = "English"               # DataForSEO language_name
    language_code: str = "en"
    business_type: str = ""
    goal: str = "generate qualified leads"
    target_customer: str = ""
    products: list[str] = Field(default_factory=list)
    competitors: list[str] = Field(default_factory=list)       # user-supplied business competitors
    exclude_domains: list[str] = Field(default_factory=lambda: [
        "wikipedia.org", "youtube.com", "reddit.com", "quora.com", "linkedin.com", "medium.com",
        "amazon.com", "amazon.in", "facebook.com", "twitter.com", "x.com", "instagram.com",
        "github.com", "google.com", "microsoft.com", "forbes.com", "indeed.com", "glassdoor.com",
    ])
    exclude_keywords: list[str] = Field(default_factory=list)  # extra negative terms
    negative_terms: list[str] = Field(default_factory=lambda: list(DEFAULT_NEGATIVE_TERMS))
    intent_value: dict[str, float] = Field(default_factory=lambda: dict(DEFAULT_INTENT_VALUE))
    brand_terms: list[str] = Field(default_factory=list)       # your own brand names (navigational)

    llm: LLMModels = Field(default_factory=LLMModels)
    limits: Limits = Field(default_factory=Limits)
    weights: ScoreWeights = Field(default_factory=ScoreWeights)
    thresholds: Thresholds = Field(default_factory=Thresholds)

    # Regexes (matched against the URL path) for pages the crawler should skip,
    # e.g. faceted/price/discount collections on e-commerce sites.
    crawl_exclude: list[str] = Field(default_factory=list)

    output_dir: str = "runs"

    @field_validator("domain")
    @classmethod
    def _clean_domain(cls, v: str) -> str:
        v = v.strip().lower()
        for p in ("https://", "http://"):
            if v.startswith(p):
                v = v[len(p):]
        if v.startswith("www."):
            v = v[4:]
        return v.split("/")[0]

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.domain.replace(".", "_")

    @property
    def all_negative_terms(self) -> list[str]:
        return [t.lower() for t in self.negative_terms + self.exclude_keywords]


class Secrets(BaseModel):
    dataforseo_login: Optional[str] = None
    dataforseo_password: Optional[str] = None
    openrouter_api_key: Optional[str] = None

    @classmethod
    def from_env(cls) -> "Secrets":
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        return cls(
            dataforseo_login=os.getenv("DATAFORSEO_LOGIN"),
            dataforseo_password=os.getenv("DATAFORSEO_PASSWORD"),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
        )


def load_config(path: str | Path | None = None, **overrides) -> RunConfig:
    data: dict = {}
    if path:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    data.update({k: v for k, v in overrides.items() if v is not None})
    return RunConfig(**data)
