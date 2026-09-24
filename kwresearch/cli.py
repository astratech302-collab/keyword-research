"""kwresearch CLI.

  kwresearch run --config config.yaml              # full run
  kwresearch run --config config.yaml --resume     # continue the latest run (skips finished phases)
  kwresearch run --config config.yaml --resume --force mapping score   # redo some phases
  kwresearch run --domain acme.io --mock           # offline demo, no API keys needed
  kwresearch report --config config.yaml           # re-render the latest run's report
  kwresearch costs --config config.yaml            # list saved costs for every run (no API calls)
"""
from __future__ import annotations

import argparse
import logging
import shlex
import sys
from pathlib import Path

from .config import Secrets, load_config
from .context import Ctx
from .db import DB
from .gsc import read_csv
from .runner import PHASES, run_pipeline


def format_run_cost(summary: dict) -> str:
    label = "at least " if summary["unpriced_calls"] else ""
    parts = [f"{label}${summary['total_usd']:.6f} total"]
    parts += [f"{provider} ${data['known_usd']:.6f}"
              for provider, data in summary["providers"].items()]
    if summary["unpriced_calls"]:
        parts.append(f"{summary['unpriced_calls']} call(s) without a returned price")
    if summary["cache_hits"]:
        parts.append(f"{summary['cache_hits']} local cache hit(s)")
    return ", ".join(parts)


def build_ctx(args, *, report_only: bool = False) -> Ctx:
    cfg = load_config(args.config, domain=args.domain, country=getattr(args, "country", None),
                      language=getattr(args, "language", None))
    if getattr(args, "mock", False):
        cfg.output_dir = cfg.output_dir.rstrip("/") + "_mock"
    gsc_paths = [Path(p).expanduser().resolve() for p in getattr(args, "gsc_csv", [])]
    if not report_only:
        for path in gsc_paths:
            read_csv(path, cfg.domain)
    db = DB(cfg.run_dir / "kwresearch.db")

    if report_only or getattr(args, "resume", False):
        last = db.latest_run(cfg.domain)
        if not last:
            sys.exit(f"No previous run for {cfg.domain}")
        run_id = last["id"]
        if not report_only:
            db.resume_run(run_id)
    else:
        run_id = db.create_run(cfg.domain, cfg.model_dump_json())

    if getattr(args, "mock", False):
        from .mock import FakeDataForSEO, FakeLLM, fake_fetch
        dfs, llm = FakeDataForSEO(), FakeLLM()
        ctx = Ctx(cfg, db, run_id, dfs, llm)
        ctx.cache["fetch"] = fake_fetch(cfg.domain)
    elif report_only:
        ctx = Ctx(cfg, db, run_id, None, None)
    else:
        from .clients.dataforseo import DataForSEO
        from .clients.llm import LLM
        s = Secrets.from_env()
        dfs = DataForSEO(s.dataforseo_login, s.dataforseo_password, db, run_id, cfg.limits.max_cost_usd,
                         cfg.country, cfg.language, cfg.language_code)
        llm = LLM(s.openrouter_api_key, db, run_id, cfg.llm.temperature)
        ctx = Ctx(cfg, db, run_id, dfs, llm)

    if not report_only and not getattr(args, "no_embeddings", False):
        from .clients.embeddings import get_embedder
        ctx.embedder = get_embedder()
    ctx.gsc_csv = gsc_paths
    return ctx


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="kwresearch", description="SEO opportunity finder")
    sub = p.add_subparsers(dest="cmd", required=True)
    phase_names = [n for n, _ in PHASES]

    r = sub.add_parser("run", help="run the research pipeline")
    r.add_argument("--config", help="YAML config (see config.example.yaml)")
    r.add_argument("--domain")
    r.add_argument("--country")
    r.add_argument("--language")
    r.add_argument("--resume", action="store_true", help="continue the latest run for this domain")
    r.add_argument("--force", nargs="*", default=[], choices=phase_names, help="re-run these phases")
    r.add_argument("--only", nargs="*", choices=phase_names, help="run only these phases")
    r.add_argument("--no-embeddings", action="store_true", help="lexical clustering only")
    r.add_argument("--mock", action="store_true", help="offline fake data, no API calls")
    r.add_argument("--gsc-csv", action="append", default=[], metavar="PATH",
                   help="Google Search Console Queries or Pages CSV; repeat for both exports")
    r.add_argument("-v", "--verbose", action="store_true")

    rep = sub.add_parser("report", help="re-render the report for the latest run")
    rep.add_argument("--config")
    rep.add_argument("--domain")
    rep.add_argument("-v", "--verbose", action="store_true")

    costs = sub.add_parser("costs", help="show locally recorded cost for each run; no API calls")
    costs.add_argument("--config")
    costs.add_argument("--domain")
    costs.add_argument("--mock", action="store_true", help="show offline demo runs")
    costs.add_argument("--run-id", type=int, help="show only this run")

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "openai", "sentence_transformers", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    for noisy in ("huggingface_hub", "transformers"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    if not args.config and not args.domain:
        p.error("--config or --domain is required")

    if args.cmd == "run":
        try:
            ctx = build_ctx(args)
        except (OSError, ValueError) as exc:
            p.error(str(exc))
        try:
            res = run_pipeline(ctx, only=args.only, force=args.force)
        except (Exception, KeyboardInterrupt) as e:
            if args.verbose:
                logging.getLogger(__name__).exception("Run failed")
            cmd = ["kwresearch", "run"]
            if args.config:
                cmd += ["--config", args.config]
            else:
                cmd += ["--domain", args.domain]
                if args.country:
                    cmd += ["--country", args.country]
                if args.language:
                    cmd += ["--language", args.language]
            if args.no_embeddings:
                cmd.append("--no-embeddings")
            if args.mock:
                cmd.append("--mock")
            for path in args.gsc_csv:
                cmd += ["--gsc-csv", path]
            if args.only:
                cmd += ["--only", *args.only]
            cmd.append("--resume")
            print(f"\nRun interrupted: {e}", file=sys.stderr)
            print(f"Recorded cost so far: {format_run_cost(ctx.db.run_cost_summary(ctx.run_id))}", file=sys.stderr)
            print(f"Resume after resolving the issue:\n  {shlex.join(cmd)}", file=sys.stderr)
            raise SystemExit(130 if isinstance(e, KeyboardInterrupt) else 1) from None
        print(f"\nReport: {res.get('report', {}).get('report')}")
        print(f"Recorded API spend: {format_run_cost(ctx.db.run_cost_summary(ctx.run_id))}")
    elif args.cmd == "report":
        ctx = build_ctx(args, report_only=True)
        from .report.render import run as render
        print(render(ctx)["report"])
    else:
        cfg = load_config(args.config, domain=args.domain)
        if args.mock:
            cfg.output_dir = cfg.output_dir.rstrip("/") + "_mock"
        path = cfg.run_dir / "kwresearch.db"
        if not path.exists():
            p.error(f"No saved runs for {cfg.domain} at {path}")
        db = DB(path)
        rows = db.query("SELECT id, status FROM runs WHERE domain=? AND (? IS NULL OR id=?) "
                        "ORDER BY id DESC", (cfg.domain, args.run_id, args.run_id))
        if not rows:
            p.error(f"No matching runs for {cfg.domain}")
        for row in rows:
            print(f"run_{row['id']} ({row['status']}): {format_run_cost(db.run_cost_summary(row['id']))}")


if __name__ == "__main__":
    main()
