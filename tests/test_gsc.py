import csv

import pytest

from kwresearch.gsc import read_csv
from kwresearch.config import RunConfig
from kwresearch.pipeline.candidates import pool
from kwresearch.pipeline.site import crawl
from kwresearch.runner import run_pipeline
from test_pipeline import make_ctx


def write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([header, "Clicks", "Impressions", "CTR", "Position"])
        writer.writerows(rows)


def test_gsc_csv_detects_queries_and_domain_pages(tmp_path):
    queries = tmp_path / "queries.csv"
    write_csv(queries, "Top queries", [["Data Governance Audit Tools", "2", "1,000", "0.20%", "13.5"]])
    kind, rows = read_csv(queries, "acmelineage.io")
    assert kind == "queries"
    assert rows[0]["key"] == "data governance audit tools"
    assert rows[0]["impressions"] == 1000
    assert rows[0]["ctr"] == pytest.approx(0.002)

    pages = tmp_path / "pages.csv"
    write_csv(pages, "Top pages", [["https://docs.acmelineage.io/guide/#chapter", 2, 100, "2%", 8]])
    assert read_csv(pages, "acmelineage.io")[0] == "pages"
    write_csv(pages, "Top pages", [["https://other.example/", 2, 100, "2%", 8]])
    with pytest.raises(ValueError, match="outside acmelineage.io"):
        read_csv(pages, "acmelineage.io")


def test_gsc_queries_flow_through_keyword_pipeline(tmp_path):
    queries = tmp_path / "queries.csv"
    write_csv(queries, "Top queries", [["data governance audit tools", 12, 1000, "1.20%", 14]])
    pages = tmp_path / "pages.csv"
    write_csv(pages, "Top pages", [["https://acmelineage.io/product/data-catalog", 7, 400, "1.75%", 17]])
    ctx = make_ctx(tmp_path)
    ctx.gsc_csv = [queries, pages]
    result = run_pipeline(ctx)
    assert result["gsc"]["files"] == 2
    assert result["gsc"]["queries"] == 1
    assert result["gsc"]["pages"] == 1
    keyword = pool(ctx)["data governance audit tools"]
    assert "gsc_queries" in keyword["sources_json"]
    assert keyword["kept"] == 1
    assert ctx.db.query("SELECT impressions FROM gsc_queries WHERE run_id=?", (ctx.run_id,))[0]["impressions"] == 1000
    assert any("data governance audit tools" in c["keywords"] and c["gsc_impressions"] >= 1000
               and "gsc_opportunity" in c["subscores"]
               for c in ctx.db.clusters(ctx.run_id))
    crawled = [p["url"] for p in crawl(ctx, ctx.cache["fetch"])]
    assert crawled.index("https://acmelineage.io/product/data-catalog") < crawled.index(
        "https://acmelineage.io/product/data-lineage")
    with (ctx.cfg.run_dir / f"run_{ctx.run_id}" / "keywords.csv").open() as handle:
        exported = {r["keyword"]: r for r in csv.DictReader(handle)}
    assert exported["data governance audit tools"]["gsc_impressions"] == "1000"
    assert exported["data governance audit tools"]["gsc_position"] == "14.0"
    assert set(run_pipeline(ctx)) == {"report"}
    write_csv(queries, "Top queries", [["data governance audit tools", 13, 1000, "1.30%", 14]])
    with pytest.raises(ValueError, match="differs from this run"):
        run_pipeline(ctx)


def test_cli_accepts_gsc_csv_path(tmp_path, monkeypatch):
    from kwresearch import cli

    queries = tmp_path / "queries.csv"
    write_csv(queries, "Top queries", [["data governance audit tools", 12, 1000, "1.20%", 14]])
    monkeypatch.setattr(cli, "load_config", lambda *args, **kwargs: RunConfig(
        domain="acmelineage.io", output_dir=str(tmp_path / "runs")))
    cli.main(["run", "--domain", "acmelineage.io", "--mock", "--no-embeddings",
              "--gsc-csv", str(queries)])
    with (tmp_path / "runs_mock" / "acmelineage_io" / "run_1" / "keywords.csv").open() as handle:
        rows = {r["keyword"]: r for r in csv.DictReader(handle)}
    assert rows["data governance audit tools"]["gsc_impressions"] == "1000"
