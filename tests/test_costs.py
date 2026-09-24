from kwresearch import cli
from kwresearch.config import RunConfig
from kwresearch.db import DB


def test_costs_command_reads_saved_runs_only(tmp_path, monkeypatch, capsys):
    cfg = RunConfig(domain="example.com", output_dir=str(tmp_path))
    db = DB(cfg.run_dir / "kwresearch.db")
    first = db.create_run(cfg.domain, cfg.model_dump_json())
    db.log_call(first, "dataforseo", "/keywords", 0.0125, False)
    db.log_call(first, "dataforseo", "/keywords", 0, True)
    db.log_call(first, "openrouter", "model", 0.00123456, False)
    db.finish_run(first)
    second = db.create_run(cfg.domain, cfg.model_dump_json())
    db.log_call(second, "openrouter", "model", None, False)
    db.finish_run(second, "interrupted")
    monkeypatch.setattr(cli, "load_config", lambda *args, **kwargs: cfg)

    cli.main(["costs", "--domain", cfg.domain])
    output = capsys.readouterr().out
    assert "run_1 (done): $0.013735 total" in output
    assert "1 local cache hit(s)" in output
    assert "run_2 (interrupted): at least $0.000000 total" in output
    assert "1 call(s) without a returned price" in output

    cli.main(["costs", "--domain", cfg.domain, "--run-id", str(first)])
    output = capsys.readouterr().out
    assert "run_1" in output and "run_2" not in output
