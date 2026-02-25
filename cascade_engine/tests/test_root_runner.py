from __future__ import annotations

from pathlib import Path

from runner import _rewrite_config_path_arg


def test_rewrite_uses_existing_argument(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = Path("config.json")
    config.write_text("{}", encoding="utf-8")

    argv = ["runner.py", "config.json"]
    assert _rewrite_config_path_arg(argv) == argv


def test_rewrite_falls_back_to_package_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    package_dir = Path("cascade_engine")
    package_dir.mkdir()
    (package_dir / "config.json").write_text("{}", encoding="utf-8")

    rewritten = _rewrite_config_path_arg(["runner.py", "config.json"])
    assert rewritten == ["runner.py", "cascade_engine/config.json"]


def test_rewrite_keeps_unknown_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    argv = ["runner.py", "missing.json"]
    assert _rewrite_config_path_arg(argv) == argv
