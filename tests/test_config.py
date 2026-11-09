"""Config loading and validation (config.py)."""

from pathlib import Path

import pytest

from meetscribe.config import (
    PROJECT_DIR,
    Config,
    config_from_dict,
    load_config,
    parse_language_option,
)
from meetscribe.console import Fatal


def test_defaults():
    cfg = Config()
    cfg.validate()
    assert cfg.model == "large-v3-turbo"
    assert cfg.language == "auto"
    assert cfg.languages == ("de", "en")
    assert cfg.language_mode == "auto (de, en)"


def test_example_config_matches_defaults():
    """meetscribe.toml.example documents the defaults; keep them in sync."""
    cfg = load_config(PROJECT_DIR / "meetscribe.toml.example")
    defaults = Config()
    for name in Config.__dataclass_fields__:
        if name == "source":
            continue
        assert getattr(cfg, name) == getattr(defaults, name), name


def test_values_from_file(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('[transcription]\nlanguage = "de"\nlanguages = "de, en"\n'
                 '[output]\ndir = "~/Meetings"\nformats = ["md"]\n')
    cfg = load_config(p)
    assert cfg.language == "de"
    assert cfg.languages == ("de", "en")
    assert cfg.output_dir == Path.home() / "Meetings"
    assert cfg.formats == ("md",)
    assert cfg.source == p


@pytest.mark.parametrize("data, message", [
    ({"transcription": {"language": "deutsch"}}, "transcription.language"),
    ({"transcription": {"languages": ["de", "xx"]}}, "transcription.languages"),
    ({"output": {"formats": ["pdf"]}}, "output.formats"),
    ({"vad": {"threshold": 1.5}}, "vad.threshold"),
    ({"vad": {"min_silence_ms": "600"}}, "vad.min_silence_ms"),
    ({"live": {"keep_awake": "yes"}}, "live.keep_awake"),
    ({"vad": {"threshhold": 0.4}}, "Unknown setting vad.threshhold"),
    ({"outptu": {}}, "Unknown section [outptu]"),
])
def test_invalid(data, message):
    with pytest.raises(Fatal, match=message.replace("[", r"\[").replace(".", r"\.")):
        config_from_dict(data)


def test_missing_file(tmp_path):
    with pytest.raises(Fatal, match="Config file not found"):
        load_config(tmp_path / "nope.toml")


def test_syntax_error(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("[transcription\n")
    with pytest.raises(Fatal, match="Syntax error"):
        load_config(p)


def test_cli_overrides_win():
    cfg = config_from_dict({"transcription": {"model": "small"}})
    cfg = cfg.with_overrides(model="large-v3", language=None)
    assert cfg.model == "large-v3"
    assert cfg.language == "auto"
    with pytest.raises(Fatal):
        cfg.with_overrides(language="klingon")


def test_language_option():
    assert parse_language_option("de") == {"language": "de"}
    assert parse_language_option("auto") == {"language": "auto"}
    assert parse_language_option("de,en") == {"language": "auto", "languages": ("de", "en")}
    assert parse_language_option("DE, FR") == {"language": "auto", "languages": ("de", "fr")}
