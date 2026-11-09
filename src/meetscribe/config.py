"""Settings: built-in defaults -> config file -> command-line options.

The config file is TOML. Every value is checked when it's loaded; an invalid
value or an unknown key stops with an error naming the setting and the file.
Keep DEFAULTS in sync with meetscribe.toml.example.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from meetscribe.console import die

# The repository folder, when running from a checkout (uv run / uv tool install -e).
PROJECT_DIR = Path(__file__).resolve().parents[2]
USER_CONFIG = Path.home() / ".config" / "meetscribe" / "config.toml"

OUTPUT_FORMATS = ("md", "txt")
BACKENDS = ("mlx",)
# Languages Whisper knows (from its tokenizer).
_WHISPER_LANGUAGES = (
    "en zh de es ru ko fr ja pt tr pl ca nl ar sv it id hi fi vi he uk el ms cs ro da hu ta no "
    "th ur hr bg lt la mi ml cy sk te fa lv bn sr az sl kn et mk br eu is hy ne mn bs kk sq sw "
    "gl mr pa si km sn yo so af oc ka be tg sd gu am yi lo uz fo ht ps tk nn mt sa lb my bo tl "
    "mg as tt haw ln ha ba jw su yue"
)
WHISPER_LANGUAGES = frozenset(_WHISPER_LANGUAGES.split())


def _is_lang(code: str) -> bool:
    return code in WHISPER_LANGUAGES


@dataclass(frozen=True)
class Config:
    # [transcription]
    backend: str = "mlx"
    model: str = "large-v3-turbo"
    language: str = "auto"
    languages: tuple[str, ...] = ("de", "en")
    prompt: str = ""
    # [audio]
    device: str = ""
    # [output]
    output_dir: Path = field(default_factory=lambda: Path.home() / "Documents" / "meetscribe")
    formats: tuple[str, ...] = ("md", "txt")
    overwrite: bool = False
    # [vad]
    vad_threshold: float = 0.5
    min_speech_ms: int = 250
    min_silence_ms: int = 600
    speech_pad_ms: int = 200
    live_max_utterance_s: float = 20.0
    file_max_utterance_s: float = 30.0
    # [live]
    keep_awake: bool = True

    source: Path | None = None  # the config file that was loaded, if any

    def with_overrides(self, **overrides: Any) -> Config:
        """Apply command-line values (None = not given) and validate again."""
        given = {k: v for k, v in overrides.items() if v is not None}
        cfg = replace(self, **given)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        for s in SETTINGS:
            s.check(self, getattr(self, s.attr))

    @property
    def language_mode(self) -> str:
        """Human-readable language setting, e.g. 'de', 'auto', or 'auto (de, en)'."""
        if self.language != "auto":
            return self.language
        return f"auto ({', '.join(self.languages)})" if self.languages else "auto"


# --- Schema -------------------------------------------------------------------


@dataclass(frozen=True)
class Setting:
    section: str
    key: str
    attr: str
    convert: Callable[[Any], Any]
    test: Callable[[Any], bool]
    expected: str

    @property
    def name(self) -> str:
        return f"{self.section}.{self.key}"

    def check(self, cfg: Config, value: Any) -> None:
        if not self.test(value):
            fail(cfg.source, self.name, value, self.expected)


class _BadType(Exception):
    pass


def _bool(v: Any) -> bool:
    if not isinstance(v, bool):
        raise _BadType
    return v


def _str(v: Any) -> str:
    if not isinstance(v, str):
        raise _BadType
    return v


def _int(v: Any) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise _BadType
    return v


def _float(v: Any) -> float:
    if isinstance(v, bool) or not isinstance(v, int | float):
        raise _BadType
    return float(v)


def _str_list(v: Any) -> tuple[str, ...]:
    if isinstance(v, str):  # also accept "de, en"
        v = [p for p in re.split(r"[,\s]+", v) if p]
    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
        raise _BadType
    return tuple(x.strip().lower() for x in v)


def _path(v: Any) -> Path:
    return Path(_str(v)).expanduser()


def _single_line(v: str) -> bool:
    return "\n" not in v


SETTINGS: tuple[Setting, ...] = (
    Setting("transcription", "backend", "backend", _str,
            lambda v: v in BACKENDS, "supported: " + ", ".join(BACKENDS)),
    Setting("transcription", "model", "model", _str,
            lambda v: bool(v.strip()) and _single_line(v),
            "a model name like large-v3-turbo, a Hugging Face repo or a local folder"),
    Setting("transcription", "language", "language", _str,
            lambda v: v == "auto" or _is_lang(v),
            "a language code like de or en, or auto"),
    Setting("transcription", "languages", "languages", _str_list,
            lambda v: all(_is_lang(x) for x in v),
            "a list of language codes like [\"de\", \"en\"], or [] for any language"),
    Setting("transcription", "prompt", "prompt", _str, lambda v: True, "text"),
    Setting("audio", "device", "device", _str, _single_line,
            "part of a device name, a device number, or \"\" for the system default"),
    Setting("output", "dir", "output_dir", _path, lambda v: str(v) != "", "a folder path"),
    Setting("output", "formats", "formats", _str_list,
            lambda v: all(x in OUTPUT_FORMATS for x in v),
            "a list of formats, supported: " + ", ".join(OUTPUT_FORMATS)),
    Setting("output", "overwrite", "overwrite", _bool, lambda v: True, "true or false"),
    Setting("vad", "threshold", "vad_threshold", _float, lambda v: 0 < v < 1,
            "a number between 0 and 1, like 0.5"),
    Setting("vad", "min_speech_ms", "min_speech_ms", _int, lambda v: v >= 0,
            "milliseconds, like 250"),
    Setting("vad", "min_silence_ms", "min_silence_ms", _int, lambda v: 100 <= v <= 5000,
            "milliseconds between 100 and 5000, like 600"),
    Setting("vad", "speech_pad_ms", "speech_pad_ms", _int, lambda v: 0 <= v <= 1000,
            "milliseconds between 0 and 1000, like 200"),
    Setting("vad", "live_max_utterance_s", "live_max_utterance_s", _float,
            lambda v: 3 <= v <= 30, "seconds between 3 and 30, like 20"),
    Setting("vad", "file_max_utterance_s", "file_max_utterance_s", _float,
            lambda v: 3 <= v <= 30, "seconds between 3 and 30, like 30"),
    Setting("live", "keep_awake", "keep_awake", _bool, lambda v: True, "true or false"),
)


def fail(source: Path | None, name: str, value: Any, expected: str) -> None:
    shown = value if not isinstance(value, Path) else str(value)
    if isinstance(shown, tuple):
        shown = list(shown)
    where = f"\nCheck {source}" if source else ""
    die(f"Invalid setting {name} = {shown!r} ({expected}){where}")


# --- Loading ------------------------------------------------------------------


def default_config_path() -> Path | None:
    """meetscribe.toml in the project folder, else ~/.config/meetscribe/config.toml."""
    for candidate in (PROJECT_DIR / "meetscribe.toml", USER_CONFIG):
        if candidate.is_file():
            return candidate
    return None


def load_config(path: Path | None = None) -> Config:
    """Defaults, then the given config file (or the default one if it exists)."""
    if path is not None:
        path = path.expanduser()
        if not path.is_file():
            die(f"Config file not found: {path}")
    else:
        path = default_config_path()
    if path is None:
        cfg = Config()
        cfg.validate()
        return cfg

    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as e:
        die(f"Syntax error in config file: {path}\n{e}")
    return config_from_dict(data, path)


def config_from_dict(data: dict[str, Any], source: Path | None = None) -> Config:
    by_section: dict[str, dict[str, Setting]] = {}
    for s in SETTINGS:
        by_section.setdefault(s.section, {})[s.key] = s

    values: dict[str, Any] = {}
    where = f"\nCheck {source}" if source else ""
    for section, table in data.items():
        if section not in by_section:
            die(f"Unknown section [{section}] in config file{where}\n"
                f"Known sections: {', '.join(by_section)}")
        if not isinstance(table, dict):
            die(f"Expected [{section}] to be a section with settings{where}")
        for key, raw in table.items():
            setting = by_section[section].get(key)
            if setting is None:
                die(f"Unknown setting {section}.{key}{where}\n"
                    f"Known settings in [{section}]: {', '.join(by_section[section])}")
            try:
                values[setting.attr] = setting.convert(raw)
            except _BadType:
                fail(source, setting.name, raw, setting.expected)

    cfg = Config(source=source, **values)
    cfg.validate()
    return cfg


def parse_language_option(value: str) -> dict[str, Any]:
    """--language: 'de' -> fixed, 'auto' -> auto, 'de,en' -> auto limited to those."""
    parts = [p for p in re.split(r"[,\s]+", value.strip().lower()) if p]
    if len(parts) > 1:
        return {"language": "auto", "languages": tuple(parts)}
    if parts == ["auto"]:
        return {"language": "auto"}
    return {"language": parts[0] if parts else value}
