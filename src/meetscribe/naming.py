"""Where transcripts and recordings go and how they're named.

    <output dir>/<series>/<YYYY-MM-DD>_<HH-MM>_<series>.{md,txt,json,wav}

Without a meeting name, files go straight into the output folder and have no
name suffix. Pure path logic, tested in tests/test_naming.py.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from pathlib import Path

STEM_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2})(?:_(.+?))?(?:-(\d+))?$")
# Every file a session can have. Used to avoid name collisions.
SESSION_SUFFIXES = (".md", ".txt", ".json", ".jsonl", ".wav")

_GERMAN = {"ä": "ae", "ö": "oe", "ü": "ue", "Ä": "Ae", "Ö": "Oe", "Ü": "Ue", "ß": "ss"}


def slugify(name: str) -> str:
    """'Weekly Team / Jour fixe' -> 'weekly-team-jour-fixe', 'Büro' -> 'buero'."""
    text = "".join(_GERMAN.get(c, c) for c in name)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def series_dir(output_dir: Path, series: str | None) -> Path:
    return output_dir / series if series else output_dir


def session_stem(start: datetime, label: str | None) -> str:
    stem = start.strftime("%Y-%m-%d_%H-%M")
    return f"{stem}_{label}" if label else stem


def new_session_base(output_dir: Path, series: str | None, start: datetime,
                     label: str | None = None) -> Path:
    """Path without extension for a new session, in the series folder. The name
    ends with label (default: the series). Never collides with an existing
    session: a second meeting in the same minute gets -2, then -3, ..."""
    folder = series_dir(output_dir, series)
    stem = session_stem(start, label or series)
    candidate, n = folder / stem, 1
    while any(candidate.with_name(candidate.name + s).exists() for s in SESSION_SUFFIXES):
        n += 1
        candidate = folder / f"{stem}-{n}"
    return candidate


def parse_stem(stem: str) -> tuple[datetime, str | None] | None:
    """('2026-11-09_14-00_weekly-team') -> (datetime(2026,11,9,14,0), 'weekly-team'),
    or None if it isn't a meetscribe file name."""
    m = STEM_RE.match(stem)
    if not m:
        return None
    try:
        start = datetime.strptime(f"{m.group(1)}_{m.group(2)}", "%Y-%m-%d_%H-%M")
    except ValueError:
        return None
    return start, m.group(3)


def series_title(series: str | None) -> str:
    """'weekly-team' -> 'Weekly Team' (for headings when no title is given)."""
    if not series:
        return "Meeting"
    return " ".join(w.capitalize() for w in series.split("-"))


def find_prompt(output_dir: Path, series: str | None) -> tuple[str, Path] | None:
    """The nearest prompt.txt: the series folder first, then the output folder."""
    folders = [series_dir(output_dir, series), output_dir] if series else [output_dir]
    for folder in folders:
        p = folder / "prompt.txt"
        if p.is_file():
            text = " ".join(p.read_text(encoding="utf-8").split())
            if text:
                return text, p
    return None
