"""Naming the speakers of a transcript (used by 'meetscribe speakers')."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from meetscribe.speakers.profiles import load_profiles, remember, save_profiles
from meetscribe.transcript import Transcript


def parse_assignment(text: str, t: Transcript) -> tuple[str, str | None]:
    """'2=Anna', 'S2=Anna' or 'Speaker 2=Anna' -> ('S2', 'Anna'); '2=' removes the name."""
    m = re.match(r"^\s*(?:s|speaker\s*)?(\d+)\s*=\s*(.*?)\s*$", text, re.IGNORECASE)
    if not m:
        raise ValueError(f"Expected NUMBER=NAME like 2=Anna, got '{text}'")
    sid = f"S{int(m.group(1))}"
    if sid not in t.speakers:
        known = ", ".join(str(sp.number) for sp in t.speakers.values())
        raise ValueError(f"There's no speaker {m.group(1)} in this transcript (speakers: {known})")
    return sid, m.group(2) or None


def apply_names(t: Transcript, names: dict[str, str | None]) -> list[tuple[str, str]]:
    """Set names. Speakers that end up with the same name are merged into the first
    one (the diarization split one person in two). Returns (kept, merged) labels."""
    for sid, name in names.items():
        sp = t.speakers[sid]
        if sp.name != name:
            sp.name = name
            sp.remembered = False
    merged = []
    by_name: dict[str, str] = {}
    for sid in list(t.speakers):
        sp = t.speakers[sid]
        if not sp.name:
            continue
        key = sp.name.casefold()
        if key not in by_name:
            by_name[key] = sid
            continue
        keep = t.speakers[by_name[key]]
        merged.append((keep.label, f"Speaker {sp.number}"))
        for seg in t.segments:
            if seg.speaker == sid:
                seg.speaker = keep.id
        if keep.embedding is not None and sp.embedding is not None:
            keep.embedding = (np.asarray(keep.embedding) + np.asarray(sp.embedding)).tolist()
        keep.remembered = keep.remembered and sp.remembered
        del t.speakers[sid]
    return merged


def remember_voices(t: Transcript, folder: Path) -> tuple[list[str], Path]:
    """Add every named speaker's voice to the series' profiles, once per meeting."""
    profiles = load_profiles(folder)
    saved = []
    for sp in t.speakers.values():
        if sp.name and sp.embedding is not None and not sp.remembered:
            remember(profiles, sp.name, np.asarray(sp.embedding, dtype=np.float32))
            sp.remembered = True
            saved.append(sp.name)
    path = folder / "speakers.json"
    if saved:
        path = save_profiles(folder, profiles)
    return saved, path
