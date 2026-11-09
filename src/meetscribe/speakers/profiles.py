"""Voice profiles: named voices per meeting series, in <series folder>/speakers.json.

A profile is the average voice embedding of a person over the meetings where
they were named (with 'meetscribe speakers'). A new meeting's speakers are
matched to the profiles by cosine similarity; each profile is used at most once.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from meetscribe.transcript import write_atomic

FILE_NAME = "speakers.json"


@dataclass
class Profile:
    name: str
    embedding: np.ndarray
    meetings: int = 1
    updated: str = ""


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(_unit(np.asarray(a, dtype=np.float64)),
                        _unit(np.asarray(b, dtype=np.float64))))


def load_profiles(folder: Path) -> dict[str, Profile]:
    path = folder / FILE_NAME
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        name: Profile(name, np.asarray(p["embedding"], dtype=np.float32), int(p.get("meetings", 1)),
                      str(p.get("updated", "")))
        for name, p in data.get("speakers", {}).items()
    }


def save_profiles(folder: Path, profiles: dict[str, Profile]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    data = {
        "speakers": {
            p.name: {"meetings": p.meetings, "updated": p.updated,
                     "embedding": [round(float(x), 6) for x in p.embedding]}
            for p in sorted(profiles.values(), key=lambda p: p.name.lower())
        }
    }
    path = folder / FILE_NAME
    write_atomic(path, json.dumps(data, ensure_ascii=False, indent=1) + "\n")
    return path


def match(embeddings: dict[str, np.ndarray], profiles: dict[str, Profile],
          threshold: float) -> dict[str, tuple[str, float]]:
    """Speaker id -> (profile name, similarity) for every speaker that matches a
    profile with at least `threshold`. Best pairs first, one speaker per profile."""
    pairs = sorted(
        ((cosine(emb, p.embedding), sid, p.name)
         for sid, emb in embeddings.items() for p in profiles.values()),
        reverse=True,
    )
    result: dict[str, tuple[str, float]] = {}
    used: set[str] = set()
    for sim, sid, name in pairs:
        if sim < threshold:
            break
        if sid in result or name in used:
            continue
        result[sid] = (name, sim)
        used.add(name)
    return result


def remember(profiles: dict[str, Profile], name: str, embedding: np.ndarray) -> None:
    """Add a meeting's embedding of `name` to their profile (running average)."""
    emb = _unit(np.asarray(embedding, dtype=np.float32))
    today = date.today().isoformat()
    p = profiles.get(name)
    if p is None:
        profiles[name] = Profile(name, emb, 1, today)
        return
    # Older meetings count, but at most like 10 meetings, so voices can drift.
    weight = min(p.meetings, 10)
    p.embedding = _unit((_unit(p.embedding) * weight + emb) / (weight + 1)).astype(np.float32)
    p.meetings += 1
    p.updated = today
