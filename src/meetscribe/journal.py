"""The live session's journal: every finished segment, written to disk at once.

One JSON object per line (JSONL), appended and flushed to disk after each write,
so a crash loses at most the utterance being transcribed. The first line holds
the metadata. When the session ends normally, the journal is removed: the json
transcript has everything. If it's still there, the session didn't finish and
'meetscribe recover' can rebuild the transcript from it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from meetscribe.backends.base import Segment
from meetscribe.transcript import Meta


class Journal:
    def __init__(self, path: Path, meta: Meta):
        self.path = path
        self._f = open(path, "x", encoding="utf-8")
        self._write({"type": "meta", **meta.to_dict()})

    def _write(self, obj: dict) -> None:
        self._f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self._f.flush()
        os.fsync(self._f.fileno())

    def add(self, segments: list[Segment]) -> None:
        for s in segments:
            self._write({"type": "segment", **s.to_dict()})

    def close(self, remove: bool = True) -> None:
        if not self._f.closed:
            self._f.close()
        if remove:
            self.path.unlink(missing_ok=True)


def read_journal(path: Path) -> tuple[Meta, list[Segment]]:
    meta = None
    segments = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                break  # a line cut off by the crash
            kind = obj.pop("type", None)
            if kind == "meta":
                meta = Meta.from_dict(obj)
            elif kind == "segment":
                segments.append(Segment.from_dict(obj))
    if meta is None:
        raise ValueError(f"{path.name} has no session metadata")
    return meta, segments
