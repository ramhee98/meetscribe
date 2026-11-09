"""A transcript, its metadata, and writing it as md, txt and json.

The json file is always written. It holds the segments and metadata in a form
meetscribe can read back (re-rendering, and later diarization and minutes).
md and txt are for people. All files are written atomically: a temporary file
next to the target, then renamed, so a crash never leaves a half-written file.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from meetscribe import __version__
from meetscribe.backends.base import Segment
from meetscribe.console import format_duration, format_timestamp

PARAGRAPH_GAP_S = 2.0  # a pause this long starts a new paragraph
PARAGRAPH_MAX_S = 60.0  # and so does a paragraph getting this long

_UMASK = os.umask(0)
os.umask(_UMASK)

RECORDING, COMPLETE, RECOVERED = "recording", "complete", "recovered"


@dataclass
class Meta:
    title: str
    series: str | None
    start: datetime
    mode: str  # live or file
    backend: str
    model: str
    language: str  # the setting, e.g. "auto (de, en)"
    duration: float = 0.0
    device: str | None = None
    audio: str | None = None  # file name of the recording, next to the transcript
    source: str | None = None  # original file in file mode
    status: str = COMPLETE

    def to_dict(self) -> dict:
        d = asdict(self)
        d["start"] = self.start.isoformat(timespec="seconds")
        d["duration"] = round(self.duration, 2)
        d["meetscribe"] = __version__
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Meta:
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        known["start"] = datetime.fromisoformat(d["start"])
        return cls(**known)


@dataclass
class Speaker:
    id: str  # S1, S2, ... in order of first appearance
    name: str | None = None  # set by voice profiles or 'meetscribe speakers'
    embedding: list[float] | None = None  # voice embedding, for naming later
    remembered: bool = False  # this meeting's voice is in the series' voice profiles

    @property
    def number(self) -> int:
        return int(self.id[1:]) if self.id[1:].isdigit() else 0

    @property
    def label(self) -> str:
        return self.name or f"Speaker {self.number}"

    def to_dict(self) -> dict:
        d: dict = {"name": self.name}
        if self.remembered:
            d["remembered"] = True
        if self.embedding is not None:
            d["embedding"] = [round(float(x), 6) for x in self.embedding]
        return d


@dataclass
class Transcript:
    meta: Meta
    segments: list[Segment] = field(default_factory=list)
    speakers: dict[str, Speaker] = field(default_factory=dict)

    def speaker_label(self, sid: str | None) -> str | None:
        if not sid:
            return None
        sp = self.speakers.get(sid)
        return sp.label if sp else sid

    def talk_time(self) -> dict[str, float]:
        """Speaking time per speaker id, most first."""
        times: Counter[str] = Counter()
        for s in self.segments:
            if s.speaker:
                times[s.speaker] += max(s.end - s.start, 0.0)
        return dict(times.most_common())

    @property
    def languages(self) -> list[str]:
        """Languages by number of words, most used first."""
        words: Counter[str] = Counter()
        for s in self.segments:
            if s.language:
                words[s.language] += len(s.text.split())
        return [lang for lang, _ in words.most_common()]

    @property
    def word_count(self) -> int:
        return sum(len(s.text.split()) for s in self.segments)

    def paragraphs(self) -> list[list[Segment]]:
        out: list[list[Segment]] = []
        for s in self.segments:
            if out:
                cur = out[-1]
                if (s.start - cur[-1].end < PARAGRAPH_GAP_S
                        and s.language == cur[-1].language
                        and s.speaker == cur[-1].speaker
                        and s.end - cur[0].start <= PARAGRAPH_MAX_S):
                    cur.append(s)
                    continue
            out.append([s])
        return out

    def to_dict(self) -> dict:
        d: dict = {"meta": self.meta.to_dict()}
        if self.speakers:
            d["speakers"] = {sid: sp.to_dict() for sid, sp in self.speakers.items()}
        d["segments"] = [s.to_dict() for s in self.segments]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Transcript:
        speakers = {sid: Speaker(sid, sp.get("name"), sp.get("embedding"),
                                 bool(sp.get("remembered")))
                    for sid, sp in d.get("speakers", {}).items()}
        return cls(Meta.from_dict(d["meta"]), [Segment.from_dict(s) for s in d["segments"]],
                   speakers)


# --- Rendering ----------------------------------------------------------------


def _heading(meta: Meta) -> str:
    return f"{meta.title} – {meta.start:%Y-%m-%d %H:%M}"


def _when(meta: Meta) -> str:
    end = meta.start + timedelta(seconds=meta.duration)
    return f"{meta.start:%A, %-d %B %Y, %H:%M}–{end:%H:%M}"


def _model(meta: Meta) -> str:
    return f"{meta.model} ({meta.backend})"


def _language_line(t: Transcript) -> str:
    return ", ".join(t.languages) or t.meta.language


def _speakers_line(t: Transcript) -> str | None:
    """'Anna 45%, Speaker 2 30%, Ramon 25%' (share of the speaking time)."""
    times = t.talk_time()
    total = sum(times.values())
    if not total:
        return None
    return ", ".join(f"{t.speaker_label(sid)} {round(100 * v / total)}%"
                     for sid, v in times.items())


def _status_note(meta: Meta) -> str | None:
    if meta.status == RECORDING:
        return (f"Recording in progress since {meta.start:%H:%M}. "
                "This file is updated as the meeting goes on.")
    if meta.status == RECOVERED:
        return ("Recovered after an interruption: the end of the meeting may be missing. "
                "Run `meetscribe file` on the recording for a complete transcript.")
    return None


def _para_text(t: Transcript, para: list[Segment], main: str | None, md: bool) -> str:
    text = " ".join(s.text for s in para)
    lang = para[0].language
    if main and lang and lang != main:
        text = (f"*({lang})* " if md else f"({lang}) ") + text
    return text


def _para_head(t: Transcript, para: list[Segment], md: bool) -> str:
    """'**[00:01:23] Anna:**' / '[00:01:23] Anna:', or just the time without speakers."""
    stamp = f"[{format_timestamp(para[0].start)}]"
    who = t.speaker_label(para[0].speaker)
    head = f"{stamp} {who}:" if who else stamp
    return f"**{head}**" if md else head


def render_md(t: Transcript) -> str:
    m = t.meta
    lines = [f"# {_heading(m)}", ""]
    lines.append(f"- **Date:** {_when(m)}")
    lines.append(f"- **Duration:** {format_duration(m.duration)}")
    lines.append(f"- **Language:** {_language_line(t)}")
    speakers = _speakers_line(t)
    if speakers:
        lines.append(f"- **Speakers:** {speakers}")
    lines.append(f"- **Model:** {_model(m)}")
    if m.audio:
        lines.append(f"- **Recording:** [{m.audio}]({m.audio.replace(' ', '%20')})")
    elif m.source:
        lines.append(f"- **Source:** {m.source}")
    note = _status_note(m)
    if note:
        lines += ["", f"> {note}"]
    lines += ["", "## Transcript", ""]
    main = t.languages[0] if len(t.languages) > 1 else None
    for para in t.paragraphs():
        lines.append(f"{_para_head(t, para, True)} {_para_text(t, para, main, True)}")
        lines.append("")
    if not t.segments:
        lines += ["*No speech transcribed yet.*", ""]
    return "\n".join(lines)


def render_txt(t: Transcript) -> str:
    m = t.meta
    lines = [_heading(m), f"{_when(m)} · {format_duration(m.duration)}",
             f"Language: {_language_line(t)} · Model: {_model(m)}"]
    speakers = _speakers_line(t)
    if speakers:
        lines.append(f"Speakers: {speakers}")
    note = _status_note(m)
    if note:
        lines.append(note)
    lines.append("")
    main = t.languages[0] if len(t.languages) > 1 else None
    for para in t.paragraphs():
        lines.append(f"{_para_head(t, para, False)} {_para_text(t, para, main, False)}")
        lines.append("")
    return "\n".join(lines)


def render_json(t: Transcript) -> str:
    return json.dumps(t.to_dict(), ensure_ascii=False, indent=2) + "\n"


RENDERERS = {"md": render_md, "txt": render_txt}


def write_atomic(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".partial")
    try:
        # mkstemp creates the file private (0600); use normal permissions instead.
        os.fchmod(fd, 0o666 & ~_UMASK)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def output_paths(base: Path, formats: tuple[str, ...]) -> list[Path]:
    """All files write_outputs creates for base: the formats plus json."""
    return [base.with_name(base.name + f".{f}") for f in (*formats, "json")]


def write_outputs(base: Path, t: Transcript, formats: tuple[str, ...]) -> list[Path]:
    written = []
    for fmt in formats:
        path = base.with_name(base.name + f".{fmt}")
        write_atomic(path, RENDERERS[fmt](t))
        written.append(path)
    path = base.with_name(base.name + ".json")
    write_atomic(path, render_json(t))
    written.append(path)
    return written


def read_transcript(path: Path) -> Transcript:
    return Transcript.from_dict(json.loads(path.read_text(encoding="utf-8")))
