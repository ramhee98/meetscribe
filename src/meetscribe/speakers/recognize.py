"""Speaker recognition for a finished transcript: diarize, align, name from profiles."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from meetscribe import console
from meetscribe.console import format_duration
from meetscribe.speakers.align import assign_speakers, number_speakers
from meetscribe.speakers.diarize import Diarizer
from meetscribe.speakers.profiles import load_profiles, match
from meetscribe.transcript import Speaker, Transcript

STEP_NAMES = {
    "segmentation": "finding speech",
    "speaker_counting": "counting speakers",
    "embeddings": "analysing voices",
    "discrete_diarization": "assigning turns",
}


@dataclass
class Result:
    seconds: float
    speakers: int
    named: dict[str, tuple[str, float]]  # speaker id -> (name, similarity)
    profiles_dir: Path


def recognize(transcript: Transcript, audio: np.ndarray, diarizer: Diarizer,
              profiles_dir: Path, match_threshold: float) -> Result:
    """Add speakers to the transcript in place. Shows a progress line."""
    t0 = time.monotonic()

    def progress(step: str, fraction: float) -> None:
        label = STEP_NAMES.get(step, step)
        console.set_status(f"{console.BLUE}==>{console.OFF} Recognizing speakers: {label} "
                           f"{int(fraction * 100)}% ({format_duration(time.monotonic() - t0)})")

    if not console._tty(console.sys.stdout):
        console.info("Recognizing speakers...")
    try:
        result = diarizer.run(audio, progress)
    finally:
        console.clear_status()

    segments = assign_speakers(transcript.segments, result.turns)
    ids = number_speakers(segments)
    transcript.segments = segments
    embeddings = {ids[label]: emb for label, emb in result.embeddings.items() if label in ids}
    named = match(embeddings, load_profiles(profiles_dir), match_threshold)
    transcript.speakers = {
        sid: Speaker(sid, named[sid][0] if sid in named else None,
                     embeddings[sid].tolist() if sid in embeddings else None)
        for sid in sorted(ids.values(), key=lambda s: int(s[1:]))
    }
    return Result(time.monotonic() - t0, len(ids), named, profiles_dir)


def report(transcript: Transcript, result: Result) -> None:
    console.ok(f"Recognized {result.speakers} speaker{'s' if result.speakers != 1 else ''} "
               f"({format_duration(result.seconds)})")
    times = transcript.talk_time()
    total = sum(times.values()) or 1.0
    for sid, secs in times.items():
        sp = transcript.speakers[sid]
        how = ""
        if sid in result.named:
            how = f"  {console.DIM}(recognized, similarity {result.named[sid][1]:.2f}){console.OFF}"
        console.detail(sp.label, f"{round(100 * secs / total)}% ({format_duration(secs)}){how}")
