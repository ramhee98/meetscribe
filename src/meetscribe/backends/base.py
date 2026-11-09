"""The interface every transcription backend implements.

A backend transcribes one short piece of 16 kHz mono audio (an utterance, at
most 30 s) at a time. Everything else (cutting audio, picking the language,
prompts, filtering, timestamps on the meeting timeline) happens in
meetscribe.pipeline, so a new backend (e.g. faster-whisper on Linux) only needs
these few methods.

Backends may need to run on the thread that loaded them (MLX does), so load()
and all calls must happen on the same thread.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Word:
    start: float
    end: float
    text: str  # as Whisper writes it, usually with a leading space


@dataclass
class Segment:
    start: float  # seconds
    end: float
    text: str
    language: str | None = None
    no_speech_prob: float = 0.0
    avg_logprob: float = 0.0
    speaker: str | None = None  # id like "S1", see transcript.Speaker
    # Word timings, only when speaker recognition needs them. Kept in memory,
    # not saved: they're used to split segments where the speaker changes.
    words: list[Word] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "text": self.text,
            "language": self.language,
        }
        if self.speaker:
            d["speaker"] = self.speaker
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Segment:
        return cls(float(d["start"]), float(d["end"]), str(d["text"]), d.get("language"),
                   speaker=d.get("speaker"))


class Transcriber(ABC):
    name: str = ""

    def __init__(self, model: str):
        self.model = model

    @property
    def model_id(self) -> str:
        """The resolved model (repo or path), for the transcript's metadata."""
        return self.model

    @abstractmethod
    def load(self) -> None:
        """Load the model (downloading it if needed) and warm it up."""

    @abstractmethod
    def language_probs(self, audio: np.ndarray) -> dict[str, float]:
        """Probability of each language code for this audio."""

    @abstractmethod
    def transcribe(self, audio: np.ndarray, language: str, prompt: str | None,
                   words: bool = False) -> list[Segment]:
        """Segments with times relative to the start of `audio`, with word timings
        if words is True."""


def get_backend(name: str, model: str) -> Transcriber:
    if name == "mlx":
        from meetscribe.backends.mlx import MlxWhisper

        return MlxWhisper(model)
    raise ValueError(f"Unknown backend {name!r}")
