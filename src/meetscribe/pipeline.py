"""From utterance to transcript segments: shared by live and file mode.

For each utterance this picks the language, builds the prompt, runs the
backend, drops made-up text and moves timestamps onto the meeting timeline.
"""

from __future__ import annotations

import re
import time

from meetscribe.backends.base import Segment, Transcriber
from meetscribe.vad import Utterance

# Utterances shorter than this keep the previous language: detection on a word
# or two is unreliable (Swiss German often comes out as Dutch or Afrikaans).
MIN_DETECT_S = 2.0
PROMPT_CONTEXT_CHARS = 200

# Text Whisper is known to invent in silence or noise, mostly from subtitled TV
# it was trained on. A segment consisting of just this is dropped.
HALLUCINATIONS = re.compile(
    r"^\W*("
    r"untertitel(ung)?\b.*|.*\bamara\.org\b.*|.*\bzdf\b.*untertitel.*|"
    r"(vielen )?dank (fürs|für's|für das) zuschauen\W*|"
    r"bis zum nächsten mal\W*|"
    r"thanks? (you )?for watching\W*|subtitles by\b.*|"
    r"\[?(musik|music|applaus|applause)\]?"
    r")\W*$",
    re.IGNORECASE,
)


def is_hallucination(seg: Segment, utterance_s: float) -> bool:
    if not seg.text.strip():
        return True
    if HALLUCINATIONS.match(seg.text.strip()):
        return True
    # Text placed after the end of the audio comes from Whisper's silence padding.
    if seg.start >= utterance_s - 0.1:
        return True
    return seg.no_speech_prob > 0.6 and seg.avg_logprob < -1.0


def choose_language(probs: dict[str, float], allowed: tuple[str, ...]) -> str:
    """The most probable language among the allowed ones (any, if none are given)."""
    candidates = {k: v for k, v in probs.items() if k in allowed} if allowed else probs
    if not candidates:
        candidates = probs
    return max(candidates, key=candidates.get)


class Pipeline:
    def __init__(self, backend: Transcriber, language: str, languages: tuple[str, ...],
                 prompt: str = "", words: bool = False):
        self.backend = backend
        self.words = words  # word timings, for speaker recognition
        self.language = language
        self.languages = languages
        self.prompt = prompt.strip()
        self._last_language: str | None = None
        self._context = ""
        self._context_language: str | None = None
        self.busy_seconds = 0.0  # time spent in the backend
        self.audio_seconds = 0.0  # utterance audio transcribed

    def _pick_language(self, utt: Utterance) -> str:
        if self.language != "auto":
            return self.language
        if utt.duration < MIN_DETECT_S and self._last_language:
            return self._last_language
        return choose_language(self.backend.language_probs(utt.audio), self.languages)

    def _build_prompt(self, language: str) -> str | None:
        """Configured prompt (names, terms), then the end of what was said before.

        The previous text is only used if it's in the same language: given text in
        another language as context, Whisper tends to translate into that language."""
        context = self._context if language == self._context_language else ""
        parts = [p for p in (self.prompt, context) if p]
        return " ".join(parts) or None

    def process(self, utt: Utterance) -> list[Segment]:
        t0 = time.monotonic()
        lang = self._pick_language(utt)
        raw = self.backend.transcribe(utt.audio, lang, self._build_prompt(lang), self.words)
        self.busy_seconds += time.monotonic() - t0
        self.audio_seconds += utt.duration

        out = []
        for seg in raw:
            if is_hallucination(seg, utt.duration):
                continue
            seg.start = utt.start_s + max(seg.start, 0.0)
            seg.end = utt.start_s + min(seg.end, utt.duration)
            seg.end = max(seg.end, seg.start)
            for w in seg.words:
                w.start = utt.start_s + max(w.start, 0.0)
                w.end = utt.start_s + min(w.end, utt.duration)
            seg.language = lang
            out.append(seg)
        if out:
            self._last_language = lang
            text = " ".join(s.text for s in out)
            if lang == self._context_language:
                text = f"{self._context} {text}"
            self._context = text[-PROMPT_CONTEXT_CHARS:].strip()
            self._context_language = lang
        return out
