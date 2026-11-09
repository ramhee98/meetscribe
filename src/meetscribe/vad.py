"""Splitting audio into utterances with voice activity detection (Silero VAD).

Whisper isn't a streaming model, and it tends to invent text in silence and
room noise. So audio is first cut into utterances: stretches of speech that end
at a pause. Each utterance is transcribed on its own. Live mode gets its text a
moment after someone pauses; file mode uses the same cuts, so both modes give
the same kind of result.

An utterance ends when the speech is followed by min_silence_ms of silence. Long
monologues are cut earlier: after 60% of max_utterance_s, a short pause (200 ms)
is enough, and at max_utterance_s it's cut no matter what.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from meetscribe.audio import SAMPLE_RATE

FRAME = 512  # samples per VAD frame (32 ms), fixed by Silero
SHORT_SILENCE_MS = 200
HYSTERESIS = 0.15  # once speaking, stay "speech" until the probability drops this far below


@dataclass
class Utterance:
    start: int  # first sample, on the recording's timeline
    audio: np.ndarray

    @property
    def start_s(self) -> float:
        return self.start / SAMPLE_RATE

    @property
    def duration(self) -> float:
        return len(self.audio) / SAMPLE_RATE

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration


def silero_speech_prob() -> Callable[[np.ndarray], float]:
    """A function giving the speech probability of one 512-sample frame."""
    from pysilero_vad import SileroVoiceActivityDetector

    vad = SileroVoiceActivityDetector()
    return lambda frame: float(vad.process_samples(frame.tolist()))


class Segmenter:
    def __init__(
        self,
        speech_prob: Callable[[np.ndarray], float],
        *,
        threshold: float = 0.5,
        min_speech_ms: int = 250,
        min_silence_ms: int = 600,
        speech_pad_ms: int = 200,
        max_utterance_s: float = 20.0,
    ):
        self.speech_prob = speech_prob
        self.threshold = threshold
        self.min_speech = int(min_speech_ms * SAMPLE_RATE / 1000)
        self.min_silence = int(min_silence_ms * SAMPLE_RATE / 1000)
        self.short_silence = min(int(SHORT_SILENCE_MS * SAMPLE_RATE / 1000), self.min_silence)
        self.pad = int(speech_pad_ms * SAMPLE_RATE / 1000)
        self.max_len = int(max_utterance_s * SAMPLE_RATE)
        self.soft_len = int(self.max_len * 0.6)

        self._pending = np.zeros(0, dtype=np.float32)  # less than one frame
        self._pos = 0  # position of the next frame
        self._frames: deque[tuple[int, np.ndarray]] = deque()  # (start sample, frame)
        self._in_speech = False
        self._utt_start = 0
        self._speech_start = 0
        self._last_speech_end = 0
        self._silence = 0
        self.last_prob = 0.0

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    def feed(self, samples: np.ndarray) -> list[Utterance]:
        samples = np.asarray(samples, dtype=np.float32)
        if self._pending.size:
            samples = np.concatenate([self._pending, samples])
        n = len(samples) // FRAME * FRAME
        self._pending = samples[n:].copy()
        out = []
        for i in range(0, n, FRAME):
            utt = self._frame(samples[i : i + FRAME])
            if utt is not None:
                out.append(utt)
        return out

    def flush(self) -> list[Utterance]:
        """End of audio: return what's left of the current utterance."""
        out = []
        if self._pending.size:
            frame = np.zeros(FRAME, dtype=np.float32)
            frame[: self._pending.size] = self._pending
            self._pending = np.zeros(0, dtype=np.float32)
            utt = self._frame(frame)
            if utt is not None:
                out.append(utt)
        if self._in_speech:
            utt = self._emit(min(self._last_speech_end + self.pad, self._pos))
            if utt is not None:
                out.append(utt)
        return out

    # --- internals ------------------------------------------------------------

    def _frame(self, frame: np.ndarray) -> Utterance | None:
        pos = self._pos
        self._frames.append((pos, frame))
        self._pos += FRAME
        p = self.speech_prob(frame)
        self.last_prob = p
        end = self._pos

        if not self._in_speech:
            if p >= self.threshold:
                self._in_speech = True
                self._utt_start = max(pos - self.pad, self._frames[0][0])
                self._speech_start = pos
                self._last_speech_end = end
                self._silence = 0
            else:
                # Keep only the pre-roll (padding before the next utterance).
                keep_from = end - self.pad - FRAME
                while self._frames and self._frames[0][0] + FRAME <= keep_from:
                    self._frames.popleft()
            return None

        if p >= self.threshold - HYSTERESIS:
            self._silence = 0
            self._last_speech_end = end
        else:
            self._silence += FRAME

        length = end - self._utt_start
        needed = self.short_silence if length >= self.soft_len else self.min_silence
        if self._silence >= needed:
            return self._emit(min(self._last_speech_end + self.pad, end))
        if length + FRAME > self.max_len:
            # Hard cut before the next frame would go over the limit. Speech
            # continues in a new utterance right after.
            utt = self._emit(end)
            self._in_speech = True
            self._utt_start = end
            self._speech_start = end
            self._last_speech_end = end
            return utt
        return None

    def _emit(self, end: int) -> Utterance | None:
        start = self._utt_start
        self._in_speech = False
        self._silence = 0
        speech = self._last_speech_end - self._speech_start
        parts = [f for (p, f) in self._frames if p + FRAME > start and p < end]
        first = next((p for (p, _) in self._frames if p + FRAME > start), start)
        # Drop frames that can't be part of a later utterance's pre-roll.
        keep_from = end - self.pad - FRAME
        while self._frames and self._frames[0][0] + FRAME <= keep_from:
            self._frames.popleft()
        if speech < self.min_speech or not parts:
            return None
        audio = np.concatenate(parts)[start - first : end - first]
        return Utterance(start, audio)
