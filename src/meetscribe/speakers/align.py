"""Giving each transcript segment its speaker.

Whisper's segments come from the transcription, the speaker turns from the
diarization, and the two don't line up: one segment can hold the end of one
person's sentence and the start of the next person's. So with word timings,
every word gets the speaker it overlaps most, and segments are split where the
speaker changes. Without word timings (e.g. a recovered live session), each
segment as a whole gets the speaker it overlaps most.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict

from meetscribe.backends.base import Segment, Word
from meetscribe.speakers.diarize import Turn

NEAREST_TURN_S = 1.0  # a word outside every turn takes the nearest turn within this distance


class _Turns:
    def __init__(self, turns: list[Turn]):
        self.turns = sorted(turns, key=lambda t: t.start)
        self.starts = [t.start for t in self.turns]

    def speaker(self, start: float, end: float) -> str | None:
        """The speaker overlapping [start, end] most, else the nearest one close by."""
        if not self.turns:
            return None
        overlap: dict[str, float] = defaultdict(float)
        i = max(bisect_right(self.starts, start) - 1, 0)
        while i < len(self.turns) and self.turns[i].start < end:
            t = self.turns[i]
            o = min(end, t.end) - max(start, t.start)
            if o > 0:
                overlap[t.speaker] += o
            i += 1
        if overlap:
            return max(overlap, key=overlap.get)
        mid = (start + end) / 2
        best, dist = None, NEAREST_TURN_S
        j = bisect_right(self.starts, mid)
        for t in self.turns[max(j - 1, 0) : j + 1]:
            d = max(t.start - mid, mid - t.end, 0.0)
            if d <= dist:
                best, dist = t.speaker, d
        return best


def _join(words: list[Word]) -> str:
    return "".join(w.text for w in words).strip()


def assign_speakers(segments: list[Segment], turns: list[Turn]) -> list[Segment]:
    """New list of segments with .speaker set (pyannote labels), split at speaker changes."""
    index = _Turns(turns)
    out: list[Segment] = []
    for seg in segments:
        if not seg.words:
            seg.speaker = index.speaker(seg.start, seg.end)
            out.append(seg)
            continue
        labels = [index.speaker(w.start, max(w.end, w.start + 0.01)) for w in seg.words]
        # Words without a speaker belong to the one before them (or after, at the start).
        for i in range(1, len(labels)):
            labels[i] = labels[i] or labels[i - 1]
        for i in range(len(labels) - 2, -1, -1):
            labels[i] = labels[i] or labels[i + 1]
        runs: list[tuple[str | None, list[Word]]] = []
        for w, label in zip(seg.words, labels, strict=True):
            if runs and runs[-1][0] == label:
                runs[-1][1].append(w)
            else:
                runs.append((label, [w]))
        runs = _absorb_single_words(runs)
        if len(runs) == 1:
            seg.speaker = runs[0][0]
            out.append(seg)
            continue
        for label, words in runs:
            text = _join(words)
            if text:
                out.append(Segment(words[0].start, words[-1].end, text, seg.language,
                                   seg.no_speech_prob, seg.avg_logprob, label, words))
    # Segments without any speaker nearby take the previous segment's speaker.
    for i in range(1, len(out)):
        out[i].speaker = out[i].speaker or out[i - 1].speaker
    return out


def _absorb_single_words(runs):
    """A lone word between two runs of the same speaker is almost always a timing
    error at the turn boundary, not a real one-word interruption."""
    if len(runs) < 3:
        return runs
    merged = [runs[0]]
    i = 1
    while i < len(runs):
        label, words = runs[i]
        if (len(words) == 1 and i + 1 < len(runs) and runs[i + 1][0] == merged[-1][0]):
            merged[-1] = (merged[-1][0], merged[-1][1] + words + runs[i + 1][1])
            i += 2
            continue
        if label == merged[-1][0]:
            merged[-1] = (label, merged[-1][1] + words)
        else:
            merged.append((label, words))
        i += 1
    return merged


def number_speakers(segments: list[Segment]) -> dict[str, str]:
    """Map pyannote labels to S1, S2, ... in order of first appearance, and apply it."""
    ids: dict[str, str] = {}
    for s in segments:
        if s.speaker and s.speaker not in ids:
            ids[s.speaker] = f"S{len(ids) + 1}"
    for s in segments:
        if s.speaker:
            s.speaker = ids[s.speaker]
    return ids
