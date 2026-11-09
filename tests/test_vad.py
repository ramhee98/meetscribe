"""Cutting audio into utterances (vad.py), with a fake VAD: a frame is speech if
its first sample is 1.0. Times below are in frames of 32 ms."""

import numpy as np

from meetscribe.vad import FRAME, Segmenter


def fake_prob(frame: np.ndarray) -> float:
    return 1.0 if frame[0] == 1.0 else 0.0


def audio(pattern: str) -> np.ndarray:
    """'.' = silent frame, 'S' = speech frame."""
    out = np.zeros(len(pattern) * FRAME, dtype=np.float32)
    for i, c in enumerate(pattern):
        if c == "S":
            out[i * FRAME : (i + 1) * FRAME] = 1.0
    return out


def seg(**kw) -> Segmenter:
    opts = dict(threshold=0.5, min_speech_ms=64, min_silence_ms=320, speech_pad_ms=64,
                max_utterance_s=10.0)
    opts.update(kw)
    return Segmenter(fake_prob, **opts)


def frames(utt) -> tuple[float, float]:
    return utt.start / FRAME, (utt.start + len(utt.audio)) / FRAME


def test_one_utterance_with_padding():
    # speech at frames 10-19, then 10 silent frames (320 ms = 10 frames) end it
    s = seg()
    utts = s.feed(audio("." * 10 + "S" * 10 + "." * 12))
    assert len(utts) == 1
    assert frames(utts[0]) == (8, 22)  # 2 frames (64 ms) padding on each side


def test_short_pause_does_not_split():
    s = seg()
    utts = s.feed(audio("." * 5 + "S" * 5 + "." * 5 + "S" * 5 + "." * 12))
    assert len(utts) == 1


def test_two_utterances():
    s = seg()
    utts = s.feed(audio("." * 5 + "S" * 5 + "." * 15 + "S" * 5 + "." * 12))
    assert len(utts) == 2
    assert frames(utts[0])[0] == 3
    assert frames(utts[1])[0] == 23


def test_clicks_are_dropped():
    s = seg(min_speech_ms=200)  # ~7 frames
    assert s.feed(audio("." * 5 + "SS" + "." * 15)) == []


def test_feed_in_odd_chunks_gives_same_result():
    data = audio("." * 7 + "S" * 9 + "." * 15 + "S" * 4 + "." * 12)
    whole = seg().feed(data)
    s = seg()
    pieces = []
    for i in range(0, len(data), 777):
        pieces += s.feed(data[i : i + 777])
    assert [(u.start, len(u.audio)) for u in whole] == [(u.start, len(u.audio)) for u in pieces]
    for a, b in zip(whole, pieces, strict=True):
        assert np.array_equal(a.audio, b.audio)


def test_audio_matches_timeline():
    data = audio("." * 10 + "S" * 10 + "." * 12)
    data += np.linspace(0, 0.5, len(data), dtype=np.float32) * (data == 0)
    utt = seg().feed(data)[0]
    assert np.array_equal(utt.audio, data[utt.start : utt.start + len(utt.audio)])


def test_long_speech_is_cut():
    # max 1 s = 31.25 frames: a 100-frame monologue is cut into pieces of at most 1 s
    s = seg(max_utterance_s=1.0)
    utts = s.feed(audio("S" * 100)) + s.flush()
    assert len(utts) >= 3
    assert all(u.duration <= 1.0 + 1e-9 for u in utts)
    # no gaps or overlaps between the pieces
    for a, b in zip(utts, utts[1:], strict=False):
        assert a.start + len(a.audio) == b.start


def test_flush_returns_speech_in_progress():
    s = seg()
    assert s.feed(audio("." * 5 + "S" * 10)) == []
    utts = s.flush()
    assert len(utts) == 1
    assert frames(utts[0]) == (3, 15)
    assert s.flush() == []
