"""Rendering, the journal, the WAV writer and the pipeline's filtering."""

import struct
from datetime import datetime

import numpy as np

from meetscribe.audio.wavfile import CrashSafeWav, repair_wav
from meetscribe.backends.base import Segment
from meetscribe.journal import Journal, read_journal
from meetscribe.pipeline import Pipeline, choose_language, is_hallucination
from meetscribe.transcript import (
    RECORDING,
    Meta,
    Transcript,
    read_transcript,
    render_md,
    render_txt,
    write_outputs,
)
from meetscribe.vad import Utterance


def meta(**kw) -> Meta:
    d = dict(title="Weekly Team", series="weekly-team", start=datetime(2026, 11, 9, 14, 0),
             mode="live", backend="mlx", model="large-v3-turbo", language="auto (de, en)",
             duration=3725.0, device="MacBook Pro Microphone",
             audio="2026-11-09_14-00_weekly-team.wav")
    d.update(kw)
    return Meta(**d)


SEGMENTS = [
    Segment(5.0, 8.0, "Guten Morgen zusammen.", "de"),
    Segment(8.5, 10.0, "Fangen wir an.", "de"),
    Segment(15.0, 18.0, "Thanks, Anna.", "en"),
    Segment(18.2, 20.0, "Gut.", "de"),
]


def test_paragraphs_split_on_pause_and_language():
    t = Transcript(meta(), SEGMENTS)
    assert [[s.text for s in p] for p in t.paragraphs()] == [
        ["Guten Morgen zusammen.", "Fangen wir an."], ["Thanks, Anna."], ["Gut."]]
    assert t.languages == ["de", "en"]


def test_markdown():
    md = render_md(Transcript(meta(), SEGMENTS))
    assert md.startswith("# Weekly Team – 2026-11-09 14:00\n")
    assert "- **Date:** Monday, 9 November 2026, 14:00–15:02" in md
    assert "- **Duration:** 1:02:05" in md
    assert "**[00:00:05]** Guten Morgen zusammen. Fangen wir an." in md
    assert "**[00:00:15]** *(en)* Thanks, Anna." in md
    assert "Recording in progress" not in md
    assert "Recording in progress" in render_md(Transcript(meta(status=RECORDING), SEGMENTS))


def test_txt_single_language_has_no_tags():
    segs = [Segment(5.0, 8.0, "Guten Morgen.", "de")]
    txt = render_txt(Transcript(meta(), segs))
    assert "[00:00:05] Guten Morgen." in txt
    assert "(de)" not in txt


def test_json_round_trip(tmp_path):
    t = Transcript(meta(), SEGMENTS)
    written = write_outputs(tmp_path / "x", t, ("md", "txt"))
    assert [p.name for p in written] == ["x.md", "x.txt", "x.json"]
    back = read_transcript(tmp_path / "x.json")
    assert back.meta == t.meta
    assert [(s.start, s.text, s.language) for s in back.segments] == \
        [(s.start, s.text, s.language) for s in SEGMENTS]
    assert not list(tmp_path.glob(".*partial"))


def test_journal_survives_cut_off_line(tmp_path):
    path = tmp_path / "s.jsonl"
    j = Journal(path, meta(status=RECORDING))
    j.add(SEGMENTS[:2])
    j.close(remove=False)
    with open(path, "a") as f:
        f.write('{"type": "segment", "start": 9')  # crash mid-write
    m, segs = read_journal(path)
    assert m.title == "Weekly Team" and m.status == RECORDING
    assert [s.text for s in segs] == ["Guten Morgen zusammen.", "Fangen wir an."]


def test_wav_header_is_kept_current_and_repairable(tmp_path):
    path = tmp_path / "r.wav"
    w = CrashSafeWav(path, sync_every_s=0)
    w.write(np.zeros(16000, dtype=np.float32))
    # synced after the write: header already says 1 s
    data = path.read_bytes()
    assert struct.unpack("<I", data[40:44])[0] == 32000
    w.write(np.zeros(8000, dtype=np.float32))
    w._f.flush()
    # simulate a crash: more data than the header says, plus half a sample
    with open(path, "r+b") as f:
        f.seek(40)
        f.write(struct.pack("<I", 100))
        f.seek(0, 2)
        f.write(b"\x01")
    changed, duration = repair_wav(path)
    assert changed and duration == 1.5
    assert struct.unpack("<I", path.read_bytes()[40:44])[0] == 48000
    assert repair_wav(path) == (False, 1.5)


def test_hallucinations():
    assert is_hallucination(Segment(0, 2, "Untertitel im Auftrag des ZDF, 2020"), 5)
    assert is_hallucination(Segment(0, 2, "Vielen Dank fürs Zuschauen!"), 5)
    assert is_hallucination(Segment(0, 2, "Untertitelung des ZDF für funk, 2017"), 5)
    assert is_hallucination(Segment(5.0, 30, "Vielen Dank."), 5)  # in the padding
    assert not is_hallucination(Segment(0, 2, "Vielen Dank."), 5)
    assert not is_hallucination(Segment(0, 2, "Die Untertitel im Video sind falsch."), 5)


def test_choose_language():
    probs = {"nl": 0.5, "de": 0.3, "en": 0.2}
    assert choose_language(probs, ("de", "en")) == "de"
    assert choose_language(probs, ()) == "nl"


class FakeBackend:
    name = "fake"

    def __init__(self):
        self.prompts = []

    def language_probs(self, audio):
        return {"en": 0.9, "de": 0.1} if audio[0] > 0 else {"de": 0.9, "en": 0.1}

    def transcribe(self, audio, language, prompt):
        self.prompts.append(prompt)
        return [Segment(0.25, 1.5, f"text in {language}", language)]


def test_pipeline_offsets_and_prompt_context():
    backend = FakeBackend()
    p = Pipeline(backend, "auto", ("de", "en"), prompt="Ramon, Anna")
    de = np.zeros(48000, dtype=np.float32)
    en = np.full(48000, 0.1, dtype=np.float32)
    s1 = p.process(Utterance(16000 * 60, de))
    assert (s1[0].start, s1[0].end, s1[0].language) == (60.25, 61.5, "de")
    p.process(Utterance(16000 * 70, de))
    p.process(Utterance(16000 * 80, en))
    assert backend.prompts[0] == "Ramon, Anna"
    assert backend.prompts[1] == "Ramon, Anna text in de"
    # never German context for English speech: Whisper would translate
    assert backend.prompts[2] == "Ramon, Anna"


def test_short_utterance_keeps_previous_language():
    backend = FakeBackend()
    p = Pipeline(backend, "auto", ("de", "en"))
    p.process(Utterance(0, np.full(48000, 0.1, dtype=np.float32)))  # en
    s = p.process(Utterance(48000, np.zeros(8000, dtype=np.float32)))  # 0.5 s
    assert s[0].language == "en"
