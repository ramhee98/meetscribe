"""Speaker recognition logic without the model: alignment, profiles, naming, rendering."""

from datetime import datetime

import numpy as np
import pytest

from meetscribe.backends.base import Segment, Word
from meetscribe.speakers.align import assign_speakers, number_speakers
from meetscribe.speakers.diarize import Turn
from meetscribe.speakers.naming import apply_names, parse_assignment, remember_voices
from meetscribe.speakers.profiles import load_profiles, match, remember, save_profiles
from meetscribe.transcript import (
    Meta,
    Speaker,
    Transcript,
    read_transcript,
    render_md,
    render_txt,
    write_outputs,
)


def words(*items):
    """words((0.0, 0.4, ' Guten'), ...)"""
    return [Word(*w) for w in items]


TURNS = [Turn(0.0, 3.0, "SPEAKER_01"), Turn(3.1, 6.0, "SPEAKER_00")]


def test_segment_split_at_speaker_change():
    seg = Segment(0.0, 6.0, "Gut, machen wir so. Ja, das kann ich.", "de", words=words(
        (0.0, 0.4, " Gut,"), (0.5, 1.2, " machen"), (1.3, 1.8, " wir"), (1.9, 2.8, " so."),
        (3.2, 3.6, " Ja,"), (3.7, 4.2, " das"), (4.3, 4.8, " kann"), (4.9, 5.5, " ich.")))
    out = assign_speakers([seg], TURNS)
    assert [(s.speaker, s.text) for s in out] == [
        ("SPEAKER_01", "Gut, machen wir so."), ("SPEAKER_00", "Ja, das kann ich.")]
    assert (out[1].start, out[1].end) == (3.2, 5.5)


def test_single_word_glitch_is_absorbed():
    turns = [Turn(0.0, 2.0, "A"), Turn(2.0, 2.3, "B"), Turn(2.3, 5.0, "A")]
    seg = Segment(0.0, 5.0, "eins zwei drei vier", "de", words=words(
        (0.5, 1.0, " eins"), (1.5, 1.9, " zwei"), (2.05, 2.25, " drei"), (3.0, 3.5, " vier")))
    out = assign_speakers([seg], turns)
    assert len(out) == 1 and out[0].speaker == "A"


def test_without_words_whole_segment_by_overlap():
    segs = [Segment(0.5, 3.5, "a"), Segment(3.2, 5.0, "b"), Segment(20.0, 21.0, "c")]
    out = assign_speakers(segs, TURNS)
    # c is far from every turn: it keeps the previous speaker
    assert [s.speaker for s in out] == ["SPEAKER_01", "SPEAKER_00", "SPEAKER_00"]


def test_numbering_by_first_appearance():
    segs = [Segment(0, 1, "a", speaker="SPEAKER_03"), Segment(1, 2, "b", speaker="SPEAKER_00"),
            Segment(2, 3, "c", speaker="SPEAKER_03")]
    ids = number_speakers(segs)
    assert ids == {"SPEAKER_03": "S1", "SPEAKER_00": "S2"}
    assert [s.speaker for s in segs] == ["S1", "S2", "S1"]


def unit(*v):
    a = np.array(v, dtype=np.float32)
    return a / np.linalg.norm(a)


def test_profiles_match_best_pairs_once(tmp_path):
    profiles = {}
    remember(profiles, "Anna", unit(1, 0, 0))
    remember(profiles, "Ramon", unit(0, 1, 0))
    save_profiles(tmp_path, profiles)
    loaded = load_profiles(tmp_path)
    assert set(loaded) == {"Anna", "Ramon"}
    found = match({"S1": unit(0.1, 1, 0), "S2": unit(1, 0.2, 0), "S3": unit(0.9, 0, 0.1),
                   "S4": unit(0, 0, 1)}, loaded, 0.6)
    assert found["S1"][0] == "Ramon"
    # S2 and S3 both sound like Anna: only the closer one gets the name
    assert found["S3"][0] == "Anna" and "S2" not in found
    assert "S4" not in found


def test_remember_averages():
    profiles = {}
    remember(profiles, "Anna", unit(1, 0))
    remember(profiles, "Anna", unit(0, 1))
    p = profiles["Anna"]
    assert p.meetings == 2
    assert np.allclose(p.embedding, unit(1, 1))


def meeting():
    meta = Meta(title="Jour Fixe", series="jour-fixe", start=datetime(2026, 11, 9, 14, 0),
                mode="file", backend="mlx", model="large-v3-turbo", language="de",
                duration=20.0)
    segs = [Segment(0.0, 4.0, "Guten Morgen.", "de", speaker="S1"),
            Segment(4.5, 6.0, "Hallo.", "de", speaker="S2"),
            Segment(6.2, 12.0, "Die Offerte ist da.", "de", speaker="S3"),
            Segment(12.5, 14.0, "Gut.", "de", speaker="S1")]
    speakers = {"S1": Speaker("S1", "Anna", [1.0, 0.0, 0.0]),
                "S2": Speaker("S2", None, [0.0, 1.0, 0.0]),
                "S3": Speaker("S3", None, [0.0, 0.9, 0.1])}
    return Transcript(meta, segs, speakers)


def test_rendering_with_speakers():
    t = meeting()
    md = render_md(t)
    assert "**[00:00:00] Anna:** Guten Morgen." in md
    assert "**[00:00:04] Speaker 2:** Hallo." in md
    assert "- **Speakers:** Speaker 3 45%, Anna 43%, Speaker 2 12%" in md
    assert "[00:00:06] Speaker 3: Die Offerte ist da." in render_txt(t)


def test_json_round_trip_with_speakers(tmp_path):
    t = meeting()
    write_outputs(tmp_path / "m", t, ("md",))
    back = read_transcript(tmp_path / "m.json")
    assert back.speakers["S1"].name == "Anna"
    assert back.speakers["S2"].embedding == [0.0, 1.0, 0.0]
    assert [s.speaker for s in back.segments] == ["S1", "S2", "S3", "S1"]


def test_parse_assignment():
    t = meeting()
    assert parse_assignment("2=Ramon", t) == ("S2", "Ramon")
    assert parse_assignment("Speaker 3 = Eddy Muster", t) == ("S3", "Eddy Muster")
    assert parse_assignment("s1=", t) == ("S1", None)
    with pytest.raises(ValueError):
        parse_assignment("7=Nobody", t)
    with pytest.raises(ValueError):
        parse_assignment("Anna", t)


def test_same_name_merges_speakers():
    t = meeting()
    merged = apply_names(t, {"S2": "Eddy", "S3": "eddy"})
    assert merged == [("Eddy", "Speaker 3")]
    assert set(t.speakers) == {"S1", "S2"}
    assert [s.speaker for s in t.segments] == ["S1", "S2", "S2", "S1"]
    assert "**[00:00:04] Eddy:** Hallo. Die Offerte ist da." in render_md(t)


def test_voices_remembered_once_per_meeting(tmp_path):
    t = meeting()
    apply_names(t, {"S2": "Ramon"})
    saved, _ = remember_voices(t, tmp_path)
    assert sorted(saved) == ["Anna", "Ramon"]
    assert remember_voices(t, tmp_path)[0] == []  # running it again changes nothing
    assert load_profiles(tmp_path)["Anna"].meetings == 1
    apply_names(t, {"S2": "Eddy"})  # renamed: Eddy is new, gets remembered
    assert remember_voices(t, tmp_path)[0] == ["Eddy"]
