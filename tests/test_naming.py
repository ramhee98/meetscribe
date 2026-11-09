"""Where transcripts are saved and how they're named (naming.py). Pure path logic."""

from datetime import datetime

from meetscribe.naming import (
    find_prompt,
    new_session_base,
    parse_stem,
    series_title,
    slugify,
)

START = datetime(2026, 11, 9, 14, 0)


def test_slugify():
    assert slugify("Weekly Team") == "weekly-team"
    assert slugify("Jour fixe / Büro Zürich") == "jour-fixe-buero-zuerich"
    assert slugify("  Straße & Co. 2026 ") == "strasse-co-2026"
    assert slugify("Café Crème") == "cafe-creme"
    assert slugify("!!!") == ""


def test_series_folder_and_name(tmp_path):
    base = new_session_base(tmp_path, "weekly-team", START)
    assert base == tmp_path / "weekly-team" / "2026-11-09_14-00_weekly-team"


def test_without_series(tmp_path):
    assert new_session_base(tmp_path, None, START) == tmp_path / "2026-11-09_14-00"


def test_label_differs_from_series(tmp_path):
    base = new_session_base(tmp_path, None, START, label="aufnahme-3")
    assert base == tmp_path / "2026-11-09_14-00_aufnahme-3"


def test_collision_in_same_minute(tmp_path):
    folder = tmp_path / "weekly-team"
    folder.mkdir()
    (folder / "2026-11-09_14-00_weekly-team.wav").touch()
    assert new_session_base(tmp_path, "weekly-team", START).name == "2026-11-09_14-00_weekly-team-2"
    (folder / "2026-11-09_14-00_weekly-team-2.md").touch()
    assert new_session_base(tmp_path, "weekly-team", START).name == "2026-11-09_14-00_weekly-team-3"


def test_parse_stem():
    assert parse_stem("2026-11-09_14-00_weekly-team") == (START, "weekly-team")
    assert parse_stem("2026-11-09_14-00_weekly-team-2") == (START, "weekly-team")
    assert parse_stem("2026-11-09_14-00") == (START, None)
    assert parse_stem("meeting") is None
    assert parse_stem("2026-13-09_14-00") is None


def test_series_title():
    assert series_title("weekly-team") == "Weekly Team"
    assert series_title(None) == "Meeting"


def test_prompt_lookup_order(tmp_path):
    assert find_prompt(tmp_path, "weekly-team") is None
    (tmp_path / "prompt.txt").write_text("Ramon, Anna\n")
    assert find_prompt(tmp_path, "weekly-team")[0] == "Ramon, Anna"
    (tmp_path / "weekly-team").mkdir()
    (tmp_path / "weekly-team" / "prompt.txt").write_text("Ramon,\nAnna, Chur")
    text, path = find_prompt(tmp_path, "weekly-team")
    assert text == "Ramon, Anna, Chur"
    assert path.parent.name == "weekly-team"
    assert find_prompt(tmp_path, None)[0] == "Ramon, Anna"
