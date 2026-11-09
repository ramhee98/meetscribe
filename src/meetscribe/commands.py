"""The small commands: devices, recover, download-model."""

from __future__ import annotations

import json
from pathlib import Path

from meetscribe.audio.devices import list_input_devices
from meetscribe.config import Config
from meetscribe.console import BOLD, DIM, OFF, detail, die, emit, info, ok, warn


def show_devices(cfg: Config) -> int:
    devices = list_input_devices()
    if not devices:
        die("No audio input devices found. Connect a microphone and try again.")
    info("Audio input devices")
    for d in devices:
        default = f"  {BOLD}(system default){OFF}" if d.is_default else ""
        emit(f"   {d.index:>2}) {d.name}  {DIM}{d.channels} ch, "
             f"{d.sample_rate / 1000:g} kHz{OFF}{default}")
    emit()
    emit("Pick one with --device NAME or --device NUMBER, or set it in the config:")
    emit('  [audio]\n  device = "MacBook Pro Microphone"')
    emit(f"{DIM}Names are more stable than numbers, which can change when devices are "
         f"plugged in.{OFF}")
    if cfg.device:
        emit()
        detail("Configured", cfg.device)
    return 0


def show_config(cfg: Config) -> int:
    from meetscribe.config import SETTINGS

    if cfg.source:
        info(f"Settings from {cfg.source}")
    else:
        info("Built-in defaults (no config file found)")
        emit(f"{DIM}Create one with: cp meetscribe.toml.example meetscribe.toml{OFF}")
    section = None
    for s in SETTINGS:
        if s.section != section:
            section = s.section
            emit(f"  [{section}]")
        value = getattr(cfg, s.attr)
        if isinstance(value, tuple):
            value = list(value)
        elif not isinstance(value, bool | int | float):
            value = str(value)
        shown = json.dumps(value, ensure_ascii=False)
        emit(f"    {s.key} = {shown}")
    return 0


def _session_base(path: Path) -> Path:
    path = path.expanduser().resolve()
    name = path.name
    for suffix in (".jsonl", ".json", ".wav", ".md", ".txt"):
        if name.endswith(suffix):
            return path.with_name(name[: -len(suffix)])
    return path


def recover(cfg: Config, path: Path) -> int:
    from meetscribe.audio.wavfile import repair_wav
    from meetscribe.journal import read_journal
    from meetscribe.transcript import RECOVERED, Transcript, write_outputs

    base = _session_base(path)
    journal = base.with_name(base.name + ".jsonl")
    wav = base.with_name(base.name + ".wav")
    if not journal.exists() and not wav.exists():
        die(f"No meetscribe session found for {path}")

    duration = None
    if wav.exists():
        try:
            changed, duration = repair_wav(wav)
        except ValueError as e:
            die(str(e))
        if changed:
            ok(f"Fixed the recording's header: {wav.name} ({_dur(duration)})")
        else:
            ok(f"Recording is fine: {wav.name} ({_dur(duration)})")

    if not journal.exists():
        info("No journal left: the session ended normally, its transcript is complete")
        return 0

    try:
        meta, segments = read_journal(journal)
    except ValueError as e:
        die(str(e))
    meta.status = RECOVERED
    if duration is not None:
        meta.duration = duration
    elif segments:
        meta.duration = segments[-1].end
    t = Transcript(meta, segments)
    for p in write_outputs(base, t, cfg.formats):
        detail("Saved", str(p))
    ok(f"Recovered {len(segments)} segments")
    journal.unlink()
    if wav.exists():
        warn("The last moments before the interruption may be missing. For a complete "
             "transcript, re-transcribe the recording:")
        emit(f'  meetscribe file "{wav}" --overwrite')
    return 0


def _dur(seconds: float) -> str:
    from meetscribe.console import format_duration

    return format_duration(seconds)


def download_model(cfg: Config, speakers: bool = False) -> int:
    """Download the transcription model, and the speaker model if speakers is on."""
    if cfg.backend == "mlx":
        from meetscribe.backends import mlx

        if mlx.cached_path(cfg.model):
            ok(f"Model {cfg.model} already downloaded")
        else:
            info(f"Downloading model {cfg.model} ({mlx.resolve_model(cfg.model)})")
            path = mlx.download(cfg.model)
            ok(f"Model downloaded to {path}")
    if speakers or cfg.speakers:
        from meetscribe.speakers.diarize import MODEL, Diarizer

        info(f"Checking speaker recognition ({MODEL})")
        d = Diarizer()
        d.load()
        ok(f"Speaker recognition ready (runs on {'the GPU' if d.device == 'mps' else 'the CPU'})")
    return 0

# --- speakers -------------------------------------------------------------------

QUOTES = 2
QUOTE_CHARS = 110


def name_speakers(cfg: Config, path: Path, assignments: list[str]) -> int:
    """Name the speakers of a transcript and remember their voices for the series."""
    import sys

    from meetscribe.speakers.naming import apply_names, parse_assignment, remember_voices
    from meetscribe.transcript import read_transcript, write_outputs

    base = _session_base(path)
    json_path = base.with_name(base.name + ".json")
    if not json_path.is_file():
        die(f"No meetscribe transcript found for {path}")
    t = read_transcript(json_path)
    if not t.speakers:
        source = t.meta.audio and base.with_name(t.meta.audio) or t.meta.source or "<recording>"
        die("This transcript has no speakers yet. Recognize them by transcribing again with "
            f"speaker recognition:\n  meetscribe file \"{source}\" --speakers --overwrite")

    if assignments:
        names = {}
        for a in assignments:
            try:
                sid, name = parse_assignment(a, t)
            except ValueError as e:
                die(str(e))
            names[sid] = name
    else:
        if not sys.stdin.isatty():
            die("Give names with --set, e.g. --set 1=Anna --set 2=Ramon (or run it in a terminal)")
        names = _ask_names(t)

    merged = apply_names(t, names)
    saved, profiles_path = remember_voices(t, json_path.parent)
    write_outputs(base, t, cfg.formats)
    for keep, gone in merged:
        info(f"Merged {gone} into {keep}: same name, so the same person")
    ok(f"Saved the names in {base.name}.*")
    if saved:
        ok(f"Remembered the voices of {', '.join(saved)} for this series "
           f"({profiles_path})")
        emit(f"{DIM}Next time, they're recognized automatically.{OFF}")
    return 0


def _ask_names(t) -> dict[str, str | None]:
    from meetscribe.console import format_duration, format_timestamp

    times = t.talk_time()
    total = sum(times.values()) or 1.0
    info("Who is who? Enter a name, Enter to keep, - to remove a name, q to quit")
    names: dict[str, str | None] = {}
    for sid, secs in times.items():
        sp = t.speakers[sid]
        emit()
        emit(f"{BOLD}{sp.label}{OFF}  {DIM}{round(100 * secs / total)}% of the speaking time "
             f"({format_duration(secs)}){OFF}")
        quotes = sorted((s for s in t.segments if s.speaker == sid),
                        key=lambda s: len(s.text), reverse=True)[:QUOTES]
        for q in sorted(quotes, key=lambda s: s.start):
            text = q.text if len(q.text) <= QUOTE_CHARS else q.text[: QUOTE_CHARS - 1] + "…"
            emit(f"  {DIM}[{format_timestamp(q.start)}]{OFF} {text}")
        current = f" [{sp.name}]" if sp.name else ""
        try:
            answer = input(f"Name{current}: ").strip()
        except EOFError:
            answer = "q"
        if answer.lower() == "q":
            die("Quit, nothing saved")
        if answer == "-":
            names[sid] = None
        elif answer:
            names[sid] = answer
    emit()
    return names
