"""File mode: transcribe an existing recording."""

from __future__ import annotations

import time
from pathlib import Path

from meetscribe import console
from meetscribe.audio.decode import file_start_time, load_audio, probe, require_ffmpeg
from meetscribe.backends.base import get_backend
from meetscribe.config import Config
from meetscribe.console import detail, die, format_count, format_duration, info, ok, warn
from meetscribe.naming import find_prompt, new_session_base, parse_stem, series_title, slugify
from meetscribe.pipeline import Pipeline
from meetscribe.transcript import Meta, Transcript, output_paths, read_transcript, write_outputs
from meetscribe.vad import Segmenter, silero_speech_prob

BLOCK_S = 30  # audio fed to the VAD per step (for the progress line)


def _existing_session(path: Path) -> Meta | None:
    """Metadata if path is a recording made by meetscribe live (json next to it)."""
    sidecar = path.with_suffix(".json")
    if sidecar.is_file():
        try:
            meta = read_transcript(sidecar).meta
        except (ValueError, KeyError, TypeError):
            return None
        if meta.audio == path.name:
            return meta
    return None


def resolve_prompt(cfg: Config, series: str | None, cli_prompt: str | None) -> tuple[str, str]:
    """(prompt, where it came from): --prompt, then prompt.txt, then the config."""
    if cli_prompt is not None:
        return cli_prompt, "--prompt"
    found = find_prompt(cfg.output_dir, series)
    if found:
        return found[0], str(found[1])
    if cfg.prompt:
        return cfg.prompt, "config"
    return "", ""


def run_file(cfg: Config, path: Path, name: str | None, cli_prompt: str | None) -> int:
    require_ffmpeg()
    path = path.expanduser().resolve()
    if not path.is_file():
        die(f"File not found: {path}")

    backend = get_backend(cfg.backend, cfg.model)  # checks the model name early
    duration, _ = probe(path)
    prior = _existing_session(path)
    if prior is None and path.suffix.lower() == ".wav" and parse_stem(path.stem):
        start, series = parse_stem(path.stem)
        prior = Meta(title=series_title(series), series=series, start=start, mode="live",
                     backend="", model="", language="", audio=path.name)

    if prior is not None:
        # Re-transcribing a meetscribe recording: keep its name, date and place.
        if name:
            warn(f"Ignoring --name: {path.name} is a meetscribe recording of "
                 f"'{prior.series or 'no series'}'")
        base = path.with_suffix("")
        series, start, title = prior.series, prior.start, prior.title
        audio_name, source, device = path.name, None, prior.device
    else:
        series = slugify(name) if name else None
        if name is not None and not series:
            die(f"--name '{name}' has no letters or digits to build a file name from")
        start = file_start_time(path)
        label = series or slugify(path.stem) or None
        base = new_session_base(cfg.output_dir, series, start, label)
        title = name or path.stem
        audio_name, source, device = None, str(path), None

    existing = [p for p in output_paths(base, cfg.formats) if p.exists()]
    if existing and not cfg.overwrite:
        names = ", ".join(p.name for p in existing)
        die(f"Transcript already exists: {names}\n"
            "Use --overwrite to replace it, e.g. when re-transcribing with a better model.")

    prompt, prompt_src = resolve_prompt(cfg, series, cli_prompt)

    info(f"Transcribing {path.name}" + (f" ({format_duration(duration)})" if duration else ""))
    detail("Model", f"{cfg.model} ({cfg.backend})")
    detail("Language", cfg.language_mode)
    if prompt:
        detail("Prompt", f"{prompt[:60]}{'…' if len(prompt) > 60 else ''} ({prompt_src})")
    if cfg.speakers:
        detail("Speakers", speakers_setting(cfg))
    detail("Output", str(base.parent))

    diarizer = load_diarizer(cfg) if cfg.speakers else None  # fails early, before the work
    t_start = time.monotonic()
    audio = load_audio(path)
    audio_s = len(audio) / 16000
    if audio_s < 0.5:
        die(f"{path.name} contains no audio")

    info("Loading model")
    backend.load()
    pipeline = Pipeline(backend, cfg.language, cfg.languages, prompt, words=cfg.speakers)
    segmenter = Segmenter(
        silero_speech_prob(),
        threshold=cfg.vad_threshold,
        min_speech_ms=cfg.min_speech_ms,
        min_silence_ms=cfg.min_silence_ms,
        speech_pad_ms=cfg.speech_pad_ms,
        max_utterance_s=cfg.file_max_utterance_s,
    )

    meta = Meta(title=title, series=series, start=start, mode="file", backend=backend.name,
                model=cfg.model, language=cfg.language_mode, duration=audio_s, device=device,
                audio=audio_name, source=source)
    transcript = Transcript(meta)

    t_transcribe = time.monotonic()
    tty = console._tty(console.sys.stdout)
    if not tty:
        info("Transcribing...")
    block = BLOCK_S * 16000
    for i in range(0, len(audio), block):
        utts = segmenter.feed(audio[i : i + block])
        if i + block >= len(audio):
            utts += segmenter.flush()
        for utt in utts:
            transcript.segments += pipeline.process(utt)
            _progress(min(utt.end_s, audio_s) / audio_s, t_transcribe)
        _progress(min((i + block) / 16000, audio_s) / audio_s, t_transcribe)
    console.clear_status()
    elapsed = time.monotonic() - t_transcribe
    ok(f"Transcribing done ({format_duration(elapsed)})")

    base.parent.mkdir(parents=True, exist_ok=True)
    if diarizer is not None:
        add_speakers(transcript, audio, diarizer, cfg, base.parent)
    written = write_outputs(base, transcript, cfg.formats)

    total = time.monotonic() - t_start
    info("Summary")
    detail("Audio", format_duration(audio_s))
    detail("Speech", f"{format_duration(pipeline.audio_seconds)} in "
                     f"{format_count(len(transcript.segments))} segments")
    detail("Transcribed", f"{format_duration(elapsed)} (total {format_duration(total)} "
                          "incl. decoding and model loading)")
    detail("Speed", f"{audio_s / elapsed:.1f}x realtime" if elapsed > 0 else "n/a")
    detail("Words", format_count(transcript.word_count))
    if transcript.languages:
        detail("Languages", ", ".join(transcript.languages))
    for p in written:
        detail("Saved", str(p))
    naming_hint(transcript, written)
    return 0


def speakers_setting(cfg: Config) -> str:
    if cfg.num_speakers:
        return f"on ({cfg.num_speakers} people)"
    if cfg.max_speakers:
        return f"on (up to {cfg.max_speakers} people)"
    return "on"


def load_diarizer(cfg: Config):
    from meetscribe.speakers.diarize import Diarizer

    diarizer = Diarizer(cfg.num_speakers, cfg.max_speakers)
    info("Loading speaker recognition")
    diarizer.load()
    return diarizer


def add_speakers(transcript: Transcript, audio, diarizer, cfg: Config, folder: Path) -> bool:
    """Speaker recognition on a finished transcript. A failure only costs the
    speaker labels, never the transcript."""
    from meetscribe.speakers.recognize import recognize, report

    if not transcript.segments:
        return False
    try:
        result = recognize(transcript, audio, diarizer, folder, cfg.match_threshold)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        warn(f"Speaker recognition failed, saving the transcript without speakers: {e}")
        return False
    report(transcript, result)
    return True


def naming_hint(transcript: Transcript, written: list[Path]) -> None:
    unnamed = [sp for sp in transcript.speakers.values() if not sp.name]
    if not unnamed or not written:
        return
    console.emit()
    console.emit(f"{console.DIM}Give the speakers names (remembered for this series):{console.OFF}")
    console.emit(f'  meetscribe speakers "{written[0]}"')


def _progress(fraction: float, since: float) -> None:
    console.set_status(f"{console.BLUE}==>{console.OFF} Transcribing... "
                       f"{int(fraction * 100)}% ({format_duration(time.monotonic() - since)})")
