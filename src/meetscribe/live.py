"""Live mode: record the meeting and transcribe it as it happens.

Threads:
  - PortAudio's audio callback and the recorder thread (audio/recorder.py):
    record, save the WAV, cut utterances
  - the main thread: loads the model and transcribes (MLX must stay on the
    thread that loaded the model), writes the journal and the transcript files
  - a status thread: redraws the status line (time, level) a few times a second

LiveSession does the work and reports through a Listener, so a menu bar app or
web UI can drive it later. run_live is the command-line front end.

Nothing is lost if meetscribe crashes: the WAV is synced every few seconds and
each transcribed segment goes to the journal right away (see recover).
"""

from __future__ import annotations

import queue
import signal
import threading
import time
from datetime import datetime
from pathlib import Path

from meetscribe import console
from meetscribe.audio import permissions
from meetscribe.audio.devices import InputDevice, resolve_device
from meetscribe.audio.recorder import Recorder
from meetscribe.audio.wavfile import CrashSafeWav
from meetscribe.backends.base import Segment, Transcriber, get_backend
from meetscribe.config import Config
from meetscribe.console import (
    DIM,
    OFF,
    RED,
    detail,
    die,
    emit,
    format_count,
    format_duration,
    format_timestamp,
    info,
    ok,
    warn,
)
from meetscribe.filemode import resolve_prompt
from meetscribe.journal import Journal
from meetscribe.naming import new_session_base, series_title, slugify
from meetscribe.pipeline import Pipeline
from meetscribe.power import keep_awake
from meetscribe.transcript import COMPLETE, RECORDING, Meta, Transcript, write_outputs
from meetscribe.vad import Segmenter, Utterance, silero_speech_prob


class Listener:
    """Receives what happens in a live session. Override what you need."""

    def segments(self, segments: list[Segment]) -> None: ...

    def warning(self, message: str) -> None: ...


class LiveSession:
    def __init__(self, cfg: Config, backend: Transcriber, device: InputDevice,
                 series: str | None, title: str, prompt: str, listener: Listener):
        self.cfg = cfg
        self.listener = listener
        self.device = device
        self.start_time = datetime.now()
        self.base = new_session_base(cfg.output_dir, series, self.start_time)
        self.base.parent.mkdir(parents=True, exist_ok=True)
        self.wav_path = self.base.with_name(self.base.name + ".wav")
        self.meta = Meta(title=title, series=series, start=self.start_time, mode="live",
                         backend=backend.name, model=cfg.model, language=cfg.language_mode,
                         device=device.name, audio=self.wav_path.name, status=RECORDING)
        self.transcript = Transcript(self.meta)
        self.pipeline = Pipeline(backend, cfg.language, cfg.languages, prompt)
        self.utterances: queue.Queue[Utterance | None] = queue.Queue()
        self.wav = CrashSafeWav(self.wav_path)
        self.journal = Journal(self.base.with_name(self.base.name + ".jsonl"), self.meta)
        segmenter = Segmenter(
            silero_speech_prob(),
            threshold=cfg.vad_threshold,
            min_speech_ms=cfg.min_speech_ms,
            min_silence_ms=cfg.min_silence_ms,
            speech_pad_ms=cfg.speech_pad_ms,
            max_utterance_s=cfg.live_max_utterance_s,
        )
        self.recorder = Recorder(device, self.wav, segmenter, self.utterances)
        self.stopping = False
        self.stopping_requested = False  # set by the Ctrl+C handler
        self.finished = False
        self._warned_zero = False
        self._warned_device = False
        self.written: list[Path] = []
        self._save()

    @property
    def seconds(self) -> float:
        return self.recorder.seconds

    @property
    def backlog(self) -> int:
        return self.utterances.qsize()

    @property
    def speaking(self) -> bool:
        return self.recorder.segmenter.in_speech

    def start(self) -> None:
        try:
            self.recorder.start()
        except RuntimeError as e:
            self.journal.close(remove=False)
            die(str(e))

    def stop(self) -> None:
        """Stop recording. Utterances already cut are still transcribed by step()."""
        if not self.stopping:
            self.stopping = True
            self.recorder.stop()

    def step(self, timeout: float = 0.2) -> bool:
        """Transcribe the next utterance, if any. False once everything is done."""
        self._check_input()
        try:
            utt = self.utterances.get(timeout=timeout)
        except queue.Empty:
            return True
        if utt is None:
            return False
        segments = self.pipeline.process(utt)
        if segments:
            self.journal.add(segments)
            self.transcript.segments += segments
            self.listener.segments(segments)
        self._save()
        return True

    def finish(self) -> list[Path]:
        """After step() returned False: write the final transcript."""
        if self.recorder.error:
            self.listener.warning(f"Recording stopped with an error: {self.recorder.error}")
        self.meta.status = COMPLETE
        self._save()
        self.journal.close(remove=True)
        self.finished = True
        return self.written

    def abandon(self) -> None:
        """Hard stop (second Ctrl+C): keep the WAV and journal for 'meetscribe recover'."""
        self.recorder.stop()
        self.journal.close(remove=False)

    def _save(self) -> None:
        self.meta.duration = max(self.wav.duration, self.meta.duration)
        self.written = write_outputs(self.base, self.transcript, self.cfg.formats)

    def _check_input(self) -> None:
        if self.recorder.all_zero and not self._warned_zero:
            self._warned_zero = True
            self.listener.warning(permissions.silent_input_message())
        if not self.stopping and self.recorder._stream is not None and not self.recorder.active \
                and not self._warned_device:
            self._warned_device = True
            self.listener.warning(f"The input device {self.device.name} stopped delivering "
                                  "audio (unplugged?). Ending the recording.")
            self.stop()


# --- Command line -------------------------------------------------------------


class ConsoleListener(Listener):
    def __init__(self, cfg: Config):
        if cfg.language != "auto":
            self.main_language = cfg.language
        else:
            self.main_language = cfg.languages[0] if cfg.languages else None

    def segments(self, segments: list[Segment]) -> None:
        for s in segments:
            tag = f"({s.language}) " if s.language and s.language != self.main_language else ""
            emit(f"{DIM}[{format_timestamp(s.start)}]{OFF} {tag}{s.text}")

    def warning(self, message: str) -> None:
        warn(message)


def _level_bar(db: float, width: int = 10) -> str:
    filled = max(0, min(width, round((db + 60) / 60 * width)))
    return "▮" * filled + "▯" * (width - filled)


def _status_loop(session: LiveSession, done: threading.Event) -> None:
    while not done.wait(0.25):
        if session.stopping:
            left = session.backlog
            text = f"{DIM}Finishing transcription{OFF}" + (f" ({left} left)" if left else "")
        else:
            dot = f"{RED}●{OFF}" if int(time.monotonic() * 2) % 2 else " "
            behind = f"  {session.backlog} behind" if session.backlog > 1 else ""
            talk = "speech" if session.speaking else "      "
            text = (f"{dot} REC {format_duration(session.seconds)}  "
                    f"{_level_bar(session.recorder.level_db)}  {DIM}{talk}{behind}  "
                    f"Ctrl+C to end{OFF}")
        console.set_status(text)


def check_microphone() -> None:
    status = permissions.microphone_status()
    if status == permissions.NOT_DETERMINED:
        info(f"macOS will ask whether {permissions.host_app()} may use the microphone. "
             "Click Allow.")
        if not permissions.request_microphone():
            die(permissions.denied_message())
        ok("Microphone access allowed")
    elif status in (permissions.DENIED, permissions.RESTRICTED):
        die(permissions.denied_message())


def run_live(cfg: Config, name: str | None, cli_prompt: str | None) -> int:
    series = slugify(name) if name else None
    if name is not None and not series:
        die(f"--name '{name}' has no letters or digits to build a file name from")
    title = name.strip() if name and not name.islower() else series_title(series)

    backend = get_backend(cfg.backend, cfg.model)  # checks the model name early
    device = resolve_device(cfg.device)
    check_microphone()
    prompt, prompt_src = resolve_prompt(cfg, series, cli_prompt)

    info(f"Loading model {cfg.model}")
    t0 = time.monotonic()
    backend.load()
    ok(f"Model ready ({format_duration(time.monotonic() - t0)})")

    session = LiveSession(cfg, backend, device, series, title, prompt, ConsoleListener(cfg))
    info(f"Recording {title}" if series else "Recording")
    detail("Device", device.name)
    detail("Language", cfg.language_mode)
    if prompt:
        detail("Prompt", f"{prompt[:60]}{'…' if len(prompt) > 60 else ''} ({prompt_src})")
    detail("Saving to", str(session.base.parent))
    emit(f"{DIM}Press Ctrl+C to end the meeting.{OFF}")
    emit()

    interrupts = 0

    def on_sigint(signum, frame):
        nonlocal interrupts
        interrupts += 1
        if interrupts == 1:
            session.stopping_requested = True
        else:
            raise KeyboardInterrupt

    old_handler = signal.signal(signal.SIGINT, on_sigint)
    done = threading.Event()
    status = threading.Thread(target=_status_loop, args=(session, done), daemon=True)
    try:
        with keep_awake(cfg.keep_awake) as awake:
            session.start()
            status.start()
            if cfg.keep_awake and not awake:
                warn("caffeinate not found: the Mac may go to sleep during the meeting")
            while True:
                if session.stopping_requested and not session.stopping:
                    session.stop()
                    console.clear_status()
                    emit()
                    info("Meeting ended, transcribing the rest "
                         "(press Ctrl+C again to stop right away)")
                if not session.step():
                    break
            written = session.finish()
    except KeyboardInterrupt:
        done.set()
        console.clear_status()
        session.abandon()
        emit()
        warn("Stopped before the transcription was finished. The recording is complete.")
        emit(f'  Rebuild the transcript with: meetscribe recover "{session.wav_path}"')
        return 130
    finally:
        done.set()
        signal.signal(signal.SIGINT, old_handler)
        console.clear_status()

    _summary(session, written)
    return 0


def _summary(session: LiveSession, written: list[Path]) -> None:
    p = session.pipeline
    t = session.transcript
    info("Summary")
    detail("Duration", format_duration(session.meta.duration))
    detail("Speech", format_duration(p.audio_seconds))
    if p.audio_seconds:
        detail("Speed", f"{p.audio_seconds / max(p.busy_seconds, 1e-6):.1f}x realtime")
    detail("Words", format_count(t.word_count))
    if t.languages:
        detail("Languages", ", ".join(t.languages))
    if session.recorder.overflows:
        detail("Dropouts", f"{session.recorder.overflows} (audio buffer overflows)")
    detail("Recording", str(session.wav_path))
    for path in written:
        detail("Saved", str(path))
    emit()
    emit(f"{DIM}Re-transcribe later with a bigger model:{OFF}")
    emit(f'  meetscribe file "{session.wav_path}" --model large-v3 --overwrite')
