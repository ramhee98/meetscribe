"""Reading recordings with ffmpeg: any format (wav, mp3, m4a, mp4, webm, ...)."""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np

from meetscribe.audio import SAMPLE_RATE
from meetscribe.console import die

FFMPEG_HINT = "Install it with: brew install ffmpeg"


def require_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            die(f"{tool} not found. {FFMPEG_HINT}")


def probe(path: Path) -> tuple[float | None, datetime | None]:
    """(duration in seconds, recording start from the file's metadata), each None if unknown."""
    cmd = ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams",
           "-select_streams", "a", str(path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        msg = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "unknown error"
        die(f"Can't read {path.name}: {msg}")
    info = json.loads(proc.stdout or "{}")
    if not info.get("streams"):
        die(f"{path.name} has no audio track")
    fmt = info.get("format", {})
    try:
        duration = float(fmt["duration"])
    except (KeyError, ValueError):
        duration = None
    created = None
    raw = fmt.get("tags", {}).get("creation_time")
    if raw:
        try:
            # ffprobe gives UTC like 2026-11-09T13:00:12.000000Z. Shown in local time.
            created = datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone()
            created = created.replace(tzinfo=None)
        except ValueError:
            created = None
    return duration, created


def file_start_time(path: Path) -> datetime:
    """When the recording started: metadata, else the file's creation time."""
    _, created = probe(path)
    if created:
        return created
    st = path.stat()
    return datetime.fromtimestamp(getattr(st, "st_birthtime", st.st_mtime))


def load_audio(path: Path) -> np.ndarray:
    """Decode to 16 kHz mono float32."""
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
           "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        msg = proc.stderr.decode(errors="replace").strip()
        die(f"ffmpeg couldn't decode {path.name}:\n{msg}")
    return np.frombuffer(proc.stdout, dtype=np.float32)
