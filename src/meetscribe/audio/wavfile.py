"""A WAV writer that survives crashes.

A normal WAV writer only fills in the file's length fields when it's closed, so
after a crash the file looks empty to most players. This one writes 16-bit PCM
and updates the length fields every few seconds (and flushes to disk), so at
most the last few seconds are missing from the header. repair_wav() fixes the
header of a file that wasn't closed, using the real file size.
"""

from __future__ import annotations

import os
import struct
import time
from pathlib import Path

import numpy as np

HEADER_SIZE = 44


def _header(sample_rate: int, channels: int, data_bytes: int) -> bytes:
    block_align = channels * 2
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_bytes, b"WAVE",
        b"fmt ", 16, 1, channels, sample_rate, sample_rate * block_align, block_align, 16,
        b"data", data_bytes,
    )


class CrashSafeWav:
    def __init__(self, path: Path, sample_rate: int = 16000, sync_every_s: float = 5.0):
        self.path = path
        self.sample_rate = sample_rate
        self.sync_every_s = sync_every_s
        self.data_bytes = 0
        self._f = open(path, "xb")  # never overwrite an existing recording
        self._f.write(_header(sample_rate, 1, 0))
        self._last_sync = time.monotonic()

    @property
    def duration(self) -> float:
        return self.data_bytes / 2 / self.sample_rate

    def write(self, samples: np.ndarray) -> None:
        """Append float32 samples in [-1, 1]."""
        pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        self._f.write(pcm)
        self.data_bytes += len(pcm)
        if time.monotonic() - self._last_sync >= self.sync_every_s:
            self.sync()

    def sync(self) -> None:
        """Update the length fields and flush everything to disk."""
        self._f.seek(4)
        self._f.write(struct.pack("<I", 36 + self.data_bytes))
        self._f.seek(40)
        self._f.write(struct.pack("<I", self.data_bytes))
        self._f.seek(0, os.SEEK_END)
        self._f.flush()
        os.fsync(self._f.fileno())
        self._last_sync = time.monotonic()

    def close(self) -> None:
        if not self._f.closed:
            self.sync()
            self._f.close()


def repair_wav(path: Path) -> tuple[bool, float]:
    """Fix the length fields of a CrashSafeWav file that wasn't closed.
    Returns (changed, duration in seconds)."""
    size = path.stat().st_size
    with open(path, "r+b") as f:
        head = f.read(HEADER_SIZE)
        if len(head) < HEADER_SIZE or head[:4] != b"RIFF" or head[36:40] != b"data":
            raise ValueError(f"{path.name} is not a meetscribe WAV recording")
        channels, rate = struct.unpack("<HI", head[22:28])
        data_bytes = size - HEADER_SIZE
        data_bytes -= data_bytes % (2 * channels)  # drop a half-written sample
        stored = struct.unpack("<I", head[40:44])[0]
        changed = stored != data_bytes
        if changed:
            f.seek(4)
            f.write(struct.pack("<I", 36 + data_bytes))
            f.seek(40)
            f.write(struct.pack("<I", data_bytes))
            f.truncate(HEADER_SIZE + data_bytes)
    return changed, data_bytes / 2 / channels / rate
