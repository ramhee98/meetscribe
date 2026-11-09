"""Keeping the Mac awake while recording, with macOS's caffeinate.

caffeinate -w <pid> ends by itself when meetscribe exits, even after a crash,
so the Mac never stays awake by accident. -i prevents idle sleep, -s system
sleep while on power. Closing the lid still sleeps the Mac (unless an external
display is connected), so keep it open during the meeting.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def keep_awake(enabled: bool = True) -> Iterator[bool]:
    """Yields True if caffeinate is running."""
    exe = shutil.which("caffeinate") if enabled else None
    if exe is None:
        yield False
        return
    proc = subprocess.Popen([exe, "-i", "-s", "-w", str(os.getpid())],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        yield True
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
