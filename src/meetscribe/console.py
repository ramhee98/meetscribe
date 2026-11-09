"""Terminal output in the same style as podscribe: ==> info, ✓ ok, ! warning, Error:.

All output goes through one lock, so the live status line (redrawn from another
thread) never gets mixed into a transcript line.
"""

from __future__ import annotations

import shutil
import sys
import threading
from typing import NoReturn

_lock = threading.RLock()
_status = ""  # current status line, redrawn after every printed line


def _tty(stream) -> bool:
    try:
        return stream.isatty()
    except (AttributeError, ValueError):
        return False


COLOR = _tty(sys.stdout)
BLUE, GREEN, YELLOW, RED, BOLD, DIM, OFF = (
    ("\033[1;34m", "\033[1;32m", "\033[1;33m", "\033[1;31m", "\033[1m", "\033[2m", "\033[0m")
    if COLOR
    else ("",) * 7
)


class Fatal(Exception):
    """An error that ends the program with a message (raised by die)."""


def _clear_status(stream) -> None:
    if _status and _tty(stream):
        stream.write("\r\033[K")


def _redraw_status() -> None:
    if _status and _tty(sys.stdout):
        sys.stdout.write(_status)
    sys.stdout.flush()


def emit(text: str = "", stream=None) -> None:
    """Print a line without disturbing the status line."""
    stream = stream or sys.stdout
    with _lock:
        _clear_status(sys.stdout)
        sys.stdout.flush()
        stream.write(text + "\n")
        stream.flush()
        _redraw_status()


def info(msg: str) -> None:
    emit(f"{BLUE}==>{OFF} {msg}")


def ok(msg: str) -> None:
    emit(f"{GREEN}✓{OFF} {msg}")


def warn(msg: str) -> None:
    emit(f"{YELLOW}!{OFF} {msg}", sys.stderr)


def error(msg: str) -> None:
    emit(f"{RED}Error:{OFF} {msg}", sys.stderr)


def die(msg: str) -> NoReturn:
    """Stop with an error message. cli.main prints it and exits with code 1."""
    raise Fatal(msg)


def detail(label: str, value: str) -> None:
    """One line of a summary block, like podscribe's '    Audio:            42:17'."""
    emit(f"    {label + ':':<18}{value}")


def set_status(text: str) -> None:
    """Show (or replace) a single status line at the bottom. Only on a terminal."""
    global _status
    if not _tty(sys.stdout):
        return
    width = shutil.get_terminal_size((80, 20)).columns
    with _lock:
        _clear_status(sys.stdout)
        # Cut to the terminal width so the line never wraps (it couldn't be cleared).
        _status = _fit(text, width - 1)
        _redraw_status()


def clear_status() -> None:
    global _status
    with _lock:
        _clear_status(sys.stdout)
        _status = ""
        sys.stdout.flush()


def _fit(text: str, width: int) -> str:
    """Cut text to width visible characters, keeping ANSI color codes intact."""
    out, visible, i = [], 0, 0
    while i < len(text):
        if text[i] == "\033":
            end = text.find("m", i)
            if end == -1:
                break
            out.append(text[i : end + 1])
            i = end + 1
            continue
        if visible >= width:
            break
        out.append(text[i])
        visible += 1
        i += 1
    return "".join(out) + (OFF if COLOR else "")


def format_duration(seconds: float) -> str:
    """mm:ss, or h:mm:ss from one hour on."""
    t = int(seconds + 0.5)
    h, m, s = t // 3600, t % 3600 // 60, t % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def format_timestamp(seconds: float) -> str:
    """Always hh:mm:ss, for transcript lines."""
    t = int(max(seconds, 0.0))
    return f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"


def format_size(n: float) -> str:
    """Human-readable size in decimal units, as in Finder."""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1000 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1000
    return f"{n:.1f} GB"


def format_count(n: int) -> str:
    """Swiss thousands separator, as in podscribe: 6'180."""
    return f"{n:,}".replace(",", "'")
