"""macOS microphone permission.

When access is denied, macOS doesn't report an error: the microphone simply
delivers silence. So the permission is checked up front with AVFoundation. The
permission belongs to the app meetscribe runs in (Terminal, iTerm, VS Code, ...),
not to meetscribe itself.
"""

from __future__ import annotations

import os
import sys
import threading

AUTHORIZED, DENIED, RESTRICTED, NOT_DETERMINED, UNKNOWN = (
    "authorized", "denied", "restricted", "not_determined", "unknown")

SETTINGS_URL = "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone"

_TERMINALS = {
    "Apple_Terminal": "Terminal",
    "iTerm.app": "iTerm",
    "vscode": "Visual Studio Code (or Cursor)",
    "WarpTerminal": "Warp",
    "ghostty": "Ghostty",
    "WezTerm": "WezTerm",
    "Hyper": "Hyper",
    "tmux": "the terminal app running tmux",
}


def host_app() -> str:
    """The app that macOS asks for permission, as far as we can tell."""
    return _TERMINALS.get(os.environ.get("TERM_PROGRAM", ""), "your terminal app")


def _av():
    if sys.platform != "darwin":
        return None
    try:
        import AVFoundation
    except ImportError:
        return None
    return AVFoundation


def microphone_status() -> str:
    av = _av()
    if av is None:
        return UNKNOWN
    status = av.AVCaptureDevice.authorizationStatusForMediaType_(av.AVMediaTypeAudio)
    return {0: NOT_DETERMINED, 1: RESTRICTED, 2: DENIED, 3: AUTHORIZED}.get(int(status), UNKNOWN)


def request_microphone(timeout: float = 120.0) -> bool:
    """Show macOS's permission dialog and wait for the answer."""
    av = _av()
    if av is None:
        return True
    answer = threading.Event()
    granted = []

    def done(ok):
        granted.append(bool(ok))
        answer.set()

    av.AVCaptureDevice.requestAccessForMediaType_completionHandler_(av.AVMediaTypeAudio, done)
    answer.wait(timeout)
    return bool(granted and granted[0])


def denied_message() -> str:
    app = host_app()
    return (
        f"meetscribe has no access to the microphone.\n\n"
        f"macOS gives microphone access per app, and meetscribe runs inside {app}.\n"
        f"To allow it:\n"
        f"  1. Open System Settings > Privacy & Security > Microphone\n"
        f"  2. Turn on {app}\n"
        f"  3. Quit {app} completely (Cmd+Q) and open it again\n"
        f"  4. Run meetscribe again\n\n"
        f"Open the settings page directly with:\n"
        f"  open \"{SETTINGS_URL}\""
    )


def silent_input_message() -> str:
    return (
        "The microphone delivers pure digital silence. This usually means macOS is "
        f"blocking microphone access for {host_app()}.\n"
        "Check System Settings > Privacy & Security > Microphone, or pick another "
        "device with --device (see 'meetscribe devices')."
    )
