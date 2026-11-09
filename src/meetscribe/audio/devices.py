"""Audio input devices: listing and picking one by name or number."""

from __future__ import annotations

from dataclasses import dataclass

from meetscribe.console import die


@dataclass(frozen=True)
class InputDevice:
    index: int
    name: str
    channels: int
    sample_rate: float
    is_default: bool


def _sd():
    try:
        import sounddevice
    except OSError as e:  # PortAudio missing
        die(f"Can't access audio devices: {e}")
    return sounddevice


def list_input_devices() -> list[InputDevice]:
    sd = _sd()
    try:
        default_in = sd.default.device[0]
    except Exception:
        default_in = -1
    devices = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            devices.append(InputDevice(i, d["name"], d["max_input_channels"],
                                       float(d["default_samplerate"]), i == default_in))
    return devices


def resolve_device(spec: str) -> InputDevice:
    """'' -> the system default input; '3' -> device number 3; otherwise the one
    device whose name contains spec (case-insensitive)."""
    devices = list_input_devices()
    if not devices:
        die("No audio input devices found. Connect a microphone and try again.")
    hint = "\nRun 'meetscribe devices' to see the available devices."

    spec = spec.strip()
    if not spec:
        for d in devices:
            if d.is_default:
                return d
        return devices[0]

    if spec.isdigit():
        for d in devices:
            if d.index == int(spec):
                return d
        die(f"No input device with number {spec}.{hint}")

    matches = [d for d in devices if spec.lower() in d.name.lower()]
    exact = [d for d in matches if d.name.lower() == spec.lower()]
    if len(exact) == 1:
        return exact[0]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        die(f"No input device matches '{spec}'.{hint}")
    names = "\n".join(f"  {d.index}) {d.name}" for d in matches)
    die(f"'{spec}' matches several devices:\n{names}\nUse more of the name, or the number.")
