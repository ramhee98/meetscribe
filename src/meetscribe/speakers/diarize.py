"""Who speaks when, with pyannote's speaker-diarization-community-1 pipeline.

The model is gated on Hugging Face: it needs a free account, accepting the
model's terms and 'hf auth login' once. After the first download it's loaded
from the cache.
"""

from __future__ import annotations

import contextlib
import io
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from meetscribe.audio import SAMPLE_RATE
from meetscribe.console import die

MODEL = "pyannote/speaker-diarization-community-1"

SETUP_HELP = f"""Speaker recognition uses the pyannote model {MODEL}, which needs a one-time setup:
  1. Log in (or create a free account) on https://huggingface.co and accept the terms on
     https://hf.co/{MODEL}
  2. Create an access token (type "Read") on https://hf.co/settings/tokens
  3. Run: uvx --from huggingface_hub hf auth login   (and paste the token)
Or turn speaker recognition off with --no-speakers."""

NOT_INSTALLED = """Speaker recognition is not installed. Install it with:
  ./install.sh   (answer yes to speaker recognition)
or: uv tool install --editable '.[speakers]'   (in the meetscribe folder)"""


@dataclass
class Turn:
    start: float
    end: float
    speaker: str  # pyannote's label, e.g. SPEAKER_00


@dataclass
class Diarization:
    turns: list[Turn]  # without overlaps, sorted by time
    embeddings: dict[str, np.ndarray]  # one voice embedding per speaker label


def is_installed() -> bool:
    try:
        import pyannote.audio  # noqa: F401
    except ImportError:
        return False
    return True


class Diarizer:
    def __init__(self, num_speakers: int = 0, max_speakers: int = 0):
        self.num_speakers = num_speakers or None
        self.max_speakers = max_speakers or None
        self._pipeline = None
        self.device = "cpu"

    def load(self) -> None:
        """Load the pipeline (downloading it the first time). Stops with setup help
        if it's not installed or the model isn't accessible."""
        if not is_installed():
            die(NOT_INSTALLED)
        import os
        import warnings

        # pyannote and its libraries are chatty about versions and deprecations.
        warnings.filterwarnings("ignore", module=r"(pyannote|torchaudio|torch|lightning)")
        os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "false")
        import torch
        from pyannote.audio import Pipeline

        try:
            # pyannote prints its own (less specific) help on failure; ours follows.
            with contextlib.redirect_stdout(io.StringIO()):
                pipeline = Pipeline.from_pretrained(MODEL)
        except Exception as e:
            die(_load_error(e))
        if pipeline is None:
            die(f"Can't load the speaker recognition model.\n\n{SETUP_HELP}")
        if torch.backends.mps.is_available():
            try:
                pipeline.to(torch.device("mps"))
                self.device = "mps"
            except Exception:
                self.device = "cpu"
        self._pipeline = pipeline

    def run(self, audio: np.ndarray,
            progress: Callable[[str, float], None] | None = None) -> Diarization:
        """Diarize 16 kHz mono audio. progress(step, fraction) is called now and then."""
        import torch

        def hook(step_name, step_artifact, file=None, total=None, completed=None):
            if progress is not None and total:
                progress(step_name, (completed or 0) / total)

        waveform = torch.from_numpy(np.array(audio, dtype=np.float32))[None]  # writable copy
        file = {"waveform": waveform, "sample_rate": SAMPLE_RATE}
        try:
            output = self._run(file, hook)
        except RuntimeError as e:
            if self.device != "mps":
                raise
            # Some operations aren't supported on the GPU in every torch version.
            self._pipeline.to(torch.device("cpu"))
            self.device = "cpu"
            if progress:
                progress(f"GPU failed ({e}), using CPU", 0.0)
            output = self._run(file, hook)
        return _convert(output)

    def _run(self, file, hook):
        return self._pipeline(file, num_speakers=self.num_speakers,
                              max_speakers=self.max_speakers, hook=hook)


def _load_error(e: Exception) -> str:
    """Explain why the model can't be loaded, as precisely as we can tell."""
    from huggingface_hub import get_token

    name = type(e).__name__
    if name == "GatedRepoError" and get_token():
        return (f"You're logged in to Hugging Face, but have no access to {MODEL} yet.\n"
                f"  - Accept the terms on https://hf.co/{MODEL} (fill in and submit the form)\n"
                "  - If you use a fine-grained token, it needs \"Read access to contents of all "
                "public gated repos you can access\". A token of type \"Read\" has it.\n"
                "Or turn speaker recognition off with --no-speakers.")
    if name in ("GatedRepoError", "RepositoryNotFoundError") or not get_token():
        return f"Can't load the speaker recognition model ({name}).\n\n{SETUP_HELP}"
    return (f"Can't load the speaker recognition model: {e}\n"
            "If you're offline: it's downloaded once, then it works without internet.")


def _convert(output) -> Diarization:
    turns = [Turn(float(seg.start), float(seg.end), str(label))
             for seg, _, label in output.exclusive_speaker_diarization.itertracks(yield_label=True)]
    turns.sort(key=lambda t: t.start)
    embeddings = {}
    if output.speaker_embeddings is not None:
        for label, emb in zip(output.speaker_diarization.labels(), output.speaker_embeddings,
                              strict=False):
            if np.all(np.isfinite(emb)):
                embeddings[str(label)] = np.asarray(emb, dtype=np.float32)
    return Diarization(turns, embeddings)
