"""mlx-whisper backend: runs Whisper on the Apple Silicon GPU."""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import numpy as np

from meetscribe.backends.base import Segment, Transcriber
from meetscribe.console import die

# Short names for the converted models on https://huggingface.co/mlx-community
MODELS = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}

MODEL_HELP = """Common models (downloaded automatically on first use):
  tiny, base, small, medium   faster, less accurate
  large-v3-turbo              (~1.6 GB, default) fast and accurate
  large-v3                    (~3.1 GB) slower, a bit more accurate
You can also give any Hugging Face repo with an MLX Whisper model, e.g.
  mlx-community/whisper-large-v3-turbo-q4, or a local folder."""


def resolve_model(model: str) -> str:
    if model in MODELS:
        return MODELS[model]
    if "/" in model or Path(model).expanduser().exists():
        return str(Path(model).expanduser()) if Path(model).expanduser().exists() else model
    die(f"Unknown model '{model}'.\n\n{MODEL_HELP}")


def cached_path(model: str) -> Path | None:
    """The local folder of a downloaded model, or None (no network access)."""
    repo = resolve_model(model)
    if Path(repo).exists():
        return Path(repo)
    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(repo_id=repo, local_files_only=True))
    except Exception:
        return None


def download(model: str) -> Path:
    """The model's local folder, downloading it first if needed."""
    path = cached_path(model)
    if path is not None:
        return path
    repo = resolve_model(model)
    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(repo_id=repo))
    except Exception as e:
        die(f"Model download failed for {repo}: {e}\n\n{MODEL_HELP}")


class MlxWhisper(Transcriber):
    name = "mlx"

    def __init__(self, model: str):
        super().__init__(model)
        self.repo = resolve_model(model)
        self._model = None
        self._dtype = None

    @property
    def model_id(self) -> str:
        return self.repo

    def load(self) -> None:
        if sys.platform != "darwin" or platform.machine() != "arm64":
            die("The mlx backend needs a Mac with Apple Silicon (M1 or newer).")
        try:
            import mlx.core as mx
            from mlx_whisper.transcribe import ModelHolder
        except ImportError as e:
            die(f"mlx-whisper is not installed ({e}). Run ./install.sh or 'uv sync'.")
        self._mx = mx
        self._dtype = mx.float16
        # A local path: no network check once the model is downloaded.
        self._path = str(download(self.model))
        self._model = ModelHolder.get_model(self._path, self._dtype)
        # Warm-up: the first run compiles GPU kernels, which takes a few seconds.
        warm = np.zeros(16000, dtype=np.float32)
        self.language_probs(warm)
        self.transcribe(warm, "en", None)

    def language_probs(self, audio: np.ndarray) -> dict[str, float]:
        from mlx_whisper.audio import N_FRAMES, N_SAMPLES, log_mel_spectrogram, pad_or_trim

        mel = log_mel_spectrogram(audio, n_mels=self._model.dims.n_mels, padding=N_SAMPLES)
        segment = pad_or_trim(mel, N_FRAMES, axis=-2).astype(self._dtype)
        _, probs = self._model.detect_language(segment)
        return {k: float(v) for k, v in probs.items()}

    def transcribe(self, audio: np.ndarray, language: str, prompt: str | None) -> list[Segment]:
        import mlx_whisper

        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self._path,
            language=language,
            initial_prompt=prompt or None,
            condition_on_previous_text=False,
            verbose=None,
        )
        return [
            Segment(
                start=float(s["start"]),
                end=float(s["end"]),
                text=s["text"].strip(),
                language=language,
                no_speech_prob=float(s.get("no_speech_prob", 0.0)),
                avg_logprob=float(s.get("avg_logprob", 0.0)),
            )
            for s in result["segments"]
        ]
