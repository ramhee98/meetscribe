"""Recording from a microphone.

The audio callback (PortAudio's thread) only copies each block into a queue.
A worker thread converts it to 16 kHz, appends it to the WAV file, measures the
level and feeds the VAD, which hands finished utterances to the main thread
for transcription. A None in the utterance queue means the recording ended.
"""

from __future__ import annotations

import math
import queue
import threading

import numpy as np

from meetscribe.audio import SAMPLE_RATE
from meetscribe.audio.devices import InputDevice
from meetscribe.audio.wavfile import CrashSafeWav
from meetscribe.vad import Segmenter, Utterance

ZERO_CHECK_S = 3.0  # this much pure digital silence at the start means: no permission


class Recorder:
    def __init__(self, device: InputDevice, wav: CrashSafeWav, segmenter: Segmenter,
                 utterances: queue.Queue[Utterance | None]):
        self.device = device
        self.wav = wav
        self.segmenter = segmenter
        self.utterances = utterances
        self.level_db = -120.0
        self.overflows = 0
        self.error: BaseException | None = None
        self.all_zero: bool | None = None  # None until ZERO_CHECK_S of audio arrived
        self._raw: queue.Queue[np.ndarray | None] = queue.Queue()
        self._samples = 0
        self._peak = 0.0
        self._stream = None
        self._thread: threading.Thread | None = None
        self._stopped = False

    @property
    def seconds(self) -> float:
        return self._samples / SAMPLE_RATE

    @property
    def active(self) -> bool:
        """False if the stream stopped by itself (e.g. the device was unplugged)."""
        return self._stream is not None and bool(self._stream.active)

    def start(self) -> None:
        import sounddevice as sd

        rate = int(self.device.sample_rate)
        try:
            self._stream = sd.InputStream(
                device=self.device.index, channels=1, samplerate=rate, dtype="float32",
                blocksize=rate // 20, callback=self._callback)
            self._stream.start()
        except Exception as e:  # PortAudioError and friends
            raise RuntimeError(f"Can't open {self.device.name}: {e}") from e
        self._thread = threading.Thread(target=self._run, args=(rate,), name="recorder",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop recording, process what's left and close the WAV. Idempotent."""
        if self._stopped:
            return
        self._stopped = True
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
        self._raw.put(None)
        if self._thread is not None:
            self._thread.join()

    def _callback(self, indata, frames, time_info, status) -> None:
        if status and status.input_overflow:
            self.overflows += 1
        self._raw.put(indata[:, 0].copy())

    def _run(self, rate: int) -> None:
        import soxr

        try:
            resampler = soxr.ResampleStream(rate, SAMPLE_RATE, 1, dtype="float32")
            while True:
                block = self._raw.get()
                if block is None:
                    break
                self._handle(resampler.resample_chunk(block))
            self._handle(resampler.resample_chunk(np.zeros(0, dtype=np.float32), last=True))
            for utt in self.segmenter.flush():
                self.utterances.put(utt)
        except BaseException as e:
            self.error = e
        finally:
            try:
                self.wav.close()
            except Exception as e:
                self.error = self.error or e
            self.utterances.put(None)

    def _handle(self, x: np.ndarray) -> None:
        if not x.size:
            return
        self.wav.write(x)
        self._samples += x.size
        rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
        self.level_db = 20 * math.log10(rms) if rms > 1e-6 else -120.0
        if self.all_zero is None:
            self._peak = max(self._peak, float(np.max(np.abs(x))))
            if self.seconds >= ZERO_CHECK_S:
                self.all_zero = self._peak == 0.0
        for utt in self.segmenter.feed(x):
            self.utterances.put(utt)
