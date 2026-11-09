"""Speaker recognition (who said what), optional.

- diarize.py: finds who speaks when (pyannote), with one voice embedding per speaker
- align.py: gives every transcript segment its speaker, splitting segments where
  the speaker changes mid-sentence (using Whisper's word timings)
- profiles.py: remembers named voices per meeting series, so the next meeting of
  the series gets names instead of "Speaker 1, 2, ..."
"""
