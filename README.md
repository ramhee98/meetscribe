# meetscribe

Transcribe recurring meetings (weekly team meetings, jour fixes, ...) locally on macOS (Apple Silicon) with [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper). Audio never leaves your machine.

- **Live mode:** records from the microphone and shows the transcript while the meeting runs. Audio and transcript are saved continuously, so nothing is lost if the Mac or meetscribe crashes
- **File mode:** transcribes an existing recording (wav, mp3, m4a, mp4, webm, ...)
- Model: `large-v3-turbo` on the Apple Silicon GPU (downloaded automatically on first run)
- Languages: detected per utterance, limited to German and English by default, so meetings can switch between them
- Voice activity detection (Silero) skips silence and room noise, so Whisper doesn't make up text
- Optional [speaker recognition](#speaker-recognition): who said what, with names that are remembered per meeting series
- Output: Markdown and text with timestamps, plus a json file, grouped by meeting series:
  `~/Documents/meetscribe/weekly-team/2026-11-09_14-00_weekly-team.md`
- Settings in a config file, overridable on the command line

meetscribe is a sibling of [podscribe](https://github.com/ramhee98/podscribe), which transcribes podcast episodes.

## Installation

Requires a Mac with Apple Silicon (M1 or newer) and [Homebrew](https://brew.sh).

```bash
git clone https://github.com/ramhee98/meetscribe
cd meetscribe
./install.sh
```

The installer:

1. Checks that you're on macOS with Apple Silicon
2. Checks that Homebrew is installed, and prints the install command if it isn't
3. Installs `ffmpeg` and [`uv`](https://docs.astral.sh/uv/) via brew, skipping any that are already available
4. Asks whether to include [speaker recognition](#speaker-recognition) (optional, about 200 MB more), then installs the `meetscribe` command with `uv tool install --editable`. uv downloads its own Python (the one macOS ships is too old) and all libraries (about 1 GB the first time). The command lives in `~/.local/bin`; if that isn't on your `PATH`, run `uv tool update-shell` and open a new terminal
5. Creates `meetscribe.toml` from `meetscribe.toml.example` (an existing config is never overwritten)
6. Shows the settings and pauses so you can change the model, language or output folder. Press Enter to continue, or `o` to open the config in `$EDITOR` (or your default text editor)
7. Downloads the model (~1.6 GB for the default), unless it's already there. With speaker recognition, also checks access to the speaker model and downloads it (see [Setup](#setup))

`./install.sh --speakers` includes speaker recognition without asking, `./install.sh --no-speakers` removes it again. Once installed, re-running the installer keeps it.

Run `./install.sh` again at any time, e.g. after changing the model or pulling a new version. Steps that are already done are skipped. Because the install is editable, changes in this folder take effect without reinstalling (except new dependencies: run `./install.sh` again).

<details>
<summary>Manual installation</summary>

```bash
brew install ffmpeg uv
uv tool install --editable .              # or '.[speakers]' with speaker recognition
cp meetscribe.toml.example meetscribe.toml
meetscribe download-model
```

Or, without installing the command, run it from the project folder with `uv run meetscribe ...`.
</details>

### Microphone permission

macOS gives microphone access per app. meetscribe runs inside your terminal app (Terminal, iTerm, VS Code, ...), so that app needs the permission. The first time you record, macOS asks; click **Allow**.

If access was denied before, meetscribe stops with an explanation instead of recording silence:

1. Open **System Settings > Privacy & Security > Microphone**
2. Turn on your terminal app
3. Quit the terminal app completely (Cmd+Q) and open it again

If the permission can't be checked and the microphone delivers pure digital silence, meetscribe warns you during the recording.

## Usage

```bash
meetscribe devices                         # list microphones
meetscribe live --name weekly-team         # record and transcribe a meeting
meetscribe file ~/Downloads/meeting.m4a    # transcribe a recording
```

### Live mode

```bash
meetscribe live --name weekly-team
```

```
==> Loading model large-v3-turbo
✓ Model ready (00:05)
==> Recording Weekly Team
    Device:           Jabra Speak 510
    Language:         auto (de, en)
    Saving to:        /Users/ramon/Documents/meetscribe/weekly-team
Press Ctrl+C to end the meeting.

[00:00:01] Guten Morgen zusammen, willkommen zum Jour fixe. Heute besprechen wir das Budget.
[00:00:13] (en) Thanks Anna. I have prepared the numbers for the marketing budget.
[00:00:21] Gut, dann machen wir das so. Ramon übernimmt die Einladung.
● REC 00:00:24  ▮▮▮▮▮▮▯▯▯▯  speech  Ctrl+C to end
```

- Text appears a moment after someone pauses (typically 1 to 3 seconds). Each utterance is transcribed on its own, see [How it works](#how-it-works)
- The bottom line shows the recording time, the input level and whether speech is detected. If transcription falls behind (unlikely on an M1 or newer), it shows how many utterances are waiting
- Press **Ctrl+C** to end the meeting. meetscribe stops recording, transcribes what's left and writes the final files. Press Ctrl+C a second time to stop right away; the recording is complete either way, and `meetscribe recover` rebuilds the transcript
- The Mac is kept from sleeping while recording (with `caffeinate`). **Closing the lid still puts it to sleep** (unless an external display is connected), so keep it open. Use `--no-keep-awake` to turn this off
- The recording is kept as `<date>_<time>_<name>.wav` (16 kHz mono, about 115 MB per hour), so you can re-transcribe it later with a bigger model:

  ```bash
  meetscribe file ~/Documents/meetscribe/weekly-team/2026-11-09_14-00_weekly-team.wav --model large-v3 --overwrite
  ```

| Option | Description |
| --- | --- |
| `--name NAME` | Meeting name / series, e.g. `weekly-team` or `"Jour fixe"`. Transcripts of the same series go into one folder |
| `--device DEVICE` | Input device: part of its name (e.g. `Jabra`) or its number from `meetscribe devices` (default: system default) |
| `--model MODEL` | Whisper model, see [Models](#models) |
| `--language LANG` | `de`, `en`, ... or `auto`. A list like `de,en` means auto, limited to those languages |
| `--prompt "TEXT"` | Names and terms to help Whisper spell them, see [Prompts](#prompts) |
| `--output-dir DIR` | Where transcripts go (default `~/Documents/meetscribe`) |
| `--speakers`, `--no-speakers` | Recognize who said what, or don't (default: off, see `[speakers]` in the config) |
| `--num-speakers N` | Number of people in the meeting, if known. Implies `--speakers` |
| `--no-keep-awake` | Don't stop the Mac from sleeping |
| `--config PATH` | Use this config file instead of the default one |

### File mode

```bash
meetscribe file ~/Downloads/meeting.m4a --name weekly-team
```

Any format ffmpeg can read works: wav, mp3, m4a (e.g. Voice Memos), mp4, webm, ...

- The meeting's date and time are taken from the file's metadata, or else from when the file was created. The transcript goes into the series folder like a live recording: `weekly-team/2026-11-09_14-00_weekly-team.md`. Without `--name`, it goes into the output folder, named after the file: `2026-11-09_14-00_meeting.md`. The recording itself isn't copied
- Given a recording made by `meetscribe live`, the transcript is written next to it and keeps the meeting's name, date and device. Since that transcript already exists, add `--overwrite` to replace it (typically to re-transcribe with a bigger model)
- Existing transcripts are never replaced without `--overwrite` (or `overwrite = true` in the config)
- Transcripts are written only once transcription has finished

File mode takes the same options as live mode (except `--device` and `--no-keep-awake`), plus `--overwrite`.

On an M1 Pro, `large-v3-turbo` transcribes about 6x faster than realtime, so a 60-minute meeting takes about 10 minutes.

### Other commands

| Command | Description |
| --- | --- |
| `meetscribe devices` | List audio input devices with their numbers, marking the system default |
| `meetscribe speakers <transcript>` | Name the speakers of a transcript; their voices are remembered for the series, see [Naming speakers](#naming-speakers) |
| `meetscribe recover <file>` | Rebuild the transcript of a live session that ended unexpectedly, see [Crashes](#crashes-and-recover) |
| `meetscribe config` | Show the settings in effect and which config file they come from |
| `meetscribe download-model [--model M] [--speakers]` | Download the model ahead of time (e.g. before a meeting without internet). `--speakers` also downloads the speaker model and checks access to it |

## Output

```
~/Documents/meetscribe/
├── prompt.txt                                  # optional, see Prompts
├── weekly-team/
│   ├── prompt.txt                              # optional, for this series
│   ├── speakers.json                           # remembered voices, see Naming speakers
│   ├── 2026-11-02_14-00_weekly-team.md
│   ├── 2026-11-02_14-00_weekly-team.txt
│   ├── 2026-11-02_14-00_weekly-team.json
│   ├── 2026-11-02_14-00_weekly-team.wav        # live mode only
│   └── 2026-11-09_14-00_weekly-team.md ...
└── jour-fixe/
    └── ...
```

- The meeting name is turned into a file name (`"Jour fixe / Büro"` → `jour-fixe-buero`). Its original spelling is used as the transcript's title
- A second meeting of the same series in the same minute gets `-2`, `-3`, ...
- `.md` and `.txt` are for reading (choose with `formats` in the config). `.json` is always written: it holds the segments and metadata for meetscribe itself (and for future features like speaker labels and meeting minutes)

The Markdown transcript:

```markdown
# Weekly Team – 2026-11-09 14:00

- **Date:** Monday, 9 November 2026, 14:00–15:02
- **Duration:** 1:02:05
- **Language:** de, en
- **Model:** large-v3-turbo (mlx)
- **Recording:** [2026-11-09_14-00_weekly-team.wav](2026-11-09_14-00_weekly-team.wav)

## Transcript

**[00:00:05]** Guten Morgen zusammen. Fangen wir an.

**[00:00:15]** *(en)* Thanks, Anna. I have prepared the numbers.
```

Sentences are grouped into paragraphs, with a new paragraph after a pause of 2 seconds, after a minute, or when the language changes. In meetings with several languages, paragraphs in other languages than the main one are marked like `*(en)*`.

## Speaker recognition

Optionally, meetscribe finds out who said what, even from a single laptop microphone:

```markdown
- **Speakers:** Anna 43%, Eddy 31%, Speaker 3 26%

**[00:00:00] Anna:** Guten Morgen zusammen. Als erstes schauen wir auf die Pendenzen.

**[00:00:08] Eddy:** Die Offerte für den neuen Server ist da.

**[00:00:15] Speaker 3:** Das ist zu viel. Können wir nicht verhandeln?
```

It uses [pyannote](https://github.com/pyannote/pyannote-audio)'s `speaker-diarization-community-1` model, locally like everything else. Turn it on per meeting with `--speakers` (or `--num-speakers N`), or always with `enabled = true` under `[speakers]` in the config.

- **Live mode:** the transcript appears live as usual, without names. When the meeting ends, the speakers are recognized from the recording and added (roughly 4 minutes per hour of meeting on an M1 Pro)
- **File mode:** after transcribing
- The header shows each speaker's share of the speaking time
- If recognition fails, the transcript is saved without speakers, never lost

### Setup

The model is free, but you have to accept its terms on Hugging Face once:

1. Log in to (or create a free account on) [huggingface.co](https://huggingface.co), open [pyannote/speaker-diarization-community-1](https://hf.co/pyannote/speaker-diarization-community-1) and fill in and submit the form to get access
2. Create an access token of type **Read** on [hf.co/settings/tokens](https://hf.co/settings/tokens). A "Fine-grained" token works only with "Read access to contents of all public gated repos you can access" ticked
3. Log in on your Mac and paste the token:

   ```bash
   uvx --from huggingface_hub hf auth login
   ```

4. Check that it works (this also downloads the model, about 65 MB):

   ```bash
   meetscribe download-model --speakers
   ```

After that, no internet connection is needed. If the setup is missing, meetscribe says so before the meeting starts, not after it.

### Naming speakers

At first, speakers are called "Speaker 1", "Speaker 2", ... After a meeting, name them once:

```bash
meetscribe speakers ~/Documents/meetscribe/weekly-team/2026-11-09_14-00_weekly-team.md
```

```
==> Who is who? Enter a name, Enter to keep, - to remove a name, q to quit

Speaker 1  43% of the speaking time (00:15)
  [00:00:00] Guten Morgen zusammen. Ich eröffne unseren Jour fixe. Als erstes schauen wir auf…
  [00:00:31] Gut, dann übernimmst du das Sommerfest, und Eddy kümmert sich um den Server.
Name: Anna
...
✓ Remembered the voices of Anna, Eddy for this series (.../weekly-team/speakers.json)
```

- The transcript files are updated with the names
- The voices are remembered in `speakers.json` in the series folder. In the next meeting of the same series, they're **recognized automatically**: the summary shows `Anna: 43% (recognized, similarity 0.91)`. Only new people stay "Speaker N"
- Run `meetscribe speakers` again after a meeting to confirm the names (Enter keeps them). Every confirmed meeting improves the remembered voice
- Give two speakers the same name to merge them, when one person was split in two
- Without a terminal: `meetscribe speakers <transcript> --set 1=Anna --set 2=Eddy`
- To forget a voice, delete its entry from `speakers.json`

### How accurate is it?

With one microphone for the whole room, expect most but not all of the text to get the right speaker. In tests with four synthetic voices (two of them similar male voices) in a 4-minute meeting:

| Setting | Speaking time with the right speaker |
| --- | --- |
| Number of people detected automatically | 78% (the two similar voices were merged into one) |
| `--num-speakers 4` | 93% |

So **tell meetscribe how many people there are** (`--num-speakers` or `num_speakers` in the config) when you know it, or at least `max_speakers`. Real voices usually differ more than synthetic ones, and longer meetings give the model more to learn each voice from, but no test with real meetings has been done yet.

What goes wrong most often:

- **Short answers right after someone else** ("Ja, das mache ich", "Nein") are often given to the previous speaker
- Similar voices get merged when the number of people isn't given
- People talking at the same time: each word gets only one speaker
- People sitting far from the microphone

A USB conference microphone in the middle of the table helps. It's still one track, but everyone is about equally loud.

Under the hood: Whisper also returns the time of every word, each word gets the speaker that pyannote found at that moment, and sentences are split where the speaker changes (a single word that seems to belong to someone else in the middle of a turn is ignored as a timing glitch).

## Languages

Whisper detects the language of each utterance. By default it may only choose German or English (`languages = ["de", "en"]`): without this limit, Swiss German is often detected as Dutch or Afrikaans.

- **Swiss German** is transcribed into standard German. Whisper has no Swiss German written form, so dialect words are "translated", and accuracy is lower than for standard German. A [prompt](#prompts) with names and local terms helps a lot
- For meetings that are always in one language, set `language = "de"` (or `--language de`). That's a little faster and avoids misdetected short utterances
- Utterances under 2 seconds keep the previous utterance's language, since detection on a word or two is unreliable
- `languages = []` allows every language Whisper knows

## Prompts

A prompt tells Whisper how to spell names and terms that come up in the meeting. It's chosen in this order:

1. `--prompt "..."` on the command line
2. `prompt.txt` in the series folder (e.g. `~/Documents/meetscribe/weekly-team/prompt.txt`)
3. `prompt.txt` in the output folder (for all series)
4. `prompt` in the config

```
Ramon, Anna Caflisch, Jour fixe, Rhätische Bahn, Chur, Sprint-Review
```

Keep it short: a list of names and terms works better than full sentences. meetscribe adds the end of the previous utterance (in the same language) as context automatically.

## Models

| Model | Size | Notes |
| --- | --- | --- |
| `large-v3-turbo` | ~1.6 GB | Default. Fast and accurate, good for live mode |
| `large-v3` | ~3.1 GB | A bit more accurate, about 2–3x slower. Good for re-transcribing |
| `medium`, `small`, `base`, `tiny` | 1.5 GB – 75 MB | Faster, noticeably less accurate in German |

Models come from [mlx-community](https://huggingface.co/mlx-community) on Hugging Face and are stored in the Hugging Face cache (`~/.cache/huggingface`). Any repo with an MLX Whisper model works, e.g. `--model mlx-community/whisper-large-v3-turbo-q4`, and so does a local folder.

## Configuration

All settings are optional. Without a config file, meetscribe uses the built-in defaults shown in `meetscribe.toml.example`.

meetscribe reads the first config file it finds:

1. `--config <path>`
2. `meetscribe.toml` in the project folder (created by `./install.sh`, gitignored)
3. `~/.config/meetscribe/config.toml`

Settings are applied in this order, with later ones taking priority:

**built-in defaults → config file → command-line options**

The file is [TOML](https://toml.io). Every value is checked when the file is loaded; an invalid value, a misspelled setting or an unknown section stops meetscribe with an error naming the setting. `meetscribe config` shows the settings in effect.

| Setting | Default | Description |
| --- | --- | --- |
| `transcription.backend` | `"mlx"` | Transcription engine. Only `mlx` for now |
| `transcription.model` | `"large-v3-turbo"` | See [Models](#models). Overridden by `--model` |
| `transcription.language` | `"auto"` | `"de"`, `"en"`, ... or `"auto"`. Overridden by `--language` |
| `transcription.languages` | `["de", "en"]` | With `"auto"`: only choose between these. `[]` = any |
| `transcription.prompt` | `""` | Prompt used when there's no `--prompt` and no `prompt.txt` |
| `audio.device` | `""` | Input device: part of its name, its number, or `""` for the system default. Overridden by `--device` |
| `output.dir` | `"~/Documents/meetscribe"` | Where transcripts and recordings go. Overridden by `--output-dir` |
| `output.formats` | `["md", "txt"]` | Formats besides json: `md`, `txt` |
| `output.overwrite` | `false` | `true` lets `meetscribe file` replace existing transcripts, like `--overwrite` |
| `vad.threshold` | `0.5` | How sure the VAD must be that a moment is speech (0–1). Lower (e.g. `0.35`) if quiet speakers get cut off, higher in noisy rooms |
| `vad.min_speech_ms` | `250` | Speech shorter than this is ignored (clicks, coughs) |
| `vad.min_silence_ms` | `600` | A pause this long ends an utterance. Shorter = live text sooner, longer = more context for Whisper |
| `vad.speech_pad_ms` | `200` | Extra audio kept around each utterance, so words aren't clipped |
| `vad.live_max_utterance_s` | `20.0` | Longest utterance in live mode before it's cut (shorter = text sooner during monologues) |
| `vad.file_max_utterance_s` | `30.0` | Longest utterance in file mode (Whisper works on up to 30 s) |
| `live.keep_awake` | `true` | Keep the Mac awake while recording. Overridden by `--no-keep-awake` |
| `speakers.enabled` | `false` | Recognize who said what. Overridden by `--speakers` / `--no-speakers` |
| `speakers.num_speakers` | `0` | Number of people, if known (`0` = detect). Overridden by `--num-speakers` |
| `speakers.max_speakers` | `0` | The most people expected (`0` = no limit) |
| `speakers.match_threshold` | `0.6` | How similar (0–1) a voice must be to a remembered one to get its name. Lower it if known people stay "Speaker N", raise it if names get mixed up |

Example `meetscribe.toml` for German-only meetings with a USB conference mic:

```toml
[transcription]
language = "de"
prompt = "Ramon, Anna, Jour fixe"

[audio]
device = "Jabra"
```

## How it works

1. **Recording** (live mode): the microphone's audio is converted to 16 kHz mono and appended to the WAV file as it comes in
2. **Utterances:** [Silero VAD](https://github.com/snakers4/silero-vad) (the same model whisper.cpp and podscribe use) finds speech and cuts the audio at pauses into utterances of up to 20 s (live) or 30 s (file). After 60% of that length, a short pause is enough to cut, so long monologues are split at natural breaks
3. **Language:** for each utterance, Whisper's language detection picks the most likely of the allowed languages
4. **Transcription:** mlx-whisper transcribes the utterance on the GPU, with the prompt and the end of the previous utterance as context
5. **Filtering:** text Whisper is known to invent in silence ("Untertitel im Auftrag des ZDF", "Vielen Dank fürs Zuschauen", ...) and text placed after the end of the audio are dropped
6. **Output:** segments get timestamps on the meeting's timeline and are written to the transcript files
7. **Speakers** (optional): after the transcription, see [Speaker recognition](#speaker-recognition)

Live and file mode use the same steps, so a live transcript and a later re-transcription of the same recording are directly comparable.

Example summary after a file:

```
==> Summary
    Audio:            07:44
    Speech:           05:21 in 45 segments
    Transcribed:      01:18 (total 01:24 incl. decoding and model loading)
    Speed:            5.9x realtime
    Words:            843
    Languages:        de, en
```

### Crashes and recover

Live mode is built so that a crash, a forced quit or a power loss loses at most a few seconds:

- The WAV file's header is updated and flushed to disk every 5 seconds
- Every transcribed segment is appended to a journal (`<session>.jsonl`) and flushed to disk immediately
- The `.md`, `.txt` and `.json` files are rewritten after every utterance (atomically: a temporary file, then renamed), marked as "Recording in progress"

When the session ends normally, the journal is deleted. If it's still there, the session was interrupted:

```bash
meetscribe recover ~/Documents/meetscribe/weekly-team/2026-11-09_14-00_weekly-team.wav
```

This fixes the WAV header, rebuilds the transcript from the journal (marked as recovered) and deletes the journal. Any file of the session works as the argument. The utterances that were still waiting for transcription are missing, so meetscribe suggests re-transcribing the complete recording with `meetscribe file ... --overwrite`.

## Development

```bash
uv sync --extra speakers # creates .venv with dev tools (and speaker recognition)
uv run pytest            # tests: naming, config, VAD cuts, rendering, journal, WAV repair, speakers
uv run ruff check .      # lint
uv run meetscribe ...    # run from the checkout
```

The tests only cover logic that needs no microphone or model, so they run in under a second.

### Files

| File | Purpose |
| --- | --- |
| `install.sh` | Installer (dependencies, command, config, model) |
| `meetscribe.toml.example` | Documented config with all defaults |
| `meetscribe.toml` | Your local config (gitignored) |
| `src/meetscribe/cli.py` | Command line |
| `src/meetscribe/config.py` | Settings: defaults, loading, validation |
| `src/meetscribe/live.py` | Live mode (`LiveSession` reports to a `Listener`, so a menu bar app or web UI can drive it) |
| `src/meetscribe/filemode.py` | File mode |
| `src/meetscribe/pipeline.py` | Utterance → segments: language, prompt, filtering, timestamps |
| `src/meetscribe/vad.py` | Cutting audio into utterances |
| `src/meetscribe/backends/` | Transcription backends behind a small interface (`base.py`). `mlx.py` for Apple Silicon |
| `src/meetscribe/audio/` | Devices, microphone permission, recording, crash-safe WAV, decoding |
| `src/meetscribe/speakers/` | Speaker recognition: diarization (`diarize.py`), assigning speakers to words (`align.py`), remembered voices (`profiles.py`), naming (`naming.py`) |
| `src/meetscribe/transcript.py` | Transcript model and md/txt/json output |
| `src/meetscribe/journal.py` | Live session journal (crash safety) |
| `src/meetscribe/naming.py` | Folder and file names |
| `src/meetscribe/power.py` | Keeping the Mac awake |
| `tests/` | Tests |

## Roadmap

Not built yet, but the structure is ready for it:

- **Speakers in live mode**: names while the meeting runs, not only afterwards
- **Meeting minutes** with decisions and action items, generated by an LLM, using the previous meeting of the same series (the newest transcript in the series folder) as context
- **Menu bar app or small web UI** driving `LiveSession`
- **Linux support** with a faster-whisper backend
