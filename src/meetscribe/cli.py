"""Command line: meetscribe live | file | devices | recover | download-model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from meetscribe import __version__, console
from meetscribe.config import Config, load_config, parse_language_option
from meetscribe.console import Fatal

DESCRIPTION = """\
Transcribe meetings locally on your Mac, live from the microphone or from a recording.

Settings are read from meetscribe.toml in the project folder or
~/.config/meetscribe/config.toml (see meetscribe.toml.example).
Command-line options override the config."""

EPILOG = """\
examples:
  meetscribe devices
  meetscribe live --name weekly-team
  meetscribe live --name jour-fixe --device "Jabra" --language de
  meetscribe file ~/Downloads/meeting.m4a --name weekly-team
  meetscribe file ~/Documents/meetscribe/weekly-team/2026-11-09_14-00_weekly-team.wav \\
      --model large-v3 --overwrite
  meetscribe recover ~/Documents/meetscribe/weekly-team/2026-11-09_14-00_weekly-team.wav
"""


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", type=Path, metavar="PATH",
                   help="use this config file instead of the default one")


def _transcription(p: argparse.ArgumentParser) -> None:
    p.add_argument("--name", metavar="NAME",
                   help="meeting name / series, e.g. weekly-team. Transcripts of the same "
                        "series go into one folder")
    p.add_argument("--model", metavar="MODEL",
                   help="whisper model, e.g. large-v3-turbo (default) or large-v3")
    p.add_argument("--language", metavar="LANG",
                   help="de, en, ... or auto. A list like de,en means auto, limited to those")
    p.add_argument("--prompt", metavar="TEXT",
                   help="names and terms to help whisper spell them (default: prompt.txt "
                        "in the series folder, then the config)")
    p.add_argument("--output-dir", type=Path, metavar="DIR",
                   help="where transcripts go (default: ~/Documents/meetscribe)")
    p.add_argument("--speakers", dest="speakers", action="store_true", default=None,
                   help="recognize who said what (needs the speakers extra, see README)")
    p.add_argument("--no-speakers", dest="speakers", action="store_false",
                   help="don't recognize speakers, even if enabled in the config")
    p.add_argument("--num-speakers", type=int, metavar="N",
                   help="number of people in the meeting, if known (helps speaker recognition)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="meetscribe", description=DESCRIPTION, epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=f"meetscribe {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    p = sub.add_parser("live", help="record from the microphone and transcribe as you go",
                       description="Record from the microphone and show the transcript live. "
                                   "Press Ctrl+C to end the meeting.")
    _transcription(p)
    p.add_argument("--device", metavar="DEVICE",
                   help="input device: part of its name or its number (see 'meetscribe devices')")
    p.add_argument("--no-keep-awake", dest="keep_awake", action="store_false", default=None,
                   help="don't stop the Mac from sleeping while recording")
    _common(p)

    p = sub.add_parser("file", help="transcribe an existing recording",
                       description="Transcribe a recording (wav, mp3, m4a, mp4, webm, ...). "
                                   "Given a meetscribe recording, its transcript is replaced "
                                   "(with --overwrite).")
    p.add_argument("path", type=Path, help="the recording")
    _transcription(p)
    p.add_argument("--overwrite", action="store_true", default=None,
                   help="replace an existing transcript")
    _common(p)

    p = sub.add_parser("devices", help="list audio input devices")
    _common(p)

    p = sub.add_parser("recover", help="rebuild the transcript of an interrupted live session",
                       description="Rebuild the transcript and fix the recording of a live "
                                   "session that ended unexpectedly (crash, power loss).")
    p.add_argument("path", type=Path, help="any file of the session (.wav, .md, .jsonl, ...)")
    _common(p)

    p = sub.add_parser("speakers", help="name the speakers of a transcript",
                       description="Name the speakers of a transcript made with --speakers. "
                                   "Their voices are remembered for the meeting series, so the "
                                   "next meeting gets the names automatically.")
    p.add_argument("path", type=Path, help="the transcript (.md, .json, ...) or its recording")
    p.add_argument("--set", dest="assignments", action="append", default=[], metavar="N=NAME",
                   help="name speaker N without asking, e.g. --set 1=Anna (repeatable). "
                        "Same name twice merges the speakers")
    _common(p)

    p = sub.add_parser("config", help="show the settings in effect and where they come from")
    _common(p)

    p = sub.add_parser("download-model", help="download the configured model ahead of time")
    p.add_argument("--model", metavar="MODEL", help="model to download instead")
    p.add_argument("--speakers", action="store_true", default=None,
                   help="also download (and check access to) the speaker recognition model")
    _common(p)
    return parser


def _apply(cfg: Config, args: argparse.Namespace) -> Config:
    over: dict = {}
    if getattr(args, "model", None):
        over["model"] = args.model
    if getattr(args, "language", None):
        over.update(parse_language_option(args.language))
    if getattr(args, "output_dir", None):
        over["output_dir"] = args.output_dir.expanduser()
    if getattr(args, "device", None) is not None:
        over["device"] = args.device
    if getattr(args, "keep_awake", None) is not None:
        over["keep_awake"] = args.keep_awake
    if getattr(args, "overwrite", None):
        over["overwrite"] = True
    if getattr(args, "speakers", None) is not None:
        over["speakers"] = args.speakers
    if getattr(args, "num_speakers", None) is not None:
        over["num_speakers"] = args.num_speakers
        if args.num_speakers and getattr(args, "speakers", None) is None:
            over["speakers"] = True  # --num-speakers implies --speakers
    return cfg.with_overrides(**over)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 2
    try:
        cfg = _apply(load_config(args.config), args)
        if args.command == "devices":
            from meetscribe.commands import show_devices

            return show_devices(cfg)
        if args.command == "file":
            from meetscribe.filemode import run_file

            return run_file(cfg, args.path, args.name, args.prompt)
        if args.command == "live":
            from meetscribe.live import run_live

            return run_live(cfg, args.name, args.prompt)
        if args.command == "recover":
            from meetscribe.commands import recover

            return recover(cfg, args.path)
        if args.command == "speakers":
            from meetscribe.commands import name_speakers

            return name_speakers(cfg, args.path, args.assignments)
        if args.command == "config":
            from meetscribe.commands import show_config

            return show_config(cfg)
        if args.command == "download-model":
            from meetscribe.commands import download_model

            return download_model(cfg, bool(args.speakers))
    except Fatal as e:
        console.clear_status()
        console.error(str(e))
        return 1
    except KeyboardInterrupt:
        console.clear_status()
        console.emit()
        console.warn("Aborted")
        return 130
    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
