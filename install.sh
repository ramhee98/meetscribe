#!/usr/bin/env bash
#
# meetscribe installer: installs dependencies and the meetscribe command,
# creates meetscribe.toml and downloads the configured model. Safe to run again.
#
# Usage: ./install.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/meetscribe.toml"
EXAMPLE_CONFIG="${SCRIPT_DIR}/meetscribe.toml.example"
STEPS=7

if [[ -t 1 ]]; then
    C_BLUE=$'\033[1;34m' C_GREEN=$'\033[1;32m' C_YELLOW=$'\033[1;33m' C_RED=$'\033[1;31m'
    C_BOLD=$'\033[1m' C_OFF=$'\033[0m'
else
    C_BLUE="" C_GREEN="" C_YELLOW="" C_RED="" C_BOLD="" C_OFF=""
fi

info()  { printf '%s==>%s %s\n' "$C_BLUE" "$C_OFF" "$*"; }
ok()    { printf '%s✓%s %s\n' "$C_GREEN" "$C_OFF" "$*"; }
warn()  { printf '%s!%s %s\n' "$C_YELLOW" "$C_OFF" "$*" >&2; }
die()   { printf '%sError:%s %s\n' "$C_RED" "$C_OFF" "$*" >&2; exit 1; }
step()  { printf '\n%s[%s/%s]%s %s%s%s\n' "$C_BLUE" "$1" "$STEPS" "$C_OFF" "$C_BOLD" "$2" "$C_OFF"; }

trap 'echo; warn "Aborted"; exit 130' INT

# --- 1. Platform --------------------------------------------------------------

step 1 "Checking platform"
[[ "$(uname -s)" == "Darwin" ]] || die "meetscribe currently only supports macOS (detected $(uname -s))"
[[ "$(uname -m)" == "arm64" ]] \
    || die "meetscribe needs a Mac with Apple Silicon (M1 or newer) for mlx-whisper (detected $(uname -m))"
ok "macOS $(sw_vers -productVersion) on Apple Silicon"

# --- 2. Homebrew --------------------------------------------------------------

step 2 "Checking Homebrew"
if ! command -v brew >/dev/null 2>&1; then
    # Installed but not on PATH yet (common right after installing Homebrew).
    if [[ -x /opt/homebrew/bin/brew ]]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
        warn "Homebrew found at /opt/homebrew/bin/brew but not on your PATH. See 'brew shellenv' to fix this for new shells."
    fi
fi
if ! command -v brew >/dev/null 2>&1; then
    die "Homebrew is not installed. Install it with:

    /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"

Then run ./install.sh again."
fi
ok "Homebrew $(brew --version | head -n 1 | awk '{print $2}')"

# --- 3. Dependencies ----------------------------------------------------------

step 3 "Installing dependencies"
# "formula:command" pairs. A dependency counts as installed if its command is
# already on PATH, however it got there, so we never install a duplicate.
for dep in ffmpeg:ffmpeg uv:uv; do
    pkg="${dep%%:*}"
    cmd="${dep#*:}"
    if command -v "$cmd" >/dev/null 2>&1; then
        ok "$pkg already installed ($(command -v "$cmd"))"
    elif brew list --formula "$pkg" >/dev/null 2>&1; then
        ok "$pkg already installed"
    else
        info "Installing $pkg"
        brew install "$pkg"
        ok "$pkg installed"
    fi
done

# --- 4. meetscribe command ----------------------------------------------------

step 4 "Installing the meetscribe command"
# uv brings its own Python (the one macOS ships is too old), and installs
# meetscribe as an editable tool: changes in this folder take effect right away.
info "This downloads Python and the libraries (about 1 GB) the first time"
uv tool install --quiet --force --editable "$SCRIPT_DIR"
TOOL_BIN="$(uv tool dir --bin)"
if command -v meetscribe >/dev/null 2>&1; then
    ok "meetscribe installed ($(command -v meetscribe))"
else
    export PATH="${TOOL_BIN}:${PATH}"
    ok "meetscribe installed in ${TOOL_BIN}"
    warn "${TOOL_BIN} is not on your PATH yet. Run 'uv tool update-shell' and open a new terminal window."
fi

# --- 5. Config ----------------------------------------------------------------

step 5 "Setting up config"
if [[ -f "$CONFIG" ]]; then
    ok "Keeping existing $(basename "$CONFIG")"
else
    cp "$EXAMPLE_CONFIG" "$CONFIG"
    ok "Created $(basename "$CONFIG") from $(basename "$EXAMPLE_CONFIG")"
fi

show_settings() {
    meetscribe config --config "$CONFIG" || true
}

open_config() {
    if [[ -n "${EDITOR:-}" ]]; then
        # Unquoted on purpose so EDITOR="code -w" works.
        # shellcheck disable=SC2086
        $EDITOR "$CONFIG"
    else
        open -t "$CONFIG"
        # open -t returns immediately, so wait for the user to finish editing.
        read -r -p "Press Enter when you've saved your changes... " _ || { echo; die "Aborted"; }
    fi
}

# --- 6. Model choice ----------------------------------------------------------

step 6 "Choosing model"
info "Current settings:"
show_settings

# Only pause when someone is at the keyboard (not when piped or in CI).
if [[ -t 0 ]]; then
    editor_name="text editor"
    if [[ -n "${EDITOR:-}" ]]; then
        editor_name="$(basename "${EDITOR%% *}")"
    fi
    while true; do
        echo "Edit $(basename "$CONFIG") now if you want a different model, language or folder."
        read -r -p "Press Enter to continue, or o to open it in ${editor_name}: " answer \
            || { echo; die "Aborted"; }
        case "$answer" in
            "")
                break
                ;;
            o|O)
                open_config
                info "Current settings:"
                show_settings
                ;;
            *)
                warn "Unknown choice '$answer'"
                ;;
        esac
    done
fi

# --- 7. Model download ---------------------------------------------------------

step 7 "Downloading model"
meetscribe download-model --config "$CONFIG"

# --- Done ---------------------------------------------------------------------

cat <<EOF

${C_GREEN}${C_BOLD}meetscribe is ready.${C_OFF}

Usage:
  meetscribe devices                      list microphones
  meetscribe live --name weekly-team      record and transcribe a meeting
  meetscribe file <recording>             transcribe a recording

The first time you record, macOS asks whether your terminal app may use the
microphone. Click Allow.

Settings: ${CONFIG}
EOF
