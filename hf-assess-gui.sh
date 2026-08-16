#!/usr/bin/env bash
# Hugging Face Assessment launcher
# Offers a real local model download, then opens the operator GUI.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYZER="$SCRIPT_DIR/hf_assess.py"
GUI_PY="$SCRIPT_DIR/hf-assess-gui.py"
VENV="$SCRIPT_DIR/.hf-assess-venv"
REQ="$SCRIPT_DIR/hf_assess_requirements.txt"
PY_SYS="${PYTHON:-python3}"

usage() {
  cat <<'EOF'
Usage: ./hf-assess-gui.sh [option] [target-or-file ...]

  (no args)          Open the one-box GUI. If no model is on disk, offer a download.
  10.0.0.8           Open the GUI with that host already detected
  nmap.xml           Open the GUI with that scan already loaded
  --gui [...]        Same as no args, skip the first-run download prompt
  --download         Offer / download a local model, then exit
  --list             Show catalog and which models are already on disk
  --bootstrap        Create the venv and install transformers
  --check            Print environment status
  --probe TEXT...    Classify input in the terminal and exit
  -h, --help         Show this help

Paste an IP, URL, nmap dump, JSON, CVE list, or file path. The GUI fills the rest.
EOF
}

need_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing $1" >&2
    exit 2
  fi
}

venv_python() {
  if [[ -x "$VENV/bin/python" ]]; then
    echo "$VENV/bin/python"
  else
    echo "$PY_SYS"
  fi
}

model_cached() {
  local repo="$1"
  local slug="models--${repo//\//--}"
  local dir="$HOME/.cache/huggingface/hub/$slug"
  [[ -d "$dir" ]] && find "$dir" -name 'config.json' -print -quit | grep -q .
}

status_mark() {
  if model_cached "$1"; then
    echo "ON DISK"
  else
    echo "not downloaded"
  fi
}

catalog() {
  cat <<'EOF'
1|HuggingFaceTB/SmolLM2-360M-Instruct|SmolLM2 360M Instruct|~720 MB|Recommended. Small instruct model for analysis prompts.
2|Qwen/Qwen2.5-0.5B-Instruct|Qwen2.5 0.5B Instruct|~1.0 GB|Stronger instruct model. Still laptop-friendly.
3|microsoft/DialoGPT-medium|DialoGPT medium|~1.4 GB|Original script default. Weaker at structured analysis.
4|sshleifer/tiny-gpt2|tiny-gpt2|~1 MB|Smoke-test weights only. Not useful for analysis.
EOF
}

print_catalog() {
  echo "Local model catalog"
  echo
  while IFS='|' read -r num repo label size why; do
    printf '  %s) %-36s  %8s  [%s]\n      %s\n      %s\n\n' \
      "$num" "$label" "$size" "$(status_mark "$repo")" "$repo" "$why"
  done < <(catalog)
}

ensure_venv() {
  if [[ -x "$VENV/bin/python" ]]; then
    if ! "$VENV/bin/python" -c "import transformers, huggingface_hub" >/dev/null 2>&1; then
      echo "Venv exists but transformers is missing. Installing requirements…"
      if [[ -f "$REQ" ]]; then
        "$VENV/bin/pip" install -r "$REQ"
      else
        "$VENV/bin/pip" install "transformers>=4.45" "huggingface_hub>=0.25" "accelerate>=1.0" "safetensors" "requests"
      fi
    fi
    return
  fi
  echo "Creating virtualenv at $VENV"
  "$PY_SYS" -m venv "$VENV"
  "$VENV/bin/python" -m pip install --upgrade pip
  if [[ -f "$REQ" ]]; then
    "$VENV/bin/pip" install -r "$REQ"
  else
    "$VENV/bin/pip" install "transformers>=4.45" "huggingface_hub>=0.25" "accelerate>=1.0" "safetensors" "requests"
  fi
}

download_repo() {
  local repo="$1"
  local py
  py="$(venv_python)"
  echo
  echo "Downloading $repo"
  echo "This is a real Hub snapshot into ~/.cache/huggingface/hub"
  echo
  REPO_ID="$repo" "$py" -u - <<'PY'
from huggingface_hub import snapshot_download
import os

repo = os.environ["REPO_ID"]
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY") or None
path = snapshot_download(repo_id=repo, token=token)
print("Saved:", path)
PY
}

pick_model_macos() {
  osascript <<'APPLESCRIPT'
set theChoices to {"SmolLM2 360M Instruct  — recommended, ~720 MB", "Qwen2.5 0.5B Instruct  — ~1.0 GB", "DialoGPT medium  — original default, ~1.4 GB", "tiny-gpt2  — smoke test only, ~1 MB", "Skip download and open the operator"}
set thePick to choose from list theChoices with prompt "Download a local Hugging Face model before opening the operator?" with title "HF Assessment — local model" default items {"SmolLM2 360M Instruct  — recommended, ~720 MB"} OK button name "Continue" cancel button name "Quit"
if thePick is false then
  return "QUIT"
end if
return item 1 of thePick
APPLESCRIPT
}

pick_model_terminal() {
  print_catalog
  echo "0) Skip download and open the operator"
  echo
  read -r -p "Choose a model [1]: " choice
  case "${choice:-1}" in
    1) echo "SmolLM2 360M Instruct  — recommended, ~720 MB" ;;
    2) echo "Qwen2.5 0.5B Instruct  — ~1.0 GB" ;;
    3) echo "DialoGPT medium  — original default, ~1.4 GB" ;;
    4) echo "tiny-gpt2  — smoke test only, ~1 MB" ;;
    0) echo "Skip download and open the operator" ;;
    *) echo "SmolLM2 360M Instruct  — recommended, ~720 MB" ;;
  esac
}

label_to_repo() {
  case "$1" in
    SmolLM2*) echo "HuggingFaceTB/SmolLM2-360M-Instruct" ;;
    Qwen*) echo "Qwen/Qwen2.5-0.5B-Instruct" ;;
    DialoGPT*) echo "microsoft/DialoGPT-medium" ;;
    tiny-gpt2*) echo "sshleifer/tiny-gpt2" ;;
    Skip*|QUIT|"") echo "" ;;
    *) echo "" ;;
  esac
}

offer_download() {
  local pick repo
  pick=""
  if [[ "$(uname -s)" == "Darwin" ]] && command -v osascript >/dev/null 2>&1; then
    pick="$(pick_model_macos 2>/dev/null | tr -d '\r')" || pick=""
  fi
  if [[ -z "$pick" || "$pick" == "false" ]]; then
    if [[ -t 0 ]]; then
      pick="$(pick_model_terminal)"
    else
      echo "No GUI picker available; open with --gui after downloading a model."
      return 0
    fi
  fi
  if [[ "$pick" == "QUIT" ]]; then
    echo "Cancelled."
    exit 0
  fi
  repo="$(label_to_repo "$pick")"
  if [[ -z "$repo" ]]; then
    echo "Skipping download."
    return 0
  fi
  if model_cached "$repo"; then
    echo "$repo is already on disk."
    CHOSEN_REPO="$repo"
    return 0
  fi
  download_repo "$repo"
  CHOSEN_REPO="$repo"
}

any_model_cached() {
  model_cached "HuggingFaceTB/SmolLM2-360M-Instruct" \
    || model_cached "Qwen/Qwen2.5-0.5B-Instruct" \
    || model_cached "microsoft/DialoGPT-medium"
}

open_gui() {
  need_file "$GUI_PY"
  local py extra=()
  py="$(venv_python)"
  if ! "$py" -c "import tkinter" >/dev/null 2>&1; then
    echo "tkinter is not available in $py" >&2
    echo "Install python-tk, or run: brew install python-tk" >&2
    exit 2
  fi
  if [[ -n "${CHOSEN_REPO:-}" ]]; then
    extra=(--repo "$CHOSEN_REPO")
  fi
  echo "Opening one-box operator…"
  exec "$py" "$GUI_PY" "${extra[@]}" "$@"
}

print_check() {
  local py
  py="$(venv_python)"
  echo "Script dir : $SCRIPT_DIR"
  echo "Analyzer   : $ANALYZER  $([[ -f $ANALYZER ]] && echo OK || echo MISSING)"
  echo "GUI        : $GUI_PY  $([[ -f $GUI_PY ]] && echo OK || echo MISSING)"
  echo "Python     : $py"
  if [[ -x "$VENV/bin/python" ]]; then
    echo "Venv       : $VENV"
    "$py" -c "import transformers, huggingface_hub, torch; print('Packages   : transformers', transformers.__version__, '| torch', torch.__version__, '| hub', huggingface_hub.__version__)"
  else
    echo "Venv       : missing (run --bootstrap)"
  fi
  echo
  print_catalog
}

CHOSEN_REPO=""

main() {
  cd "$SCRIPT_DIR"
  local cmd="${1:-}"
  case "$cmd" in
    -h|--help)
      usage
      ;;
    --list)
      print_catalog
      ;;
    --check)
      print_check
      ;;
    --bootstrap)
      ensure_venv
      echo "Bootstrap complete. Use: $VENV/bin/python $ANALYZER --check-deps"
      ;;
    --download)
      need_file "$ANALYZER"
      ensure_venv
      offer_download
      ;;
    --probe)
      shift || true
      need_file "$ANALYZER"
      "$PY_SYS" "$ANALYZER" --probe "$@"
      ;;
    --gui)
      shift || true
      need_file "$ANALYZER"
      ensure_venv
      open_gui "$@"
      ;;
    "")
      need_file "$ANALYZER"
      ensure_venv
      if any_model_cached; then
        echo "Local model already on disk — skipping download prompt."
      else
        offer_download
      fi
      open_gui
      ;;
    -*)
      echo "Unknown option: $cmd" >&2
      usage >&2
      exit 2
      ;;
    *)
      need_file "$ANALYZER"
      ensure_venv
      if ! any_model_cached; then
        offer_download
      fi
      open_gui "$@"
      ;;
  esac
}

main "$@"
