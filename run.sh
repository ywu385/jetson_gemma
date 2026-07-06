#!/usr/bin/env bash
# One-command runner for the laptop side. First run creates a local .venv and
# installs the dependencies; after that it just starts the web app.
#
#   ./run.sh                      # -> http://127.0.0.1:8000
#   PORT=8080 ./run.sh            # any config.py env var passes straight through
#
# Reminder: the Jetson tunnel is yours to open (in another terminal):
#   ssh -f -N -L 11434:localhost:11434 <user>@<jetson-ip>
set -euo pipefail

cd "$(dirname "$0")"    # always run from the repo root, wherever it was called from

if [ ! -d .venv ]; then
    echo "[setup] creating .venv (first run only)"
    python3 -m venv .venv
fi

# (Re)install deps only when requirements.txt is newer than the last install.
if [ requirements.txt -nt .venv/.deps-installed ]; then
    echo "[setup] installing dependencies from requirements.txt"
    .venv/bin/pip install -q -r requirements.txt
    touch .venv/.deps-installed
fi

exec .venv/bin/python -m globe_bayes.frontend.webform
