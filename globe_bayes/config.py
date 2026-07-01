"""
Runtime configuration. Everything is env-overridable so the same code runs in
dev (Jetson interpreter only) and demo (Jetson with cloud fallback) without edits.

  DEVICE = laptop   (this process: Flask + bayes + persistence)
  SERVER = Jetson   (Ollama / gemma4:e2b, reached over HTTP)
"""

import os

# --- run mode -----------------------------------------------------------------
# dev  : Jetson interpreter only; if it's down, show the numbers + a visible
#        "narration unavailable" notice (fail honestly while developing).
# demo : try Jetson, then automatically fall back to cloud backends (wired later)
#        so a brittle Jetson never breaks a live demo.
APP_MODE = os.environ.get("APP_MODE", "dev").lower()

# --- Jetson interpreter (Ollama) ----------------------------------------------
# Default assumes an SSH tunnel from this laptop to the Jetson:
#     ssh -L 11434:localhost:11434 <user>@<jetson-host>
# For a standalone demo, point this at the Jetson's IP and run Ollama with
# OLLAMA_HOST=0.0.0.0 on the Jetson instead.
JETSON_OLLAMA_URL = os.environ.get("JETSON_OLLAMA_URL", "http://localhost:11434")
JETSON_MODEL = os.environ.get("JETSON_MODEL", "gemma4:e2b")
# First token on an edge GPU can be slow on a cold load (the model loads into
# memory on the first request); subsequent requests are warm and faster.
JETSON_TIMEOUT = float(os.environ.get("JETSON_TIMEOUT", "120"))
# How long Ollama keeps the model resident in GPU after a request. Long enough that
# a demo never re-pays the cold load between readings ("-1" = keep loaded forever).
JETSON_KEEP_ALIVE = os.environ.get("JETSON_KEEP_ALIVE", "30m")

# --- device-side persistence --------------------------------------------------
STATE_PATH = os.environ.get("STATE_PATH", "model_state.json")
# Seed realistic readings the first time we run (never clobbers existing data).
SEED_ON_FIRST_RUN = os.environ.get("SEED_ON_FIRST_RUN", "1") == "1"

# --- domain -------------------------------------------------------------------
# Turbidity-tube length (cm) = the most a reading can be for this deployment. Varies
# by site/tube (single value for now). A reading at this length is right-CENSORED: the
# water was clearer than the tube can measure, so the true clarity is "at least" this.
# Kept here, not in model.py, so model.py stays the upstream copy.
TUBE_LENGTH_CM = float(os.environ.get("TUBE_LENGTH_CM", "45"))

# --- web server ---------------------------------------------------------------
# Browser is on the laptop too, so localhost is fine. Override to 0.0.0.0 only
# if you want another machine to reach the form directly.
HOST = os.environ.get("HOST", "127.0.0.1")
# Default 8000, not 5000: macOS AirPlay Receiver squats on port 5000, which hijacks
# http://localhost:5000 in the browser (IPv6) even when Flask binds 127.0.0.1 (IPv4).
PORT = int(os.environ.get("PORT", "8000"))

# --- cloud fallbacks (wired in a later step; placeholders for demo mode) ------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
