"""
Device-side JSON persistence for the hierarchical bayes model.

This lives on the DEVICE (laptop) — the durable, deterministic side of the
system. The Jetson interpreter is stateless; all readings accumulate here so
the demo survives a Jetson outage (or a swap to a cloud interpreter) without
losing a single measurement.

The model core (model.py) stays pure; this thin layer serialises ModelState
to/from a small JSON file so sequential turbidity readings accumulate across
submissions and restarts. Swap for SQLite later if we want per-observation
history.
"""

import json
import os
from pathlib import Path

from globe_bayes import config
from globe_bayes.bayes.model import Hyperparams, ModelState, from_dict, to_dict, update

DEFAULT_PATH = "model_state.json"

# Realistic opening readings so a fresh demo isn't empty (center water clearer).
# All within the configured tube length so none are rejected when seeding.
SEED_READINGS = {
    "site_1": [28, 31, 26],
    "site_2": [33, 29],
    "site_3": [30],
    "site_5": [40, 44, 42],   # center water is clearer, but still inside a 45 cm tube
    "site_6": [43, 41],
}


def _fresh_model() -> ModelState:
    """A new model using the deployment's configured tube length (config-driven, so
    model.py stays the upstream copy)."""
    return ModelState(hp=Hyperparams(tube_length=config.TUBE_LENGTH_CM))


def load_model(path: str = DEFAULT_PATH) -> ModelState:
    """Load saved state, or a fresh model if the file doesn't exist yet."""
    p = Path(path)
    if not p.exists():
        return _fresh_model()
    return from_dict(json.loads(p.read_text()))


def save_model(state: ModelState, path: str = DEFAULT_PATH) -> None:
    """Persist state atomically (write temp + rename) so a crash can't corrupt it."""
    p = Path(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(to_dict(state), indent=2))
    os.replace(tmp, p)  # atomic on POSIX — matters on hardware that loses power


def record(site_id: str, observations, path: str = DEFAULT_PATH) -> dict:
    """Load -> fold in a new reading -> save. Returns the pooling report.

    Augments the model report with 'n_prior_readings' (how many readings the site
    already had) so the narrator can get extra-excited about high-value submissions
    at data-starved sites. Computed here, not in model.py, to keep that file in sync
    with the upstream copy.
    """
    state = load_model(path)
    n_prior = len(state.readings.get(site_id, []))
    state, report = update(state, site_id, observations)
    report["n_prior_readings"] = n_prior
    save_model(state, path)
    return report


def seed_if_empty(path: str = DEFAULT_PATH) -> bool:
    """Populate a fresh state file with demo readings. No-op if it already exists.

    Returns True if seeding happened. Lets the demo open with realistic numbers
    instead of flat priors, without ever clobbering accumulated real data.
    """
    if Path(path).exists():
        return False
    state = _fresh_model()
    for site_id, readings in SEED_READINGS.items():
        state, _ = update(state, site_id, readings)
    save_model(state, path)
    return True
