"""
Generate Ollama /api/chat request payloads for several reading scenarios, using the
REAL production prompt (SYSTEM_PROMPT + model card), so we can eyeball narration
quality on the Jetson and iterate on the prompt:

    python make_test_payloads.py [out_dir]   # default out_dir = current dir

Each scenario writes <out_dir>/payload_<name>.json, ready for:
    curl -s http://localhost:11434/api/chat -d @payload_<name>.json
"""

import copy
import json
import os
import sys

from globe_bayes import config
from globe_bayes.bayes import store
from globe_bayes.bayes.model import update
from globe_bayes.narrator import SYSTEM_PROMPT, _user_prompt

# (site, observations) chosen to exercise each narration path (tube length from config).
SCENARIOS = {
    "normal":     ("site_1", [30.0]),   # well-sampled site, consistent -> trusted impact
    "sparse":     ("site_4", [31.0]),   # site_4 has no seed data -> high-value, excitement
    "censored":   ("site_2", [45.0]),   # at the tube max -> right-censored, ask to mark it
    "impossible": ("site_4", [85.0]),   # outside [0, tube] -> all rejected, ask to re-measure
    "suspicious": ("site_1", [5.0]),    # site_1 seeded ~28-31 -> extreme low, down-weighted
}


def _seeded():
    """A model seeded with the demo readings, matching what the app starts from."""
    m = store._fresh_model()
    for site, readings in store.SEED_READINGS.items():
        m, _ = update(m, site, readings)
    return m


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out, exist_ok=True)
    base = _seeded()
    for name, (site, obs) in SCENARIOS.items():
        state = copy.deepcopy(base)
        n_prior = len(state.readings[site])
        _, report = update(state, site, obs)
        report["n_prior_readings"] = n_prior  # mirror what store.record() attaches
        payload = {
            "model": config.JETSON_MODEL,
            "think": False,
            "stream": False,
            "keep_alive": "30m",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _user_prompt(report)},
            ],
        }
        path = os.path.join(out, f"payload_{name}.json")
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"wrote {path}  ({name}: {site} {obs} -> "
              f"accepted={report['n_accepted']} rejected={report['n_rejected']} "
              f"surprise={report['surprise']})")


if __name__ == "__main__":
    main()
