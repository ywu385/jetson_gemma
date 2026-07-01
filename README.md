# globe_bayes

A citizen-science **water-clarity demo**. A field operator submits a turbidity-tube
reading; a deterministic **Bayesian model** folds it into what we already know about
that site and lake; then a local **Gemma** LLM (nicknamed **Ivonabot**) explains the
result back to them in plain, encouraging language — including what the reading means
for the water and how much their submission helped.

It blends two sibling projects:
- the **GLOBE curation** Flask front-end, and
- the **Bayesian water-clarity model** (`jetson_bayes/bayes`).

---

## The one idea to understand first

The system is split into a **device** and a **server**, and they are deliberately unequal:

| | Runs on | Owns | If it dies |
|---|---|---|---|
| **DEVICE** | your **laptop** | the Flask form, the Bayesian math, and **all the data** (the truth) | nothing works — this is the brain |
| **SERVER** | the **Jetson** | **only** the LLM narration (stateless) | you lose the *prose*, never the *data* |

The Jetson is the most fragile part (edge hardware, a cable, a warm model), so it's
treated as a **swappable inference endpoint** — a brain you can unplug and replace. The
LLM is wired in behind a stable interface (`narrate(report) -> text`), so a Jetson,
a cloud model (GPT/Opus), or a stub can all drop into the same slot. Everything durable
and deterministic stays on the laptop.

> Practical upshot: if the Jetson is off, the form still records readings and shows the
> numbers. You just get an "interpreter offline" notice instead of Ivonabot's prose.

---

## Quick start (TL;DR)

> **First time on a fresh Jetson?** It won't be reachable at `10.42.0.1` until you turn
> its wired port into a DHCP server. Run this **once, on the Jetson**, then plug in the
> cable (see [Deploying to your own Jetson](#deploying-to-your-own-jetson) for the full
> steps):
> ```bash
> sudo bash scripts/setup_jetson_direct_net.sh     # Jetson becomes 10.42.0.1
> ```

```bash
# --- one-time setup (laptop) ---
cd globe_bayes
python -m venv .venv && source .venv/bin/activate
pip install -e .                     # installs deps + the globe-bayes-* commands

# --- every run ---
# 1) open the tunnel to the Jetson (separate terminal, leave it open):
#    (assumes the Jetson is already set up as a DHCP server at 10.42.0.1 — see note above)
ssh -f -N -L 11434:localhost:11434 jwu385@10.42.0.1

# 2) start the web app:
globe-bayes-web                      # -> http://127.0.0.1:8000
```

Then open **http://127.0.0.1:8000**, pick a site, enter a value, and submit.

---

## Running it — the two halves

You need **both** halves up for full narration.

### ① The Jetson (narration engine)

Ollama runs `gemma4:e2b` as a service on the Jetson's `:11434`. It's bound to the
Jetson's *localhost*, so you reach it by forwarding that port to your laptop over SSH:

```bash
ssh -f -N -L 11434:localhost:11434 jwu385@10.42.0.1     # direct cable
# or over Wi-Fi:  ssh -f -N -L 11434:localhost:11434 jwu385@jwu385-jetson.local
```

- `-f -N` runs it backgrounded with no shell. Drop those flags to keep it in a visible
  window you can `Ctrl-C`.
- Once open, your laptop's `localhost:11434` **is** the Jetson's Ollama.

### ② The laptop (web app)

```bash
APP_MODE=dev globe-bayes-web         # http://127.0.0.1:8000
```

On startup it seeds demo data (first run only) and **pre-warms** the model in the
background so the first reading isn't an ~80 s cold load.

---

## What each part does

Installable package under `src/globe_bayes/` (editable install: `pip install -e .`).

```
src/globe_bayes/
├── config.py            # all env-overridable settings (one place)
├── narrator.py          # THE SWAPPABLE LLM LAYER — Ivonabot
├── bayes/               # deterministic core (the "truth")
│   ├── model.py         #   the Bayesian math — upstream copy, keep in sync
│   ├── store.py         #   load → update → save; persistence + seeding
│   ├── model_card.py    #   plain-language model description fed to the LLM
│   └── make_test_payloads.py   # generate LLM test requests per scenario
└── frontend/            # the web app
    ├── webform.py       #   Flask form → record → narrate → render
    └── bgp.jpg          #   background image
```

| Part | What it does | Notes for you |
|---|---|---|
| **`bayes/model.py`** | The hierarchical Bayesian update (site → cluster → lake), Student-t outlier down-weighting, physical-range gate. | **Canonical copy of `jetson_bayes/bayes/model.py` — keep in sync.** Pure & deterministic; no LLM, no I/O. |
| **`bayes/store.py`** | Persists readings to `model_state.json` (`record()` = load → update → atomic save). Seeds demo data; attaches `n_prior_readings`. Builds fresh models at the configured tube length. | This is the device-side "database" (swap for SQLite later if needed). |
| **`bayes/model_card.py`** | Turns the live model constants into a plain-English description that's injected into the LLM prompt, so Ivonabot can *reason* about the model instead of parroting numbers. | Auto-stays in sync with `model.py`. |
| **`narrator.py`** | The swappable interpreter. Defines the `Narrator` interface, `JetsonNarrator` (Ollama), `FallbackNarrator`, the **situation router**, and the prompt. | This is the "harness around the model." See below. |
| **`frontend/webform.py`** | The Flask app: renders the form, converts the input, calls `store.record()` then `narrator.narrate()`, and renders the report + Ivonabot's prose. Degrades gracefully if the Jetson is down. | The only user-facing surface. |
| **`config.py`** | Every knob (mode, Jetson URL/model, timeouts, port, tube length, paths) with env-var overrides. | Change behavior here, not in code. |
| **`model_state.json`** | The accumulated readings. **Lives on the laptop, gitignored.** | Delete it to reset the demo. |

---

## Taking parts of it (reuse map)

The code is **layered on purpose** so you can lift only what you need. Bottom layers have
no internal dependencies; each layer only imports from the ones below it (`config.py` is a
shared leaf). Nothing reaches "up."

```
LAYER                          FILE(S)                              EXTERNAL DEPS
──────────────────────────────────────────────────────────────────────────────────
4  Web app / UI          ┐  frontend/webform.py                    flask
   Dev/test tool         ┘  bayes/make_test_payloads.py            (none)
3  LLM narrator (Ivonabot)   narrator.py                           requests + a running Ollama
2  Persistence               bayes/store.py                        (stdlib only)
1  Model description         bayes/model_card.py                   (none)
0  Bayesian math             bayes/model.py         ◄ standalone   numpy
   Runtime config            config.py              ◄ leaf         (none)
   Jetson direct network     scripts/setup_jetson_direct_net.sh    bash + NetworkManager
```

### "I just want…"

| Take just… | Files | Also needs | External |
|---|---|---|---|
| **The Bayesian model** (fold in readings, get posteriors) | `bayes/model.py` | — | `numpy` |
| **Model + on-disk persistence** | `bayes/model.py`, `bayes/store.py` | `config.py` | `numpy` |
| **The LLM narrator** (turn any structured report → prose) | `narrator.py`, `bayes/model_card.py` | `bayes/model.py`, `config.py` | `requests` + Ollama |
| **The direct-cable Jetson network** (no Python at all) | `scripts/setup_jetson_direct_net.sh` | — | bash, NetworkManager |
| **The whole demo** | everything | — | see `pyproject.toml` |

**Notes for lifting pieces:**
- `bayes/model.py` is deliberately dependency-free (just `numpy`) and is the **canonical
  upstream copy** — it's the cleanest thing to grab, a single self-contained file.
- `config.py` is a tiny leaf (just `os.environ` reads). If you take `store.py` or
  `narrator.py` without it, swap the handful of `config.X` lookups for your own values.
- `narrator.py` doesn't care *which* model produced the report — it just needs a dict.
  To reuse Ivonabot with a different model, keep the keys its router reads
  (`n_accepted`, `n_rejected`, `accepted`, `tube_length`, `surprise`, `weights_assigned`,
  `n_prior_readings`, `hierarchy`) — or edit `_situation_hint` to match your report shape.

---

## How one submission flows

```
   [browser form]
        │  site + measurement
        ▼
 ntu_to_cm(...)                     # frontend: input → cm (clamped to tube length)
        │
        ▼
 store.record(site, [cm])           # DEVICE: fold into the Bayesian model, persist
        │  returns a structured "report" (before/after estimates, surprise, weights)
        ▼
 narrator.narrate(report)           # SERVER: Ivonabot turns the report into prose
        │      ├─ _situation_hint(report)   ← code decides the situation (below)
        │      └─ Gemma on the Jetson       ← writes the friendly explanation
        ▼
   [rendered page]  =  Ivonabot's prose  +  a collapsible "Model details" table
```

Everything left of `narrate()` is deterministic and on the laptop. Only the wording is
delegated to the LLM.

---

## Ivonabot: the narrator (the "harness around the model")

`narrator.py` is a thin, model-agnostic wrapper. The model itself is swappable behind
one interface:

```
Narrator (interface):  narrate(report) -> text
  ├─ JetsonNarrator     → Gemma via Ollama (current)
  ├─ FallbackNarrator   → Jetson, then cloud (for demo mode)
  └─ (future) cloud     → GPT / Opus
```

**Why the LLM is kept on a short leash:** a 2B-class edge model can't be trusted to infer
context from raw JSON. So the code classifies each report **deterministically** and hands
the model one explicit directive — the LLM only chooses the *wording*, never the facts.

### Situation routing

| Situation | Trigger | What Ivonabot is told to do |
|---|---|---|
| **all rejected** | no reading accepted | warm, **no** impact claims; ask them to re-measure/confirm |
| **partial reject** | some readings out of range | narrate the good one(s); ask to re-check the bad |
| **censored (tube max)** | accepted reading ≥ tube length | explain it's a "≥" (water clearer than the tube can read); ask exact-vs-censored; reassure; ask them to mark it |
| **surprising** | large surprise / down-weighted | kept but treated cautiously; invite a second reading |
| **sparse / high-value** | site has ≤ 5 prior readings | genuine excitement — this reading really moves the estimate |
| **normal** | everything else | thank them; point to the concrete impact (tightened interval) |

Every accepted reading also gets a short, **hedged** ecological note (what the clarity
suggests for the lake/river and plausible causes), drawn from the model's own aquatic
knowledge.

> **Iterating on prompts:** edit `narrator.py`, run `globe-bayes-payloads /tmp/payloads`
> to regenerate per-scenario request files, `scp` them to the Jetson, and `curl` each
> against `/api/chat` to compare wordings without touching the app.

---

## Configuration (env vars)

Set any of these before `globe-bayes-web` (e.g. `TUBE_LENGTH_CM=60 globe-bayes-web`):

| Var | Default | Meaning |
|---|---|---|
| `APP_MODE` | `dev` | `dev` = Jetson only (fail visibly); `demo` = Jetson → cloud fallback |
| `JETSON_OLLAMA_URL` | `http://localhost:11434` | where Ollama is reachable (the tunnel endpoint) |
| `JETSON_MODEL` | `gemma4:e2b` | Ollama model tag |
| `JETSON_TIMEOUT` | `120` | seconds before the interpreter is considered down |
| `JETSON_KEEP_ALIVE` | `30m` | how long Ollama keeps the model warm in GPU (`-1` = forever) |
| `TUBE_LENGTH_CM` | `45` | turbidity-tube max length (cm); a reading **at** it is right-censored |
| `PORT` | `8000` | web-server port (5000 is taken by macOS AirPlay) |
| `STATE_PATH` | `model_state.json` | the device-side data file |
| `SEED_ON_FIRST_RUN` | `1` | seed demo readings if no state file exists yet |

---

## Deploying to *your own* Jetson

1. **Install Ollama + pull the model** (over SSH on the Jetson):
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ollama pull gemma4:e2b        # ~1.6 GB loaded, fits the 8 GB Orin Nano
   ollama ps                     # confirm it loads on the GPU
   ```
2. **Set up the direct-cable network** (so you don't depend on Wi-Fi/DHCP at a demo):
   ```bash
   sudo bash scripts/setup_jetson_direct_net.sh      # Jetson becomes 10.42.0.1, serves DHCP
   #   undo with:  sudo bash scripts/setup_jetson_direct_net.sh --reset
   ```
   Plug a cable from your laptop into that port (laptop on default/Automatic settings);
   you'll pull a `10.42.0.x` lease and reach the Jetson at `10.42.0.1`.
3. **Point the app at it** (if not using the default tunnel): open the SSH tunnel to your
   Jetson's user/IP, or set `JETSON_OLLAMA_URL`.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| "**Interpreter offline**" in the prose block | The tunnel is down. Re-run the `ssh -f -N -L 11434:...` command. Check with `curl localhost:11434/api/tags`. |
| **First reading takes ~80 s** | Cold model load. It's a one-time cost; `keep_alive` + startup pre-warm keep it fast after. Pre-warm manually: `curl localhost:11434/api/generate -d '{"model":"gemma4:e2b","keep_alive":"30m"}'`. |
| **Port 8000 in use** | Stop the old server: `lsof -ti tcp:8000 | xargs kill`. |
| **Want a clean demo** | Delete `model_state.json` and restart — it reseeds fresh. |
| **Can't reach the Jetson** | `ping 10.42.0.1`, then `nc -z 10.42.0.1 22`. Cable in the right port? Script run? |

---

## Open questions / not-yet-done

- **Input units are unresolved.** The form currently takes **NTU** and converts to cm
  (`600/NTU`). But a turbidity *tube* is read directly in **cm** — NTU comes from an
  electronic sensor. **This needs a decision** (see `narrator.py` routing — the censored
  logic assumes cm). If the operator reads the tube in cm, we switch the form to cm input.
- **Cloud fallback** (`demo` mode) — the machinery exists (`FallbackNarrator`) but no
  cloud backend is wired yet.
- **True censored inference** — the model currently folds an accepted `45` in as the exact
  value; it doesn't yet treat it as "≥45" in the posterior math (a `model.py` change).
- UI is a functional MVP; optional field-notes capture and TTS ("read it aloud") are ideas.

See `CLAUDE.md` for deeper architecture notes.
