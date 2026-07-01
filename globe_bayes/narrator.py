"""
The interpreter layer: narrate(report) -> plain-language text.

The bayes report is structured and deterministic; turning it into prose is the
only swappable, possibly-flaky step. Every backend implements the same one-method
interface, so the Jetson (local Gemma) and cloud models (GPT / Opus, added later)
are interchangeable — and a dead Jetson can fall back without touching the data
path. The bayes math and the stored readings never depend on any of this working.
"""

from __future__ import annotations

import json
import re
from typing import Protocol

import requests

from globe_bayes import config
from globe_bayes.bayes.model import Hyperparams
from globe_bayes.bayes.model_card import build_model_card


class NarratorUnavailable(RuntimeError):
    """A backend couldn't produce a narration (unreachable, timeout, bad reply)."""


class Narrator(Protocol):
    name: str

    def narrate(self, report: dict) -> str: ...


# The narration task: what we want the model to DO with a report. The tone is
# deliberately warm and encouraging — this is a citizen-science "AI for Good"
# effort, so a contributor should feel their reading mattered, and a bad value
# should read as "let's double-check together," never a cold rejection.
NARRATION_TASK = """\
You are a warm, encouraging assistant for a citizen-science water-monitoring effort
(a NASA GLOBE project). A field operator has just submitted a turbidity-tube
reading. You are given the JSON report from the Bayesian model described below. Reply
in 4-7 plain, friendly sentences for someone who is not a statistician, covering:

1. APPRECIATION & IMPACT — Thank them and make the value of their submission concrete:
   how it sharpened our estimate or confidence (look at how the 'after' variance and
   95% credible interval tightened versus 'before'), or what it confirms about the site
   or the whole lake. Encourage them to keep monitoring. (If NO reading was accepted,
   skip the praise and go straight to point 4.)

2. WHAT IT MEANS — State the reading and explain it plainly (higher cm = clearer
   water), whether the model found it surprising, and how it nudged the site and lake
   estimates — using the model description to explain *why* (e.g. shrinkage toward
   neighbors, or outlier robustness).

3. WHAT IT MEANS FOR THE WATER — Using your own general knowledge of freshwater /
   aquatic ecology, briefly say what this level of clarity suggests for the health of
   the lake or river they are observing (light penetration, algae or suspended
   sediment, habitat for aquatic life), and what could plausibly drive a reading or
   change like this (recent rain or runoff, sediment, an algal bloom, seasonal
   turnover). Keep it qualitative and hedged ("may indicate", "could suggest") — offer
   plausible context, not firm conclusions about this specific water body.

4. DATA QUALITY, GENTLY — If a reading was REJECTED as physically impossible (outside
   the [0, 45] cm tube range), do not scold or just say "rejected": note it looks like
   it may be a slip and kindly ask them to re-measure or confirm that value. If a
   reading was accepted but heavily DOWN-WEIGHTED (low weight) or flagged as a
   notable/extreme surprise, say the model kept it but is treating it cautiously, and
   invite a second reading to confirm.

Be concrete with the numbers from the report and never invent report values. For the
ecological context, drawing on general aquatic knowledge is welcome — keep it cautious
and avoid overclaiming about this specific water body. Write plain, friendly prose —
no markdown, bullet points, or LaTeX/math notation; write numbers plainly (e.g. "15 cm").
"""

# Inject the model card so the LLM understands the generative model it's narrating
# — this is what lets it reason rather than parrot. The card is built from the
# live model definition, so it stays in sync with model.py.
SYSTEM_PROMPT = NARRATION_TASK + "\n\n" + build_model_card(
    Hyperparams(tube_length=config.TUBE_LENGTH_CM)
)


def _situation_hint(report: dict) -> str:
    """Classify the report deterministically and hand the model one unambiguous
    directive. A small edge model can't be trusted to infer 'all readings were
    rejected' from raw JSON, so we route it explicitly — and forbid the false
    'it confirmed things' spin when nothing was actually recorded.
    """
    n_acc = report.get("n_accepted", 0)
    n_rej = report.get("n_rejected", 0)
    weights = report.get("weights_assigned") or []
    down_weighted = any(w < 0.5 for w in weights)
    tube = report.get("tube_length", config.TUBE_LENGTH_CM)
    accepted = report.get("accepted") or []
    at_tube_max = bool(accepted) and max(accepted) >= tube - 1e-9

    if n_acc == 0:
        return (
            f"SITUATION: every submitted reading was physically impossible (outside the "
            f"[0, {tube:g}] cm range) and was NOT recorded. There is NO impact and NO change "
            "to any estimate — do not claim the reading helped, confirmed, or shifted "
            "anything. Briefly and warmly acknowledge their effort, explain the value "
            "looks out of range (likely a measuring slip), and clearly ask them to "
            "re-measure or confirm it. Keep it short."
        )
    if n_rej > 0:
        return (
            "SITUATION: at least one reading was recorded, but one or more were out of "
            "range and set aside. Narrate the recorded reading(s) normally, then gently "
            "ask them to re-check the out-of-range value(s)."
        )
    if at_tube_max:
        return (
            f"SITUATION: the reading sits right at the top of the turbidity tube "
            f"({tube:g} cm) — the most the tube can measure. This usually means the water "
            f"was actually CLEARER than the tube can capture (the marker was still visible "
            f"at the bottom), so the true clarity is AT LEAST {tube:g} cm — a right-CENSORED "
            f"reading, not necessarily exactly {tube:g}. Warmly thank them and explain this "
            f"gently, then ask them to clarify: did the water disappear right at the "
            f"{tube:g} cm mark, or was it still clear at the very bottom of the tube? "
            f"Reassure them that an at-or-beyond-the-tube reading is completely fine and "
            f"still useful — just ask them to note it (e.g. record it as '{tube:g} cm or "
            f"clearer', or mark the top of the tube) so we capture it correctly."
        )
    if report.get("surprise") in ("notable", "extreme") or down_weighted:
        return (
            "SITUATION: the reading was recorded but is surprising versus this site's "
            "history, so the model down-weighted it and it barely moved the estimate. "
            "Acknowledge it, say it was kept but is being treated cautiously, and invite "
            "a second reading to confirm."
        )
    n_prior = report.get("n_prior_readings")
    if n_prior is not None and n_prior <= 5:
        return (
            "SITUATION: this site still has relatively few readings so far, so a solid "
            "reading here is especially valuable and noticeably improves our estimate. "
            "Respond with genuine EXCITEMENT — this is a great, high-impact reading "
            "because data has been scarce at this site. Point to how much it tightened "
            "the estimate, and encourage them to add any field notes (weather, time of "
            "day, conditions) to go with such a valuable data point."
        )
    return (
        "SITUATION: a normal, trusted reading. Thank them and make its concrete impact "
        "clear — e.g. the tightened 95% credible interval or what it confirms about the "
        "site."
    )


# Reinforced next to the data (the situation directive otherwise crowds it out on a
# small model). Only added when a reading was actually accepted — no point asking for
# ecological meaning when everything was rejected.
_ECOLOGY_REMINDER = (
    "Also include one or two hedged sentences on what this clarity level suggests for "
    "the health of the lake or river, and what could plausibly be driving it (general "
    "aquatic knowledge — qualitative, not firm claims about this specific site)."
)


def _user_prompt(report: dict) -> str:
    parts = [_situation_hint(report)]
    if report.get("n_accepted", 0) > 0:
        parts.append(_ECOLOGY_REMINDER)
    parts.append("Model report to narrate:\n\n" + json.dumps(report, indent=2))
    return "\n\n".join(parts)


# gemma4 is a reasoning model: it can emit a chain-of-thought. We disable that with
# `think: false` (faster, cleaner on the edge); this strips any trace defensively in
# case a model/Ollama version inlines a <think>...</think> block into the content.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


class JetsonNarrator:
    """Local Gemma on the Jetson via Ollama's HTTP chat API."""

    def __init__(self, url: str | None = None, model: str | None = None,
                 timeout: float | None = None, keep_alive: str | None = None):
        self.url = (url or config.JETSON_OLLAMA_URL).rstrip("/")
        self.model = model or config.JETSON_MODEL
        self.timeout = timeout or config.JETSON_TIMEOUT
        self.keep_alive = keep_alive or config.JETSON_KEEP_ALIVE
        self.name = f"jetson:{self.model}"

    def narrate(self, report: dict) -> str:
        try:
            resp = requests.post(
                f"{self.url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": _user_prompt(report)},
                    ],
                    "stream": False,
                    "think": False,
                    "keep_alive": self.keep_alive,
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return _strip_thinking(resp.json()["message"]["content"])
        except (requests.RequestException, KeyError, ValueError) as e:
            raise NarratorUnavailable(f"{self.name} unavailable: {e}") from e

    def warm_up(self) -> bool:
        """Best-effort: load the model into the Jetson's GPU now (with no prompt) so
        the first real reading isn't a ~80 s cold load. Returns True if it's ready."""
        try:
            resp = requests.post(
                f"{self.url}/api/generate",
                json={"model": self.model, "keep_alive": self.keep_alive},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return True
        except requests.RequestException:
            return False


class FallbackNarrator:
    """Try each backend in order; return the first success.

    `last_used` records which backend produced the text, so the UI can show
    'narrated by jetson:gemma4:e2b' vs a cloud fallback during a demo.
    """

    def __init__(self, backends: list[Narrator]):
        if not backends:
            raise ValueError("FallbackNarrator needs at least one backend")
        self.backends = backends
        self.last_used: str | None = None
        self.name = "fallback(" + ", ".join(b.name for b in backends) + ")"

    def narrate(self, report: dict) -> str:
        errors = []
        for backend in self.backends:
            try:
                text = backend.narrate(report)
                self.last_used = backend.name
                return text
            except NarratorUnavailable as e:
                errors.append(str(e))
        self.last_used = None
        raise NarratorUnavailable("all backends failed: " + "; ".join(errors))

    def warm_up(self) -> bool:
        """Pre-load any backend that supports it (the Jetson). Best-effort."""
        ok = False
        for backend in self.backends:
            warm = getattr(backend, "warm_up", None)
            if callable(warm):
                ok = warm() or ok
        return ok


def build_narrator(mode: str | None = None) -> FallbackNarrator:
    """Assemble the backend chain for the active mode.

      dev  -> Jetson only (fail visibly if it's down)
      demo -> Jetson, then cloud fallbacks (OpenAI / Anthropic, added later)
    """
    mode = (mode or config.APP_MODE).lower()
    chain: list[Narrator] = [JetsonNarrator()]
    if mode == "demo":
        # Cloud backends drop in here once wired, e.g.:
        #   if config.ANTHROPIC_API_KEY: chain.append(AnthropicNarrator())
        #   if config.OPENAI_API_KEY:    chain.append(OpenAINarrator())
        pass
    return FallbackNarrator(chain)
