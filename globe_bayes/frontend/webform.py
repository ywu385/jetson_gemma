"""
GLOBE curation bayes demo — device-side web app.

Flow per submission (all on the DEVICE / laptop):
    form (site + NTU) -> ntu_to_cm -> store.record() -> bayes report
                                                     -> narrator.narrate(report)
                                                     -> render numbers + prose

The narrator talks to the Jetson (Gemma) over HTTP. If the Jetson is down, the
numeric report still renders and the prose block shows a clear notice (dev mode)
or falls back to a cloud interpreter (demo mode).

Run:  python webform.py
"""

import threading
from html import escape
from pathlib import Path

from flask import Flask, request, send_from_directory

from globe_bayes import config
from globe_bayes.bayes import store
from globe_bayes.bayes.model import SITES
from globe_bayes.narrator import NarratorUnavailable, build_narrator

_NTU_TO_CM_CALIBRATION = 600.0
_MIN_NTU = 1
_PROJECT_DIR = Path(__file__).parent

_PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>GLOBE Curation AI Demo</title>
  <style>
    html, body {{ min-height: 100vh; margin: 0; }}
    body {{
      box-sizing: border-box; padding: 2rem;
      display: flex; flex-direction: column;
      background-image: url("/bgp.jpg");
      background-size: cover; background-position: center; background-repeat: no-repeat;
      color: white; font-family: Arial, sans-serif;
    }}
    .panel {{
      max-width: 40rem; background: rgba(0,0,0,0.55);
      padding: 1.25rem 1.5rem; border-radius: 0.75rem;
      /* push toward the bottom: top margin soaks up the space, panel floats 12vh
         above the bottom edge; collapses to top-aligned if taller than the screen */
      margin-top: auto; margin-bottom: 45vh;
    }}
    button {{
      background: white; border: 1px solid white; border-radius: 0.5rem;
      color: black; cursor: pointer; padding: 0.3rem 0.7rem;
    }}
    .report {{ font-size: 0.9rem; line-height: 1.4; }}
    .narration {{ margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid rgba(255,255,255,0.3); }}
    .meta {{ opacity: 0.7; font-size: 0.8rem; }}
    .warn {{ color: #ffd27f; }}
    table {{ border-collapse: collapse; margin: 0.5rem 0; }}
    td, th {{ padding: 0.15rem 0.6rem; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
  </style>
</head>
<body>
  <div class="panel">
    <h2>Submit a water transparency tube reading</h2>
    <form method="post">
      <label for="site_id">Site/Tube ID:</label>
      <select name="site_id" id="site_id">{options}</select>
      &nbsp;&nbsp;
      <label for="measurement">Turbidity (NTU):</label>
      <input type="number" step="any" min="{min_ntu}" name="measurement" id="measurement" required>
      &nbsp;&nbsp;
      <button type="submit">Submit</button>
    </form>
    <p class="meta">mode: {mode} &middot; interpreter: {interpreter}</p>
    {result}
  </div>
</body>
</html>
"""


def ntu_to_cm(ntu: float, tube_length: float) -> float:
    """Convert an NTU turbidity reading to centimeters of tube clarity."""
    if ntu <= 0:
        raise ValueError("NTU must be greater than 0")
    measurement_cm = _NTU_TO_CM_CALIBRATION / ntu
    return max(0.0, min(tube_length, measurement_cm))


def _options() -> str:
    return "\n".join(
        f'<option value="{escape(s)}">{escape(s)}</option>' for s in SITES
    )


def _level_row(level: dict) -> str:
    b, a = level["before"], level["after"]
    return (
        f"<tr><td>{escape(level['id'])} ({escape(level['level'])})</td>"
        f"<td>{b['mean']}</td><td>{a['mean']}</td>"
        f"<td>{level['delta']['mean_shift']:+}</td>"
        f"<td>{a['ci95'][0]}–{a['ci95'][1]}</td></tr>"
    )


def _render_report(report: dict) -> str:
    rows = "".join(_level_row(l) for l in report["hierarchy"])
    rejected = ""
    if report["n_rejected"]:
        vals = ", ".join(str(r["value"]) for r in report["rejected"])
        rejected = f'<p class="warn">Rejected (out of tube range): {escape(vals)}</p>'
    return (
        '<div class="report">'
        f"<p>Accepted {report['n_accepted']} reading(s) at "
        f"<b>{escape(report['site_id'])}</b> "
        f"(cluster: {escape(report['cluster'])}) &middot; "
        f"surprise: <b>{escape(str(report['surprise']))}</b> "
        f"(z={report['predictive_z']}) &middot; "
        f"weights: {report['weights_assigned']}</p>"
        f"{rejected}"
        "<table><tr><th>level</th><th>before</th><th>after</th>"
        "<th>shift</th><th>95% CI</th></tr>"
        f"{rows}</table>"
        "</div>"
    )


def _render_result(report: dict, narration: str, source: str, ok: bool) -> str:
    cls = "" if ok else "warn"
    prose = (
        f'<div class="narration {cls}">{escape(narration)}'
        f'<p class="meta">— {escape(source)}</p></div>'
    )
    return _render_report(report) + prose
# update the fonts and the spacing

def create_app() -> Flask:
    app = Flask(__name__)
    options = _options()
    tube_length = config.TUBE_LENGTH_CM
    narrator = build_narrator()

    if config.SEED_ON_FIRST_RUN and store.seed_if_empty(config.STATE_PATH):
        print(f"[device] seeded fresh state at {config.STATE_PATH}")

    # Pre-warm the interpreter in the background so the first reading isn't a cold
    # model load (~80 s on the edge GPU). Non-blocking; failure is harmless.
    def _prewarm() -> None:
        if narrator.warm_up():
            print(f"[device] interpreter pre-warmed ({narrator.name})")
        else:
            print("[device] interpreter pre-warm skipped (interpreter offline)")

    threading.Thread(target=_prewarm, daemon=True).start()

    @app.route("/bgp.jpg")
    def background_image():
        return send_from_directory(_PROJECT_DIR, "bgp.jpg")

    def page(result: str = "") -> str:
        return _PAGE.format(
            options=options, min_ntu=_MIN_NTU, mode=config.APP_MODE,
            interpreter=escape(narrator.name), result=result,
        )

    @app.route("/", methods=["GET", "POST"])
    def form() -> str:
        if request.method != "POST":
            return page()

        site_id = request.form["site_id"]
        measurement_ntu = float(request.form["measurement"])
        measurement_cm = ntu_to_cm(measurement_ntu, tube_length)

        # 1) deterministic: fold the reading in and persist on the device
        report = store.record(site_id, [measurement_cm], path=config.STATE_PATH)
        print(f"[device] recorded {measurement_ntu:g} NTU ({measurement_cm:g} cm) "
              f"at {site_id}")

        # 2) swappable: ask the interpreter to narrate (degrade gracefully)
        try:
            narration = narrator.narrate(report)
            source = f"narrated by {narrator.last_used}"
            ok = True
        except NarratorUnavailable as e:
            narration = ("Interpreter unavailable — showing the model results "
                         "without narration. Start Ollama on the Jetson (or open "
                         "the SSH tunnel) and resubmit.")
            source = f"interpreter offline: {e}"
            ok = False

        return page(_render_result(report, narration, source, ok))

    return app


def main() -> None:
    app = create_app()
    print(f" * mode={config.APP_MODE}  interpreter={config.JETSON_OLLAMA_URL} "
          f"({config.JETSON_MODEL})")
    print(f" * open http://{config.HOST}:{config.PORT}/ and submit readings...")
    app.run(host=config.HOST, port=config.PORT, use_reloader=False)


if __name__ == "__main__":
    main()
