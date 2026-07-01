"""
A plain-language description of the Bayesian model, injected into the LLM so it
can *explain and reason about* the model's outputs — not just recite numbers.

Kept separate from narrator.py on purpose: the same grounding context feeds both
one-shot narration and any future Q&A / chatbot path. Structural facts and prior
scales are read live from model.py, so this card never drifts from the model that
actually produced the report.
"""

from globe_bayes.bayes.model import CLUSTERS, SITE_CLUSTER, SITES, Hyperparams


def _structure_block() -> str:
    """Render the lake → cluster → site tree from the live structure constants."""
    lines = ["      lake (the whole water body)"]
    for cluster in CLUSTERS:
        members = ", ".join(s for s in SITES if SITE_CLUSTER[s] == cluster)
        lines.append(f"        - {cluster}: {members}")
    return "\n".join(lines)


def build_model_card(hp: Hyperparams | None = None) -> str:
    """The model explanation, with scales filled in from the current hyperparams."""
    hp = hp or Hyperparams()
    return f"""\
HOW THE MODEL WORKS (use this to explain and reason — do not recite it verbatim):

Domain: water clarity measured with a turbidity tube, in centimeters (cm).
Higher cm = clearer water. The physically valid range is [0, {hp.tube_length:g}] cm;
readings outside it are impossible and are rejected before inference.
Rough ecological guide on this {hp.tube_length:g} cm tube: below ~{hp.tube_length / 3:.0f} cm is turbid /
murky (heavy suspended sediment or algae, little light penetration); about
{hp.tube_length / 3:.0f}-{2 * hp.tube_length / 3:.0f} cm is moderate; above ~{2 * hp.tube_length / 3:.0f} cm is clear (good light
penetration). Use this to judge what a reading means ecologically.

Structure — a 3-level hierarchy (a tree):
{_structure_block()}
Each site belongs to a cluster; each cluster belongs to the lake.

Inference — exact Bayesian partial pooling (closed form, no sampling):
- Every level is Gaussian, so a site's estimate is a precision-weighted blend of
  its own readings and the estimates of its cluster and the lake.
- SHRINKAGE: a site with few or noisy readings leans toward its cluster and the
  lake; a site with many consistent readings trusts its own data. A brand-new
  site is therefore estimated near its neighbors, not from nothing.
- More consistent data -> lower variance: the model grows more certain, so the
  'after' sd / credible interval is tighter than 'before'.
- Information flows upward too: a site's readings also nudge its cluster and lake.

Prior scales (the model's starting beliefs, in cm):
- lake prior mean ~{hp.m0:g} cm (sd ~{hp.v0 ** 0.5:.0f})
- between-cluster spread sd ~{hp.tau2_cluster ** 0.5:.0f}; between-site spread sd ~{hp.tau2_site ** 0.5:.0f}
- measurement noise of a single reading sd ~{hp.sigma2 ** 0.5:.0f} cm

Robustness to bad readings (Student-t errors, dof {hp.nu:g}):
- Each reading gets an influence weight: ~1.0 = a normal, fully trusted reading;
  near 0 = treated as a likely outlier and largely ignored.
- A lone surprising reading barely moves the estimate. If later readings confirm
  the shift, the weights recover and the model adapts — one blunder can't dominate.

How to read the report fields:
- 'surprise' / 'predictive_z': how unexpected the new reading was versus what the
  model predicted for that site (z is in sd units; |z|<1 unsurprising, >3 extreme).
- 'weights_assigned': the influence each new reading received (see robustness).
- 'accepted' / 'rejected': which readings passed the physical [0, {hp.tube_length:g}] gate.
- 'hierarchy': before vs after estimates at site / cluster / lake, each with mean,
  sd, variance, and a 95% credible interval (ci95); 'delta' is the change.
"""
