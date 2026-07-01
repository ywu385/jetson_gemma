"""
Hierarchical Gaussian conjugate model for turbidity-tube water clarity (cm),
with robust (Student-t) down-weighting of outlier readings.

Hierarchy (v1, hardcoded):

    lake                      (top / water body)
    ├── shoreline (cluster)   ── site_1, site_2, site_3, site_4
    └── center    (cluster)   ── site_5, site_6

Each level is Gaussian, so the *pooling* is exact closed form (the Normal is its
own conjugate prior for a mean). We assemble the Gaussian Markov tree as a joint
precision / information matrix over the latent means and condition on the site
readings:

    reading  x ~ N(mu_site,    sigma2)       sigma2 fixed (tube measurement noise)
    mu_site    ~ N(mu_cluster, tau2_site)    within-cluster spread between sites
    mu_cluster ~ N(mu_lake,    tau2_cluster) between-cluster spread within lake
    mu_lake    ~ N(m0, v0)                   top prior

Robustness: a plain Normal likelihood lets one bad reading dilute the model
without bound. Instead we fit Student-t errors via iteratively reweighted least
squares (IRLS): each reading gets weight w = (nu+1)/(nu + r^2/sigma2), where r is
its residual from the fitted site mean. Outliers (large r) get w -> 0, so their
influence is bounded; if later readings confirm the shift, the weights recover
and the model adapts. A hard physical-bounds gate ([0, tube_length]) rejects
impossible values first. All quantities are in cm / cm^2. sigma2 is fixed in v1.
"""

from dataclasses import asdict, dataclass, field

import numpy as np

# --- structure ---------------------------------------------------------------

SITES = ["site_1", "site_2", "site_3", "site_4", "site_5", "site_6"]
CLUSTERS = ["shoreline", "center"]
SITE_CLUSTER = {
    "site_1": "shoreline",
    "site_2": "shoreline",
    "site_3": "shoreline",
    "site_4": "shoreline",
    "site_5": "center",
    "site_6": "center",
}

# node ordering for the joint precision matrix: lake, clusters, then sites
_NODES = ["lake"] + CLUSTERS + SITES
_IDX = {name: i for i, name in enumerate(_NODES)}

_MAX_IRLS_ITERS = 10
_IRLS_TOL = 1e-4


@dataclass
class Hyperparams:
    """Fixed structural hyperparameters (cm / cm^2), tuned for a 60 cm tube."""

    m0: float = 30.0            # lake prior mean (mid-range of a 0-60 tube)
    v0: float = 225.0          # lake prior variance (sd 15)
    tau2_cluster: float = 100.0  # between-cluster spread within the lake (sd 10)
    tau2_site: float = 64.0      # between-site spread within a cluster (sd 8)
    sigma2: float = 25.0         # measurement noise of one tube reading (sd 5)
    nu: float = 4.0              # Student-t degrees of freedom (outlier robustness)
    tube_length: float = 60.0    # physical max reading; [0, tube_length] is valid


@dataclass
class ModelState:
    """Structure + the raw readings per site (kept so outliers can be reweighted)."""

    hp: Hyperparams = field(default_factory=Hyperparams)
    readings: dict = field(default_factory=lambda: {s: [] for s in SITES})


def build_default_model() -> ModelState:
    """Fresh model: 6 sites / 2 clusters / 1 lake, default priors, no readings yet."""
    return ModelState()


def to_dict(state: ModelState) -> dict:
    """Plain-dict view of the state, ready to JSON-serialize. Pure, no I/O."""
    return {"hp": asdict(state.hp), "readings": state.readings}


def from_dict(d: dict) -> ModelState:
    """Rebuild a ModelState from to_dict() output, tolerating missing sites."""
    raw = d.get("readings", {})
    return ModelState(
        hp=Hyperparams(**d.get("hp", {})),
        readings={s: [float(x) for x in raw.get(s, [])] for s in SITES},
    )


# --- inference ---------------------------------------------------------------

def _solve_with_stats(hp: Hyperparams, n_eff: dict, sum_eff: dict) -> dict:
    """Exact tree posterior given (possibly weighted) per-site sufficient stats."""
    p = len(_NODES)
    lam = np.zeros((p, p))   # joint precision (information) matrix
    h = np.zeros(p)          # information vector (precision @ mean)

    # lake top prior
    li = _IDX["lake"]
    lam[li, li] += 1.0 / hp.v0
    h[li] += hp.m0 / hp.v0

    # lake <-> cluster links
    for c in CLUSTERS:
        ci = _IDX[c]
        prec = 1.0 / hp.tau2_cluster
        lam[li, li] += prec
        lam[ci, ci] += prec
        lam[li, ci] -= prec
        lam[ci, li] -= prec

    # cluster <-> site links
    for s in SITES:
        si, ci = _IDX[s], _IDX[SITE_CLUSTER[s]]
        prec = 1.0 / hp.tau2_site
        lam[ci, ci] += prec
        lam[si, si] += prec
        lam[ci, si] -= prec
        lam[si, ci] -= prec

    # weighted site data: effective count n_eff adds precision n_eff/sigma2
    for s in SITES:
        if n_eff[s] > 0:
            si = _IDX[s]
            lam[si, si] += n_eff[s] / hp.sigma2
            h[si] += sum_eff[s] / hp.sigma2

    cov = np.linalg.inv(lam)
    mean = cov @ h
    return {name: (float(mean[i]), float(cov[i, i])) for i, name in enumerate(_NODES)}


def _weighted_stats(readings: dict, weights: dict) -> tuple:
    n_eff = {s: float(sum(weights[s])) for s in SITES}
    sum_eff = {s: float(sum(w * x for w, x in zip(weights[s], readings[s]))) for s in SITES}
    return n_eff, sum_eff


def _solve_robust(state: ModelState) -> tuple:
    """IRLS fit with Student-t errors. Returns (posterior, weights-per-reading)."""
    hp = state.hp
    weights = {s: [1.0] * len(xs) for s, xs in state.readings.items()}

    for _ in range(_MAX_IRLS_ITERS):
        post = _solve_with_stats(hp, *_weighted_stats(state.readings, weights))
        max_delta = 0.0
        new_weights = {}
        for s in SITES:
            mu = post[s][0]
            nw = [(hp.nu + 1.0) / (hp.nu + (x - mu) ** 2 / hp.sigma2) for x in state.readings[s]]
            new_weights[s] = nw
            for old, new in zip(weights[s], nw):
                max_delta = max(max_delta, abs(old - new))
        weights = new_weights
        if max_delta < _IRLS_TOL:
            break

    post = _solve_with_stats(hp, *_weighted_stats(state.readings, weights))
    return post, weights


def _summary(mean: float, var: float) -> dict:
    sd = var ** 0.5
    return {
        "mean": round(mean, 2),
        "variance": round(var, 2),
        "sd": round(sd, 2),
        "ci95": [round(mean - 1.96 * sd, 2), round(mean + 1.96 * sd, 2)],
    }


def _surprise_label(abs_z: float) -> str:
    if abs_z < 1.0:
        return "unsurprising"
    if abs_z < 2.0:
        return "mild"
    if abs_z < 3.0:
        return "notable"
    return "extreme"


def update(state: ModelState, site_id: str, observations) -> tuple:
    """
    Fold new turbidity readings for one site into the model (robustly).

    Physically impossible readings (outside [0, tube_length]) are rejected up
    front. Accepted readings are fitted with Student-t down-weighting, so a lone
    blunder barely moves the estimate. Returns (new_state, report); `report`
    carries the before/after ladder up the hierarchy plus per-reading weights and
    a surprise score — the context an LLM needs to narrate what happened.
    """
    if site_id not in SITES:
        raise ValueError(f"unknown site {site_id!r}; known sites: {SITES}")
    observations = [float(x) for x in observations]
    if not observations:
        raise ValueError("no observations provided")

    hp = state.hp
    accepted, rejected = [], []
    for x in observations:
        if 0.0 <= x <= hp.tube_length:
            accepted.append(x)
        else:
            rejected.append({"value": x, "reason": f"outside tube range [0, {hp.tube_length:g}]"})

    before, _ = _solve_robust(state)

    # surprise of the accepted batch vs the site's prior predictive (for narration)
    b_mean, b_var = before[site_id]
    if accepted:
        obs_mean = sum(accepted) / len(accepted)
        pred_sd = (b_var + hp.sigma2 / len(accepted)) ** 0.5
        z = (obs_mean - b_mean) / pred_sd
        predictive_z, surprise = round(z, 2), _surprise_label(abs(z))
    else:
        predictive_z, surprise = None, "n/a (all readings rejected)"

    state.readings[site_id].extend(accepted)
    after, weights = _solve_robust(state)

    # influence of the readings we just added, normalized so 1.0 = an ordinary
    # reading and ~0 = effectively ignored as an outlier
    w_full = (hp.nu + 1.0) / hp.nu
    new_weights = [round(w / w_full, 3) for w in weights[site_id][-len(accepted):]] if accepted else []

    cluster = SITE_CLUSTER[site_id]
    hierarchy = []
    for level, name in [("site", site_id), ("cluster", cluster), ("lake", "lake")]:
        b_m, b_v = before[name]
        a_m, a_v = after[name]
        hierarchy.append({
            "level": level,
            "id": name,
            "before": _summary(b_m, b_v),
            "after": _summary(a_m, a_v),
            "delta": {
                "mean_shift": round(a_m - b_m, 2),
                "variance_change": round(a_v - b_v, 2),
            },
        })

    report = {
        "site_id": site_id,
        "cluster": cluster,
        "units": "cm (turbidity tube clarity; higher = clearer)",
        "tube_length": hp.tube_length,
        "observations": observations,
        "accepted": accepted,
        "rejected": rejected,
        "n_accepted": len(accepted),
        "n_rejected": len(rejected),
        "predictive_z": predictive_z,
        "surprise": surprise,
        "weights_assigned": new_weights,
        "hierarchy": hierarchy,
    }
    return state, report


# --- demo --------------------------------------------------------------------

def demo() -> None:
    """Show a normal reading vs. a suspicious one vs. an impossible one."""
    import copy
    import json

    def seeded() -> ModelState:
        m = build_default_model()
        seed = {
            "site_1": [28, 31, 26],
            "site_2": [33, 29],
            "site_3": [30],
            "site_5": [48, 52, 46],   # center water is clearer
            "site_6": [50, 47],
        }
        for s, xs in seed.items():
            m, _ = update(m, s, xs)
        return m

    base = seeded()
    print("=== state after seeding (60 cm tube) ===")
    snap, _ = _solve_robust(base)
    for name in _NODES:
        m, v = snap[name]
        print(f"  {name:<10} mean={m:5.2f} cm   sd={v ** 0.5:4.2f}")

    print("\n=== one new reading at site_4 (fresh shoreline site) ===")
    for label, reading in [("normal 31", 31), ("suspicious 59", 59), ("impossible 85", 85)]:
        _, r = update(copy.deepcopy(base), "site_4", [reading])
        site, lake = r["hierarchy"][0], r["hierarchy"][2]
        print(
            f"  {label:<14} acc/rej={r['n_accepted']}/{r['n_rejected']}  "
            f"z={r['predictive_z']}  {r['surprise']:<12}  weight={r['weights_assigned']}\n"
            f"                 site_4 {site['before']['mean']:>5} -> {site['after']['mean']:<5} | "
            f"lake {lake['before']['mean']:>5} -> {lake['after']['mean']}"
        )

    print("\n=== same 59 cm reading, but at site_1 (already has 3 readings ~28-31) ===")
    _, r = update(copy.deepcopy(base), "site_1", [59])
    site, lake = r["hierarchy"][0], r["hierarchy"][2]
    print(
        f"  z={r['predictive_z']}  {r['surprise']}  weight={r['weights_assigned']}\n"
        f"                 site_1 {site['before']['mean']:>5} -> {site['after']['mean']:<5} | "
        f"lake {lake['before']['mean']:>5} -> {lake['after']['mean']}   "
        f"(corroborating data crushes the outlier)"
    )

    _, r = update(copy.deepcopy(base), "site_4", [59])
    print("\n=== full report for the suspicious 59 cm reading (fresh site) ===")
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    demo()
