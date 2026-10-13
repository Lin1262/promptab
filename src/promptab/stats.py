"""Core statistics for comparing prompt / model variants on a shared test set.

Design choices (see README for references):
- Binary scores: Beta posteriors with a Jeffreys prior, Wilson intervals as a
  frequentist cross-check. CLT intervals are avoided because they are badly
  calibrated at the sample sizes typical of prompt evals (tens to hundreds).
- Paired binary comparison: Dirichlet posterior over the 2x2 outcome table, so
  the comparison uses the pairing (same test cases) instead of treating the two
  variants as independent samples. Exact McNemar test reported alongside.
- Continuous scores: Bayesian bootstrap (Rubin 1981) on per-case differences.
- Early stopping: expected-loss rule. Stop when the expected score you would
  give up by picking the current leader falls below a user-set threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

JEFFREYS = 0.5


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float


@dataclass(frozen=True)
class Comparison:
    n_pairs: int
    diff: Interval  # mean(B) - mean(A)
    prob_b_better: float
    expected_loss_a: float  # expected shortfall if you ship A
    expected_loss_b: float  # expected shortfall if you ship B
    p_value: float | None  # exact McNemar (binary) / Wilcoxon (continuous)
    method: str


def _is_binary(x: np.ndarray) -> bool:
    return bool(np.all((x == 0) | (x == 1)))


def wilson(k: int, n: int, level: float = 0.95) -> Interval:
    if n == 0:
        return Interval(float("nan"), 0.0, 1.0)
    z = stats.norm.ppf(0.5 + level / 2)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return Interval(p, max(0.0, centre - half), min(1.0, centre + half))


def beta_interval(k: int, n: int, level: float = 0.95) -> Interval:
    a, b = k + JEFFREYS, n - k + JEFFREYS
    lo, hi = stats.beta.ppf([(1 - level) / 2, (1 + level) / 2], a, b)
    return Interval(k / n if n else float("nan"), float(lo), float(hi))


def summarize(scores, level: float = 0.95, seed: int = 0) -> Interval:
    """Mean score with a credible interval suited to small samples."""
    x = np.asarray(scores, dtype=float)
    if x.size == 0:
        raise ValueError("no scores")
    if _is_binary(x):
        return beta_interval(int(x.sum()), x.size, level)
    draws = _bayes_bootstrap_means(x, seed=seed)
    lo, hi = np.quantile(draws, [(1 - level) / 2, (1 + level) / 2])
    return Interval(float(x.mean()), float(lo), float(hi))


def _bayes_bootstrap_means(x: np.ndarray, draws: int = 20000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = rng.dirichlet(np.ones(x.size), size=draws)
    return w @ x


def compare(a, b, level: float = 0.95, draws: int = 40000, seed: int = 0) -> Comparison:
    """Paired comparison of variant B against variant A on the same test cases."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("a and b must be non-empty and aligned by test case")
    rng = np.random.default_rng(seed)
    q = [(1 - level) / 2, (1 + level) / 2]

    if _is_binary(a) and _is_binary(b):
        both = int(np.sum((a == 1) & (b == 1)))
        a_only = int(np.sum((a == 1) & (b == 0)))
        b_only = int(np.sum((a == 0) & (b == 1)))
        neither = a.size - both - a_only - b_only
        theta = rng.dirichlet(np.array([both, a_only, b_only, neither]) + JEFFREYS, size=draws)
        d = theta[:, 2] - theta[:, 1]  # P(B pass) - P(A pass)
        discordant = a_only + b_only
        p = float(stats.binomtest(b_only, discordant).pvalue) if discordant else 1.0
        method = "dirichlet-2x2 + exact McNemar"
    else:
        diffs = b - a
        d = _bayes_bootstrap_means(diffs, draws=draws, seed=seed)
        p = float(stats.wilcoxon(diffs).pvalue) if np.any(diffs != 0) else 1.0
        method = "bayesian-bootstrap + Wilcoxon"

    lo, hi = np.quantile(d, q)
    return Comparison(
        n_pairs=a.size,
        diff=Interval(float(b.mean() - a.mean()), float(lo), float(hi)),
        prob_b_better=float(np.mean(d > 0)),
        expected_loss_a=float(np.mean(np.maximum(d, 0))),
        expected_loss_b=float(np.mean(np.maximum(-d, 0))),
        p_value=p,
        method=method,
    )


def decide(cmp: Comparison, threshold: float = 0.005) -> str:
    """Expected-loss stopping rule: 'ship A', 'ship B', or 'keep testing'.

    threshold is in score units, e.g. 0.005 = half a percentage point of pass rate.
    """
    if cmp.expected_loss_b <= threshold and cmp.expected_loss_b <= cmp.expected_loss_a:
        return "ship B"
    if cmp.expected_loss_a <= threshold:
        return "ship A"
    return "keep testing"


def plan(p_a: float, lift: float, concordance: float = 0.8, threshold: float = 0.005,
         max_n: int = 2000, sims: int = 300, seed: int = 0) -> dict:
    """Simulate how many paired binary test cases are needed before decide() stops.

    concordance: share of cases where A and B agree (typical prompts tweaks: 0.7-0.9).
    Returns the median and 90th percentile stopping n and the rate of correct calls.
    """
    p_b = p_a + lift
    # Joint cell probabilities consistent with the marginals and concordance.
    disc = 1 - concordance
    b_only = (disc + lift) / 2
    a_only = (disc - lift) / 2
    both = p_a - a_only
    neither = 1 - both - a_only - b_only
    cells = np.array([both, a_only, b_only, neither])
    if np.any(cells < 0) or not 0 < p_b < 1:
        raise ValueError("inconsistent p_a / lift / concordance")
    rng = np.random.default_rng(seed)
    checkpoints = list(range(20, max_n + 1, 20))
    stops, correct = [], 0
    for s in range(sims):
        draws = rng.choice(4, size=max_n, p=cells)
        a = np.isin(draws, [0, 1]).astype(float)
        b = np.isin(draws, [0, 2]).astype(float)
        verdict, n_stop = "keep testing", max_n
        for n in checkpoints:
            verdict = decide(compare(a[:n], b[:n], draws=4000, seed=s), threshold)
            if verdict != "keep testing":
                n_stop = n
                break
        stops.append(n_stop)
        correct += (verdict == "ship B") if lift > 0 else (verdict == "ship A")
    return {
        "median_n": int(np.median(stops)),
        "p90_n": int(np.quantile(stops, 0.9)),
        "correct_rate": correct / sims,
    }
