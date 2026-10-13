import json

import numpy as np
import pytest
from click.testing import CliRunner

from promptab import compare, decide, summarize
from promptab import io
from promptab.cli import main
from promptab.stats import wilson


def test_wilson_known_value():
    # 8/10 at 95%: textbook Wilson interval is approx [0.490, 0.943]
    w = wilson(8, 10)
    assert w.low == pytest.approx(0.490, abs=1e-3)
    assert w.high == pytest.approx(0.943, abs=1e-3)


def test_summary_interval_contains_estimate():
    s = summarize([1, 1, 0, 1, 0, 1, 1, 1])
    assert s.low < s.estimate < s.high


def test_binary_interval_calibration():
    """Jeffreys intervals should cover the true rate ~95% of the time at small n."""
    rng = np.random.default_rng(1)
    p, n, hits, trials = 0.8, 30, 0, 2000
    for _ in range(trials):
        s = summarize(rng.binomial(1, p, n))
        hits += s.low <= p <= s.high
    assert 0.93 <= hits / trials <= 0.98


def test_clear_winner_detected():
    rng = np.random.default_rng(2)
    a = rng.binomial(1, 0.6, 300)
    b = np.where(rng.random(300) < 0.5, 1, a)  # B fixes half of A's failures
    c = compare(a, b)
    assert c.prob_b_better > 0.99
    assert c.diff.low > 0
    assert c.p_value < 0.01
    assert decide(c, threshold=0.01) == "ship B"


def test_identical_variants_not_shipped_early():
    a = np.array([1, 0, 1, 1, 0] * 4)
    c = compare(a, a.copy())
    assert c.prob_b_better == pytest.approx(0.5, abs=0.05)
    assert c.p_value == 1.0


def test_continuous_scores():
    rng = np.random.default_rng(3)
    a = rng.normal(0.5, 0.1, 80)
    c = compare(a, a + 0.05 + rng.normal(0, 0.02, 80))
    assert c.method.startswith("bayesian-bootstrap")
    assert c.prob_b_better > 0.99


def test_misaligned_raises():
    with pytest.raises(ValueError):
        compare([1, 0], [1])


def test_promptfoo_loader(tmp_path):
    rows = []
    for t in range(4):
        rows.append({"promptIdx": 0, "testIdx": t, "success": t != 3,
                     "prompt": {"label": "v1"}, "provider": {"id": "gpt"}})
        rows.append({"promptIdx": 1, "testIdx": t, "success": True,
                     "prompt": {"label": "v2"}, "provider": {"id": "gpt"}})
    f = tmp_path / "results.json"
    f.write_text(json.dumps({"results": {"version": 3, "results": rows}}))
    table = io.load(f)
    assert set(table) == {"v1 | gpt", "v2 | gpt"}
    a, b = io.aligned(table, "v1 | gpt", "v2 | gpt")
    assert a == [1, 1, 1, 0] and b == [1, 1, 1, 1]


def test_cli_end_to_end(tmp_path):
    f = tmp_path / "r.csv"
    lines = ["case,variant,score"]
    for i in range(60):
        lines.append(f"{i},old,{int(i % 3 != 0)}")
        lines.append(f"{i},new,1")
    f.write_text("\n".join(lines))
    runner = CliRunner()
    out = runner.invoke(main, ["compare", str(f), "-a", "old", "-b", "new"])
    assert out.exit_code == 0, out.output
    assert "SHIP B" in out.output
    out = runner.invoke(main, ["summary", str(f)])
    assert out.exit_code == 0 and "old" in out.output
