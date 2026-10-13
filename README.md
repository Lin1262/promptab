# promptab

**Statistically sound A/B testing for LLM prompts, with Bayesian early stopping.**

Most prompt comparisons look like this: run 50 test cases, see 82% vs 78%, ship the "winner". At that sample size the difference is usually noise. `promptab` tells you whether a prompt change is real, how sure you can be, and when you have run enough cases to stop paying for more API calls.

```
$ promptab summary examples/results.csv
variant                                      n   score  95% interval
v1                                         120  72.5%  [ 64.0%,  79.9%]
v2                                         120  80.8%  [ 73.1%,  87.1%]

$ promptab compare examples/results.csv -a v1 -b v2
paired test cases : 120
B - A             : +0.083  [+0.025, +0.146]
P(B better)       : 0.997
expected loss     : ship A 0.0820 | ship B 0.0000
p-value           : 0.0129  (dirichlet-2x2 + exact McNemar)
decision          : SHIP B
```

The two intervals in `summary` overlap, so an unpaired comparison would call this a toss-up. Pairing on the same 120 test cases shows the improvement is real. (`examples/results.csv` is simulated data.)

## Why another eval tool

| | Typical eval dashboards | promptab |
|---|---|---|
| Uncertainty on a pass rate | none, or a CLT interval that is miscalibrated below a few hundred cases | Jeffreys Beta interval, calibrated at small n (tested) |
| Comparing two prompts | two independent percentages | **paired** analysis on the same test cases, which is far more sensitive |
| When to stop | fixed n, chosen by gut | **expected-loss stopping rule**: stop once picking the leader costs you < 0.5 pp in expectation |
| Planning | none | simulation of how many cases you will need |

## Early stopping saves most of the test budget

Simulated with `promptab plan` (baseline pass rate 70%, the two prompts agree on 80% of cases, 200 simulations each):

| True improvement | promptab median stop | promptab 90th pct | Fixed-n McNemar test at 80% power |
|---|---|---|---|
| +10 pp | 40 cases | 80 cases | ~170 cases |
| +5 pp | 60 cases | 160 cases | ~670 cases |

Caveat: the expected-loss rule bounds the *cost of a wrong call*, not the false-positive rate. With a +5 pp true lift it picked the better prompt in 90% of simulations; when it picks wrong, the expected cost is under the threshold you set. If you need classical error-rate guarantees, use the p-value column with a pre-registered n.

## Install

```
pip install -e .
```

## Usage

Input is either a promptfoo results file (`promptfoo eval -o results.json`) or a CSV with columns `case,variant,score` (score is 0/1 or any number).

```
promptab summary results.json                          # every variant, with intervals
promptab compare results.csv -a old -b new             # paired comparison + decision
promptab plan --baseline 0.7 --lift 0.05               # how many cases will I need?
```

Python API:

```python
from promptab import compare, decide
c = compare(scores_a, scores_b)   # aligned by test case
decide(c, threshold=0.005)        # "ship A" | "ship B" | "keep testing"
```

## Methods

- **Binary scores:** Beta posterior with Jeffreys prior; Wilson interval available as a frequentist cross-check.
- **Paired binary comparison:** Dirichlet posterior over the 2x2 table (both pass / A only / B only / neither); the difference in pass rates is `θ_B-only − θ_A-only`. Exact McNemar test alongside.
- **Continuous scores:** Bayesian bootstrap (Rubin, 1981) over per-case differences; Wilcoxon signed-rank test alongside.
- **Stopping rule:** expected loss `E[max(θ_other − θ_chosen, 0)]` below a threshold, as in Bayesian A/B testing practice.

## References

- Miller, E. (2024). *Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations.* arXiv:2411.00640.
- Bowyer, S. et al. (2025). *Position: Don't Use the CLT in LLM Evals With Fewer Than a Few Hundred Datapoints.* arXiv:2503.01747.
- Rubin, D. B. (1981). *The Bayesian Bootstrap.* Annals of Statistics 9(1).

## License

MIT
