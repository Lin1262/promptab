"""promptab command line interface."""

from __future__ import annotations

import click

from . import io, stats


def _pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


@click.group()
@click.version_option(package_name="promptab")
def main() -> None:
    """Statistically sound A/B testing for LLM prompts."""


@main.command()
@click.argument("results", type=click.Path(exists=True, dir_okay=False))
@click.option("--level", default=0.95, show_default=True, help="Credible level.")
def summary(results: str, level: float) -> None:
    """Score and credible interval for every variant."""
    table = io.load(results)
    click.echo(f"{'variant':40} {'n':>5} {'score':>7}  {int(level * 100)}% interval")
    for variant, scores in sorted(table.items()):
        s = stats.summarize(list(scores.values()), level)
        click.echo(f"{variant[:40]:40} {len(scores):5} {_pct(s.estimate)}  [{_pct(s.low)}, {_pct(s.high)}]")


@main.command()
@click.argument("results", type=click.Path(exists=True, dir_okay=False))
@click.option("-a", "variant_a", required=True, help="Baseline variant.")
@click.option("-b", "variant_b", required=True, help="Challenger variant.")
@click.option("--threshold", default=0.005, show_default=True,
              help="Acceptable expected loss, in score units (0.005 = 0.5 pp).")
@click.option("--level", default=0.95, show_default=True)
def compare(results: str, variant_a: str, variant_b: str, threshold: float, level: float) -> None:
    """Paired comparison of B vs A, with a ship / keep-testing call."""
    a, b = io.aligned(io.load(results), variant_a, variant_b)
    c = stats.compare(a, b, level)
    click.echo(f"paired test cases : {c.n_pairs}")
    click.echo(f"B - A             : {c.diff.estimate:+.3f}  [{c.diff.low:+.3f}, {c.diff.high:+.3f}]")
    click.echo(f"P(B better)       : {c.prob_b_better:.3f}")
    click.echo(f"expected loss     : ship A {c.expected_loss_a:.4f} | ship B {c.expected_loss_b:.4f}")
    click.echo(f"p-value           : {c.p_value:.4f}  ({c.method})")
    click.echo(f"decision          : {stats.decide(c, threshold).upper()}")


@main.command()
@click.option("--baseline", "p_a", required=True, type=float, help="Baseline pass rate, e.g. 0.7")
@click.option("--lift", required=True, type=float, help="Improvement you care about, e.g. 0.05")
@click.option("--concordance", default=0.8, show_default=True,
              help="Share of cases where both variants agree.")
@click.option("--threshold", default=0.005, show_default=True)
def plan(p_a: float, lift: float, concordance: float, threshold: float) -> None:
    """Estimate how many test cases you need before you can stop."""
    r = stats.plan(p_a, lift, concordance, threshold)
    click.echo(f"median stop  : {r['median_n']} cases")
    click.echo(f"90th pct stop: {r['p90_n']} cases")
    click.echo(f"correct call : {r['correct_rate']:.0%} of simulations")
