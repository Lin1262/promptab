"""Load eval results into a case x variant score table."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

Table = dict[str, dict[str, float]]  # variant -> case_id -> score


def load(path: str | Path) -> Table:
    path = Path(path)
    if path.suffix.lower() == ".json":
        return load_promptfoo(path)
    return load_csv(path)


def load_csv(path: str | Path) -> Table:
    """CSV with columns: case, variant, score (score is 0/1 or a number)."""
    table: Table = defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = {"case", "variant", "score"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing columns: {sorted(missing)}")
        for row in reader:
            table[row["variant"]][row["case"]] = float(row["score"])
    return dict(table)


def load_promptfoo(path: str | Path) -> Table:
    """Results file from `promptfoo eval -o results.json`.

    A variant is one prompt x provider pair; a case is one test (testIdx).
    Uses `score` when present, otherwise `success` as 0/1.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = data.get("results", data)
    if isinstance(rows, dict):
        rows = rows.get("results", [])
    table: Table = defaultdict(dict)
    for r in rows:
        prompt = r.get("prompt") or {}
        provider = r.get("provider") or {}
        if isinstance(provider, str):
            provider = {"id": provider}
        p_label = prompt.get("label") or f"prompt{r.get('promptIdx', 0)}"
        v_label = provider.get("label") or provider.get("id") or "provider"
        variant = f"{p_label} | {v_label}"
        case = str(r.get("testIdx", r.get("id")))
        score = r.get("score")
        if score is None:
            score = 1.0 if r.get("success") else 0.0
        table[variant][case] = float(score)
    if not table:
        raise ValueError("no results found in promptfoo file")
    return dict(table)


def aligned(table: Table, a: str, b: str) -> tuple[list[float], list[float]]:
    """Scores for variants a and b on the test cases both were run on."""
    for v in (a, b):
        if v not in table:
            raise KeyError(f"unknown variant {v!r}; have {sorted(table)}")
    cases = sorted(set(table[a]) & set(table[b]))
    if not cases:
        raise ValueError("variants share no test cases")
    return [table[a][c] for c in cases], [table[b][c] for c in cases]
