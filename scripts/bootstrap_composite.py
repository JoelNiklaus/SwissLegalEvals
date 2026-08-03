"""Bootstrap a confidence interval for each model's composite score.

The composite averages four task groups, so a ranking gap only means something
if it survives the spread of the tasks inside those groups. Resampling tasks
within each group with replacement gives the interval the blog post quotes when
it calls the leading models tied.

    python scripts/bootstrap_composite.py [--resamples 10000]
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from swiss_legal_evals.plot import PRIMARY_METRICS  # noqa: E402

logger = logging.getLogger(__name__)

SUMMARY_LONG_CSV = pathlib.Path("results/summary_long.csv")
# The composite averages these four groups equally, and each group averages its
# families equally, so the bootstrap has to preserve both levels of weighting.
GROUPS = {
    "slds": ["slds"],
    "lexam_oq": ["lexam_oq"],
    "translation": ["sdst", "slt", "sscprt"],
    "mcq": ["lexam_mcq_4_idk", "lexam_mcq_8_idk", "lexam_mcq_16_idk"],
}
MCQ_SCALE = 100.0


def _task_scores(df: pd.DataFrame) -> pd.DataFrame:
    """One primary-metric score per model, family, and task."""
    rows = []
    for (model, family, task), group in df.groupby(["model", "family", "task"]):
        if family not in PRIMARY_METRICS:
            continue
        available = group["metric"].tolist()
        metric = next((m for m in PRIMARY_METRICS[family] if m in available), None)
        if metric is None:
            continue
        value = group.loc[group["metric"] == metric, "value"].iloc[0]
        # MCQ accuracy is 0-1 while the judges score 0-100.
        scaled = value * MCQ_SCALE if family.startswith("lexam_mcq") else value
        rows.append({"model": model, "family": family, "task": task, "score": scaled})
    return pd.DataFrame(rows)


def _composite(family_means: dict[str, float]) -> float:
    return float(np.mean([np.mean([family_means[f] for f in fams]) for fams in GROUPS.values()]))


def bootstrap_model(scores: pd.DataFrame, resamples: int, rng: np.random.Generator) -> tuple[float, float]:
    """Composite score and the half-width of its 95% interval for one model."""
    by_family = {f: sub["score"].to_numpy() for f, sub in scores.groupby("family")}
    draws = np.empty(resamples)
    for i in range(resamples):
        # Resample tasks inside each family, so every family keeps its weight.
        draws[i] = _composite({f: rng.choice(v, size=len(v), replace=True).mean() for f, v in by_family.items()})
    low, high = np.percentile(draws, [2.5, 97.5])
    return _composite({f: v.mean() for f, v in by_family.items()}), float((high - low) / 2)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--summary-long", type=pathlib.Path, default=SUMMARY_LONG_CSV)
    args = parser.parse_args()

    scores = _task_scores(pd.read_csv(args.summary_long))
    rng = np.random.default_rng(0)
    rows = []
    needed = {family for families in GROUPS.values() for family in families}
    for model, sub in scores.groupby("model"):
        # Translation-only models have no composite to speak of.
        if not needed <= set(sub["family"]):
            continue
        composite, half_width = bootstrap_model(sub, args.resamples, rng)
        rows.append({"model": model, "composite": composite, "half_width_95": half_width})

    table = pd.DataFrame(rows).sort_values("composite", ascending=False).round(2)
    logger.info("\n%s", table.to_string(index=False))


if __name__ == "__main__":
    main()
