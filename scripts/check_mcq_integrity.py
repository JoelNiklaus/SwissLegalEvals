"""Verify that MCQ results were scored against the prompt the model actually saw.

Each details row stores both the doc used for scoring and the prompt that was sent.
If the two disagree, the run scored generations against a different option order and
its MCQ numbers are meaningless. This caught exactly that in two published runs,
caused by lighteval shuffling LEXam choices from one shared RNG stream (see
``_patch_deterministic_mcq_shuffle`` in ``tasks.py``).

Run after any evaluation that resumed from a cache::

    python scripts/check_mcq_integrity.py [--details-dir results/details]
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

logger = logging.getLogger(__name__)
DEFAULT_DETAILS_DIR = pathlib.Path("results/details")


def _prompt_text(model_response: dict) -> str:
    """Flatten the recorded prompt; API backends store messages, vLLM a rendered string."""
    raw = model_response["input"]
    messages = [raw] if isinstance(raw, str) else list(raw)
    return " ".join(str(m["content"]) if isinstance(m, dict) else str(m) for m in messages)


def check_file(path: pathlib.Path) -> dict[str, object]:
    """Count rows whose scored question does not appear in the prompt that was sent."""
    df = pd.read_parquet(path, columns=["doc", "model_response"])
    mismatched = sum(
        row["doc"]["query"] not in _prompt_text(row["model_response"]) for _, row in df.iterrows()
    )
    model_dir = path.parts[-4:-2]
    return {
        "model": "/".join(model_dir),
        "run": path.parts[-2],
        "rows": len(df),
        "mismatched": mismatched,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--details-dir", type=pathlib.Path, default=DEFAULT_DETAILS_DIR)
    args = parser.parse_args()

    files = sorted(args.details_dir.rglob("details_lexam_mcq*.parquet"))
    if not files:
        raise FileNotFoundError(f"No MCQ detail files under {args.details_dir}")
    logger.info("Checking %d MCQ detail files", len(files))

    rows = Parallel(n_jobs=8)(delayed(check_file)(f) for f in tqdm(files))
    per_run = pd.DataFrame(rows).groupby(["model", "run"])[["rows", "mismatched"]].sum().reset_index()
    per_run["pct_mismatched"] = (100 * per_run["mismatched"] / per_run["rows"]).round(1)
    per_run = per_run.sort_values("pct_mismatched", ascending=False)
    logger.info("\n%s", per_run.to_string(index=False))

    corrupted = per_run[per_run["mismatched"] > 0]
    if not corrupted.empty:
        logger.error(
            "%d of %d runs scored generations against a different option order; "
            "their MCQ metrics must be regenerated.",
            len(corrupted),
            len(per_run),
        )
        return 1
    logger.info("All %d runs scored the prompts they generated from.", len(per_run))
    return 0


if __name__ == "__main__":
    sys.exit(main())
