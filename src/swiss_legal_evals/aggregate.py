"""Aggregate lighteval JSON results into tidy pandas tables."""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "results"
DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "results" / "summary_long.csv"

FAMILY_PATTERN = re.compile(
    r"^(slds|sdst|slt|sscprt|lexam_oq|lexam_mcq_\d+(?:_idk)?)"
)


def parse_family(task_name: str) -> str:
    """Map a lighteval task full name to a benchmark family."""
    base = task_name.split("|")[0]
    match = FAMILY_PATTERN.match(base)
    if match:
        return match.group(1)
    raise ValueError(f"Cannot parse family from task name: {task_name}")


def _is_lighteval_summary_row(task_name: str) -> bool:
    """Detect aggregate rows already computed by lighteval."""
    base = task_name.split("|")[0]
    return base == "all" or base.endswith(":_average")


def load_results_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _discover_latest_results(results_root: Path) -> dict[str, Path]:
    """Map model directory name to the latest results_*.json (per plan glob)."""
    by_model: dict[str, list[Path]] = {}
    for path in sorted(results_root.rglob("results_*.json")):
        by_model.setdefault(path.parent.name, []).append(path)
    latest: dict[str, Path] = {}
    for model_name, paths in by_model.items():
        latest[model_name] = paths[-1]
    return latest


def results_to_long_df(results_dir: Path) -> pd.DataFrame:
    """Parse results/results/**/results_*.json into a long DataFrame."""
    rows: list[dict[str, str | float]] = []

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    latest_files = _discover_latest_results(results_dir)
    if not latest_files:
        raise ValueError(f"No results_*.json files under {results_dir}")

    for model_name, results_file in sorted(latest_files.items()):
        payload = load_results_json(results_file)
        metrics = payload["results"]

        for task_full_name, task_metrics in metrics.items():
            if _is_lighteval_summary_row(task_full_name):
                continue
            family = parse_family(task_full_name)
            for metric_name, value in task_metrics.items():
                rows.append(
                    {
                        "model": model_name,
                        "family": family,
                        "task": task_full_name,
                        "metric": metric_name,
                        "value": float(value),
                    }
                )

    if not rows:
        raise ValueError(f"No result rows parsed from {results_dir}")

    return pd.DataFrame(rows)


def family_mean_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    """Mean metric value per model and family."""
    return (
        long_df.groupby(["model", "family", "metric"], as_index=False)["value"]
        .mean()
        .rename(columns={"value": "mean_value"})
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Aggregate lighteval results.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing per-model result subfolders",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="Path for long-format CSV",
    )
    parser.add_argument(
        "--family-summary",
        type=Path,
        default=None,
        help="Optional path for family-mean summary CSV",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    long_df = results_to_long_df(args.results_dir)
    summary_df = family_mean_summary(long_df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(args.output, index=False)
    logger.info("Wrote long results to %s (%d rows)", args.output, len(long_df))

    family_path = args.family_summary or args.output.with_name("summary_family_mean.csv")
    summary_df.to_csv(family_path, index=False)
    logger.info("Wrote family summary to %s", family_path)


if __name__ == "__main__":
    main()
