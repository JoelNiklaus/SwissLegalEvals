"""Plot aggregated benchmark results with Plotly."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import plotly.express as px

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FAMILY_SUMMARY = PROJECT_ROOT / "results" / "summary_family_mean.csv"
DEFAULT_PLOTS_DIR = PROJECT_ROOT / "plots"
DEFAULT_SUMMARY_CSV = PROJECT_ROOT / "results" / "summary.csv"

# Primary metric per family (first match wins if multiple metrics exist).
PRIMARY_METRICS: dict[str, list[str]] = {
    "slds": ["slds_judge_deepseek_v4_pro", "slds_judge_deepseek_v3.2", "slds_judge"],
    "sdst": ["slt_judge_gpt-4o-mini", "slt_judge"],
    "slt": ["slt_judge_gpt-4o-mini", "slt_judge"],
    "sscprt": ["slt_judge_gpt-4o-mini", "slt_judge"],
    "lexam_oq": ["lexam_oq_judge_deepseek_r1", "lexam_oq_judge"],
    "lexam_mcq_4": ["acc", "trad_score"],
    "lexam_mcq_4_idk": ["acc", "trad_score"],
    "lexam_mcq_8": ["acc", "trad_score"],
    "lexam_mcq_8_idk": ["acc", "trad_score"],
    "lexam_mcq_16": ["acc", "trad_score"],
    "lexam_mcq_16_idk": ["acc", "trad_score"],
    "lexam_mcq_32": ["acc", "trad_score"],
    "lexam_mcq_32_idk": ["acc", "trad_score"],
}


def _pick_primary_metric(family: str, available: list[str]) -> str:
    if family in PRIMARY_METRICS:
        for name in PRIMARY_METRICS[family]:
            if name in available:
                return name
    if len(available) == 1:
        return available[0]
    raise ValueError(
        f"No primary metric for family={family}; available={available}"
    )


def _model_family_pivot(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Wide table: rows=models, columns=families, values=primary metric."""
    records: list[dict[str, str | float]] = []
    for (model, family), group in summary_df.groupby(["model", "family"]):
        metrics = group["metric"].unique().tolist()
        primary = _pick_primary_metric(family, metrics)
        row = group[group["metric"] == primary].iloc[0]
        records.append(
            {
                "model": model,
                "family": family,
                "metric": primary,
                "value": row["mean_value"],
            }
        )
    pivot = pd.DataFrame(records)
    wide = pivot.pivot(index="model", columns="family", values="value")
    wide.index.name = "model"
    return wide.sort_index()


def plot_family_bars(summary_df: pd.DataFrame, plots_dir: Path) -> None:
    """One grouped bar chart per family."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    for family, group in summary_df.groupby("family"):
        metrics = group["metric"].unique().tolist()
        primary = _pick_primary_metric(family, metrics)
        subset = group[group["metric"] == primary].sort_values("mean_value", ascending=True)
        fig = px.bar(
            subset,
            x="model",
            y="mean_value",
            title=f"{family} — {primary}",
            labels={"mean_value": primary, "model": "Model"},
        )
        out = plots_dir / f"{family}.html"
        fig.write_html(out)
        logger.info("Wrote plot %s", out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot Swiss legal eval summaries.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_FAMILY_SUMMARY,
        help="Family-mean summary CSV from aggregate.py",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=DEFAULT_PLOTS_DIR,
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=DEFAULT_SUMMARY_CSV,
        help="Model x family pivot table output",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    summary_df = pd.read_csv(args.input)
    wide = _model_family_pivot(summary_df)

    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(args.summary_csv)
    logger.info("Wrote model x family summary to %s", args.summary_csv)

    plot_family_bars(summary_df, args.plots_dir)


if __name__ == "__main__":
    main()
