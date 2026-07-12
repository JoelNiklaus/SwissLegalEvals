"""Render the bar charts embedded in the Swiss legal evals blog post.

Reads the full-run summary produced by ``swiss-legal-evals-plot``
(``results/summary.csv``, mirrored in the public HF bucket) and writes PNGs to
``blog/figures/``. Charts use one brand colour per model and place each model's
Hugging Face org avatar (cached in ``blog/logos/``) under its bar, in the style
of the Artificial Analysis index. Composite scores follow the definitions in
the bucket's ``REPORT.md``.

Usage::

    hf buckets sync hf://buckets/joelniklaus/SwissLegalEvals/summary.csv results
    uv run python blog/make_figures.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from PIL import Image

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_CSV = PROJECT_ROOT / "results" / "summary.csv"
SUMMARY_LONG_CSV = PROJECT_ROOT / "results" / "summary_long.csv"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
LOGOS_DIR = Path(__file__).resolve().parent / "logos"

# Judge metric per family, used to read the per-task language scores.
LANG_JUDGE = {
    "slds": "slds_judge_deepseek_v4_pro",
    "sdst": "slt_judge_gpt-4o-mini",
    "slt": "slt_judge_gpt-4o-mini",
    "sscprt": "slt_judge_gpt-4o-mini",
}
LANG_LABELS = {"de": "German", "fr": "French", "it": "Italian"}

# MCQ columns hold ``trad_score`` on a 0-1 scale; everything else is 0-100.
MCQ_COLUMNS = ["lexam_mcq_4_idk", "lexam_mcq_8_idk", "lexam_mcq_16_idk"]
TRANSLATION_COLUMNS = ["sdst", "slt", "sscprt"]

# Map the raw hub names in summary.csv to clean display labels for the charts.
DISPLAY_NAMES = {
    "DeepSeek-V4-Pro": "DeepSeek V4 Pro",
    "DeepSeek-V4-Flash": "DeepSeek V4 Flash",
    "GLM-5.2": "GLM 5.2",
    "Hy-MT2-30B-A3B": "Hunyuan MT2 30B",
    "Kimi-K2.6": "Kimi K2.6",
    "Llama-3.3-70B-Instruct": "Llama 3.3 70B",
    "LFM2.5-8B-A1B": "LFM2.5 8B",
    "MiniMax-M3": "MiniMax M3",
    "NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4": "Nemotron 3 Ultra 550B",
    "Olmo-3.1-32B-Think": "OLMo 3.1 32B Think",
    "Qwen3.5-35B-A3B": "Qwen3.5 35B",
    "gemma-4-31B-it": "Gemma 4 31B",
    "gpt-oss-120b": "gpt-oss 120B",
}

# Display name -> Hugging Face org that publishes it (source of the logo).
ORG_BY_MODEL = {
    "DeepSeek V4 Pro": "deepseek-ai",
    "DeepSeek V4 Flash": "deepseek-ai",
    "GLM 5.2": "zai-org",
    "Hunyuan MT2 30B": "tencent",
    "Kimi K2.6": "moonshotai",
    "Llama 3.3 70B": "meta-llama",
    "LFM2.5 8B": "LiquidAI",
    "MiniMax M3": "MiniMaxAI",
    "Nemotron 3 Ultra 550B": "nvidia",
    "OLMo 3.1 32B Think": "allenai",
    "Qwen3.5 35B": "Qwen",
    "Gemma 4 31B": "google",
    "gpt-oss 120B": "openai",
}

# One distinct, roughly brand-aligned colour per model.
BRAND_COLORS = {
    "Nemotron 3 Ultra 550B": "#76B900",
    "Kimi K2.6": "#6D5AE6",
    "DeepSeek V4 Pro": "#4D6BFE",
    "DeepSeek V4 Flash": "#8AA0FF",
    "MiniMax M3": "#E1275C",
    "GLM 5.2": "#12A594",
    "Llama 3.3 70B": "#0467DF",
    "Gemma 4 31B": "#4285F4",
    "gpt-oss 120B": "#202124",
    "Qwen3.5 35B": "#B03CE6",
    "OLMo 3.1 32B Think": "#F0529C",
    "LFM2.5 8B": "#F59E0B",
    "Hunyuan MT2 30B": "#0052D9",
}

GROUP_PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]

WIDTH, HEIGHT, SCALE = 1120, 680, 2


def load_scores() -> pd.DataFrame:
    """Load per-model family scores and derive the composite columns."""
    df = pd.read_csv(SUMMARY_CSV)
    df["model"] = df["model"].replace(DISPLAY_NAMES)

    df["translation"] = df[TRANSLATION_COLUMNS].mean(axis=1)
    # Rescale MCQ accuracy from 0-1 to the 0-100 range of the judge metrics.
    df["mcq"] = df[MCQ_COLUMNS].mean(axis=1) * 100
    # skipna=False keeps the composite blank for translation-only models
    # (Hy-MT2-30B) so they never appear in the overall ranking.
    df["overall"] = df[["slds", "lexam_oq", "translation", "mcq"]].mean(axis=1, skipna=False)
    return df


def _logo(model: str) -> Image.Image:
    """Return the cached HF org avatar for a model, downloading it once."""
    org = ORG_BY_MODEL[model]
    path = LOGOS_DIR / f"{org}.png"
    if not path.exists():
        LOGOS_DIR.mkdir(parents=True, exist_ok=True)
        overview = requests.get(f"https://huggingface.co/api/organizations/{org}/overview", timeout=30)
        overview.raise_for_status()
        avatar_url = overview.json()["avatarUrl"]
        raw = requests.get(avatar_url, timeout=30)
        raw.raise_for_status()
        path.write_bytes(raw.content)
        logger.info("Cached logo for %s", org)
    return Image.open(path).convert("RGBA")


def _add_logos_and_names(fig: go.Figure, models: list[str]) -> None:
    """Place each model's logo and rotated name below its bar (paper coords)."""
    for model in models:
        fig.add_layout_image(
            source=_logo(model),
            xref="x",
            yref="paper",
            x=model,
            y=-0.02,
            sizex=0.9,
            sizey=0.13,
            xanchor="center",
            yanchor="top",
            sizing="contain",
            layer="above",
        )
        fig.add_annotation(
            x=model,
            xref="x",
            y=-0.17,
            yref="paper",
            text=model,
            showarrow=False,
            textangle=-30,
            xanchor="right",
            yanchor="top",
            font=dict(size=12, color="#374151"),
        )


def _style(fig: go.Figure, title: str, y_title: str) -> None:
    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, x=0.5, font=dict(size=20, color="#111827")),
        margin=dict(l=50, r=25, t=80, b=210),
        yaxis=dict(title=y_title, gridcolor="#E5E7EB", zeroline=False),
        xaxis=dict(title="", showticklabels=False, showgrid=False, showline=False, ticks=""),
        plot_bgcolor="white",
    )


def _save(fig: go.Figure, name: str) -> None:
    out = FIGURES_DIR / name
    fig.write_image(out, width=WIDTH, height=HEIGHT, scale=SCALE)
    logger.info("Wrote %s", out)


def _ranked_bar(df: pd.DataFrame, column: str, title: str, y_title: str, out: str) -> None:
    """Vertical bar chart of one score column, one brand colour per model."""
    ranked = df.dropna(subset=[column]).sort_values(column, ascending=False)
    models = ranked["model"].tolist()
    fig = px.bar(
        ranked,
        x="model",
        y=column,
        color="model",
        color_discrete_map=BRAND_COLORS,
        category_orders={"model": models},
        text=ranked[column].map(lambda value: f"{value:.1f}"),
    )
    fig.update_traces(
        marker_cornerradius=6,
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(color="white", size=15, family="Arial Black"),
        showlegend=False,
    )
    _style(fig, title, y_title)
    _add_logos_and_names(fig, models)
    _save(fig, out)


def plot_overall(df: pd.DataFrame) -> None:
    """Composite ranking over the four task groups (comparable models only)."""
    _ranked_bar(
        df, "overall", "Overall score on Swiss legal tasks (composite, 0-100)",
        "Composite score", "overall_ranking.png",
    )


def plot_translation(df: pd.DataFrame) -> None:
    """SwiLTra-Bench translation average across all 12 models."""
    _ranked_bar(
        df, "translation", "Legal translation quality (SwiLTra-Bench, judge score 0-100)",
        "Translation average", "translation.png",
    )


def _grouped_bar(
    long: pd.DataFrame, models: list[str], series_order: list[str], title: str, y_title: str, out: str
) -> None:
    """Grouped vertical bars (one bar group per model), logos under each group."""
    fig = px.bar(
        long,
        x="model",
        y="score",
        color="series",
        barmode="group",
        color_discrete_sequence=GROUP_PALETTE,
        category_orders={"model": models, "series": series_order},
    )
    fig.update_traces(marker_cornerradius=4)
    _style(fig, title, y_title)
    fig.update_layout(
        legend=dict(title="", orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    _add_logos_and_names(fig, models)
    _save(fig, out)


def plot_task_group_profile(df: pd.DataFrame) -> None:
    """Per-group scores for the seven strongest models to show specialization."""
    top = df.dropna(subset=["overall"]).nlargest(7, "overall")["model"].tolist()
    groups = {
        "slds": "SLDS",
        "translation": "SwiLTra-Bench",
        "lexam_oq": "LEXam OQ",
        "mcq": "LEXam MCQ",
    }
    long = (
        df[df["model"].isin(top)]
        .melt(id_vars="model", value_vars=list(groups), var_name="series", value_name="score")
        .assign(series=lambda d: d["series"].map(groups))
    )
    _grouped_bar(
        long, top, list(groups.values()),
        "Where the frontier models differ (top 7 by composite)", "Score (0-100)",
        "task_group_profile.png",
    )


def plot_mcq_scaling(df: pd.DataFrame) -> None:
    """MCQ accuracy as the number of answer options grows from 4 to 16."""
    labels = {"lexam_mcq_4_idk": "4 options", "lexam_mcq_8_idk": "8 options", "lexam_mcq_16_idk": "16 options"}
    mcq = df.dropna(subset=MCQ_COLUMNS).sort_values("lexam_mcq_4_idk", ascending=False)
    models = mcq["model"].tolist()
    long = (
        mcq.melt(id_vars="model", value_vars=MCQ_COLUMNS, var_name="series", value_name="acc")
        .assign(score=lambda d: d["acc"] * 100, series=lambda d: d["series"].map(labels))
    )
    _grouped_bar(
        long, models, list(labels.values()),
        "MCQ accuracy drops as answer options increase (LEXam, trad_score)", "Accuracy (%)",
        "mcq_scaling.png",
    )


def plot_translation_subsets(df: pd.DataFrame) -> None:
    """Judge scores on the three SwiLTra subsets across all 12 models."""
    labels = {"sdst": "Decision summaries", "slt": "Laws", "sscprt": "Press releases"}
    tx = df.dropna(subset=TRANSLATION_COLUMNS).sort_values("translation", ascending=False)
    models = tx["model"].tolist()
    long = (
        tx.melt(id_vars="model", value_vars=list(labels), var_name="series", value_name="score")
        .assign(series=lambda d: d["series"].map(labels))
    )
    _grouped_bar(
        long, models, list(labels.values()),
        "Translation quality by subset (SwiLTra-Bench, judge score 0-100)", "Judge score (0-100)",
        "translation_subsets.png",
    )


def load_language_scores() -> pd.DataFrame:
    """Per-model mean judge score by output language (German, French, Italian).

    Averaged over SLDS and the three SwiLTra translation subsets, the benchmarks
    that cover all three languages (LEXam is en/de only and is excluded). Each
    task is attributed to the language the model must produce: the SLDS headnote
    language and the translation target language. Translation-only models (no
    SLDS) are dropped so every language rests on the same benchmark mix.
    """
    df = pd.read_csv(SUMMARY_LONG_CSV)
    df = df[df["family"].isin(LANG_JUDGE)]
    df = df[df["metric"] == df["family"].map(LANG_JUDGE)].copy()

    def output_lang(row: pd.Series) -> str:
        code = row["task"].split(":")[1].split("|")[0]
        return code.split("_")[1] if row["family"] == "slds" else code.split("-")[1]

    df["lang"] = df.apply(output_lang, axis=1)
    df = df[df["lang"].isin(LANG_LABELS)]
    df["group"] = df["family"].where(df["family"] == "slds", "translation")

    # Mean per group, then mean of the two groups so SLDS and translation weigh equally.
    group_mean = df.groupby(["model", "lang", "group"])["value"].mean().reset_index()
    lang_mean = group_mean.groupby(["model", "lang"])["value"].mean().reset_index()
    has_slds = group_mean.loc[group_mean["group"] == "slds", "model"].unique()
    lang_mean = lang_mean[lang_mean["model"].isin(has_slds)]
    lang_mean["model"] = lang_mean["model"].replace(DISPLAY_NAMES)
    return lang_mean.rename(columns={"value": "score"})


def plot_language_comparison() -> None:
    """Mean judge score by Swiss language across SLDS and translation."""
    lang = load_language_scores()
    lang["series"] = lang["lang"].map(LANG_LABELS)
    order = lang.groupby("model")["score"].mean().sort_values(ascending=False).index.tolist()
    _grouped_bar(
        lang, order, list(LANG_LABELS.values()),
        "Quality by Swiss language (SLDS + translation, judge score 0-100)", "Judge score (0-100)",
        "language_comparison.png",
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_scores()
    plot_overall(df)
    plot_translation(df)
    plot_translation_subsets(df)
    plot_task_group_profile(df)
    plot_mcq_scaling(df)
    plot_language_comparison()


if __name__ == "__main__":
    main()
