#!/usr/bin/env python3
"""Compare 16k vs 32k cap experiment results for rambling models."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from lighteval.models.model_output import ModelResponse

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TASKS: list[tuple[str, str, str]] = [
    ("slds", "slds:de_de|0", "slds_judge_deepseek_v4_pro"),
    ("swiltrabench", "slt-paragraph_level:de-fr|0", "slt_judge_gpt-4o-mini"),
    ("lexam_mcq", "lexam_mcq_16_idk:en|0", "trad_score"),
]

RAMBLING_MODELS = [
    "minimax-m3",
    "kimi-k2.6",
    "glm-5.2",
]


def _hub_folder(model_name: str) -> Path | None:
    """Locate results json under results_cap_compare_{16,32}k/results/."""
    mapping = {
        "minimax-m3": "MiniMaxAI/MiniMax-M3",
        "kimi-k2.6": "moonshotai/Kimi-K2.6",
        "glm-5.2": "zai-org/GLM-5.2",
    }
    rel = mapping[model_name]
    return Path(rel)


def _detail_dir(result_root: Path, hub_rel: Path) -> Path | None:
    org, model = hub_rel.parts
    base = result_root / "details" / org / model
    if not base.exists():
        return None
    ts_dirs = sorted(base.iterdir())
    return ts_dirs[-1] if ts_dirs else None


def _output_stats(detail_dir: Path, task_name: str, cap_tokens: int) -> dict[str, float | int]:
    files = list(detail_dir.glob(f"details_{task_name}_*.parquet"))
    if not files:
        return {"n": 0, "avg_chars": 0.0, "max_chars": 0, "near_cap_pct": 0.0}

    cap_chars = cap_tokens * 4
    chars: list[int] = []
    near = 0
    for fp in files:
        df = pd.read_parquet(fp)
        for _, row in df.iterrows():
            text = (ModelResponse(**row["model_response"]).final_text[0] or "").strip()
            n = len(text)
            chars.append(n)
            if n >= cap_chars * 0.9:
                near += 1
    return {
        "n": len(chars),
        "avg_chars": sum(chars) / len(chars),
        "max_chars": max(chars),
        "near_cap_pct": 100 * near / len(chars),
    }


def main() -> None:
    rows: list[dict[str, object]] = []
    for cap_label, cap_tokens in (("16k", 16384), ("32k", 32768)):
        result_root = PROJECT_ROOT / f"results_cap_compare_{cap_label}"
        if not result_root.exists():
            continue
        for model_name in RAMBLING_MODELS:
            hub_rel = _hub_folder(model_name)
            results_path = result_root / "results" / hub_rel
            json_files = sorted(results_path.glob("results_*.json"))
            if not json_files:
                continue
            data = json.loads(json_files[-1].read_text())
            detail_dir = _detail_dir(result_root, hub_rel)
            if detail_dir is None:
                continue

            for family, task_name, score_key in TASKS:
                metrics = data["results"][task_name]
                stats = _output_stats(detail_dir, task_name, cap_tokens)
                rows.append(
                    {
                        "model": model_name,
                        "cap": cap_label,
                        "family": family,
                        "score": metrics[score_key],
                        "extract_fail": metrics["extract_fail"] if "extract_fail" in metrics else 0.0,
                        **stats,
                    }
                )

    if not rows:
        print("No cap-compare results found under results_cap_compare_{16,32}k/")
        return

    print(
        f"{'model':22} {'cap':>4} {'family':12} {'score':>8} {'xfail':>6} "
        f"{'avg_chr':>8} {'max_chr':>8} {'near%':>6}"
    )
    for row in sorted(rows, key=lambda r: (r["model"], r["family"], r["cap"])):
        print(
            f"{row['model']:22} {row['cap']:>4} {row['family']:12} "
            f"{row['score']:8.2f} {row['extract_fail']:6.2f} "
            f"{row['avg_chars']:8.0f} {row['max_chars']:8.0f} {row['near_cap_pct']:6.0f}"
        )

    print("\n=== Deltas (32k minus 16k) ===")
    by_key: dict[tuple[str, str], dict[str, dict[str, object]]] = {}
    for row in rows:
        by_key.setdefault((row["model"], row["family"]), {})[row["cap"]] = row
    for (model, family), caps in sorted(by_key.items()):
        if "16k" not in caps or "32k" not in caps:
            continue
        a, b = caps["16k"], caps["32k"]
        print(
            f"{model:22} {family:12} "
            f"score {float(b['score']) - float(a['score']):+6.2f}  "
            f"xfail {float(b['extract_fail']) - float(a['extract_fail']):+5.2f}  "
            f"avg_chars {int(b['avg_chars']) - int(a['avg_chars']):+6d}  "
            f"near% {float(b['near_cap_pct']) - float(a['near_cap_pct']):+5.0f}"
        )


if __name__ == "__main__":
    main()
