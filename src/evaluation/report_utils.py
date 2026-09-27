from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

TASK_SIZES = {"humaneval": 164, "humaneval-unstripped": 164, "humaneval-chat": 164, "humanevalplus": 164}

PROVENANCE_KEYS = (
    "tasks",
    "model",
    "peft_model",
    "load_in_4bit",
    "load_in_8bit",
    "precision",
    "do_sample",
    "repetition_penalty",
    "temperature",
    "top_p",
    "n_samples",
    "max_length_generation",
    "limit",
)

MUST_MATCH = (
    "tasks",
    "load_data_path", 
    "prefix",
    "load_in_4bit",
    "load_in_8bit",
    "precision",
    "do_sample",
    "repetition_penalty",
    "temperature",
    "top_p",
    "n_samples",
    "max_length_generation",
    "limit",
)

CASE_ORDER = ["A", "B", "B_tied", "C", "C_tied"]
ID_COLUMNS = ("problem_idx", "task_id", "problem_id", "problem", "doc_id", "idx", "index")

def load_json(path) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def pass_at_1(metrics: dict | None, task: str) -> float | None:
    if not metrics:
        return None
    value = metrics.get(task, {}).get("pass@1")
    return None if value is None else float(value)

def n_problems(metrics: dict | None, task: str) -> int | None:
    limit = (metrics or {}).get("config", {}).get("limit")
    if limit:
        return int(limit)
    return TASK_SIZES.get(task)

def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return float("nan"), float("nan")
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)

def pass_at_1_table(metrics_by_model: dict[str, dict | None], task: str) -> pd.DataFrame:
    rows = []
    for name, m in metrics_by_model.items():
        p, n = pass_at_1(m, task), n_problems(m, task)
        if p is None:
            rows.append({"Model": name, "pass@1 (%)": np.nan, "CI low": np.nan, "CI high": np.nan,
                         "n problems": n, "status": "pending"})
            continue
        lo, hi = wilson_ci(p, n) if n else (np.nan, np.nan)
        rows.append({"Model": name, "pass@1 (%)": 100 * p, "CI low": 100 * lo, "CI high": 100 * hi,
                     "n problems": n, "status": "ok"})
    return pd.DataFrame(rows).set_index("Model")

def provenance_table(metrics_by_model: dict[str, dict | None]) -> pd.DataFrame:
    rows = []
    for name, m in metrics_by_model.items():
        cfg = (m or {}).get("config", {})
        rows.append({"Model": name, **{k: (cfg.get(k, "—") if m else "pending") for k in PROVENANCE_KEYS}})
    return pd.DataFrame(rows).set_index("Model")

def provenance_warnings(metrics_by_model: dict[str, dict | None], task: str) -> list[str]:
    present = {n: m.get("config", {}) for n, m in metrics_by_model.items() if m}
    warnings = []
    for key in MUST_MATCH:
        values = {n: cfg.get(key) for n, cfg in present.items()}
        if len({json.dumps(v, default=str) for v in values.values()}) > 1:
            warnings.append(f"'{key}' differs across models: {values}")
    for n, cfg in present.items():
        if cfg.get("limit"):
            warnings.append(f"{n}: evaluated with --limit {cfg['limit']} (smoke test), not the full {task} set")
    return warnings

def plot_pass_at_1(ax, table: pd.DataFrame, colors: dict[str, str], title: str) -> None:
    done = table[table["status"] == "ok"]
    if done.empty:
        ax.text(0.5, 0.5, "No metrics yet", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    names = list(done.index)
    scores = done["pass@1 (%)"].to_numpy()
    yerr = np.vstack([scores - done["CI low"].to_numpy(), done["CI high"].to_numpy() - scores])
    bars = ax.bar(names, scores, color=[colors.get(n, "#999999") for n in names], width=0.55,
                  yerr=yerr, capsize=6, ecolor="#333333")
    for bar, score, hi in zip(bars, scores, done["CI high"].to_numpy(), strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, hi + 1.5, f"{score:.1f}%", ha="center", fontweight="bold")
    ax.set_ylabel("pass@1 (%) — error bars: 95% Wilson CI")
    ax.set_ylim(0, 105)
    ax.set_title(title)

def find_trainer_state(run_dir) -> Path | None:
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return None
    best, best_step = None, -1
    for path in run_dir.glob("**/trainer_state.json"):
        try:
            step = json.loads(path.read_text(encoding="utf-8")).get("global_step", -1)
        except (OSError, json.JSONDecodeError):
            continue
        if step > best_step:
            best, best_step = path, step
    return best

def load_log_history(path) -> tuple[pd.DataFrame, dict]:
    state = json.loads(Path(path).read_text(encoding="utf-8"))
    history = pd.DataFrame(state.get("log_history", []))
    return history, state

def smooth(series: pd.Series, window: int = 10) -> pd.Series:
    return series.rolling(window, min_periods=1).mean()

def count_lines(path) -> int:
    with open(path, "rb") as f:
        return sum(1 for _ in f)

def summarize_preference_reports(pref_dir="data/preferences") -> pd.DataFrame:
    rows = []
    for path in sorted(Path(pref_dir).glob("*report*.jsonl")):
        cases, total, kept = Counter(), 0, 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                total += 1
                cases[record.get("case", "?")] += 1
                kept += not record.get("discarded", False)
        row = {"report file": path.name, "prompts": total, "pairs kept": kept}
        row.update({f"case {c}": cases.get(c, 0) for c in CASE_ORDER})
        rows.append(row)
    return pd.DataFrame(rows)

def list_preference_datasets(pref_dir="data/preferences") -> pd.DataFrame:
    rows = [{"dataset file": p.name, "pairs": count_lines(p)}
            for p in sorted(Path(pref_dir).glob("*.jsonl")) if "report" not in p.name]
    return pd.DataFrame(rows)

def bootstrap_mean_diff(df: pd.DataFrame, model_a: str, model_b: str, metric: str, n_boot: int = 5000, seed: int = 0) -> dict | None:
    rng = np.random.default_rng(seed)
    sub = df[df["model"].isin([model_a, model_b])].dropna(subset=[metric])
    id_col = next((c for c in ID_COLUMNS if c in sub.columns), None)
    if id_col is not None:
        wide = sub.groupby([id_col, "model"])[metric].mean().unstack("model")
        if model_a in wide.columns and model_b in wide.columns:
            d = (wide[model_a] - wide[model_b]).dropna().to_numpy(dtype=float)
            if len(d) > 1:
                boots = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
                lo, hi = np.percentile(boots, [2.5, 97.5])
                return {"diff": d.mean(), "lo": lo, "hi": hi, "n": len(d), "paired_on": id_col}
    a = sub.loc[sub["model"] == model_a, metric].to_numpy(dtype=float)
    b = sub.loc[sub["model"] == model_b, metric].to_numpy(dtype=float)
    if len(a) < 2 or len(b) < 2:
        return None
    boots = (rng.choice(a, size=(n_boot, len(a))).mean(axis=1)
             - rng.choice(b, size=(n_boot, len(b))).mean(axis=1))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"diff": a.mean() - b.mean(), "lo": lo, "hi": hi, "n": min(len(a), len(b)), "paired_on": None}

def describe_diff(result: dict | None, lower_is_better: bool = True) -> str:
    if result is None:
        return "not enough data"
    ci = f"{result['diff']:+.2f} [95% CI {result['lo']:+.2f}, {result['hi']:+.2f}]"
    if result["lo"] <= 0 <= result["hi"]:
        return f"{ci} → CI includes 0: no detectable difference at n={result['n']}"
    better = (result["diff"] < 0) == lower_is_better
    return f"{ci} → {'lower' if result['diff'] < 0 else 'higher'} ({'better' if better else 'worse'})"