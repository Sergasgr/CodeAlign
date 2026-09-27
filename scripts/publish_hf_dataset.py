import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_curation.curation_config import ALLOWED_LICENSES, DS_PATH, REPORT_PATH  
from src.data_curation.pii import find_secrets, has_real_email, redact_emails 

def content_key(language: str, code: str) -> str:
    return hashlib.sha1(f"{language}\0{code}".encode("utf-8")).hexdigest()

def read_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def upstream_provenance(needed: set[str]) -> dict[str, dict]:
    from src.data_curation.dataset import load_languages

    ds = load_languages().filter(lambda row: (row["license"] or "").lower() in ALLOWED_LICENSES)
    ds = ds.select_columns(["target_language", "new_contents", "commit", "repos", "new_file"])
    found: dict[str, dict] = {}
    for row in ds:
        key = content_key(row["target_language"], row["new_contents"])
        if key in needed and key not in found:  # identical contents in several commits: keep the first
            found[key] = {"commit": row["commit"], "repos": row["repos"], "new_file": row["new_file"]}
    return found

def to_row(record: dict, provenance: dict[str, dict], annotated: bool) -> dict:
    meta = record["metadata"]
    key = content_key(meta["language"], record["messages"][1]["content"])
    prov = provenance.get(key, {})
    row = {
        "messages": [{"role": m["role"], "content": redact_emails(m["content"])} for m in record["messages"]],
        "language": meta["language"],
        "prompt_type": meta["prompt_type"],
        "lint_errors": meta["lint_errors"],
        "cyclomatic_complexity": meta["cyclomatic_complexity"],
        "license": meta["license"],
        "commit": meta.get("commit") or prov.get("commit"),
        "repos": meta.get("repos") or prov.get("repos"),
        "new_file": meta.get("new_file") or prov.get("new_file"),
    }
    if annotated:
        row.update({"status": meta["status"], "error": meta["error"], "is_valid_syntax": meta["is_valid_syntax"]})
    return row

def build(path: Path, provenance: dict[str, dict], annotated: bool, stats: Counter) -> list[dict]:
    rows = []
    for record in read_jsonl(path):
        text = record["messages"][0]["content"] + "\n" + record["messages"][1]["content"]
        if find_secrets(text):
            stats[f"{'annotated' if annotated else 'sft'}: dropped (secret)"] += 1
            continue
        if has_real_email(text):
            stats[f"{'annotated' if annotated else 'sft'}: e-mail redacted"] += 1
        rows.append(to_row(record, provenance, annotated))
    return rows

def card_body(sft: list[dict], annotated: list[dict], stats: Counter, github_url: str) -> str:
    langs = Counter(r["language"] for r in sft)
    licenses = Counter(r["license"] for r in sft)
    edit_share = sum(r["prompt_type"] == "edit" for r in sft) / len(sft)
    with_prov = sum(bool(r["commit"]) for r in sft) / len(sft)
    lang_table = "\n".join(f"| {k} | {v:,} |" for k, v in langs.most_common())
    lic_table = "\n".join(f"| {k} | {v:,} |" for k, v in licenses.most_common())
    return f"""
# CodeAlign — curated CommitPackFT (8 languages)

Instruction/code pairs from [`bigcode/commitpackft`](https://huggingface.co/datasets/bigcode/commitpackft), filtered for syntax validity (tree-sitter), per-language lint errors, cyclomatic complexity, internal duplication and cross-sample near-duplicates (MinHash/LSH). Built as the SFT set of [CodeAlign]({github_url}); the pipeline, thresholds and the full curation report live in that repository.

| config | rows | contents |
|---|---|---|
| `sft` (default) | {len(sft):,} | accepted samples — the SFT training set minus the rows dropped for secrets (see Privacy) |
| `annotated` | {len(annotated):,} | every processed sample with its curation verdict (`status`, `error`) |

```python
from datasets import load_dataset
sft = load_dataset("<repo-id>", "sft", split="train")
```

## Format

`messages` is ChatML (`user` prompt, `assistant` target file). Two prompt types, derived from the commit itself: `new_file` (write-from-spec; commit created the file) and `edit` (existing file + commit message as instruction). {edit_share:.1%} of `sft` rows are `edit`.

## Languages (`sft`)

| language | rows |
|---|---|
{lang_table}

## Licensing and provenance

Only samples whose upstream `license` is one of {", ".join(sorted(ALLOWED_LICENSES))} are included; the licence of every sample is in its `license` column and applies to that sample's code.

| license | `sft` rows |
|---|---|
{lic_table}

As in CommitPackFT, each row keeps its source `commit` and `repos` so the copyright holder can be identified ({with_prov:.1%} of `sft` rows carry provenance). Code authors who want their code removed can open an issue on the GitHub repository.

## Known issues in the quality columns

`lint_errors` holds the values computed by the v1.0 curation run, which had linter bugs (the data was not re-curated):
- **C++, JavaScript, TypeScript:** `lint_errors` is always 0, so these languages were filtered on syntax, complexity and duplication only. cpplint's total was parsed from the wrong output stream (fixed in the repository afterwards); ESLint ≥ 9 rejects the `--no-eslintrc` command line (still open).
- **Python:** every value includes ruff's summary line (+1, or +2 when ruff also printed a fix hint), so the actual number of violations is 1–2 lower (fixed in the repository afterwards).

## Privacy

- Rows containing a high-confidence secret (private key blocks, AWS / GitHub / Slack / Google / Stripe credentials) were dropped: {stats.get("sft: dropped (secret)", 0):,} from `sft`, {stats.get("annotated: dropped (secret)", 0):,} from `annotated`.
- E-mail addresses (other than documentation and GitHub no-reply domains) were replaced with `<EMAIL>` in {stats.get("sft: e-mail redacted", 0):,} `sft` rows.
- Detection is pattern-based and will miss some personal data; do not use this dataset to identify individuals.

## Decontamination

13-gram overlap of every HumanEval and MBPP problem (prompt + canonical solution) against the Python samples is reported in `src/notebooks/01_curation_report.ipynb` of the GitHub repository.

## Citation

Upstream data: Muennighoff et al., *OctoPack: Instruction Tuning Code Large Language Models* (2023), arXiv:2308.07124.
"""

def apply_card(card, body: str, repo_id: str):
    card.data.license = "other"
    card.data.license_name = "per-sample-permissive" 
    card.data.language = ["code", "en"]
    card.data.pretty_name = "CodeAlign curated CommitPackFT"
    card.data.source_datasets = ["bigcode/commitpackft"]
    card.data.task_categories = ["text-generation"]
    card.data.tags = ["code", "sft", "commitpackft", "code-quality"]
    card.text = body.replace("<repo-id>", repo_id)
    return card

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", required=True, help="<user>/<dataset-name> on the Hub")
    parser.add_argument("--github-url", required=True, help="URL of the CodeAlign GitHub repo (linked from the card)")
    parser.add_argument("--dry-run", action="store_true", help="write parquet + README.md locally, upload nothing")
    parser.add_argument("--out-dir", type=Path, default=Path("data/hf_release"))
    parser.add_argument("--public", action="store_true", help="push as public (default: private)")
    parser.add_argument("--skip-provenance", action="store_true", help="don't re-join commit/repos from CommitPackFT")
    args = parser.parse_args()

    from datasets import Dataset
    from huggingface_hub import DatasetCard

    provenance: dict[str, dict] = {}
    if not args.skip_provenance:
        needed = {content_key(r["metadata"]["language"], r["messages"][1]["content"])
                  for r in read_jsonl(REPORT_PATH) if not r["metadata"].get("commit")}
        if needed:
            print(f"Re-joining provenance for {len(needed):,} samples from CommitPackFT...")
            provenance = upstream_provenance(needed)
            print(f"  matched {len(provenance):,} / {len(needed):,}")

    stats: Counter = Counter()
    sft = build(DS_PATH, provenance, annotated=False, stats=stats)
    annotated = build(REPORT_PATH, provenance, annotated=True, stats=stats)
    for k, v in sorted(stats.items()):
        print(f"{k}: {v:,}")

    body = card_body(sft, annotated, stats, args.github_url)
    if args.dry_run:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        Dataset.from_list(sft).to_parquet(args.out_dir / "sft.parquet")
        Dataset.from_list(annotated).to_parquet(args.out_dir / "annotated.parquet")
        card = apply_card(DatasetCard("---\nlicense: other\n---\n"), body, args.repo_id)
        (args.out_dir / "README.md").write_text(str(card), encoding="utf-8")
        print(f"Dry run: wrote {args.out_dir}/sft.parquet, annotated.parquet, README.md — nothing uploaded.")
        return

    private = not args.public
    Dataset.from_list(sft).push_to_hub(args.repo_id, config_name="sft", set_default=True, split="train", private=private)
    Dataset.from_list(annotated).push_to_hub(args.repo_id, config_name="annotated", split="train", private=private)
    card = apply_card(DatasetCard.load(args.repo_id, repo_type="dataset"), body, args.repo_id)  # keeps the configs YAML
    card.push_to_hub(args.repo_id, repo_type="dataset")
    print(f"Pushed https://huggingface.co/datasets/{args.repo_id} ({'private' if private else 'public'})")

if __name__ == "__main__":
    main()
