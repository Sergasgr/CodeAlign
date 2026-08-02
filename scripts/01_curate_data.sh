#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python -m src.data_curation.main