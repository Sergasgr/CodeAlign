#!/usr/bin/env bash
# Phase 1 — Data curation.
# Runs inside the executor image: PMD, clippy, eslint, golangci-lint, dotnet and cpplint only exist
# there, and a missing linter binary raises FileNotFoundError and aborts the run on the host.
# Prerequisite: docker build -f docker/Dockerfile.executor -t codealign-executor:latest .
set -euo pipefail
cd "$(dirname "$0")/.."

docker run --rm \
  -u "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e DOTNET_NOLOGO=1 \
  -e DOTNET_CLI_TELEMETRY_OPTOUT=1 \
  -e UV_CACHE_DIR=/tmp \
  -v "$(pwd):/app" \
  -w /app \
  codealign-executor:latest \
  /opt/venv/bin/uv run python -m src.data_curation.main
