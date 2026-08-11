#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python -m src.data_curation.main

#OPCION 2
#!/bin/bash
#set -e

#echo "Building executor image (needed for native linters)..."
#docker build -f docker/Dockerfile.executor -t codealign-executor:latest .

#echo "Running data curation inside the container..."
#docker run --rm -v "$(pwd):/app" -w /app codealign-executor:latest bash -c "/opt/venv/bin/pip install uv && /opt/venv/bin/uv run python -m src.data_curation.main"