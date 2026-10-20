#!/usr/bin/env bash
set -euo pipefail
bench=$(cd "$(dirname "$0")/.." && pwd)
exec bash "$bench/run.sh" ring64 "${1:?provide a new run name; output goes to benchmarks/runs/}"
