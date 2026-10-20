#!/usr/bin/env bash
set -euo pipefail
out=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$out/../.." && pwd)
cd "$repo/src"
"$repo/.local/bin/bazel-7.7.1" --batch --output_user_root=/tmp/malicious-spu-bazel \
  build -c opt //libspu/mpc/spdzwise:circuit_bench --lockfile_mode=off --jobs=16 \
  --extra_toolchains=@rules_foreign_cc//toolchains:preinstalled_cmake_toolchain \
  > "$out/build.log" 2>&1
./bazel-bin/libspu/mpc/spdzwise/circuit_bench 65536 16 7 ring64 \
  > "$out/w65536-d16.stdout" 2> "$out/w65536-d16.log"
python3 "$out/summarize.py" > "$out/summary.txt"
