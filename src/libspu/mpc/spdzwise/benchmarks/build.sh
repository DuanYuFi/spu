#!/usr/bin/env bash
set -euo pipefail
bench=$(cd "$(dirname "$0")" && pwd)
bazel_bin=${BAZEL:-bazel}
if [[ -z ${BAZEL:-} && -x "$bench/tools/bin/bazel-7.7.1" ]]; then
  bazel_bin="$bench/tools/bin/bazel-7.7.1"
fi
cd "$bench"
"$bazel_bin" --batch \
  --output_user_root="${BENCHMARK_BAZEL_ROOT:-/tmp/malicious-spu-bazel}" \
  build -c opt //:circuit_bench \
  --lockfile_mode=update --jobs="${BENCHMARK_JOBS:-4}" \
  --extra_toolchains=@rules_foreign_cc//toolchains:preinstalled_cmake_toolchain "$@"
