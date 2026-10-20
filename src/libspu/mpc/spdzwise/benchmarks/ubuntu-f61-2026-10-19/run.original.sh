#!/usr/bin/env bash
set -euo pipefail
out=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$out/../.." && pwd)
cd "$repo/src"
bazel_spu() {
  "$repo/.local/bin/bazel-7.7.1" --batch --output_user_root=/tmp/malicious-spu-bazel "$@" \
    --lockfile_mode=off --jobs=16 \
    --extra_toolchains=@rules_foreign_cc//toolchains:preinstalled_cmake_toolchain
}
bazel_spu build -c opt //libspu/mpc/spdzwise:circuit_bench > "$out/build.log" 2>&1
bazel_spu test -c opt //libspu/mpc/spdzwise:protocol_test \
  //libspu/mpc/spdzwise:io_test //libspu/mpc:protocol_session_test \
  //libspu/kernel/hal:ring_test > "$out/native-tests.log" 2>&1
python3 libspu/mpc/spdzwise/field_reference_test.py > "$out/field-test.log" 2>&1
python3 libspu/mpc/spdzwise/reference_test.py > "$out/reference-test.log" 2>&1
for shape in 65536:16 1048576:1 16384:64; do
  width=${shape%:*}
  depth=${shape#*:}
  ./bazel-bin/libspu/mpc/spdzwise/circuit_bench "$width" "$depth" 7 \
    > "$out/w$width-d$depth.stdout" 2> "$out/w$width-d$depth.log"
done
python3 "$out/summarize.py" > "$out/summary.txt"
