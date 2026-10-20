#!/usr/bin/env bash
set -euo pipefail
bench=$(cd "$(dirname "$0")" && pwd)
mode=${1:?usage: run.sh f61|ring64 new-run-name}
run_name=${2:?usage: run.sh f61|ring64 new-run-name}
[[ $# == 2 && "$run_name" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]] || {
  echo 'Provide a new run name using letters, digits, dots, underscores or hyphens.' >&2
  exit 1
}
case "$mode" in
  f61)
    shapes=(65536:16 1048576:1 16384:64)
    summarizer="$bench/macbook-air-f61-2026-10-19/summarize.py"
    ;;
  ring64)
    shapes=(65536:16)
    summarizer="$bench/ubuntu-aby3-ring64-2026-10-20/summarize.py"
    ;;
  *) echo 'Mode must be f61 or ring64.' >&2; exit 1 ;;
esac
mkdir -p "$bench/runs"
out="$bench/runs/$run_name"
mkdir "$out"
bash "$bench/build.sh" > "$out/build.log" 2>&1
python3 - "$mode" "$out" "$bench" <<'PY'
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

mode, out, bench = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
metadata = {
    "date": datetime.datetime.now().astimezone().isoformat(),
    "platform": platform.platform(),
    "logical_cpus": os.cpu_count(),
    "YACL_NUM_THREADS": os.environ.get("YACL_NUM_THREADS"),
    "compiler": subprocess.check_output([os.environ.get("CXX", "c++"), "--version"], text=True).splitlines()[0],
    "mode": mode,
    "protocols": ["aby3-f61-adapter", "spdzwise-f61"] if mode == "f61" else ["aby3-ring64"],
    "domain": "F_p, p=2^61-1" if mode == "f61" else "Z/(2^64)",
    "storage": "FM64",
    "build": "-c opt; requested Bazel version in .bazelversion; see build.log for resolved toolchains",
    "transport": "FactoryMem, three parties in one process",
    "warmups": 1,
    "repetitions": 7,
    "shapes": [[65536, 16], [1048576, 1], [16384, 64]] if mode == "f61" else [[65536, 16]],
    "secret_input_elements": 2,
    "multiplication_gates_per_shape": 1048576,
    "inputs_signed": [2**60 - 8, -13],
    "binary_sha256": hashlib.sha256((bench / "bazel-bin/circuit_bench").read_bytes()).hexdigest(),
    "source_sha256": {name: hashlib.sha256((bench / name).read_bytes()).hexdigest()
                      for name in ("circuit_bench.cc", "BUILD.bazel", "MODULE.bazel", ".bazelrc", "build.sh", "run.sh")},
}
(out / "parameters.json").write_text(json.dumps(metadata, indent=2) + "\n")
PY
for shape in "${shapes[@]}"; do
  width=${shape%:*}
  depth=${shape#*:}
  args=("$width" "$depth" 7)
  if [[ "$mode" == ring64 ]]; then
    args+=(ring64)
  fi
  "$bench/bazel-bin/circuit_bench" "${args[@]}" \
    > "$out/w$width-d$depth.stdout" 2> "$out/w$width-d$depth.log"
done
cp "$summarizer" "$out/summarize.py"
python3 "$out/summarize.py" > "$out/summary.txt"
echo "Results: $out"
