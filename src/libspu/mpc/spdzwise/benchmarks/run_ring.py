"""Benchmark RING SPDZWISE C1-C3, preserving raw data and source provenance.

Build //:ring_circuit_bench first, then:
  YACL_NUM_THREADS=12 python3 run_ring.py NEW_OUTPUT_DIRECTORY
An existing output directory containing only build.log is also accepted.
"""

import csv
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

BENCH = Path(__file__).resolve().parent
ROOT = BENCH.parents[4]
CIRCUITS = {"C1": (1048576, 1), "C2": (65536, 16), "C3": (16384, 64)}
PHASES = ("input_ms", "expand_ms", "circuit_ms", "verify_ms", "mul_ms")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: run_ring.py NEW_OUTPUT_DIRECTORY")
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(p.name != "build.log" for p in out.iterdir()):
        raise SystemExit("Refusing to overwrite an existing experiment")
    binary = BENCH / "bazel-bin/ring_circuit_bench"
    sources = sorted(p for p in BENCH.parent.glob("*.cc") if not p.name.endswith("_test.cc"))
    sources += sorted(BENCH.parent.glob("*.h"))
    sources += [BENCH.parent / "BUILD.bazel"]
    sources += [BENCH / n for n in ("ring_circuit_bench.cc", "run_ring.py", "BUILD.bazel", "MODULE.bazel", ".bazelrc")]
    sources += [ROOT / "src" / n for n in ("libspu/mpc/protocol_session.cc", "libspu/mpc/common/prg_state.cc", "libspu/mpc/utils/simulate.h", "MODULE.bazel", ".bazelrc")]
    parameters = {
        "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "protocol": "SPDZWISE + FM64 + RING", "internal_bits": 128,
        "output_bits": 64, "extension_bits": 64, "check_zero": "Protocol 9, BLAKE3-256",
        "circuits": CIRCUITS, "gates": 1048576, "secret_input_words": 2,
        "inputs": "x=2^60-8, y=uint64(-13)",
        "recurrence": "z[0,i]=x+i; z[l+1,i]=z[l,i]*(z[l,i]+y)",
        "warmups": 1, "repetitions": 7, "fresh_session_per_iteration": True,
        "checks_per_iteration": "one immediate input check, one final check of all 1048576 products",
        "transport": "FactoryMem: three parties in one process",
        "excluded": "client sharing, context/session/key/PRSS initialization, barriers, checked export/commit, reconstruction and plaintext oracle",
        "included": "input includes copy check, MAC generation and immediate verification; circuit includes secret additions, multiplications, PRSS masks, communication and saved verification records; verification includes copy checks, secret-coefficient aggregation, Protocol 9 and status propagation",
        "aggregation": "per iteration take the maximum over parties for each phase; core=circuit+verify, total=input+expand+core; report median/min/max over iterations 1..7 (exclude warmup 0)",
        "communication": "sent payload bytes per party; excludes transport headers and untimed setup/export",
        "YACL_NUM_THREADS": os.environ.get("YACL_NUM_THREADS"),
        "platform": platform.platform(), "logical_cpus": os.cpu_count(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)), "loadavg_at_start": os.getloadavg(),
        "compiler": subprocess.check_output(["c++", "--version"], text=True).splitlines()[0],
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "binary_sha256": digest(binary),
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sources},
    }
    (out / "parameters.json").write_text(json.dumps(parameters, indent=2) + "\n")
    (out / "lscpu.txt").write_text(subprocess.check_output(["lscpu"], text=True))
    patch = subprocess.check_output(["git", "diff", "--binary", "HEAD", "--"] + [str(p.relative_to(ROOT)) for p in sources], cwd=ROOT)
    for path in sources:
        relative = str(path.relative_to(ROOT))
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", relative], cwd=ROOT, capture_output=True)
        if tracked.returncode:
            diff = subprocess.run(["git", "diff", "--no-index", "--", "/dev/null", relative], cwd=ROOT, capture_output=True)
            if diff.returncode != 1:
                raise RuntimeError(diff.stderr.decode())
            patch += diff.stdout
    (out / "source.patch").write_bytes(patch)
    summaries = {}
    for name, (width, depth) in CIRCUITS.items():
        command = [str(binary), str(width), str(depth), "7"]
        start = time.monotonic()
        with (out / f"{name}.stdout").open("w") as stdout, (out / f"{name}.log").open("w") as stderr:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
            _, status, usage = os.wait4(process.pid, 0)
            process.returncode = os.waitstatus_to_exitcode(status)
        resource = {
            "command": command, "exit_code": process.returncode,
            "wall_seconds": time.monotonic() - start,
            "user_seconds": usage.ru_utime, "system_seconds": usage.ru_stime,
            "max_rss_kib": usage.ru_maxrss, "loadavg_at_end": os.getloadavg(),
        }
        (out / f"{name}-resources.json").write_text(json.dumps(resource, indent=2) + "\n")
        if process.returncode:
            raise SystemExit(f"{name} failed; see {name}.log")
        lines = [line for line in (out / f"{name}.stdout").read_text().splitlines(True)
                 if line.startswith(("protocol,", "spdzwise-ring128,"))]
        (out / f"{name}.csv").write_text("".join(lines))
        rows = list(csv.DictReader(lines))
        assert len(rows) == 24
        measured = []
        for iteration in range(8):
            batch = [r for r in rows if int(r["iteration"]) == iteration]
            assert len(batch) == 3 and {int(r["rank"]) for r in batch} == {0, 1, 2}
            for row in batch:
                for key, value in {"width": width, "depth": depth, "gates":1048576,
                                   "secret_inputs": 2, "checks": 2, "input_bytes":242,
                                   "expand_bytes":0, "circuit_bytes":33554432+4*depth,
                                   "verify_bytes":182}.items():
                    assert int(row[key]) == value, (name, key, row)
                assert float(row["mul_ms"]) <= float(row["circuit_ms"])
            maximum = {key: max(float(r[key]) for r in batch) for key in PHASES}
            maximum["core_ms"] = maximum["circuit_ms"] + maximum["verify_ms"]
            maximum["total_ms"] = maximum["input_ms"] + maximum["expand_ms"] + maximum["core_ms"]
            if iteration:
                measured.append({"iteration": iteration, **maximum})
        phases = {key: {"median_ms": statistics.median(r[key] for r in measured),
                        "min_ms": min(r[key] for r in measured),
                        "max_ms": max(r[key] for r in measured)}
                  for key in (*PHASES, "core_ms", "total_ms")}
        summaries[name] = {
            "width": width, "depth": depth, "iterations": measured, "phases": phases,
            "bytes_per_party": {key: int(rows[0][key]) for key in ("input_bytes", "expand_bytes", "circuit_bytes", "verify_bytes")},
            "validated_iterations": 8, "validated_outputs_per_iteration": width,
        }
        print(name, json.dumps(phases), flush=True)
    (out / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    # Detect accidental edits or replacement of the executable during the run.
    assert digest(binary) == parameters["binary_sha256"]
    assert all(digest(ROOT / name) == sha for name, sha in parameters["source_sha256"].items())
    (out / "SHA256SUMS").write_text("".join(f"{digest(p)}  {p.name}\n" for p in sorted(out.iterdir()) if p.is_file()))


if __name__ == "__main__":
    main()
