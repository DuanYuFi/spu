"""Run the server suite with exactly 64 YACL background workers.

Build circuit_bench, ring_circuit_bench and binary_bench first. Then run
  python3 run_server.py NEW_OUTPUT_DIRECTORY [--ring-only]
YACL_NUM_THREADS includes the caller, so 65 creates 64 pool workers.
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

BENCH = Path(__file__).resolve().parent
ROOT = BENCH.parents[4]
CIRCUITS = {"C1": (1048576, 1), "C2": (65536, 16), "C3": (16384, 64)}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rows, protocol, circuit, normal, binary):
    rows = [r for r in rows if r["protocol"] == protocol]
    iterations = range(8) if normal or binary else range(7)
    assert len(rows) == len(iterations) * 3
    width, depth = CIRCUITS[circuit]
    wise = protocol.startswith("spdzwise")
    ring = protocol == "spdzwise-ring128"
    if binary:
        expected_bytes = {"online_bytes": 8388608,
                          "verify_bytes": 218110266 if wise else 0}
        phase_map = {"circuit_ms": "online_ms", "verify_ms": "verify_ms"}
    else:
        expected_bytes = {
            "input_bytes" if normal else "input_mac_bytes": (242 if ring else 96) if normal else 0,
            "expand_bytes": 0,
            "circuit_bytes": 1048576 * (32 if ring else 16 if wise else 8) + (4 * depth if ring else 0),
            "verify_bytes": 182 if ring else 64 if wise else 0,
        }
        phase_map = {"input_ms": "input_ms" if normal else "input_mac_ms",
                     "expand_ms": "expand_ms", "circuit_ms": "circuit_ms",
                     "verify_ms": "verify_ms", "mul_ms": "mul_ms"}
    measured = []
    for iteration in iterations:
        batch = [r for r in rows if int(r["iteration"]) == iteration]
        assert len(batch) == 3 and {int(r["rank"]) for r in batch} == {0, 1, 2}
        for row in batch:
            assert int(row["width"]) == width and int(row["depth"]) == depth
            assert int(row["word_ands" if binary else "gates"]) == 1048576
            assert int(row["checks"]) == (int(wise) if binary else 2 if wise else 0)
            for key, value in expected_bytes.items():
                assert int(row[key]) == value, (protocol, circuit, key, row)
            if not binary:
                assert float(row["mul_ms"]) <= float(row["circuit_ms"])
        maxima = {key: max(float(r[column]) for r in batch)
                  for key, column in phase_map.items()}
        maxima["core_ms"] = maxima["circuit_ms"] + maxima["verify_ms"]
        maxima["total_ms"] = maxima["core_ms"] + maxima.get("input_ms", 0) + maxima.get("expand_ms", 0)
        if iteration or not (normal or binary):
            measured.append({"iteration": iteration, **maxima})
    assert len(measured) == 7
    return {
        "protocol": protocol, "circuit": circuit, "width": width, "depth": depth,
        "iterations": measured,
        "phases": {key: {"median_ms": statistics.median(r[key] for r in measured),
                         "min_ms": min(r[key] for r in measured),
                         "max_ms": max(r[key] for r in measured)} for key in maxima},
        "bytes_per_party": expected_bytes,
        "output_validation": "all outputs in all 8 iterations" if normal or binary else "all outputs in final iteration",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--ring-only", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, "YACL_NUM_THREADS": "65"}
    binaries = {n: BENCH / "bazel-bin" / n for n in
                ("circuit_bench", "ring_circuit_bench", "binary_bench")}
    sources = sorted(BENCH.parent.glob("*.cc")) + sorted(BENCH.parent.glob("*.h"))
    sources += [BENCH.parent / "BUILD.bazel"]
    sources += [BENCH / n for n in ("circuit_bench.cc", "ring_circuit_bench.cc", "binary_bench.cc", "run_server.py", "BUILD.bazel", "MODULE.bazel", "MODULE.bazel.lock", ".bazelrc")]
    sources += [ROOT / "src" / n for n in (".bazelrc", "MODULE.bazel", "libspu/core/parallel_utils.h", "libspu/mpc/common/prg_state.h", "libspu/mpc/common/prg_state.cc", "libspu/mpc/utils/simulate.h", "libspu/mpc/aby3/arithmetic.cc", "libspu/mpc/aby3/boolean.cc")]
    parameters = {
        "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "YACL_NUM_THREADS": "65", "background_workers": 64,
        "thread_semantics": "YACL subtracts one caller from YACL_NUM_THREADS; the pool is shared by three parties",
        "platform": platform.platform(), "logical_cpus": os.cpu_count(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)), "loadavg_at_start": os.getloadavg(),
        "compiler": subprocess.check_output(["c++", "--version"], text=True).splitlines()[0],
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "warmups": 1, "repetitions": 7, "circuits": CIRCUITS,
        "transport": "FactoryMem, three parties in one process; benchmarks run sequentially",
        "aggregation": "per iteration phase maxima across parties, then median/min/max over seven measured iterations; total sums phase maxima before taking median",
        "arithmetic": "two secret inputs, z[0,i]=2^60-8+i; z[l+1,i]=z[l,i]*(z[l,i]-13); integer circuit, no fixed-point truncation",
        "spdzwise_timing": "both FIELD and RING use normal import with immediate input check, fresh session each iteration; final check covers 1048576 products; setup/export/reconstruction/oracle excluded",
        "aby3_timing": "existing circuit_bench adapters, session reused after warmup; no authentication or verification; setup/reconstruction/oracle excluded",
        "binary_timing": "C2 packed uint64 AND; ABFL verification includes random triple generation and all checks; setup/import/export/oracle excluded",
        "binary_sha256": {n: digest(p) for n, p in binaries.items()},
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sources},
    }
    (out / "parameters.json").write_text(json.dumps(parameters, indent=2) + "\n")
    (out / "lscpu.txt").write_text(subprocess.check_output(["lscpu"], text=True))
    (out / "meminfo.txt").write_text(Path("/proc/meminfo").read_text())
    # Include untracked implementation files without touching the Git index.
    patch = subprocess.check_output(["git", "diff", "--binary", "HEAD", "--"] + [str(p.relative_to(ROOT)) for p in sources], cwd=ROOT)
    for path in sources:
        relative = str(path.relative_to(ROOT))
        if subprocess.run(["git", "ls-files", "--error-unmatch", relative], cwd=ROOT, capture_output=True).returncode:
            diff = subprocess.run(["git", "diff", "--no-index", "--", "/dev/null", relative], cwd=ROOT, capture_output=True)
            assert diff.returncode == 1
            patch += diff.stdout
    (out / "source.patch").write_bytes(patch)
    jobs = []
    for circuit, (width, depth) in CIRCUITS.items():
        base = [str(width), str(depth), "7"]
        if not args.ring_only:
            jobs += [("aby3-f61-adapter", circuit, "circuit_bench", base, False, False),
                     ("aby3-ring64", circuit, "circuit_bench", base + ["ring64"], False, False),
                     ("spdzwise-f61-normal", circuit, "ring_circuit_bench", base + ["f61"], True, False)]
        jobs += [("spdzwise-ring128", circuit, "ring_circuit_bench", base, True, False)]
    if not args.ring_only:
        jobs += [("aby3-boolean", "C2", "binary_bench", ["--aby3"], False, True),
                 ("spdzwise-abfl", "C2", "binary_bench", [], False, True)]
    summaries = []
    for protocol, circuit, executable, options, normal, binary in jobs:
        name = f"{protocol}-{circuit}"
        command = [str(binaries[executable]), *options]
        print("Running", name, flush=True)
        start = time.monotonic()
        with (out / f"{name}.stdout").open("w") as stdout, (out / f"{name}.log").open("w") as stderr:
            process = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr)
            _, status, usage = os.wait4(process.pid, 0)
            process.returncode = os.waitstatus_to_exitcode(status)
        resource = {"command": command, "exit_code": process.returncode,
                    "wall_seconds": time.monotonic() - start, "user_seconds": usage.ru_utime,
                    "system_seconds": usage.ru_stime, "max_rss_kib": usage.ru_maxrss,
                    "loadavg_at_end": os.getloadavg()}
        (out / f"{name}-resources.json").write_text(json.dumps(resource, indent=2) + "\n")
        if process.returncode:
            raise RuntimeError(f"{name} failed; see archived log")
        stdout_text = (out / f"{name}.stdout").read_text()
        pool_sizes = {int(n) for n in re.findall(r"Create a fixed thread pool with size (\d+)", stdout_text)}
        # Small batches can stay on the callers without creating the lazy pool.
        assert not pool_sizes or pool_sizes == {64}, (name, pool_sizes)
        resource["observed_pool_workers"] = sorted(pool_sizes)
        (out / f"{name}-resources.json").write_text(json.dumps(resource, indent=2) + "\n")
        lines = [line for line in stdout_text.splitlines(True)
                 if line.startswith(("protocol,", protocol + ","))]
        (out / f"{name}.csv").write_text("".join(lines))
        summary = summarize(list(csv.DictReader(lines)), protocol, circuit, normal, binary)
        summaries.append(summary)
        print(name, {k: round(v["median_ms"], 3) for k, v in summary["phases"].items()}, flush=True)
        (out / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    assert all(digest(p) == parameters["binary_sha256"][n] for n, p in binaries.items())
    assert all(digest(ROOT / n) == sha for n, sha in parameters["source_sha256"].items())
    (out / "SHA256SUMS").write_text("".join(f"{digest(p)}  {p.name}\n" for p in sorted(out.iterdir()) if p.is_file()))


if __name__ == "__main__":
    main()
