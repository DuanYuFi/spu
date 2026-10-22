"""Archive HAL comparison measurements including all malicious checks and output preparation.

Build: bash build.sh //:comparison_bench
Run: python3 run_comparison.py NEW_OUTPUT_DIRECTORY
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
import tarfile
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

BENCH = Path(__file__).resolve().parent
ROOT = BENCH.parents[4]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def summarize(rows, count, repetitions, protocol):
    assert len(rows) == 3 * (repetitions + 1)
    phases = ("total_ms", "circuit_ms", "finalize_output_ms")
    traffic = ("sent_bytes", "sent_actions", "circuit_bytes", "finalize_output_bytes")
    samples = []
    wise = protocol == "spdzwise"
    expected = 241 * ((count + 63) // 64) * 64 if wise else 0
    for iteration in range(repetitions + 1):
        batch = [r for r in rows if int(r["iteration"]) == iteration]
        assert len(batch) == 3 and {int(r["rank"]) for r in batch} == {0, 1, 2}
        for row in batch:
            assert row["protocol"] == protocol
            assert int(row["count"]) == int(row["outputs_verified"]) == count
            assert int(row["check_mac_calls"]) == int(row["check_boolean_calls"]) == int(wise)
            assert int(row["mac_batches"]) == 0
            assert int(row["boolean_batches"]) == int(wise)
            assert int(row["and_bits"]) == expected
            assert float(row["total_ms"]) >= float(row["circuit_ms"]) > 0
            assert abs(float(row["total_ms"]) - float(row["circuit_ms"]) - float(row["finalize_output_ms"])) < 1e-4
            assert int(row["sent_bytes"]) == int(row["circuit_bytes"]) + int(row["finalize_output_bytes"])
            assert (int(row["finalize_output_bytes"]) > 0) == wise
        if iteration:
            samples.append({"iteration": iteration,
                            **{k: max(float(r[k]) for r in batch) for k in phases},
                            **{k: sum(int(r[k]) for r in batch) for k in traffic}})
    times = [s["total_ms"] for s in samples]
    median = statistics.median(times)
    return {"samples": samples, "median_ms": median,
            "min_ms": min(times), "max_ms": max(times),
            "comparisons_per_second": count * 1000 / median,
            "phases": {k: {"median_ms": statistics.median(s[k] for s in samples),
                           "min_ms": min(s[k] for s in samples),
                           "max_ms": max(s[k] for s in samples)} for k in phases},
            "three_party_traffic": {k: sorted({s[k] for s in samples}) for k in traffic},
            "three_party_sent_bytes": sorted({s["sent_bytes"] for s in samples}),
            "recorded_and_bits_per_party": expected,
            "check_mac_calls_per_party": int(wise),
            "check_boolean_calls_per_party": int(wise),
            "nonempty_mac_batches_per_party": 0,
            "nonempty_boolean_batches_per_party": int(wise),
            "outputs_verified": count * (repetitions + 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, default=100000)
    parser.add_argument("--repetitions", type=int, default=7)
    args = parser.parse_args()
    assert args.count > 0 and args.repetitions > 0
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    binary = BENCH / "bazel-bin/comparison_bench"
    sources = sorted(BENCH.parent.glob("*.cc")) + sorted(BENCH.parent.glob("*.h"))
    sources += [BENCH.parent / "BUILD.bazel"]
    sources += [BENCH / n for n in ("comparison_bench.cc", "run_comparison.py", "BUILD.bazel",
                                    "build.sh", "MODULE.bazel", "MODULE.bazel.lock", ".bazelrc")]
    sources += [ROOT / "src" / n for n in (
        ".bazelrc", "MODULE.bazel", "MODULE.bazel.lock", "libspu/mpc/ab_api.cc",
        "libspu/mpc/utils/tiling_util.h", "libspu/mpc/aby3/arithmetic.cc",
        "libspu/kernel/hal/integer.cc", "libspu/kernel/hal/ring.cc",
        "libspu/kernel/hal/prot_wrapper.cc", "libspu/mpc/protocol_session.cc",
        "libspu/mpc/aby3/conversion.cc", "libspu/mpc/aby3/conversion.h",
        "libspu/mpc/aby3/boolean.cc", "libspu/mpc/aby3/protocol.cc",
        "libspu/mpc/common/prg_state.cc", "libspu/core/parallel_utils.h")]
    parameters = {
        "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "operation": "HAL i_less(x,y), then checked output preparation", "count": args.count,
        "repetitions": args.repetitions, "warmups": 1,
        "YACL_NUM_THREADS": 65, "background_workers": 64,
        "transport": "FactoryMem, three parties in one process; protocols run serially",
        "timing": "Both protocols call HAL i_less on two prepared DT_I64/FM64 arithmetic secret-share arrays and return a serializable one-bit Boolean secret-share array with DT_I1. Time includes subtraction, native MSB, and output preparation. SPDZ-wise prepareOutputs invokes finalize_protocol -> checkMac + checkBoolean inside timing, including all random material generation and verification communication. Excludes input sharing/authentication, context/PRSS setup, reconstruction and plaintext comparison.",
        "aggregation": "Maximum whole checked interval across parties per iteration, then median/min/max across measured iterations. Phase maxima/medians are diagnostic and must not be summed. Traffic sums sent bytes across parties including all finalization traffic.",
        "input": "Two arrays of signed 63-bit inputs in [-2^62, 2^62-1], SplitMix64 seed 0x4d5342434d503634 and 25 edge pairs. Identical plaintext across protocols, fresh native shares and contexts per iteration.",
        "validation": "All outputs including warmup checked against modular MSB and signed less-than outside timing. Inside timing, one checkMac call and one checkBoolean call through normal output finalization; all pending Boolean gates are verified. Already authenticated inputs and linear subtraction leave zero new arithmetic MAC batches; checkMac still performs its count/status exchange, included in timing.",
        "security": "Checked SPDZ-wise comparison versus native semi-honest ABY3 at the same HAL input/output abstraction. All comparison security checks are timed. Protocol-specific internal circuits/layouts remain different; this does not isolate security-model overhead.",
        "platform": platform.platform(), "affinity": sorted(os.sched_getaffinity(0)),
        "compiler": subprocess.check_output(["c++", "--version"], text=True).splitlines()[0],
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "binary_sha256": digest(binary),
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sources},
    }
    save(out / "parameters.json", parameters)
    (out / "lscpu.txt").write_text(subprocess.check_output(["lscpu"], text=True))
    (out / "meminfo.txt").write_text(Path("/proc/meminfo").read_text())
    (out / "git-status.txt").write_text(subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True))
    with tarfile.open(out / "sources.tar.gz", "w:gz") as archive:
        for path in sources:
            archive.add(path, arcname=str(path.relative_to(ROOT)))
    results = {}
    for protocol in ("spdzwise", "aby3"):
        command = [str(binary), protocol, str(args.count), str(args.repetitions)]
        start = time.monotonic()
        with (out / (protocol + ".stdout")).open("w") as stdout, (out / (protocol + ".log")).open("w") as stderr:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr,
                                       env=dict(os.environ, YACL_NUM_THREADS="65"))
            _, status, usage = os.wait4(process.pid, 0)
            process.returncode = os.waitstatus_to_exitcode(status)
        resources = {"command": command, "exit_code": process.returncode,
                     "wall_seconds": time.monotonic() - start,
                     "user_seconds": usage.ru_utime, "system_seconds": usage.ru_stime,
                     "max_rss_kib": usage.ru_maxrss}
        save(out / (protocol + "-resources.json"), resources)
        if process.returncode:
            raise SystemExit(f"{protocol} failed; inspect {out / (protocol + '.log')}")
        stdout_text = (out / (protocol + ".stdout")).read_text()
        pools = {int(n) for n in re.findall(r"Create a fixed thread pool with size (\d+)", stdout_text)}
        assert pools <= {64} and (args.count < 32768 or pools == {64}), pools
        lines = [s for s in stdout_text.splitlines(True) if s.startswith(("protocol,", protocol + ","))]
        (out / (protocol + ".csv")).write_text("".join(lines))
        results[protocol] = summarize(list(csv.DictReader(lines)), args.count, args.repetitions, protocol)
        results[protocol]["resources"] = resources
        print(protocol, results[protocol], flush=True)
    ratio = results["spdzwise"]["median_ms"] / results["aby3"]["median_ms"]
    results["spdzwise"]["vs_aby3_checked_time_ratio"] = ratio
    save(out / "summary.json", results)
    table = ["| Protocol | Checked total median ms | Range ms | Comparisons/s | Three-party sent bytes |",
             "|---|---:|---:|---:|---:|"]
    for name, r in results.items():
        table.append(f"| {name} | {r['median_ms']:.3f} | {r['min_ms']:.3f}–{r['max_ms']:.3f} | {r['comparisons_per_second']:,.0f} | {', '.join(str(n) for n in r['three_party_sent_bytes'])} |")
    phase_table = ["| Protocol | Circuit ms | Finalization + output ms | Circuit sent bytes | Finalization sent bytes |",
                   "|---|---:|---:|---:|---:|"]
    for name, r in results.items():
        phase_table.append(f"| {name} | {r['phases']['circuit_ms']['median_ms']:.3f} | {r['phases']['finalize_output_ms']['median_ms']:.3f} | {r['three_party_traffic']['circuit_bytes']} | {r['three_party_traffic']['finalize_output_bytes']} |")
    report = f"""# Checked SPDZ-wise ring comparison versus ABY3

{args.count:,} pairs; FM64; three-party FactoryMem; 64 background workers; one warmup and {args.repetitions} measured iterations. Fresh native shares and contexts each iteration.

{parameters['timing']}

{parameters['security']}

{chr(10).join(table)}

SPDZ-wise / ABY3 checked total median time: **{ratio:.3f}x**.

{parameters['aggregation']}

{chr(10).join(phase_table)}

{parameters['input']}

{parameters['validation']}

Reproduce from this benchmark directory with `bash build.sh //:comparison_bench` and `python3 run_comparison.py NEW_OUTPUT_DIRECTORY --count {args.count} --repetitions {args.repetitions}`. Commands, source archive and fingerprints, binary fingerprint, environment, raw samples and resource records are stored alongside this report. These measurements use local-memory transport.
"""
    (out / "RESULTS.md").write_text(report)
    parameters["finished_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    save(out / "parameters.json", parameters)
    (out / "SHA256SUMS").write_text("".join(f"{digest(p)}  {p.name}\n" for p in sorted(out.iterdir()) if p.is_file()))


if __name__ == "__main__":
    main()
