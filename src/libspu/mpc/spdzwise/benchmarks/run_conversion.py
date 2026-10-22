"""Archive RING split-3 A2B and native ABY3 conversion measurements.

Build: bash build.sh //:conversion_bench
Run: python3 run_conversion.py NEW_OUTPUT_DIRECTORY
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


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def summarize(rows, count, repetitions, wise):
    assert len(rows) == 3 * (repetitions + 1)
    samples = []
    for iteration in range(repetitions + 1):
        batch = [r for r in rows if int(r["iteration"]) == iteration]
        assert len(batch) == 3 and {int(r["rank"]) for r in batch} == {0, 1, 2}
        for r in batch:
            assert int(r["count"]) == int(r["outputs_verified"]) == count
            assert int(r["input_checks"]) == int(r["final_checks"]) == int(wise)
            assert int(r["sent_bytes"]) == int(r["online_bytes"]) + int(r["verify_bytes"])
            if wise:
                assert abs(float(r["total_ms"]) - float(r["online_ms"]) - float(r["verify_ms"])) < 1e-4
                assert int(r["and_bits"]) > 0
            else:
                assert float(r["verify_ms"]) == int(r["verify_bytes"]) == int(r["and_bits"]) == 0
        if iteration:
            samples.append({
                "iteration": iteration,
                **{key: max(float(r[key]) for r in batch) for key in ("total_ms", "online_ms", "verify_ms")},
                **{key: sum(int(r[key]) for r in batch) for key in ("sent_bytes", "online_bytes", "verify_bytes", "sent_actions")},
            })
    return {
        "samples": samples,
        "phases": {key: {"median_ms": statistics.median(s[key] for s in samples),
                          "min_ms": min(s[key] for s in samples),
                          "max_ms": max(s[key] for s in samples)} for key in ("total_ms", "online_ms", "verify_ms")},
        "three_party_traffic": {key: sorted({s[key] for s in samples}) for key in ("sent_bytes", "online_bytes", "verify_bytes", "sent_actions")},
        "recorded_and_bits_per_party": sorted({int(r["and_bits"]) for r in rows}),
        "outputs_verified": count * (repetitions + 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, default=100000)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--protocols", nargs="+", choices=["spdzwise", "aby3"], default=["spdzwise", "aby3"])
    args = parser.parse_args()
    assert args.count > 0 and args.repetitions > 0
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    binary = BENCH / "bazel-bin/conversion_bench"
    sources = sorted(BENCH.parent.glob("*.cc")) + sorted(BENCH.parent.glob("*.h"))
    sources += [BENCH.parent / "BUILD.bazel"]
    sources += [BENCH / n for n in ("conversion_bench.cc", "run_conversion.py", "BUILD.bazel", "build.sh", "MODULE.bazel", "MODULE.bazel.lock", ".bazelrc")]
    sources += [ROOT / "src" / n for n in (".bazelrc", "MODULE.bazel", "MODULE.bazel.lock", "libspu/mpc/aby3/conversion.cc", "libspu/mpc/aby3/conversion.h", "libspu/mpc/aby3/boolean.cc", "libspu/mpc/aby3/protocol.cc", "libspu/mpc/common/prg_state.cc", "libspu/core/parallel_utils.h")]
    parameters = {
        "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "count": args.count, "repetitions": args.repetitions, "warmups": 1,
        "protocols": args.protocols, "YACL_NUM_THREADS": 65, "background_workers": 64,
        "transport": "FactoryMem, three parties in one process; protocols run serially",
        "timing": "A2B plus one final ABFL verification; native ABY3 A2B only. Exclude input sharing/authentication, context/PRSS setup, export/reconstruction/comparison.",
        "aggregation": "maximum whole interval across parties per iteration, then median/min/max; phase medians must not be summed",
        "input": "full-width uint64 SplitMix64 seed 0x53504c4954334132 with fixed edge prefix; identical plaintext across protocols; fresh protocol-native shares and context each iteration",
        "security": "SPDZwise uses existing ABFL checks; ABY3 is the native semi-honest baseline",
        "platform": platform.platform(), "affinity": sorted(os.sched_getaffinity(0)),
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "binary_sha256": digest(binary),
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sources},
    }
    save(out / "parameters.json", parameters)
    (out / "lscpu.txt").write_text(subprocess.check_output(["lscpu"], text=True))
    (out / "meminfo.txt").write_text(Path("/proc/meminfo").read_text())
    (out / "git-status.txt").write_text(subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True))
    results = {}
    for protocol in args.protocols:
        command = [str(binary), protocol, str(args.count), str(args.repetitions)]
        start = time.monotonic()
        with (out / (protocol + ".stdout")).open("w") as stdout, (out / (protocol + ".log")).open("w") as stderr:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=dict(os.environ, YACL_NUM_THREADS="65"))
            _, status, usage = os.wait4(process.pid, 0)
            process.returncode = os.waitstatus_to_exitcode(status)
        resources = {"command": command, "exit_code": process.returncode,
                     "wall_seconds": time.monotonic() - start, "user_seconds": usage.ru_utime,
                     "system_seconds": usage.ru_stime, "max_rss_kib": usage.ru_maxrss}
        save(out / (protocol + "-resources.json"), resources)
        if process.returncode:
            raise SystemExit(f"{protocol} failed; inspect {out / (protocol + '.log')}")
        stdout_text = (out / (protocol + ".stdout")).read_text()
        pools = {int(n) for n in re.findall(r"Create a fixed thread pool with size (\d+)", stdout_text)}
        assert pools <= {64} and (args.count < 32768 or pools == {64}), pools
        lines = [s for s in stdout_text.splitlines(True) if s.startswith(("protocol,", protocol + ","))]
        (out / (protocol + ".csv")).write_text("".join(lines))
        results[protocol] = summarize(list(csv.DictReader(lines)), args.count, args.repetitions, protocol != "aby3")
        results[protocol]["resources"] = resources
        print(protocol, results[protocol]["phases"], flush=True)
    if "aby3" in results:
        base = results["aby3"]
        for name, r in results.items():
            if name != "aby3":
                r["vs_aby3"] = {
                    "total_time_ratio": r["phases"]["total_ms"]["median_ms"] / base["phases"]["total_ms"]["median_ms"],
                    "online_time_ratio": r["phases"]["online_ms"]["median_ms"] / base["phases"]["total_ms"]["median_ms"],
                    "traffic_ratio": r["three_party_traffic"]["sent_bytes"][0] / base["three_party_traffic"]["sent_bytes"][0],
                }
    save(out / "summary.json", results)
    table = ["| Protocol | Total ms | Range ms | Conversion ms | Verification ms | Total sent bytes (3 parties) | Peak RSS MiB |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, r in results.items():
        p = r["phases"]
        t = p["total_ms"]
        table.append(f"| {name} | {t['median_ms']:.3f} | {t['min_ms']:.3f}–{t['max_ms']:.3f} | {p['online_ms']['median_ms']:.3f} | {p['verify_ms']['median_ms']:.3f} | {r['three_party_traffic']['sent_bytes'][0]:,} | {r['resources']['max_rss_kib']/1024:.1f} |")
    report = f"""# RING split-3 A2B versus native ABY3

{args.count:,} full-width 64-bit inputs; three-party FactoryMem; 64 background workers; one warmup and {args.repetitions} measured iterations. Fresh context and native input shares every iteration. All outputs, including warmup, were reconstructed and compared exactly, outside timing.

SPDZwise timing includes conversion and one final ABFL verification, including fresh verification material. Native ABY3 timing includes its complete registered A2B operation. Input authentication, protocol initialization and output reconstruction are excluded. ABY3 is a semi-honest baseline.

{chr(10).join(table)}

Each sample is the maximum whole interval across the three parties; the table reports medians and ranges. Separately aggregated phase medians cannot be summed. CSV includes per-party phase traffic and message counts. Peak RSS covers the whole three-party process, including input preparation.
"""
    for name, r in results.items():
        if "vs_aby3" in r:
            ratio = r["vs_aby3"]
            report += f"\n{name} / ABY3: total time **{ratio['total_time_ratio']:.3f}x**, conversion time **{ratio['online_time_ratio']:.3f}x**, total traffic **{ratio['traffic_ratio']:.3f}x**.\n"
    report += "\nReproduce with `bash build.sh //:conversion_bench` and `python3 run_conversion.py NEW_OUTPUT_DIRECTORY`. Commands, source/binary fingerprints, environment, raw samples and resources are archived alongside this report. These are local-memory transport results.\n"
    (out / "RESULTS.md").write_text(report)
    parameters["finished_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    save(out / "parameters.json", parameters)
    (out / "SHA256SUMS").write_text("".join(f"{digest(p)}  {p.name}\n" for p in sorted(out.iterdir()) if p.is_file()))


if __name__ == "__main__":
    main()
