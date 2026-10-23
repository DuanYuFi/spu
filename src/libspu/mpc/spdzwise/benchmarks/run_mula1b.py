"""Run native SPDZ-wise daBit and ABY3 MulA1B, including necessary checks."""
import argparse
import csv
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
from zoneinfo import ZoneInfo

BENCH = Path(__file__).resolve().parent
ROOT = BENCH.parents[4]
TIME_KEYS = ("total_ms", "operation_ms", "final_check_ms", "dabit_ms",
             "precheck_ms", "bit2a_ms", "multiply_ms")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, default=1000000)
    parser.add_argument("--repetitions", type=int, default=7)
    args = parser.parse_args()
    assert args.count > 0 and args.repetitions > 0
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    binary = BENCH / "bazel-bin/mula1b_bench"
    env = dict(os.environ, YACL_NUM_THREADS="65")
    sources = sorted(BENCH.parent.glob("*.cc")) + sorted(BENCH.parent.glob("*.h"))
    sources += [BENCH.parent / "BUILD.bazel"]
    sources += [BENCH / x for x in ("mula1b_bench.cc", "run_mula1b.py", "BUILD.bazel", "build.sh", "MODULE.bazel", "MODULE.bazel.lock", ".bazelrc")]
    sources += [ROOT / "src/libspu/mpc/aby3" / x for x in ("arithmetic.cc", "arithmetic.h", "ot.cc", "protocol.cc")]
    save(out / "parameters.json", {
        "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "count": args.count, "repetitions": args.repetitions, "warmups": 1,
        "protocols": ["spdzwise", "aby3"], "YACL_NUM_THREADS": 65,
        "background_workers": 64, "inter_op_parallelism": False, "intra_op_parallelism": False,
        "transport": "FactoryMem, three parties in one process, protocol runs serial",
        "timing": "mul_a1b + final checkMac for SPDZ-wise; includes fresh daBit, bitness and pre-opening checks, Bit2A, MulAA. Excludes input preparation/import/verification and output export/reconstruction. ABY3 uses native semi-honest mul_a1b without MACs.",
        "aggregation": "Max party whole interval per iteration, then median/min/max excluding warmup. Traffic sums three senders. Phase medians are not additive.",
        "input": "Full-width uint64 and random bits, seed 0x4d554c413142; fixed edge prefix tests both bits. Fresh shares and session per iteration.",
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "binary_sha256": digest(binary),
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sources},
    })
    (out / "lscpu.txt").write_text(subprocess.check_output(["lscpu"], text=True))
    (out / "environment.txt").write_text(
        subprocess.check_output(["uname", "-a"], text=True)
        + Path("/etc/os-release").read_text()
        + subprocess.check_output(["free", "-b"], text=True)
        + subprocess.check_output(["g++", "--version"], text=True))
    summary = {}
    for protocol in ("spdzwise", "aby3"):
        cmd = [str(binary), protocol, str(args.count), str(args.repetitions)]
        print("Running", protocol, flush=True)
        with (out / f"{protocol}.stdout").open("w") as stdout, (out / f"{protocol}.log").open("w") as stderr:
            started = time.monotonic()
            child = subprocess.Popen(cmd, env=env, stdout=stdout, stderr=stderr)
            _, status, usage = os.wait4(child.pid, 0)
            child.returncode = os.waitstatus_to_exitcode(status)
            save(out / f"{protocol}-resources.json", {
                "command": cmd, "returncode": child.returncode,
                "wall_seconds": time.monotonic() - started,
                "user_seconds": usage.ru_utime, "system_seconds": usage.ru_stime,
                "max_rss_kib": usage.ru_maxrss,
                "minor_faults": usage.ru_minflt, "major_faults": usage.ru_majflt,
                "voluntary_switches": usage.ru_nvcsw, "involuntary_switches": usage.ru_nivcsw})
            if child.returncode:
                raise subprocess.CalledProcessError(child.returncode, cmd)
        lines = (out / f"{protocol}.stdout").read_text().splitlines()
        logs = (out / f"{protocol}.log").read_text()
        if args.count >= 1000000:
            assert "Create a fixed thread pool with size 64" in "\n".join(lines) + logs
        data = "\n".join(s for s in lines if s.startswith(("protocol,", protocol + ","))) + "\n"
        (out / f"{protocol}.csv").write_text(data)
        rows = list(csv.DictReader(io.StringIO(data)))
        assert len(rows) == 3 * (args.repetitions + 1)
        samples = []
        for iteration in range(args.repetitions + 1):
            batch = [r for r in rows if int(r["iteration"]) == iteration]
            assert len(batch) == 3 and {int(r["rank"]) for r in batch} == {0, 1, 2}
            for r in batch:
                assert int(r["count"]) == int(r["outputs_verified"]) == args.count
                assert int(r["input_checks"]) == int(protocol == "spdzwise")
                assert int(r["mac_checks"]) == 2 * int(protocol == "spdzwise")
                assert int(r["boolean_checks"]) == 0
            if iteration:
                samples.append({"iteration": iteration,
                    **{k: max(float(r[k]) for r in batch) for k in TIME_KEYS},
                    "sent_bytes": sum(int(r["sent_bytes"]) for r in batch),
                    "sent_actions": sum(int(r["sent_actions"]) for r in batch)})
        stats = {k: {"median": statistics.median(s[k] for s in samples),
                     "min": min(s[k] for s in samples), "max": max(s[k] for s in samples)}
                 for k in (*TIME_KEYS, "sent_bytes", "sent_actions")}
        summary[protocol] = {"samples": samples, "stats": stats,
            "pairs_per_second": args.count * 1000 / stats["total_ms"]["median"],
            "outputs_verified": args.count * (args.repetitions + 1)}
        save(out / "summary.json", summary)
        print(protocol, stats["total_ms"], flush=True)
    files = sorted(p for p in out.iterdir() if p.is_file() and p.name != "SHA256SUMS")
    (out / "SHA256SUMS").write_text("".join(f"{digest(p)}  {p.name}\n" for p in files))


if __name__ == "__main__":
    main()
