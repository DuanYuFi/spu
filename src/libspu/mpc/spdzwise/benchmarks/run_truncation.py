"""Archive signed RING truncation versus the normally registered ABY3 TruncAPr2.

Build with `bash build.sh //:truncation_bench`, then run:
  python3 run_truncation.py NEW_OUTPUT_DIRECTORY
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


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def summarize(rows, count, repetitions, wise):
    assert len(rows) == (repetitions + 1) * 3
    iterations = []
    for iteration in range(repetitions + 1):
        batch = [r for r in rows if int(r["iteration"]) == iteration]
        assert len(batch) == 3 and {int(r["rank"]) for r in batch} == {0, 1, 2}
        for r in batch:
            assert int(r["count"]) == count and int(r["bits"]) == 16
            assert int(r["outputs_verified"]) == count
            assert int(r["input_checks"]) == int(wise)
            assert int(r["final_checks"]) == int(wise)
            assert abs(float(r["total_ms"]) - float(r["material_ms"]) - float(r["online_ms"])) < 1e-4
            assert sum(int(r[k]) for k in ("error_minus1", "error_zero", "error_plus1")) == count
        if iteration:
            iterations.append({
                "iteration": iteration,
                **{key: max(float(r[key]) for r in batch)
                   for key in ("total_ms", "material_ms", "online_ms", "verify_ms")},
                "total_sent_bytes": sum(int(r["sent_bytes"]) for r in batch),
                "total_verify_bytes": sum(int(r["verify_bytes"]) for r in batch),
            })
    phases = {key: {"median_ms": statistics.median(r[key] for r in iterations),
                    "min_ms": min(r[key] for r in iterations),
                    "max_ms": max(r[key] for r in iterations)}
              for key in ("total_ms", "material_ms", "online_ms", "verify_ms")}
    median = phases["total_ms"]["median_ms"]
    return {
        "iterations": iterations, "phases": phases,
        "ns_per_element": median * 1e6 / count,
        "elements_per_second": count * 1e3 / median,
        "bytes_per_party": sorted({int(r["sent_bytes"]) for r in rows}),
        "total_sent_bytes": sorted({r["total_sent_bytes"] for r in iterations}),
        "validation": "all elements, including warmup; final authentication checks excluded from primary timing",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, default=100000)
    parser.add_argument("--repetitions", type=int, default=7)
    args = parser.parse_args()
    assert args.count >= 20 and args.count % 2 == 0 and args.repetitions > 0
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    binary = BENCH / "bazel-bin/truncation_bench"
    env = dict(os.environ, YACL_NUM_THREADS="65")
    sources = sorted(BENCH.parent.glob("*.cc")) + sorted(BENCH.parent.glob("*.h"))
    sources += [BENCH.parent / "BUILD.bazel"]
    sources += [BENCH / n for n in ("truncation_bench.cc", "run_truncation.py", "build.sh", "BUILD.bazel", "MODULE.bazel", "MODULE.bazel.lock", ".bazelrc")]
    sources += [ROOT / "src" / n for n in (".bazelrc", "MODULE.bazel", "libspu/mpc/aby3/arithmetic.cc", "libspu/mpc/aby3/arithmetic.h", "libspu/mpc/aby3/protocol.cc", "libspu/mpc/common/prg_state.cc", "libspu/mpc/common/prg_state.h", "libspu/core/parallel_utils.h", "libspu/mpc/utils/simulate.h")]
    parameters = {
        "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "count": args.count, "bits": 16, "application_ring_bits": 64,
        "signed_input_bits": 63, "internal_authentication_bits": 128,
        "input_range": "-2^62 < x < 2^62; balanced positive/negative pairs and zero",
        "input_generator": "SplitMix64, seed 0x5350445a5452554e, fixed edge prefix; independent fresh shares each iteration",
        "warmups": 1, "repetitions": args.repetitions,
        "YACL_NUM_THREADS": 65, "background_workers": 64,
        "transport": "FactoryMem, three parties in one process; protocols run sequentially",
        "timing": "start after input authentication; include fresh random material, bitness and opening checks; stop at authenticated output; final authentication/export/reconstruction excluded",
        "aggregation": "per iteration maximum of whole timed interval across ranks; median/min/max over measured iterations; do not sum phase maxima",
        "aby3_kernel": "normally registered TruncAPr2, SignType::Unknown, ordinary replicated shares",
        "bitness": "MP-SPDZ random linear dot product with fresh commit/reveal coefficients; full internal 128-bit CheckZero instead of low-word opening",
        "mpspdz_reference": "113c4e47dd94bca9ecab029d2f0abe649bea5c52",
        "platform": platform.platform(), "logical_cpus": os.cpu_count(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "loadavg_at_start": os.getloadavg(),
        "compiler": subprocess.check_output(["c++", "--version"], text=True).splitlines()[0],
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "binary_sha256": sha256(binary),
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sources},
    }
    write_json(out / "parameters.json", parameters)
    (out / "lscpu.txt").write_text(subprocess.check_output(["lscpu"], text=True))
    (out / "meminfo.txt").write_text(Path("/proc/meminfo").read_text())
    patch = subprocess.check_output(["git", "diff", "--binary", "HEAD", "--"] + [str(p.relative_to(ROOT)) for p in sources], cwd=ROOT)
    for p in sources:
        relative = str(p.relative_to(ROOT))
        if subprocess.run(["git", "ls-files", "--error-unmatch", relative], cwd=ROOT, capture_output=True).returncode:
            diff = subprocess.run(["git", "diff", "--no-index", "--", "/dev/null", relative], cwd=ROOT, capture_output=True)
            assert diff.returncode == 1
            patch += diff.stdout
    (out / "source.patch").write_bytes(patch)
    summaries = {}
    for protocol in ("spdzwise", "aby3"):
        command = [str(binary), protocol, str(args.count), str(args.repetitions)]
        print("Running", " ".join(command), flush=True)
        start = time.monotonic()
        with (out / f"{protocol}.stdout").open("w") as stdout, (out / f"{protocol}.log").open("w") as stderr:
            process = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr)
            _, status, usage = os.wait4(process.pid, 0)
            process.returncode = os.waitstatus_to_exitcode(status)
        write_json(out / f"{protocol}-resources.json", {
            "command": command, "exit_code": process.returncode,
            "wall_seconds": time.monotonic() - start,
            "user_seconds": usage.ru_utime, "system_seconds": usage.ru_stime,
            "max_rss_kib": usage.ru_maxrss, "loadavg_at_end": os.getloadavg(),
        })
        if process.returncode:
            raise SystemExit(f"{protocol} failed: inspect {out / (protocol + '.log')}")
        stdout_text = (out / f"{protocol}.stdout").read_text()
        pools = {int(n) for n in re.findall(r"Create a fixed thread pool with size (\d+)", stdout_text)}
        assert pools <= {64} and (args.count < 32768 or pools == {64}), pools
        lines = [line for line in stdout_text.splitlines() if line.startswith(("protocol,", "spdzwise-ring-signed,", "aby3-truncapr2,"))]
        (out / f"{protocol}.csv").write_text("\n".join(lines) + "\n")
        rows = list(csv.DictReader(lines))
        summaries[protocol] = summarize(rows, args.count, args.repetitions, protocol == "spdzwise")
    ratio = summaries["spdzwise"]["phases"]["total_ms"]["median_ms"] / summaries["aby3"]["phases"]["total_ms"]["median_ms"]
    summaries["spdzwise_over_aby3"] = ratio
    write_json(out / "summary.json", summaries)
    table = ["| 协议 | 主耗时中位数 ms | 最小–最大 ms | 随机材料 ms | 在线 ms | 计时外检查 ms | 吞吐 elements/s |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name in ("spdzwise", "aby3"):
        s = summaries[name]
        p = s["phases"]
        total = p["total_ms"]
        table.append(f"| {name} | {total['median_ms']:.3f} | {total['min_ms']:.3f}–{total['max_ms']:.3f} | {p['material_ms']['median_ms']:.3f} | {p['online_ms']['median_ms']:.3f} | {p['verify_ms']['median_ms']:.3f} | {s['elements_per_second']:.0f} |")
    report = f"""# Signed RING probabilistic truncation

{args.count:,} 个正负混合输入，m=16，双方输入满足 |x|<2^62。三方 FactoryMem；64 个 YACL 后台 worker；预热 1 次、测量 {args.repetitions} 次。

主计时从已准备好的输入 shares 开始，包含现场随机材料生成与必要检查，输出 shares 形成时停止。SPDZwise 输入已正确认证，输出保留 MAC；ABY3 为正常注册的 TruncAPr2、普通 replicated shares。输入认证、最终认证检查、输出重构均不计入主耗时。所有轮次全部输出均通过验证。

{chr(10).join(table)}

SPDZwise / ABY3 主耗时比：**{ratio:.2f}×**。各分项独立取三方最大值，不能相加重建主耗时。ABY3 的随机材料生成属于其算子内部，表中在线时间包含完整调用，并非预处理后的时间。

SPDZwise 按 MP-SPDZ 的有符号偏移、随机掩码与回绕修正实现；bitness 聚合使用现有内部 128 位零检查。主时间未包含最终延迟认证检查，不应解释为完成全部恶意安全验证的端到端时间。ABY3 保留源码中的 +1 修正，未假定与 MP-SPDZ 精确舍入分布相同。

计时内三方总发送字节：SPDZwise {summaries['spdzwise']['total_sent_bytes']}，ABY3 {summaries['aby3']['total_sent_bytes']}。此结果为本机内存通信，不能直接推断 LAN/WAN 延迟。

复现：`bash build.sh //:truncation_bench`，随后 `python3 run_truncation.py NEW_OUTPUT_DIRECTORY`。完整命令与环境见 parameters.json 和资源记录；逐轮数据见 CSV；源码改动见 source.patch。
"""
    (out / "RESULTS.md").write_text(report)
    (out / "SHA256SUMS").write_text("".join(f"{sha256(p)}  {p.relative_to(out)}\n" for p in sorted(out.iterdir()) if p.is_file() and p.name != "SHA256SUMS"))
    print(report)


if __name__ == "__main__":
    main()
