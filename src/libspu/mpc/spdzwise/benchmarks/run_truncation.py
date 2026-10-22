"""Archive TruncNaive, TruncPrRandomBit, native ABY3 TruncAPr2 or exact ABY3.

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


def summarize(rows, count, repetitions, protocol):
    wise = protocol in ("spdzwise", "random-bit")
    exact = protocol in ("spdzwise", "aby3-exact")
    naive = protocol == "spdzwise"
    assert len(rows) == (repetitions + 1) * 3
    iterations = []
    for iteration in range(repetitions + 1):
        batch = [r for r in rows if int(r["iteration"]) == iteration]
        assert len(batch) == 3 and {int(r["rank"]) for r in batch} == {0, 1, 2}
        for r in batch:
            assert int(r["count"]) == count and int(r["bits"]) == 16
            assert int(r["outputs_verified"]) == count
            assert int(r["input_checks"]) == int(wise)
            assert r["protocol"] == protocol
            assert int(r["final_checks"]) == 0
            assert int(r["internal_checks"]) == int(wise)
            assert int(r["boolean_checks"]) == int(naive)
            if exact:
                assert int(r["error_zero"]) == count
            if naive:
                assert float(r["internal_verify_ms"]) > 0
            assert abs(float(r["total_ms"]) - float(r["material_ms"]) - float(r["online_ms"])) < 1e-4
            assert sum(int(r[k]) for k in ("error_minus1", "error_zero", "error_plus1")) == count
        if iteration:
            iterations.append({
                "iteration": iteration,
                **{key: max(float(r[key]) for r in batch)
                   for key in ("total_ms", "material_ms", "online_ms", "verify_ms", "internal_verify_ms")},
                "total_sent_bytes": sum(int(r["sent_bytes"]) for r in batch),
                "total_verify_bytes": sum(int(r["verify_bytes"]) for r in batch),
            })
    phases = {key: {"median_ms": statistics.median(r[key] for r in iterations),
                    "min_ms": min(r[key] for r in iterations),
                    "max_ms": max(r[key] for r in iterations)}
              for key in ("total_ms", "material_ms", "online_ms", "verify_ms", "internal_verify_ms")}
    median = phases["total_ms"]["median_ms"]
    return {
        "iterations": iterations, "phases": phases,
        "ns_per_element": median * 1e6 / count,
        "elements_per_second": count * 1e3 / median,
        "bytes_per_party": sorted({int(r["sent_bytes"]) for r in rows}),
        "total_sent_bytes": sorted({r["total_sent_bytes"] for r in iterations}),
        "validation": "all elements, including warmup; required MAC/ABFL verification inside primary interval; post-call empty checkpoint excluded",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, default=100000)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--protocols", nargs="+",
                        choices=("spdzwise", "random-bit", "aby3", "aby3-exact"),
                        default=("spdzwise", "random-bit", "aby3"))
    args = parser.parse_args()
    assert args.count >= 20 and args.count % 2 == 0 and args.repetitions > 0
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    binary = BENCH / "bazel-bin/truncation_bench"
    env = dict(os.environ, YACL_NUM_THREADS="65")
    sources = sorted(BENCH.parent.glob("*.cc")) + sorted(BENCH.parent.glob("*.h"))
    sources += [BENCH.parent / "BUILD.bazel"]
    sources += [BENCH / n for n in ("truncation_bench.cc", "run_truncation.py", "build.sh", "BUILD.bazel", "MODULE.bazel", "MODULE.bazel.lock", ".bazelrc")]
    sources += [ROOT / "src" / n for n in (".bazelrc", "MODULE.bazel", "libspu/mpc/aby3/arithmetic.cc", "libspu/mpc/aby3/arithmetic.h", "libspu/mpc/aby3/protocol.cc", "libspu/mpc/common/prg_state.cc", "libspu/mpc/common/prg_state.h", "libspu/core/parallel_utils.h", "libspu/mpc/utils/simulate.h", "libspu/mpc/kernel.h", "libspu/mpc/ab_api.cc", "libspu/mpc/aby3/conversion.cc", "libspu/mpc/aby3/conversion.h", "libspu/mpc/aby3/boolean.cc")]
    parameters = {
        "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "protocols": args.protocols,
        "count": args.count, "bits": 16, "application_ring_bits": 64,
        "signed_input_bits": 63, "internal_authentication_bits": 128,
        "input_range": "-2^62 < x < 2^62; balanced positive/negative pairs and zero",
        "input_generator": "SplitMix64, seed 0x5350445a5452554e, fixed edge prefix; independent fresh shares each iteration",
        "warmups": 1, "repetitions": args.repetitions,
        "YACL_NUM_THREADS": 65, "background_workers": 64,
        "transport": "FactoryMem, three parties in one process; protocols run sequentially",
        "timing": "start after input authentication; include fresh material and all required arithmetic/Boolean verification before masked opening; stop at authenticated output; post-call empty checkpoint/export/reconstruction excluded",
        "aggregation": "per iteration maximum of whole timed interval across ranks; median/min/max over measured iterations; do not sum phase maxima",
        "spdzwise_kernel": "default TruncASelector -> TruncNaive, exact signed floor; A2B/local arithmetic shift/B2A",
        "random_bit_kernel": "explicit TruncPrRandomBit state entry, probabilistic rounding; current pre-opening checkpoint included",
        "phase_definitions": "material_ms: Naive B2A mask/MAC generation, RandomBit authenticated random bits; online_ms: remaining call time including checks (not preprocessing-free online time); internal_verify_ms: Naive combined MAC/ABFL check, -1 means uninstrumented for RandomBit; verify_ms: post-call empty checkpoint",
        "aby3_kernel": "normally registered TruncAPr2, SignType::Unknown, ordinary replicated shares",
        "aby3_exact": "native registered A2B -> ARShiftB(16) -> B2ASelector; exact signed floor; no custom protocol kernels or verification",
        "precise_aby3_flag": "ENABLE_PRECISE_ABY3_TRUNCPR is already enabled and selects probabilistic TruncAPr2, not exact floor",
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
    for protocol in args.protocols:
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
        lines = [line for line in stdout_text.splitlines() if line.startswith(("protocol,", "spdzwise,", "random-bit,", "aby3,", "aby3-exact,"))]
        (out / f"{protocol}.csv").write_text("\n".join(lines) + "\n")
        rows = list(csv.DictReader(lines))
        summaries[protocol] = summarize(rows, args.count, args.repetitions, protocol)
    comparisons = []
    if "spdzwise" in summaries and "random-bit" in summaries:
        ratio = (summaries["random-bit"]["phases"]["total_ms"]["median_ms"] /
                 summaries["spdzwise"]["phases"]["total_ms"]["median_ms"])
        summaries["random_bit_over_naive"] = ratio
        comparisons.append(f"TruncPrRandomBit / TruncNaive 耗时比：**{ratio:.3f}×**。")
    if "spdzwise" in summaries and "aby3" in summaries:
        ratio = (summaries["spdzwise"]["phases"]["total_ms"]["median_ms"] /
                 summaries["aby3"]["phases"]["total_ms"]["median_ms"])
        summaries["naive_over_aby3"] = ratio
        comparisons.append(f"TruncNaive / ABY3 TruncAPr2 耗时比：**{ratio:.3f}×**。")
    write_json(out / "summary.json", summaries)
    labels = {"spdzwise": "SPDZwise TruncNaive (floor)",
              "random-bit": "SPDZwise TruncPrRandomBit", "aby3": "ABY3 TruncAPr2",
              "aby3-exact": "ABY3 exact (A2B / ARShiftB / B2A)"}
    table = ["| 算子 | 主耗时中位数 ms | 最小–最大 ms | 随机材料 ms | 剩余算子时间 ms | 计时外空检查 ms | 三方发送 B | 峰值 RSS MiB |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in args.protocols:
        label = labels[name]
        s = summaries[name]
        p = s["phases"]
        total = p["total_ms"]
        rss = json.loads((out / f"{name}-resources.json").read_text())["max_rss_kib"] / 1024
        table.append(f"| {label} | {total['median_ms']:.3f} | {total['min_ms']:.3f}–{total['max_ms']:.3f} | {p['material_ms']['median_ms']:.3f} | {p['online_ms']['median_ms']:.3f} | {p['verify_ms']['median_ms']:.3f} | {s['total_sent_bytes']} | {rss:.1f} |")
    report = f"""# Truncation Benchmark

{args.count:,} 个正负混合输入，m=16，输入满足 |x|<2^62。三方 FactoryMem；64 个 YACL 后台 worker（YACL_NUM_THREADS=65）；预热 1 次、测量 {args.repetitions} 次；每轮新上下文，各组串行运行。
本次运行：{", ".join(labels[p] for p in args.protocols)}。
输入生成器、随机种子、范围和配置与此前 TruncPrRandomBit 实验一致。每轮取三方主计时区间最大值，再统计中位数及范围。

主计时从准备好的输入 shares 开始，包含算子全部转换、随机材料和通信，到输出 arithmetic shares 形成时停止。
SPDZwise 预先完成输入认证，主计时包含所有必要算术/ABFL 验证及掩码打开；输入认证、调用后的空检查、导出和重构排除。
ABY3 使用正常 IO 和注册的原生内核，无恶意验证阶段，Verification Time 为 0。

{chr(10).join(table)}

{chr(10).join(comparisons)}

`aby3-exact` 执行原生 A2B → ARShiftB(16) → B2ASelector，保留 64 位符号扩展并返回算术 shares；并未替换 ABY3 的默认 trunc_a。
仓库已开启的 ENABLE_PRECISE_ABY3_TRUNCPR 仅选择概率舍入 TruncAPr2，不能视为 exact floor 开关。
`spdzwise` 执行默认 TruncASelector → TruncNaive；`random-bit` 显式调用 truncPrRandomBit；`aby3` 使用原生 TruncAPr2 和 SignType::Unknown。

Naive 和 ABY3 exact 逐元素严格检查 floor(x/2^16)，包括负数；RandomBit 检查 floor 或 floor+1（整除时精确），ABY3 TruncAPr2 保留原生修正，误差至多 1。
包括预热在内，本次每种算子核对 **{args.count * (args.repetitions + 1):,} 个输出**，全部通过。

material_ms：Naive 的 B2A 掩码/MAC 生成，RandomBit 的认证随机位生成；ABY3 未单独细分随机性。
online_ms 为剩余算子时间，包含内部检查，不是预处理后的纯在线成本。
internal_verify_ms：Naive 的合并 MAC/ABFL 检查（含验证随机材料），RandomBit 的 -1 表示未单独细分，ABY3 为 0。
verify_ms 仅记录计时外空检查。各阶段独立聚合，不能相加重建总耗时。RSS 包含输入准备及三方整个进程。

结果仅代表当前机器 FactoryMem 和该批量。SPDZwise 与 ABY3 安全模型不同。

复现：`bash build.sh //:truncation_bench`，随后 `python3 run_truncation.py NEW_OUTPUT_DIRECTORY --count {args.count} --repetitions {args.repetitions} --protocols {" ".join(args.protocols)}`。
参数和源码/二进制指纹见 parameters.json，逐轮数据见 CSV，资源见 *-resources.json，源码改动见 source.patch。
"""
    (out / "RESULTS.md").write_text(report)
    (out / "SHA256SUMS").write_text("".join(f"{sha256(p)}  {p.relative_to(out)}\n" for p in sorted(out.iterdir()) if p.is_file() and p.name != "SHA256SUMS"))
    print(report)


if __name__ == "__main__":
    main()
