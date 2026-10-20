"""Run the ABFL C2 benchmark and archive raw samples, provenance and statistics.

Build //:binary_bench in this benchmark workspace first, then run:
  python3 run_binary.py NEW_OUTPUT_DIRECTORY [--aby3]
An existing directory containing only build.log is also accepted.
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


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if len(sys.argv) not in (2, 3) or (len(sys.argv) == 3 and sys.argv[2] != "--aby3"):
        raise SystemExit("usage: run_binary.py NEW_OUTPUT_DIRECTORY [--aby3]")
    native = len(sys.argv) == 3
    protocol = "aby3-boolean" if native else "spdzwise-abfl"
    checks = 0 if native else 1
    bucket = 0 if native else 3
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(p.name != "build.log" for p in out.iterdir()):
        raise SystemExit("Refusing to overwrite an existing experiment")
    binary = BENCH / "bazel-bin/binary_bench"
    sources = sorted((BENCH.parent).glob("*.cc")) + sorted((BENCH.parent).glob("*.h"))
    sources += [BENCH.parent / "BUILD.bazel"]
    sources += [BENCH / n for n in ("binary_bench.cc", "BUILD.bazel", "MODULE.bazel", "MODULE.bazel.lock", ".bazelrc", "run_binary.py")]
    sources += [ROOT / "src" / path for path in (
        "libspu/mpc/aby3/boolean.cc", "libspu/mpc/aby3/type.h",
        "libspu/mpc/common/communicator.h", "libspu/mpc/common/prg_state.cc",
        "libspu/core/parallel_utils.h")]
    parameters = {
        "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "protocol": protocol, "width": 65536, "depth": 16,
        "word_ands": 1048576, "bit_ands": 67108864, "nbits": 64,
        "bucket_size": bucket, "verification_unit": "none" if native else "64-bit word", "cut_and_choose_words": 0 if native else 3,
        "cut_and_choose_bits": 0 if native else 192, "random_word_triples": 0 if native else 2097155,
        "random_bit_triples": 0 if native else 134217920,
        "input_vectors": 17, "secret_input_words": 1114112,
        "input_generator": "SplitMix64 finalizer(index + 0x9e3779b97f4a7c15), index=layer*65536+i; see source",
        "circuit": "z[0,i]=x[0,i]; z[l+1,i]=z[l,i]&x[l+1,i]",
        "tiling": False, "warmups": 1, "repetitions": 7,
        "verification_tile_words": 0 if native else 65536,
        "random_generation_tile_words": 0 if native else 65536,
        "replica_check_tile_words": 0 if native else 65536,
        "replica_check": "none" if native else "BLAKE3-256 per tile, sent only to replica holder",
        "transport": "FactoryMem, three parties in one process",
        "timing": "native ABY3 AND, PRSS masks and communication only" if native else "online includes AND, PRSS masks and saved triples; verification includes random triple generation, copy checks, commit-reveal, cut-and-choose and sacrifice",
        "excluded": "input sharing, runtime/PRSS setup, input import/copy checks, output export/reconstruction and plaintext comparison",
        "aggregation": "per repetition take maximum over parties for each phase; total=sum of phase maxima; report median/min/max over seven repetitions excluding warmup",
        "platform": platform.platform(), "logical_cpus": os.cpu_count(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "loadavg_at_start": os.getloadavg(),
        "YACL_NUM_THREADS": os.environ.get("YACL_NUM_THREADS"),
        "compiler": subprocess.check_output(["c++", "--version"], text=True).splitlines()[0],
        "build": "Bazel 7.7.1, -c opt; see build.log",
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        "binary_sha256": digest(binary),
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sources},
    }
    (out / "parameters.json").write_text(json.dumps(parameters, indent=2) + "\n")
    (out / "lscpu.txt").write_text(subprocess.check_output(["lscpu"], text=True))
    (out / "meminfo.txt").write_text(Path("/proc/meminfo").read_text())
    (out / "run_binary.py").write_bytes(Path(__file__).read_bytes())
    # Archive executable inputs only, not old experiment data or report edits.
    patch = subprocess.check_output(
        ["git", "diff", "--binary", "HEAD", "--"] +
        [str(p.relative_to(ROOT)) for p in sources], cwd=ROOT)
    for path in sources:
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(path.relative_to(ROOT))], cwd=ROOT, capture_output=True)
        if tracked.returncode:
            diff = subprocess.run(["git", "diff", "--no-index", "--", "/dev/null", str(path.relative_to(ROOT))], cwd=ROOT, capture_output=True)
            if diff.returncode != 1:
                raise RuntimeError(diff.stderr.decode())
            patch += diff.stdout
    (out / "source.patch").write_bytes(patch)
    start = time.monotonic()
    with (out / "w65536-d16.stdout").open("w") as stdout, (out / "w65536-d16.log").open("w") as stderr:
        process = subprocess.Popen([str(binary)] + (["--aby3"] if native else []), stdout=stdout, stderr=stderr)
        _, status, usage = os.wait4(process.pid, 0)
        process.returncode = os.waitstatus_to_exitcode(status)
    resources = {
        "exit_code": process.returncode, "wall_seconds": time.monotonic() - start,
        "user_seconds": usage.ru_utime, "system_seconds": usage.ru_stime,
        "max_rss_kib": usage.ru_maxrss, "loadavg_at_end": os.getloadavg(),
        "finished_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
    }
    (out / "resources.json").write_text(json.dumps(resources, indent=2) + "\n")
    if process.returncode:
        raise SystemExit(f"Benchmark failed ({process.returncode}); see archived logs")
    lines = [s for s in (out / "w65536-d16.stdout").read_text().splitlines(True)
             if s.startswith(("protocol,", protocol + ","))]
    (out / "w65536-d16.csv").write_text("".join(lines))
    rows = list(csv.DictReader(lines))
    assert len(rows) == 24
    maxima = []
    for iteration in range(8):
        batch = [r for r in rows if int(r["iteration"]) == iteration]
        assert len(batch) == 3 and {int(r["rank"]) for r in batch} == {0, 1, 2}
        for r in batch:
            for key, value in {"width":65536, "depth":16, "word_ands":1048576,
                               "bit_ands":67108864, "bucket_size":bucket,
                               "online_bytes":8388608, "verify_bytes":0 if native else 218110266,
                               "checks":checks}.items():
                assert int(r[key]) == value, (key, r)
            assert abs(float(r["total_ms"]) - float(r["online_ms"]) - float(r["verify_ms"])) < 0.00001
        online, verify = [max(float(r[key]) for r in batch) for key in ("online_ms", "verify_ms")]
        if iteration:
            maxima.append({"iteration":iteration, "online_ms":online,
                           "verify_ms":verify, "total_ms":online+verify})
    payloads = {key: {int(r[key]) for r in rows} for key in ("online_bytes", "verify_bytes")}
    assert all(len(v) == 1 for v in payloads.values())
    summary = {"aggregation":parameters["aggregation"], "iterations":maxima,
               "phases":{key:{"median_ms":statistics.median(r[key] for r in maxima),
                              "min_ms":min(r[key] for r in maxima),
                              "max_ms":max(r[key] for r in maxima)}
                         for key in ("online_ms", "verify_ms", "total_ms")},
               "bytes_per_party":{k:next(iter(v)) for k,v in payloads.items()},
               "validated_output_words_per_iteration":65536, "validated_iterations":8}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "summary.txt").write_text(json.dumps(summary["phases"], indent=2) + "\n")
    phase = summary["phases"]
    comm = summary["bytes_per_party"]
    title = "ABY3 Boolean" if native else "SPDZwise-ABFL word64"
    verification = ("原生 ABY3 半诚实 AND，无最终 verification，也不保存 triple。输入使用同一 XOR replicated sharing 布局，在计时外适配为 ABY3 Boolean 类型。" if native else
                    "B=3，随机 triple 2,097,155 个 64-bit word（134,217,920 bit lanes）。置换、分桶和抽查均以 word 为单位；抽查 3 个随机 word，B 按在线 packed word 数选择。")
    report = f'''# {title} C2 — {parameters["started_at"].split("T")[0]}

三方 FactoryMem 本机实验；W=65,536，D=16，64-bit packed AND，电路不分块。
YACL_NUM_THREADS={parameters["YACL_NUM_THREADS"] or "默认"}；ABFL verification 使用 65,536-word 工作块。
共 1,048,576 个 word AND / 67,108,864 个 bit AND。
{verification}
预热 1 次，正式测量 7 次；每轮各阶段取三方最大耗时，再取七轮中位数。
总耗时按每轮的 online 最大值与 verification 最大值相加后统计。

| 阶段 | 中位数 ms | 最小 ms | 最大 ms | 每方发送 payload bytes |
|---|---:|---:|---:|---:|
| Online AND{"（含保存 triple）" if not native else ""} | {phase['online_ms']['median_ms']:.3f} | {phase['online_ms']['min_ms']:.3f} | {phase['online_ms']['max_ms']:.3f} | {comm['online_bytes']} |
| Final verification | {phase['verify_ms']['median_ms']:.3f} | {phase['verify_ms']['min_ms']:.3f} | {phase['verify_ms']['max_ms']:.3f} | {comm['verify_bytes']} |
| 合计 | {phase['total_ms']['median_ms']:.3f} | {phase['total_ms']['min_ms']:.3f} | {phase['total_ms']['max_ms']:.3f} | {comm['online_bytes']+comm['verify_bytes']} |

{"Verification 包含随机 triple 生成、copy 检查、commit–reveal、cut-and-choose 和 sacrifice。" if not native else "未运行恶意安全验证。"}
输入分享、runtime/PRSS 初始化、导入检查、输出导出及重构核对均不计入上表。
8 轮均核对全部 65,536 个输出，且断言 online payload 正确、verification 检查次数为 {checks}。ABFL 另外断言导出不额外通信。

进程峰值 RSS：{usage.ru_maxrss / 1024 / 1024:.3f} GiB（三方合计，含输入和验证缓冲区）。
进程 wall time：{resources['wall_seconds']:.3f} 秒（包含未计入阶段耗时的准备与核对）。
未隔离其他机器负载；本机内存通信结果不代表 LAN/WAN。
这是开发原型性能记录，不代表完整恶意安全后端或安全界限验收。

原始逐方数据：[CSV](w65536-d16.csv)，[stdout](w65536-d16.stdout)，[日志](w65536-d16.log)。
参数和源码/二进制指纹：[parameters.json](parameters.json)；源码差异：[source.patch](source.patch)。
环境：[lscpu.txt](lscpu.txt)、[meminfo.txt](meminfo.txt)；进程资源：[resources.json](resources.json)。
重跑：在 benchmarks 目录构建 `//:binary_bench`，执行 `{"YACL_NUM_THREADS=" + parameters["YACL_NUM_THREADS"] + " " if parameters["YACL_NUM_THREADS"] else ""}python3 run_binary.py 新目录{" --aby3" if native else ""}`。
'''
    (out / "RESULTS.md").write_text(report)
    files = sorted(p for p in out.iterdir() if p.is_file() and p.name != "SHA256SUMS")
    (out / "SHA256SUMS").write_text("".join(f"{digest(p)}  {p.name}\n" for p in files))
    print(report)

if __name__ == "__main__":
    main()
