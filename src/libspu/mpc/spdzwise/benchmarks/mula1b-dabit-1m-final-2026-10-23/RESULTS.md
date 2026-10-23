# 1M daBit MulA1B：最终保留结果

保留最后一次优化版 SPDZwise 复测与最新原生 ABY3 测量，日期为 2026-10-23。
两组来自不同的串行批次，使用同一二进制、相同设置和明文；原始 CSV/stdout/log
逐字保留，未重跑、拼接轮次或选择最快样本。较早测量及优化前数据已清理。
具体原始批次身份保存在 `parameters.json` 的 `retained_runs` 中，仅作来源记录。

1M = 1,000,000 对完整 FM64 算术整数与一位 Boolean 值。每协议预热一次、
测量七次，每轮新 session 和 shares。FactoryMem 三方同进程、64 个后台 worker，
关闭 context inter/intra 并行分片；先取每轮三方总耗时最大值，再统计中位数和范围。

| 协议 | 总耗时中位数（ms） | 范围（ms） | 三方总发送（B） | 吞吐量（对/秒） |
|---|---:|---:|---:|---:|
| SPDZwise RING daBit（含验证） | **1,031.242** | 1,011.883–1,083.038 | 321,502,535 | 969,704 |
| 原生 ABY3（半诚实） | **103.164** | 65.967–119.252 | 64,000,000 | 9,693,316 |

SPDZwise 计时包含现场 daBit、bitness 和复制检查、打开前全部依赖验证、Bit2A
一致性打开、MulAA 与最终 checkMac；不含输入认证/验证、输出导出和重构。
ABY3 为半诚实基线，计入原生 mul_a1b 的完整调用。两协议均不做截断。

## SPDZwise 阶段耗时

| 阶段 | 中位数 ms | 范围 ms |
|---|---:|---:|
| daBit 生成（含 bitness） | 637.051 | 613.581–647.652 |
| 打开前验证 | 204.983 | 189.241–208.727 |
| 打包打开及线性 Bit2A | 18.311 | 15.107–29.006 |
| MulAA | 140.411 | 135.272–175.283 |
| 最终 checkMac | 68.965 | 61.142–76.566 |

各阶段独立取三方最大值和跨轮中位数，不能相加作为主指标。
SPDZwise 每轮总发送 321,502,535 B、371 次发送动作；ABY3 为 64,000,000 B、8 次。
逐方数据及所有阶段见 [spdzwise.csv](spdzwise.csv)、[aby3.csv](aby3.csv)，
逐轮和聚合统计见 [summary.json](summary.json)。

## 验证与资源

每协议 8 轮共 8,000,000 个输出全部满足 `y=x*b mod 2^64`，输出复制一致。
SPDZwise 每方输入非空 MAC 验证 1 次在计时外，材料与最终乘积的 2 次在计时内。
输入没有上游 Boolean AND，因此非空 Boolean 批次为 0；必要同步和打开检查仍完整执行。
导出未增加非空验证。9 个测试目标共 318 个用例通过，三个未受影响目标使用缓存，
其余六个目标执行了当前实现；故障拒绝及回归证据见 [validation/](validation/)。

- spdzwise：整个三方进程峰值 RSS 2,033.9 MiB，进程 wall 30.795 s（包含输入和输出处理，不是主计时指标）。
- aby3：整个三方进程峰值 RSS 457.1 MiB，进程 wall 10.867 s（包含输入和输出处理，不是主计时指标）。

Xeon Gold 6248R、96 逻辑 CPU、GCC 13.3.0、Bazel 7.7.1、`-c opt`。
环境记录见 [environment.txt](environment.txt) 和 [lscpu.txt](lscpu.txt)，源代码及
二进制指纹见 [parameters.json](parameters.json)，全目录校验见 [SHA256SUMS](SHA256SUMS)。
环境快照来自同一配置的测量批次；它不是最后一次 SPDZwise 启动时重新抓取的快照。

```sh
cd src/libspu/mpc/spdzwise/benchmarks
bash build.sh //:mula1b_bench
python3 run_mula1b.py NEW_OUTPUT_DIRECTORY
```
