# Truncation Benchmark

100,000 个 FM64 正负混合输入，|x|<2^62，截断 16 位。三方 FactoryMem，64 个后台 worker，预热 1 次、正式测量 7 次。每轮取三方完整计时区间最大值，再统计中位数。

| Protocol | Exact | Total Time | Total Comm. | Verification Time |
| --- | :---: | ---: | ---: | ---: |
| ABY3 TruncAPr2 | No | 8.728 ms | 2,800,000 B | 0.000 ms |

Exact=Yes 表示严格等于 floor(x/2^16)，包括负数。Naive 使用认证 A2B → 本地符号扩展右移 → B2A；ABY3 exact 使用原生 A2B → ARShiftB → B2ASelector。TruncAPr2 为概率舍入，非 exact。

总时间包含完整算子调用和内部通信；Naive 包含现场随机材料、必要 MAC/ABFL 验证及掩码打开，verification 是总时间的一部分。输入准备/认证、后续空检查、输出导出和重构排除。ABY3 为半诚实，verification 为 0。

包括预热在内，每种保留算子核对 800,000 个输出，全部通过。Exact 路径逐元素验证 floor，TruncAPr2 验证误差至多 1。阶段中位数独立聚合，不能相加重建总时间。

已清理 RandomBit 性能数据；本报告只包含仍保留原始数据的算子。共享源码快照、原始构建指纹与回归记录保留用于追溯，不代表仍保留其中每条路径的性能实验。

复现：`python3 run_truncation.py NEW_OUTPUT_DIRECTORY --count 100000 --repetitions 7 --protocols aby3`。参数见 parameters.json，逐方逐轮数据见 CSV，资源统计见 *-resources.json。
