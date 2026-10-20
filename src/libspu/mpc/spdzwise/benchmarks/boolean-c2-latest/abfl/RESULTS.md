# SPDZwise-ABFL word64 C2 — 2026-10-20

三方 FactoryMem 本机实验；W=65,536，D=16，64-bit packed AND，电路不分块。
YACL_NUM_THREADS=12；ABFL verification 使用 65,536-word 工作块。
共 1,048,576 个 word AND / 67,108,864 个 bit AND。
B=3，随机 triple 2,097,155 个 64-bit word（134,217,920 bit lanes）。置换、分桶和抽查均以 word 为单位；抽查 3 个随机 word，B 按在线 packed word 数选择。
预热 1 次，正式测量 7 次；每轮各阶段取三方最大耗时，再取七轮中位数。
总耗时按每轮的 online 最大值与 verification 最大值相加后统计。

| 阶段 | 中位数 ms | 最小 ms | 最大 ms | 每方发送 payload bytes |
|---|---:|---:|---:|---:|
| Online AND（含保存 triple） | 27.966 | 24.428 | 45.135 | 8388608 |
| Final verification | 400.363 | 362.178 | 414.744 | 218110266 |
| 合计 | 425.179 | 407.313 | 439.512 | 226498874 |

Verification 包含随机 triple 生成、copy 检查、commit–reveal、cut-and-choose 和 sacrifice。
输入分享、runtime/PRSS 初始化、导入检查、输出导出及重构核对均不计入上表。
8 轮均核对全部 65,536 个输出，且断言 online payload 正确、verification 检查次数为 1。ABFL 另外断言导出不额外通信。

进程峰值 RSS：0.650 GiB（三方合计，含输入和验证缓冲区）。
进程 wall time：5.094 秒（包含未计入阶段耗时的准备与核对）。
未隔离其他机器负载；本机内存通信结果不代表 LAN/WAN。
这是开发原型性能记录，不代表完整恶意安全后端或安全界限验收。

原始逐方数据：[CSV](w65536-d16.csv)，[stdout](w65536-d16.stdout)，[日志](w65536-d16.log)。
参数和源码/二进制指纹：[parameters.json](parameters.json)；源码差异：[source.patch](source.patch)。
环境：[lscpu.txt](lscpu.txt)、[meminfo.txt](meminfo.txt)；进程资源：[resources.json](resources.json)。
重跑：在 benchmarks 目录构建 `//:binary_bench`，执行 `YACL_NUM_THREADS=12 python3 run_binary.py 新目录`。
