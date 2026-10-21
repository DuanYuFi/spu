# Malicious PPML Benchmark

实验源码、Bazel 配置、原始数据和参数统一放在本目录；总表见项目 [Benchmark](../../../../../readme-malicious-ppml.md#benchmark)。当前服务器结果统一使用 **64 个 YACL 后台 worker**：设置 `YACL_NUM_THREADS=65`，因为 YACL 将调用线程计入该参数。三方共享线程池，各组实验串行运行。

| 实验目录 | 内容 |
|---|---|
| [server-64worker-2026-10-21](server-64worker-2026-10-21/) | 当前服务器全套结果：`before/` 优化前、`after/` 优化后；四种 Arithmetic 协议各 C1–C3，ABY3 / ABFL Binary 各 C2；`validation/` 保存回归证据 |
| [spdzwise-ring128-latest](spdzwise-ring128-latest/) | 历史 SPDZwise RING C1–C3，原 `YACL_NUM_THREADS=12`；目录名保留供旧链接访问 |
| [boolean-c2-latest](boolean-c2-latest/RESULTS.md) | 历史 C2 Boolean，原 `YACL_NUM_THREADS=12`；目录名保留供旧链接访问 |
| [macbook-air-f61-2026-10-19](macbook-air-f61-2026-10-19/) | MacBook Air：ABY3-F61 / SPDZwise-F61，三种电路 |
| [ubuntu-f61-2026-10-19](ubuntu-f61-2026-10-19/) | Ubuntu Server：ABY3-F61 / SPDZwise-F61，三种电路 |
| [ubuntu-aby3-ring64-2026-10-20](ubuntu-aby3-ring64-2026-10-20/) | Ubuntu Server：原生 ABY3-ring64，65,536 × 16 |

每组预热一次、测量七次；每轮各阶段取三方最大值，再统计中位数、最小值和最大值。SPDZwise FIELD/RING 共用正常输入认证基准入口，每轮新 session，输入立即检查，电路结束后检查全部乘法记录。ABY3 沿用原有适配器与会话复用方式。Binary 的 verification 包含 ABFL 随机 triple 生成及所有检查。

从本目录复现全套服务器实验：

```sh
bash build.sh //:ring_circuit_bench //:binary_bench --distdir=/tmp/spu-distfiles
python3 run_server.py NEW_OUTPUT_DIRECTORY
```

脚本固定设置线程数，不依赖调用者的 `YACL_NUM_THREADS`，并检查日志中实际创建的线程池大小；小批次可能不创建惰性线程池。`--ring-only` 可只运行 Ring C1–C3。输出包含逐方逐轮 CSV、原始日志、`summary.json`、参数、源码/二进制指纹、源码补丁、资源记录和 `SHA256SUMS`。当前 Ring 优化为小端通信直接发送连续内存并复用接收缓冲区，消除中间编码字符串和新建解码向量。消息格式、通信量、失败传播和验证 transcript 保持一致。两个副本哈希的并行计算也已试验，但未显示稳定收益，因此未保留在最终代码中；试验数据保存在 `hash-parallel-trial/`。
