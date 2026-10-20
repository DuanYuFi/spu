# Malicious PPML Benchmark

实验源码、Bazel 配置、原始数据和参数统一放在本目录；总表见项目 [Benchmark](../../../../../readme-malicious-ppml.md#benchmark)。Boolean 仅保留最新一组 ABY3 / ABFL 对照；算术实验保持原样。

| 实验目录 | 内容 |
|---|---|
| [boolean-c2-latest](boolean-c2-latest/RESULTS.md) | 当前代码 C2 Boolean：相同二进制、12 线程，ABY3 / SPDZwise-ABFL 各预热 1 次、测量 7 次，完整输出核对 |
| [macbook-air-f61-2026-10-19](macbook-air-f61-2026-10-19/) | MacBook Air：ABY3-F61 / SPDZwise-F61，三种电路 |
| [ubuntu-f61-2026-10-19](ubuntu-f61-2026-10-19/) | Ubuntu Server：ABY3-F61 / SPDZwise-F61，三种电路 |
| [ubuntu-aby3-ring64-2026-10-20](ubuntu-aby3-ring64-2026-10-20/) | Ubuntu Server：原生 ABY3-ring64，65,536 × 16 |

本目录是独立 Bazel workspace，通过 `MODULE.bazel` 引用仓库 `src` 并导入已有编译配置。需要 Bazel 7.7.1、C++17 编译器、CMake 和 Python 3；本机缓存工具在忽略提交的 `tools/bin/`。

Boolean 源码为 `binary_bench.cc`；[最新实验说明](boolean-c2-latest/RESULTS.md)给出实际构建、运行命令与统计口径。`run_binary.py` 保存逐方逐轮 CSV、stdout/stderr、参数、源码差异、源码/二进制指纹、资源记录和校验和。源码差异只包括可执行输入，不再包含旧实验结果或文档修改。再次运行请使用新的输出目录，成功后再替换 latest，runner 不覆盖已有数据。

算术构建和运行入口（从仓库根目录执行）：

```bash
bash src/libspu/mpc/spdzwise/benchmarks/build.sh
bash src/libspu/mpc/spdzwise/benchmarks/run.sh f61 ubuntu-f61-new
bash src/libspu/mpc/spdzwise/benchmarks/run.sh ring64 ubuntu-ring64-new
```

`f61` 运行三个形状及两种协议，`ring64` 运行默认形状；均预热一次、测量七次。输出放在 `runs/<名称>/`，已有目录会被拒绝。算术原始数据及历史环境/源码指纹保留原样。
