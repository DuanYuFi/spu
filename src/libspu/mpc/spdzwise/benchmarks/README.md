# Malicious PPML Benchmark

实验源码、Bazel 配置、原始数据和参数统一放在本目录。汇总表及实验说明见项目的 [Benchmark](../../../../../readme-malicious-ppml.md#benchmark)。每个归档目录对应一次机器、协议及日期确定的实验；F61 对照实验各包含三种电路配置、两种协议。

| 实验目录 | 内容 |
|---|---|
| [macbook-air-f61-2026-10-19](macbook-air-f61-2026-10-19/) | MacBook Air：ABY3-F61 / SPDZwise-F61，三种电路 |
| [ubuntu-f61-2026-10-19](ubuntu-f61-2026-10-19/) | Ubuntu Server：ABY3-F61 / SPDZwise-F61，三种电路 |
| [ubuntu-aby3-ring64-2026-10-20](ubuntu-aby3-ring64-2026-10-20/) | Ubuntu Server：原生 ABY3-ring64，65,536 × 16 |
| [ubuntu-aby3-binary-2026-10-20](ubuntu-aby3-binary-2026-10-20/) | Ubuntu Server：原生 ABY3 Boolean AND，65,536 × 16；仅保留原文及逐轮最大耗时 |

各目录的 `parameters.json` 记录基本程序参数；`metadata.json` 保留原测量环境和指纹，内部旧路径是测量时路径。算术实验完整保留 `.stdout`、`.log`、`.csv`、`summary.json` 及历史 `RESULTS.md`；迁移后的原始文件指纹列在 `SHA256SUMS`。历史结果文档中的计时值及当时构建记录保留原样。Binary 的 `iteration-maxima.csv` 从 `recorded-readme.md` 转存，原三方日志、临时源码和构建产物已在此前清理，无法恢复；该目录明确记录了这一缺口。

本目录是独立 Bazel workspace，通过 `MODULE.bazel` 引用 `../../../..`（仓库的 `src` 目录），并导入现有 SPU 编译配置。实验目标定义在本地 `BUILD.bazel`，无需在框架中添加 benchmark 目标。需要 Bazel 7.7.1、C++17 编译器、CMake 和 Python 3；可通过 `BAZEL` 指定 Bazel 可执行文件，已缓存的本机工具位于忽略提交的 `tools/bin/`。

从仓库根目录构建或重新测量：

```sh
bash src/libspu/mpc/spdzwise/benchmarks/build.sh
bash src/libspu/mpc/spdzwise/benchmarks/run.sh f61 ubuntu-f61-new
bash src/libspu/mpc/spdzwise/benchmarks/run.sh ring64 ubuntu-ring64-new
```

`f61` 运行三个形状和两种协议，`ring64` 运行默认形状；均预热一次、测量七次。每次输出写入本目录下新的 `runs/<名称>/`，包含参数、构建日志、原始数据和汇总；名称已存在时退出，归档数据不会被覆盖。Binary 原临时程序未保留，目前没有重跑入口。
