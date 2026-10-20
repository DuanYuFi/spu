# 最新 C2 Boolean 实验 — 2026-10-20

使用当前代码重新构建的同一 `binary_bench`，先 ABY3、后 ABFL，依次执行，不与本任务编译或测试并发。两种协议均设置 `YACL_NUM_THREADS=12`，没有修改框架默认线程数或协议实现。

固定三方 replicated XOR sharing、FactoryMem 单进程；W=65,536、D=16、每元素 64 位，共 1,048,576 个 packed word AND（67,108,864 个 bit AND）。输入生成、输入向量和电路相同。各预热 1 次，正式测量 7 次；每个阶段先取当轮三方最大值，再取七轮中位数。总耗时先逐轮相加，再取中位数，不能直接把各阶段中位数相加。通信量是每方发送的应用 payload，不含 transport framing。

| 协议 | Online（ms / B） | Verification（ms / B） | 总计（ms / B） |
|---|---:|---:|---:|
| [ABY3 Boolean](aby3/RESULTS.md) | 23.340 / 8,388,608 | 0.000 / 0 | 23.340 / 8,388,608 |
| [SPDZwise-ABFL](abfl/RESULTS.md) | 27.966 / 8,388,608 | 400.363 / 218,110,266 | 425.179 / 226,498,874 |

ABY3 为原生半诚实 AND，不保存 triple、不运行最终验证；ABFL online 包含快照保存，verification 包含随机 triple 生成、副本检查、commit–reveal、cut-and-choose 和 sacrifice。ABFL 的 M=1,048,576、B=3，随机池为 2,097,155 个 64-bit word triple，抽查 3 个完整随机 word，工作块为 65,536 word；不公开抽查在线 triple。

输入分享、runtime/PRSS 初始化、导入检查及输出导出/重构核对不计时。两种协议各 8 轮全部核对 65,536 个输出，并断言门数、检查次数及通信量。每方逐轮数据、min/max、峰值 RSS、时间戳、CPU/线程配置、源码差异与指纹分别在 `aby3/`、`abfl/`；机器可读汇总为 `comparison.json`。两组记录的二进制和源码指纹均一致，并与当前文件核对通过。

本目录只包含本次最新性能实验，不合并此前测量。旧 Boolean 实验、packing 微基准与 profiling 中间目录已清理；算术实验保留。`validation/` 单独保留此前相同 C++ 源码的 276 个通过测试及 1/12 线程 Boolean 测试证据，不将它们称为本次新执行的测试。此轮没有修改协议代码，因此仅重跑性能实验及其完整输出检查。

共享机器负载未隔离，没有固定 CPU 频率；FactoryMem 结果不代表 LAN/WAN 性能。原型安全参数和完整组合安全证明的限制见 SPDZwise README。

## 复现命令

以下命令在 `src/libspu/mpc/spdzwise/benchmarks` 执行；本次实际输出目录为 `boolean-c2-latest`。再次执行时改用新的目录名，runner 拒绝覆盖已有结果。

```bash
mkdir -p boolean-c2-latest
tools/bin/bazel-7.7.1 --batch --output_user_root=/tmp/malicious-spu-bazel build -c opt --lockfile_mode=off --jobs=16 --extra_toolchains=@rules_foreign_cc//toolchains:preinstalled_cmake_toolchain //:binary_bench > boolean-c2-latest/build.log 2>&1
mkdir -p boolean-c2-latest/aby3 boolean-c2-latest/abfl
cp boolean-c2-latest/build.log boolean-c2-latest/aby3/build.log
cp boolean-c2-latest/build.log boolean-c2-latest/abfl/build.log
YACL_NUM_THREADS=12 python3 run_binary.py boolean-c2-latest/aby3 --aby3
YACL_NUM_THREADS=12 python3 run_binary.py boolean-c2-latest/abfl
```
