# Malicious PPML 开发进展

更新日期：2026-10-19。当前实现是三方 **SPDZwise-F61 算术原型**，尚未完成恶意安全 PPML 后端验收。`SPDZWISE` 已接入 C++/protobuf/Python 配置和 IO，生产协议工厂仍显式拒绝执行；单元测试和 benchmark 通过 C++ `registerArithmeticPrototype()` 入口运行。

## 目前代码进展

- **域与分享**：`p=2^61-1`，FM64 存储；秘密数采用中心化有符号解释，默认 16 位小数。实现域运算、输入范围检查、带 MAC 的内部分享及客户端输入/输出类型。
- **算术协议**：输入副本一致性检查、输入认证、加法/取负/公开乘法、秘密乘法和矩阵乘法；输入、结果进入认证队列后检查。内部 MAC 类型禁止序列化，只有检查完成并提交的客户端输出可以导出。
- **随机性与验证**：key、乘法掩码、秘密检查系数使用独立的双边 AES-CTR PRSS 流，F61 采用拒绝采样。verification 直接从队列读取记录，合并 value/MAC 点积归约及交换；每条记录本地乘法由 6 次降为 4 次，单次非空检查每方发送 5 次、共 64 B 应用 payload。
- **框架集成**：`ProtocolSession` 提供导入、最终检查、暂存输出、提交和失败后禁用导出；device/Python 路径增加相应 hook，HAL 增加协议比较分派。现有协议未安装 session 时沿用原路径。device/Python 完整集成仍未验收。
- **正确性覆盖**：C++ 协议、IO、会话和 HAL 分派测试；Python 使用真实 C++ 域头文件与大整数对照，并测试明文数值规格。篡改用例覆盖不同服务器和大批次不同位置的 MAC、失败后的会话中止和输出阻断。

当前缺口：ABFL/post-sacrifice、恶意安全 daBits、密码学 A2B/B2A、精确截断、比较及非线性算子尚未实现；Python 明文规格仅为正确性参照。LR、树、CNN 等 PPML 模型未运行；客户端完整旧输出包重放防护、真实部署、协议组合安全论证及总失败概率仍待完成。域算术测试通过不等于这些验收已完成。

实现位于 [`src/libspu/mpc/spdzwise`](src/libspu/mpc/spdzwise/)，协议参考映射和数值约定见其 [README](src/libspu/mpc/spdzwise/README.md)。

### Original SPU Modification

以下记录对原有 SPU 文件的修改及原因，便于区分协议目录内的增量实现与框架接入改动。

| 原有文件（相对仓库根目录） | 修改 | 原因 |
|---|---|---|
| `src/libspu/spu.h`、`src/libspu/spu.proto`、`src/libspu/spu.cc`；`spu/api.py`、`spu/libspu.cc`、`spu/libspu.pyi` | 新增 `SPDZWISE = 7`，补充配置打印、字符串解析和 Python 枚举绑定。 | 使 C++、protobuf 和 Python 对协议标识保持一致，支持配置与 IO 接入。 |
| `src/libspu/core/config.cc` | SPDZWISE 未指定小数位时默认使用 16 位。 | 为 F61 定点数原型显式设置默认精度，避免直接套用存储域 FM64 的默认值。 |
| `src/libspu/mpc/factory.cc`、`src/libspu/mpc/BUILD.bazel` | 增加 SPDZWISE IO 构造及依赖，协议执行分支显式抛错。 | 支持开发阶段的分享与重构，同时在混合协议及安全验收完成前关闭生产执行入口。 |
| `src/libspu/core/object.h` | 新增 `hasState()`。 | 允许框架先判断协议是否安装 `ProtocolSession`，使会话 hook 按需启用。 |
| `src/libspu/core/context.h` | 新增零参数 `bindParams(KernelEvalContext*)` 重载。 | 支持通过 `dynDispatch` 调用无参数的协议最终检查内核。 |
| `src/libspu/device/api.cc`、`src/libspu/device/BUILD.bazel` | 执行前开启会话并导入秘密输入；发布输出前完成最终检查和输出准备；成功后提交，异常由 guard 中止会话；补充构建依赖。 | 将认证、延迟检查和输出提交纳入统一执行生命周期，防止未验证结果或失败执行的结果被导出；空输出执行也必须完成检查。 |
| `spu/libspu.cc`、`spu/BUILD.bazel` | 写入变量时检查会话及外部输入，读取值、元数据和分块数时统一访问已提交的导出视图；覆盖、删除或清空变量时使相关导出失效；补充会话依赖。 | 避免 Python API 直接读取含 MAC 的内部值，或在变量更新后返回旧导出。 |
| `src/libspu/core/type.h`、`src/libspu/core/value.cc` | 增加默认允许的 `isSerializable()` 类型接口，序列化前检查实部和虚部的导出权限，并在复制数据前完成元数据检查。 | 允许内部认证类型禁止序列化，防止私有 MAC 数据通过通用 `ValueProto` 路径离开运行时。 |
| `src/libspu/device/io.cc` | SPDZWISE 重构要求三方输出的 dtype、shape 一致，且不含复数。 | 补充检查分享数据缓冲区之外的元数据，避免将不一致的输出解释为同一结果。 |
| `src/libspu/kernel/hal/ring.cc`、`src/libspu/kernel/hal/ring_test.cc` | `_less` 优先分派 `less_ss`、`less_sp`、`less_ps`，保留操作数顺序；增加分派和原有回退路径的正确性测试。 | 为域协议提供有符号比较入口，避免中心化域边界下直接用 `msb(x-y)` 判断大小；SPDZWISE 的实际比较内核仍待实现。 |
| `src/libspu/mpc/BUILD.bazel` | 增加新文件 `protocol_session.h/.cc` 的库目标及 `protocol_session_test.cc` 测试目标。 | 为上述生命周期和输出门控提供可复用的协议会话接口及正确性验证。 |

## 唯一 benchmark：两输入 F61 circuit

保留 [`2026-10-19-circuit`](src/libspu/mpc/spdzwise/benchmarks/2026-10-19-circuit/) 所定义的电路和 `circuit_bench` target，移除其余本次开发产生的性能实验及独立乘法 benchmark。

只分享两个秘密元素 x、y，然后执行：

```text
z[0,i]   = x + i                 (i = 0,...,W-1)
z[l+1,i] = z[l,i] * (z[l,i] + y) (l = 0,...,D-1)
```

默认 W=65,536、D=16，共 1,048,576 个秘密乘法门和同等数量的加法。每层依赖上一层；只对两个标量做输入 MAC，广播/公开偏移为局部线性操作。每轮断言电路结束前不发生 verification，队列包含 1,048,578 条记录，末尾只做一次非空检查，导出不增加非空检查。正常协议导入路径仍保留其立即检查，不采用 benchmark 的延迟输入检查入口。

ABY3 对照是 **F61 适配版**，与 SPDZwise 共用 F61 replicated 乘法和 PRSS 底层；它不代表原生环 ABY3 或完整 PPML 应用的性能。SPDZwise 承担 MAC 和最终验证开销，ABY3 对照不提供这些检查。

## 与 ABY3 的效率对比

以下是 2026-10-19 用当前 verification 实现重新测量的结果，单位 ms。环境：Apple M2，24 GiB 内存、8 核，三方线程同进程 `FactoryMem`；Apple Clang 16、Bazel 7.7.1、`-c opt`，YACL 默认 7 个 worker。每组预热一次、记录七次；每阶段先取三方最大耗时，再取七轮中位数。总时间先逐轮相加后取中位数，各列中位数不要求严格相加。

| 协议 | 输入 MAC | 公开展开 | 电路计算 | 其中乘法 | 最终 verification | 电路+验证 | 全部计时阶段 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ABY3-F61 适配版 | 0.000 | 0.319 | 24.572 | 15.933 | 0.000 | 24.572 | 24.903 |
| SPDZwise-F61 | 0.018 | 2.183 | 58.387 | 33.259 | 11.495 | 69.395 | 71.595 |

默认电路中，SPDZwise 的乘法内核约为 ABY3 的 **2.09 倍**；电路计算加验证约为 **2.82 倍**，全部计时阶段约为 **2.87 倍**。初始分享、PRSS/key 建立、输入副本检查、导出及重构排除；输入 MAC、公开展开、计算、最终验证分别计时。三组形状均检查最后一轮全部输出。本机内存通信结果不用于推断跨机网络或完整 PPML 性能。

每方电路发送 payload：ABY3 为 8,388,608 B，SPDZwise 为 16,777,216 B；SPDZwise 另有输入 MAC 16 B、verification 64 B，公开展开不通信。详细三组结果、原始数据及源码指纹见 [RESULTS.md](src/libspu/mpc/spdzwise/benchmarks/2026-10-19-circuit/RESULTS.md)。

## 构建准备

以下命令从仓库根目录开始，需要 C++/Bazel 工具链。`/tmp/spu-distfiles` 是本机的预下载依赖缓存；缓存不完整时 Bazel 仍可能下载依赖。本机使用预安装 CMake 工具链。

```sh
cd src

# 使用本机已有的 Bazel 和依赖缓存；其他环境可调整这些 /tmp 路径。
bazel_spu() {
  USE_BAZEL_VERSION=7.7.1 BAZELISK_HOME=/tmp/spu-spdzwise-bazelisk \
    bazel --output_user_root=/tmp/spu-spdzwise-bazel "$@" \
    --lockfile_mode=off --distdir=/tmp/spu-distfiles --jobs=4 \
    --extra_toolchains=@rules_foreign_cc//toolchains:preinstalled_cmake_toolchain
}
```

## 运行单元测试（正确性）

在上面的 `src` 目录和 shell 中运行：

```sh
bazel_spu test -c opt \
  //libspu/mpc/spdzwise:protocol_test \
  //libspu/mpc/spdzwise:io_test \
  //libspu/mpc:protocol_session_test \
  //libspu/kernel/hal:ring_test
```

在仓库根目录运行 Python 正确性测试（域测试会调用 C++17 编译器并启用 UBSan，可通过 `CXX` 指定编译器）：

```sh
python3 src/libspu/mpc/spdzwise/field_reference_test.py
python3 src/libspu/mpc/spdzwise/reference_test.py
```

本次验证：协议 9、IO 7、会话 7、HAL 4，共 **27 个 C++ 用例通过**；Python 域测试 1 个用例（20,162 条运算输入）、明文规格 7 个用例通过。测试范围和未完成验收见 [VALIDATION.md](src/libspu/mpc/spdzwise/VALIDATION.md)。

## 运行 benchmark

在前述 `src` 目录和 shell 中运行：

```sh
bazel_spu build -c opt //libspu/mpc/spdzwise:circuit_bench

# 参数：宽度 W、深度 D、记录轮数；每个进程另做一轮预热。
./bazel-bin/libspu/mpc/spdzwise/circuit_bench 65536 16 7
./bazel-bin/libspu/mpc/spdzwise/circuit_bench 1048576 1 7
./bazel-bin/libspu/mpc/spdzwise/circuit_bench 16384 64 7
```

如需覆盖目录中的原始数据并重新统计七轮结果：

```sh
(
  out=libspu/mpc/spdzwise/benchmarks/2026-10-19-circuit
  for shape in 65536:16 1048576:1 16384:64; do
    width=${shape%:*}
    depth=${shape#*:}
    ./bazel-bin/libspu/mpc/spdzwise/circuit_bench "$width" "$depth" 7 \
      > "$out/w$width-d$depth.stdout" 2> "$out/w$width-d$depth.log" || exit 1
  done
  python3 "$out/summarize.py"
)
```

统计脚本验证三方样本、门数、输入数、队列长度、检查次数和各阶段 payload，生成 `.csv` 和 `summary.json`。重新测量后还需更新结果文档和 `metadata.json` 中的指纹。
