# Malicious PPML 开发进展

更新日期：2026-10-20。当前实现是三方 **SPDZwise-F61 算术原型**，尚未完成恶意安全 PPML 后端验收。`SPDZWISE` 已接入 C++/protobuf/Python 配置和 IO，生产协议工厂仍显式拒绝执行；单元测试和 benchmark 通过 C++ `registerArithmeticPrototype()` 入口运行。

## 目前代码进展

- **域与分享**：`p=2^61-1`，FM64 存储；秘密数采用中心化有符号解释，默认 16 位小数。实现域运算、输入范围检查、带 MAC 的内部分享及客户端输入/输出类型。
- **算术协议**：输入副本一致性检查、输入认证、加法/取负/公开乘法、秘密乘法和矩阵乘法；输入、结果进入认证队列后检查。内部 MAC 类型禁止序列化，只有检查完成并提交的客户端输出可以导出。
- **随机性与验证**：key、乘法掩码、秘密检查系数使用独立的双边 AES-CTR PRSS 流，F61 采用拒绝采样。verification 直接从队列读取记录，合并 value/MAC 点积归约及交换；每条记录本地乘法由 6 次降为 4 次，单次非空检查每方发送 5 次、共 64 B 应用 payload。
- **框架集成**：`ProtocolSession` 提供导入、最终检查、暂存输出、提交和失败后禁用导出；device/Python 路径增加相应 hook，HAL 增加协议比较分派。现有协议未安装 session 时沿用原路径。device/Python 完整集成仍未验收。
- **正确性覆盖**：C++ 协议、IO、会话和 HAL 分派测试；Python 使用真实 C++ 域头文件与大整数对照，并测试明文数值规格。篡改用例覆盖不同服务器和大批次不同位置的 MAC、失败后的会话中止和输出阻断。

ABFL/post-sacrifice 的 C++ Boolean 原型已实现：三方 replicated XOR sharing、ABY3 online AND、64-bit word 分桶以及最终验证后导出，详见 [Boolean 原型说明](src/libspu/mpc/spdzwise/README.md#abfl-boolean-development-prototype)。生产 factory 仍未启用。

当前缺口：ABFL 的安全界限及组合审查、恶意安全 daBits、密码学 A2B/B2A、精确截断、比较及非线性算子尚未完成；Python 明文规格仅为正确性参照。LR、树、CNN 等 PPML 模型未运行；客户端完整旧输出包重放防护、真实部署、协议组合安全论证及总失败概率仍待完成。域算术测试通过不等于这些验收已完成。

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

## Benchmark

### Experimental Setup

实验使用两台机器，均以三方线程在同一进程中运行：

| 机器 | 硬件 | 系统与工具链 | 并行配置 |
|---|---|---|---|
| MacBook Air Laptop | Apple M2，8 核，24 GiB 内存 | macOS（未记录版本），Apple Clang 16，Bazel 7.7.1，`-c opt`（SPU `-O2`） | YACL 默认 7 个 worker，另有三方调用线程 |
| Ubuntu Server | 双路 Xeon Gold 6248R，48 物理核 / 96 逻辑 CPU（内存容量未记录） | Ubuntu 24.04（10 月 20 日记录为 24.04.3 LTS），GCC 13.3.0，Bazel 7.7.1，`-c opt` | YACL 默认 95 个 worker；未绑定 CPU 或隔离其他负载 |

通信均通过 `FactoryMem` 完成，属于本机内存通信，没有跨机链路、带宽限制或人工 RTT 设置；结果不代表 LAN/WAN 性能。通信量统一统计**每方发送的应用 payload**，不含传输协议开销；三方发送合计为表中数值的 3 倍。

对比三种协议配置：

| 协议 | Domain | 说明 |
|---|---|---|
| ABY3-ring64 | Ring：`Z/(2^64)`，FM64 存储 | SPU 原生半诚实 ABY3，调用原生算术内核；无 MAC 或恶意安全 verification。 |
| ABY3-F61 | Field：`F_p`，`p=2^61-1`，FM64 存储 | F61 适配版，与 SPDZwise 共用域乘法和 PRSS 底层；作为相同域上的半诚实对照。 |
| SPDZwise-F61 | Field：`F_p`，`p=2^61-1`，FM64 存储 | 带 MAC 的算术原型，在电路末尾执行一次批量 verification；完整恶意安全 PPML 后端尚未验收。 |

Ring64 用于测量原生 ABY3；F61 采用可用 64 位字存储的 Mersenne 素数域，用于比较同一 Field 上的 ABY3 与 SPDZwise。**FM64 是存储宽度，不代表 F61 按模 `2^64` 运算**。算术实验均为整数电路，不包含定点截断。Binary 使用原生 ABY3 Boolean `and_bb`，在 `GF(2)` 上逐位计算，每个 FM64 字打包 64 个比特。

每组预热 1 次、测量 7 次：每轮各阶段取三方最大耗时，再取七轮中位数。总耗时先逐轮相加再取中位数，各列中位数不要求严格相加。计时排除初始分享、协议/key/PRSS 初始化、输入副本检查、输出重构和明文核对；内核执行期间的 PRSS 采样计入电路耗时。

### Circuit

固定 `W×D=1,048,576`，使用三种宽度与深度配置；每层的输出依赖上一层：

| 电路 | 宽度 W | 深度 D | 算术秘密乘法门数 |
|---|---:|---:|---:|
| C1：单层宽电路 | 1,048,576 | 1 | 1,048,576 |
| C2：默认电路 | 65,536 | 16 | 1,048,576 |
| C3：深电路 | 16,384 | 64 | 1,048,576 |

算术电路只分享两个秘密标量 `x=2^60-8`、`y=-13`，随后公开展开，每个乘法门前有一次秘密加法：

```text
z[0,i]   = x + i                 (i = 0,...,W-1)
z[l+1,i] = z[l,i] * (z[l,i] + y) (l = 0,...,D-1)
```

Ring64 与 F61 分别按各自模数计算。SPDZwise 仅对两个输入生成 MAC，公开展开为局部线性操作；所有乘法结果延迟至末尾统一检查，共 `W×D+2=1,048,578` 条记录。普通协议的输入导入仍保留立即检查；此处延迟检查是 benchmark 的专用计时入口。

Binary 已测量 C2 的 AND 电路：预先分享 `D+1=17` 个 uint64 输入向量，每层一次完整向量 `and_bb` 调用，无 tiling：

```text
z[0,i]   = x[0,i]
z[l+1,i] = z[l,i] & x[l+1,i]     (l = 0,...,D-1)
```

共 1,048,576 个 uint64 AND，等价于 67,108,864 个单比特 AND。算术实验核对最后一轮全部输出；binary 原记录为每次预热和测量均核对全部 65,536 个输出，均通过。

### Arithmetic Benchmark

下表汇总全部算术测量。每个结果单元格为 **耗时 ms / 每方发送 B**。电路计算包含逐层加法和乘法；Verification 单独列出。总计沿用原始“全部计时阶段”，包含两个输入的 MAC、公开展开、电路计算与 Verification；其中 SPDZwise 输入 MAC 另发送 16 B，公开展开不通信。初始化等排除项见上文。ABY3 的 Verification 为 0，表示没有该检查阶段。

| 机器 | 协议 | 电路 | Domain | 电路计算（ms / B） | Verification（ms / B） | 总计（ms / B） |
|---|---|---|---|---:|---:|---:|
| Laptop | ABY3-F61 | C1 | F61 | 27.642 / 8,388,608 | 0.000 / 0 | 34.360 / 8,388,608 |
| Laptop | SPDZwise-F61 | C1 | F61 | 64.085 / 16,777,216 | 11.099 / 64 | 109.972 / 16,777,296 |
| Laptop | ABY3-F61 | C2 | F61 | 24.572 / 8,388,608 | 0.000 / 0 | 24.903 / 8,388,608 |
| Laptop | SPDZwise-F61 | C2 | F61 | 58.387 / 16,777,216 | 11.495 / 64 | 71.595 / 16,777,296 |
| Laptop | ABY3-F61 | C3 | F61 | 20.224 / 8,388,608 | 0.000 / 0 | 20.310 / 8,388,608 |
| Laptop | SPDZwise-F61 | C3 | F61 | 52.476 / 16,777,216 | 10.794 / 64 | 63.739 / 16,777,296 |
| Server | ABY3-F61 | C1 | F61 | 44.084 / 8,388,608 | 0.000 / 0 | 55.579 / 8,388,608 |
| Server | SPDZwise-F61 | C1 | F61 | 118.092 / 16,777,216 | 17.220 / 64 | 181.233 / 16,777,296 |
| Server | ABY3-F61 | C2 | F61 | 68.427 / 8,388,608 | 0.000 / 0 | 69.401 / 8,388,608 |
| Server | SPDZwise-F61 | C2 | F61 | 97.420 / 16,777,216 | 25.447 / 64 | 125.854 / 16,777,296 |
| Server | ABY3-F61 | C3 | F61 | 59.536 / 8,388,608 | 0.000 / 0 | 59.762 / 8,388,608 |
| Server | SPDZwise-F61 | C3 | F61 | 80.255 / 16,777,216 | 27.426 / 64 | 109.635 / 16,777,296 |
| Server | ABY3-ring64 | C2 | Ring64 | 39.344 / 8,388,608 | 0.000 / 0 | 40.096 / 8,388,608 |

原始数据与基本参数分别保存在 [Laptop F61](src/libspu/mpc/spdzwise/benchmarks/macbook-air-f61-2026-10-19/)、[Server F61](src/libspu/mpc/spdzwise/benchmarks/ubuntu-f61-2026-10-19/) 和 [Server Ring64](src/libspu/mpc/spdzwise/benchmarks/ubuntu-aby3-ring64-2026-10-20/)。各目录保留逐方逐轮数据、乘法内核及输入 MAC / 展开耗时、统计结果和原始环境/源码指纹。两台机器的架构、线程数及调度环境不同，不能仅按表中耗时作单一硬件因素归因。

### Binary Benchmark

最新测量：2026-10-20。同一构建产物、相同 C2 输入、三方 FactoryMem，均为 `YACL_NUM_THREADS=12`；W=65,536、D=16，共 1,048,576 个 64-bit packed AND。ABY3 与 ABFL 依次运行，各预热 1 次、测量 7 次，每轮各阶段取三方最大值，再取七轮中位数。

| 机器 | 协议 | 电路 | Domain | 电路计算（ms / B） | Verification（ms / B） | 总计（ms / B） |
|---|---|---|---|---:|---:|---:|
| Server | ABY3 Boolean | C2（AND） | GF(2)，64-bit 打包 | 23.340 / 8,388,608 | 0.000 / 0 | 23.340 / 8,388,608 |
| Server | SPDZwise-ABFL | C2（AND） | GF(2)，64-bit 打包 | 27.966 / 8,388,608 | 400.363 / 218,110,266 | 425.179 / 226,498,874 |

通信量为每方发送的应用 payload。总耗时先逐轮相加后取中位数，不等于阶段中位数之和。ABY3 不保存 triple、没有最终验证；ABFL online 包含快照保存，verification 包含随机池生成及全部检查。输入分享、初始化、导入检查、输出导出及重构核对不计时。两种协议各 8 轮均核对全部 65,536 个输出。

[最新 Boolean 实验及复现命令](src/libspu/mpc/spdzwise/benchmarks/boolean-c2-latest/RESULTS.md)包含 [ABY3 原始数据](src/libspu/mpc/spdzwise/benchmarks/boolean-c2-latest/aby3/)、[ABFL 原始数据](src/libspu/mpc/spdzwise/benchmarks/boolean-c2-latest/abfl/)及统一参数、源码/二进制指纹和资源记录。仅保留本次最新 Boolean 性能实验；旧版本与中间 profiling/packing 实验已清理。共享机器负载未隔离，本机结果不代表 LAN/WAN。

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

实验源码、Bazel 构建目标、数据和基本参数统一位于 [`src/libspu/mpc/spdzwise/benchmarks`](src/libspu/mpc/spdzwise/benchmarks/)。简要构建与运行入口见该目录的 [README](src/libspu/mpc/spdzwise/benchmarks/README.md)；重新测量使用新输出目录；Boolean 目录只保留最新实验，算术历史记录保留。
