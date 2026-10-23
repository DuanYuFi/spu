# Malicious PPML 开发进展

更新日期：2026-10-23。当前实现是三方 **SPDZwise FIELD/RING 算术原型**，尚未完成恶意安全 PPML 后端验收。`SPDZWISE` 已接入 C++/protobuf/Python 配置和 IO，生产协议工厂仍显式拒绝执行；单元测试和 benchmark 通过 C++ `registerArithmeticPrototype()` 入口运行。

## 目前代码进展

- **域与分享**：`p=2^61-1`，FM64 存储；秘密数采用中心化有符号解释，默认 16 位小数。实现域运算、输入范围检查、带 MAC 的内部分享及客户端输入/输出类型。
- **配置与路由**：对外统一使用 `SPDZWISE`，通过 `spdzwise_config.domain` 选择 `FIELD` 或 `RING`，两者都只允许 `FM64`。默认 `FIELD` 对应 Mersenne 素数域 `p=2^61-1`；`RING` 已有独立注册和 IO 路径，内部为 `Z/(2^128)`，应用/输出位宽为 64，不回退到 field。FIELD 与 RING 乘法内核分别位于 `spdzwise::field`、`spdzwise::ring` 命名空间，算术、ABFL Boolean 和 conversion 预留入口已分文件存放。
- **算术协议**：输入副本一致性检查、输入认证、加法/取负/公开乘法、秘密乘法和矩阵乘法；输入、结果进入认证队列后检查。内部 MAC 类型禁止序列化，只有检查完成并提交的客户端输出可以导出。
- **RING 精确截断**：`TruncASelector` 默认使用 `TruncNaive`，执行 A2B → 本地符号扩展右移 → B2A，精确输出 `floor(x/2^m)`；支持完整 int64 范围、`m=0..63`，输出保留 MAC，全部转换 AND 在 B2A 打开前统一验证。
- **RING 概率截断**：实现 MP-SPDZ `TruncPrRing(signed=True,k=63)`，支持符号未知、正负混合的认证输入 `-2^62 <= x < 2^62`；用公开偏移处理符号，输出保留 MAC。保留为显式 `TruncPrRandomBit` 路径，不再是默认方案；该概率算法仍不支持满 64 位输入。FIELD 截断未实现。
- **随机性与验证**：key、乘法掩码、秘密检查系数使用独立的双边 AES-CTR PRSS 流，F61 采用拒绝采样。verification 直接从队列读取记录，合并 value/MAC 点积归约及交换；每条记录本地乘法由 6 次降为 4 次，单次非空检查每方发送 5 次、共 64 B 应用 payload。
- **框架集成**：`ProtocolSession` 提供导入、最终检查、暂存输出、提交和失败后禁用导出；device/Python 路径增加相应 hook，HAL 增加协议比较分派。现有协议未安装 session 时沿用原路径。device/Python 完整集成仍未验收。
- **正确性覆盖**：C++ 协议、IO、会话和 HAL 分派测试；Python 使用真实 C++ 域头文件与大整数对照，并测试明文数值规格。篡改用例覆盖不同服务器和大批次不同位置的 MAC、失败后的会话中止和输出阻断。

ABFL/post-sacrifice 的 C++ Boolean 原型已实现：三方 replicated XOR sharing、ABY3 online AND、64-bit word 分桶以及最终验证后导出，详见 [Boolean 原型说明](src/libspu/mpc/spdzwise/README.md#abfl-boolean-development-prototype)。生产 factory 仍未启用。

RING/FM64 已实现 split-3 A2B：仅转换 authenticated share 的 value 低 64 位，采用 CSA + bit-sliced RCA，所有秘密 AND 纳入会话最终 ABFL 验证；不转换 MAC share。技术与论文依据见 [SPDZwise README](src/libspu/mpc/spdzwise/README.md#ring-split-3-a2b)，测量见下方 Building Blocks Benchmark。

当前缺口：ABFL 的安全界限及组合审查、恶意安全 daBits、FIELD A2B/B2A、FIELD 截断、FIELD 比较及非线性算子尚未完成；Python 明文规格仅为正确性参照。LR、树、CNN 等 PPML 模型未运行；客户端完整旧输出包重放防护、真实部署、协议组合安全论证及总失败概率仍待完成。域算术测试通过不等于这些验收已完成。

实现位于 [`src/libspu/mpc/spdzwise`](src/libspu/mpc/spdzwise/)，协议参考映射和数值约定见其 [README](src/libspu/mpc/spdzwise/README.md)。

<!-- ### Framework Design

Arithmetic multiplication: Spdzwise Ring
Binary AND: ABFL
A2B: split 3
Bit2A: dabit
B2A: split 3
TruncPr: exact truncation (A2B, local truncation, B2A)
MSBA2B: split 3
MulA1B: dabit Bit2A -> MulAA

-->

### Original SPU Modification

以下以 SPU 基线 `826537d90d60c1a94286852c5ba13c8eb470a785` 为参照，记录通用框架文件的修改位置及目的，包含新增的会话接口和本次 domain 配置接入。路径均相对仓库根目录；`src/libspu/mpc/spdzwise/` 内的协议算法和路由实现另见下文。

| 框架文件（相对仓库根目录）                                                                                                                                                                                                        | 修改位置与内容                                                                                                                                                                        | 目的                                                                                                                                                        |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`src/libspu/spu.h`](src/libspu/spu.h)、[`src/libspu/spu.proto`](src/libspu/spu.proto)                                                                                                                                            | `ProtocolKind` 增加 `SPDZWISE = 7`；新增 `SpdzWiseConfig` 及 `RuntimeConfig.spdzwise_config`（protobuf 字段号 74）。`Domain` 为 `FIELD = 0`、`RING = 1`，默认 `FIELD`。               | 在统一协议名下区分算术域，兼容未设置 domain 的旧 F61 配置；`RuntimeConfig.field` 继续表示字宽，不承担 field/ring 选择。                                     |
| [`src/libspu/spu.cc`](src/libspu/spu.cc)                                                                                                                                                                                          | `convertFromPB()`、`convertToPB()` 读写 domain；`DumpToString()` 打印协议和 domain。现有 `ParseFromJsonString()`、`ParseFromString()`、`SerializeAsString()` 通过这些转换支持新配置。 | 使 JSON、protobuf、C++ 配置及诊断输出保持一致；执行会话比较序列化配置时也会包含 domain。                                                                    |
| [`src/libspu/core/config.cc`](src/libspu/core/config.cc)                                                                                                                                                                          | `populateRuntimeConfig()` 对 SPDZWISE 校验 domain 只能为 `FIELD` 或 `RING`，且字宽只能为 `FM64`；未指定小数位时仍默认 16 位。                                                         | 在创建上下文时拒绝不支持的配置，固定当前两条路径的字宽，并保留 F61 原型的默认定点精度。生产执行仍由协议工厂门禁控制。                                       |
| [`spu/api.py`](spu/api.py)                                                                                                                                                                                                        | `parse_protocol_kind()` 识别字符串 `SPDZWISE`。                                                                                                                                       | 让使用字符串配置的 Python 调用方能够选择统一的协议标识。                                                                                                    |
| [`spu/libspu.cc`](spu/libspu.cc)、[`spu/libspu.pyi`](spu/libspu.pyi)、[`spu/__init__.py`](spu/__init__.py)                                                                                                                        | 增加 `SPDZWISE` 枚举绑定；绑定并声明 `SpdzWiseConfig`、嵌套 `Domain` 和 `RuntimeConfig.spdzwise_config` 属性；包级导出 `SpdzWiseConfig`。                                             | 支持 Python 设置 `config.spdzwise_config.domain = spu.SpdzWiseConfig.FIELD` 或 `RING`，并提供对应的类型提示与配置序列化能力。                               |
| [`src/libspu/mpc/factory.cc`](src/libspu/mpc/factory.cc)、[`src/libspu/mpc/BUILD.bazel`](src/libspu/mpc/BUILD.bazel)                                                                                                              | `CreateIO()` 的 SPDZWISE 分支将完整 `RuntimeConfig` 传给 `makeSpdzWiseIo()`，构建目标增加协议 IO 依赖；`RegisterProtocol()` 的 SPDZWISE 执行分支仍显式抛错。                          | IO 工厂必须读取 domain，避免仅按 `FM64` 将 ring 配置误接入 F61 分享与重构；同时在混合协议及安全验收完成前保留生产执行门禁。                                 |
| [`src/libspu/core/object.h`](src/libspu/core/object.h)                                                                                                                                                                            | `Object` 新增 `hasState()`。                                                                                                                                                          | 允许框架先判断协议是否安装 `ProtocolSession`，使会话 hook 按需启用；未安装会话的现有协议沿用原路径。                                                        |
| [`src/libspu/core/context.h`](src/libspu/core/context.h)                                                                                                                                                                          | 新增零参数 `bindParams(KernelEvalContext*)` 重载。                                                                                                                                    | 支持通过 `dynDispatch` 调用无参数的 `finalize_protocol` 最终检查内核。                                                                                      |
| 新增 [`src/libspu/mpc/protocol_session.h`](src/libspu/mpc/protocol_session.h)、[`protocol_session.cc`](src/libspu/mpc/protocol_session.cc)；修改 [`src/libspu/mpc/BUILD.bazel`](src/libspu/mpc/BUILD.bazel)                       | 提供 `ProtocolSession`、`getProtocolSession()` 和 `ProtocolRunGuard`；实现 `begin()`、`prepareOutputs()`、`commit()`、`abort()`、已提交导出视图及导出失效操作，并增加库目标。         | 将检查、暂存导出、提交和失败处理纳入可复用的框架接口；`prepareOutputs()` 必须调用最终检查，即使输出列表为空；异常中止会话并清除导出。                       |
| [`src/libspu/device/api.cc`](src/libspu/device/api.cc)、[`src/libspu/device/BUILD.bazel`](src/libspu/device/BUILD.bazel)                                                                                                          | `executeImpl()` 执行前开启会话，通过已有 `mpc::import_s()` 接口导入秘密输入；发布输出前调用 `prepareOutputs()`；成功后通过 guard 提交，异常时中止；补充会话依赖。                     | 将认证、延迟检查和输出提交接入实际执行生命周期，防止未验证或失败执行的结果被导出。`mpc/api.h/.cc` 中的 import/export 分派接口复用基线已有实现，未为此修改。 |
| [`spu/libspu.cc`](spu/libspu.cc)、[`spu/BUILD.bazel`](spu/BUILD.bazel)                                                                                                                                                            | `SetVar()` 检查会话及外部输入；`GetVar()`、`GetVarMeta()`、`GetVarChunksCount()` 经 `GetExportValue()` 读取已提交视图；覆盖、删除或清空变量时使相关导出失效；补充会话依赖。           | 避免 Python API 读取含 MAC 的内部值，或在变量更新后返回旧导出；值、元数据和分块访问遵循同一输出提交规则。                                                   |
| [`src/libspu/core/type.h`](src/libspu/core/type.h)、[`src/libspu/core/value.cc`](src/libspu/core/value.cc)                                                                                                                        | 增加默认允许的 `isSerializable()` 类型接口；`Value::toMetaProto()` 检查实部和虚部是否允许序列化，`toProto()` 在复制数据前完成元数据检查。                                             | 允许内部认证类型禁止序列化，防止 MAC 数据经通用 `ValueProto` 路径离开运行时。                                                                               |
| [`src/libspu/device/io.cc`](src/libspu/device/io.cc)                                                                                                                                                                              | `IoClient::combineShares()` 对 SPDZWISE 检查三方输出的 dtype、shape 一致，且不含复数。                                                                                                | 补充分享缓冲区之外的元数据检查，避免将不一致的输出解释为同一结果。                                                                                          |
| [`src/libspu/kernel/hal/ring.cc`](src/libspu/kernel/hal/ring.cc)、[`ring_test.cc`](src/libspu/kernel/hal/ring_test.cc)                                                                                                            | `_less()` 优先分派 `less_ss`、`less_sp`、`less_ps`，保留操作数顺序；测试覆盖专用分派和原有回退路径。                                                                                  | 为域协议预留专用有符号比较入口，避免中心化域边界下直接使用 `msb(x-y)`；SPDZWISE 的实际比较内核仍待实现。                                                    |
| 新增 [`src/libspu/core/config_test.cc`](src/libspu/core/config_test.cc)，修改 [`src/libspu/core/BUILD.bazel`](src/libspu/core/BUILD.bazel)                                                                                        | 新增 `config_test` 目标，覆盖旧配置默认 FIELD、两种 domain 的 protobuf/binary/JSON 往返、诊断输出、FM64 限制及其他协议字宽兼容性。                                                    | 验证 domain 在框架配置链路中不丢失，非法配置被拒绝，现有协议配置不受 SPDZWISE 专有限制影响。                                                                |
| 新增 [`src/libspu/mpc/protocol_session_test.cc`](src/libspu/mpc/protocol_session_test.cc)，修改 [`src/libspu/mpc/BUILD.bazel`](src/libspu/mpc/BUILD.bazel)；修改 [`spu/tests/spu_runtime_test.py`](spu/tests/spu_runtime_test.py) | 增加通用会话生命周期测试目标，以及 Python domain 配置、拷贝和序列化测试。                                                                                                             | 覆盖最终检查、提交和失败后禁止导出等框架约束，并为 Python 配置绑定提供回归用例。C++ 测试已通过；完整 Python 扩展集成测试尚未运行。                          |

开发原型的算术路由本体位于 [`src/libspu/mpc/spdzwise/protocol.cc`](src/libspu/mpc/spdzwise/protocol.cc)：`registerArithmeticPrototype()` 按 domain 调用 `registerFieldArithmeticPrototype()` 或 `registerRingArithmeticPrototype()`。field 分支注册 [`arithmetic.h/.cc`](src/libspu/mpc/spdzwise/arithmetic.cc) 中的 `field::MulAA/MulAP` 等 F61 内核、[`boolean.h/.cc`](src/libspu/mpc/spdzwise/boolean.cc) 中的 ABFL，以及 [`conversion.h/.cc`](src/libspu/mpc/spdzwise/conversion.cc) 中尚未注册实际转换的预留入口；ring 分支注册 `ring.cc` 中的独立环内核。IO 路由位于同目录 [`io.cc`](src/libspu/mpc/spdzwise/io.cc) 的 `makeSpdzWiseIo()`。这些文件属于协议实现，未将 F61/ABFL 算法放入通用框架。

本次配置与文件拆分的验证结果为 7 个 C++ 目标、共 **283 个用例通过**，另通过工厂文件的编译检查；完整记录见 [VALIDATION.md](src/libspu/mpc/spdzwise/VALIDATION.md#domain-routing-and-source-separation--2026-10-20)。生产工厂门禁未因新增配置和路由而开启。

## Benchmark

### Experimental Setup

服务器结果于 **2026-10-21** 全部重测，以下主表使用最终优化版本；Laptop 保留 2026-10-19 历史数据。各协议均以三方线程在同一进程中运行。

| 机器               | 硬件                                                           | 系统与工具链                                                            | 并行配置                                                                     |
| ------------------ | -------------------------------------------------------------- | ----------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| MacBook Air Laptop | Apple M2，8 核，24 GiB 内存                                    | macOS（未记录版本），Apple Clang 16，Bazel 7.7.1，`-c opt`（SPU `-O2`） | 历史配置：YACL 默认 7 个 worker，另有三方调用线程                            |
| Ubuntu Server      | 双路 Xeon Gold 6248R，48 物理核 / 96 逻辑 CPU，约 754 GiB 内存 | Ubuntu 24.04，GCC 13.3.0，Bazel 7.7.1，`-c opt`（SPU `-O2`）            | **固定 64 个后台 worker**，`YACL_NUM_THREADS=65`；三方共享线程池，未绑定 CPU |

YACL 将调用线程计入 `YACL_NUM_THREADS`，线程池大小为该值减一，因此实际 64 worker 对应参数 65。统一脚本固定该参数，并核对启动日志中的线程池大小；小批次可能仅在调用线程执行。所有 benchmark 串行运行，测量时本任务的构建和测试已结束；共享服务器的其他负载未隔离，负载、CPU affinity 和资源记录随原始数据保存。

通信通过 `FactoryMem` 完成，属于本机内存通信，没有跨机链路或人工 RTT；结果不代表 LAN/WAN 性能。通信量统计**每方发送的应用 payload**，不含传输协议开销；三方发送合计为表中数值的 3 倍。

| 协议             | Domain                            | 说明                                                                              |
| ---------------- | --------------------------------- | --------------------------------------------------------------------------------- |
| ABY3-ring64      | `Z/(2^64)`，FM64 存储             | SPU 原生半诚实算术内核，无 MAC 或最终 verification。                              |
| ABY3-F61         | `F_p`，`p=2^61-1`，FM64 存储      | F61 适配版，与 SPDZwise 共用域乘法和 PRSS 底层。                                  |
| SPDZwise-F61     | `F_p`，`p=2^61-1`，FM64 存储      | 正常输入认证并立即检查，乘法结果在电路结束后批量验证。                            |
| SPDZwise-ring128 | 内部 `Z/(2^128)`，输出 `Z/(2^64)` | 应用位宽 64、扩展位宽 64；正常输入认证、最终副本检查、秘密系数聚合及 Protocol 9。 |

**FM64 不表示 F61 按模 `2^64` 运算。**算术实验均为整数电路，不包含定点截断。Binary 每个 uint64 字打包 64 个 GF(2) 比特。

每组预热 1 次、测量 7 次；每轮各阶段取三方最大耗时，再统计七轮中位数。总计先逐轮相加再取中位数，不要求等于各列中位数之和。计时排除初始分享、协议/key/PRSS 初始化、屏障、输出导出/重构及明文核对；内核运行中的 PRSS 计入电路时间。

服务器 SPDZwise FIELD/RING 共用 `ring_circuit_bench`，每轮新 session，输入副本检查和立即认证检查均计入输入阶段。ABY3 沿用 `circuit_bench` 的会话复用方式；其 F61 runner 同时输出旧 SPDZwise 对照行，主表仅取其中 ABY3 行，SPDZwise-F61 统一取正常认证入口的数据。Laptop 仍采用原先的输入延迟检查口径。

### Circuit

固定 `W×D=1,048,576`，每层输出依赖上一层；四种服务器算术协议均运行 C1–C3。

| 电路           |    宽度 W | 深度 D | 算术秘密乘法门数 |
| -------------- | --------: | -----: | ---------------: |
| C1：单层宽电路 | 1,048,576 |      1 |        1,048,576 |
| C2：默认电路   |    65,536 |     16 |        1,048,576 |
| C3：深电路     |    16,384 |     64 |        1,048,576 |

算术电路只分享两个秘密标量 `x=2^60-8`、`y=-13`，随后公开展开，每个乘法门前有一次秘密加法：

```text
z[0,i]   = x + i                 (i = 0,...,W-1)
z[l+1,i] = z[l,i] * (z[l,i] + y) (l = 0,...,D-1)
```

各协议按各自 domain 计算。服务器两个 SPDZwise 版本均先检查两个输入，最终批次包含 `W×D=1,048,576` 条乘法记录；公开展开为局部线性操作。服务器 SPDZwise 共 48 次预热/测量均核对全部输出；ABY3 的六组算术实验各核对最后一轮全部输出。

Binary 测量 C2：预先分享 `D+1=17` 个 uint64 输入向量，每层一次完整向量 `and_bb`，电路无 tiling：

```text
z[0,i]   = x[0,i]
z[l+1,i] = z[l,i] & x[l+1,i]     (l = 0,...,D-1)
```

共 1,048,576 个 uint64 AND，即 67,108,864 个单比特 AND。ABY3 / ABFL 各 8 轮均核对全部 65,536 个输出。

### Arithmetic Benchmark

每个结果为 **耗时 ms / 每方发送 B**。电路计算包含逐层加法、乘法、PRSS、通信及认证记录保存。总计包含输入、公开展开、电路和 Verification。服务器 SPDZwise-F61 输入阶段另发送 96 B，SPDZwise-ring128 另发送 242 B；Laptop F61 历史输入 MAC 为 16 B。公开展开不通信，ABY3 无认证或最终验证。

| 机器   | 协议             | 电路 | Domain           |   电路计算（ms / B） | Verification（ms / B） |       总计（ms / B） |
| ------ | ---------------- | ---- | ---------------- | -------------------: | ---------------------: | -------------------: |
| Laptop | ABY3-F61         | C1   | F61              |   27.642 / 8,388,608 |              0.000 / 0 |   34.360 / 8,388,608 |
| Laptop | SPDZwise-F61     | C1   | F61              |  64.085 / 16,777,216 |            11.099 / 64 | 109.972 / 16,777,296 |
| Laptop | ABY3-F61         | C2   | F61              |   24.572 / 8,388,608 |              0.000 / 0 |   24.903 / 8,388,608 |
| Laptop | SPDZwise-F61     | C2   | F61              |  58.387 / 16,777,216 |            11.495 / 64 |  71.595 / 16,777,296 |
| Laptop | ABY3-F61         | C3   | F61              |   20.224 / 8,388,608 |              0.000 / 0 |   20.310 / 8,388,608 |
| Laptop | SPDZwise-F61     | C3   | F61              |  52.476 / 16,777,216 |            10.794 / 64 |  63.739 / 16,777,296 |
| Server | ABY3-F61         | C1   | F61              |   42.209 / 8,388,608 |              0.000 / 0 |   55.165 / 8,388,608 |
| Server | ABY3-ring64      | C1   | Ring64           |   48.172 / 8,388,608 |              0.000 / 0 |   53.745 / 8,388,608 |
| Server | SPDZwise-F61     | C1   | F61              | 140.952 / 16,777,216 |            20.257 / 64 | 217.920 / 16,777,376 |
| Server | SPDZwise-ring128 | C1   | Ring128 → Ring64 | 262.719 / 33,554,436 |           53.859 / 182 | 369.412 / 33,554,860 |
| Server | ABY3-F61         | C2   | F61              |   67.826 / 8,388,608 |              0.000 / 0 |   68.859 / 8,388,608 |
| Server | ABY3-ring64      | C2   | Ring64           |   42.530 / 8,388,608 |              0.000 / 0 |   43.446 / 8,388,608 |
| Server | SPDZwise-F61     | C2   | F61              | 125.499 / 16,777,216 |            26.801 / 64 | 155.211 / 16,777,376 |
| Server | SPDZwise-ring128 | C2   | Ring128 → Ring64 | 185.650 / 33,554,496 |           56.775 / 182 | 239.663 / 33,554,920 |
| Server | ABY3-F61         | C3   | F61              |   59.369 / 8,388,608 |              0.000 / 0 |   59.565 / 8,388,608 |
| Server | ABY3-ring64      | C3   | Ring64           |   62.459 / 8,388,608 |              0.000 / 0 |   62.767 / 8,388,608 |
| Server | SPDZwise-F61     | C3   | F61              | 122.224 / 16,777,216 |            26.032 / 64 | 147.488 / 16,777,376 |
| Server | SPDZwise-ring128 | C3   | Ring128 → Ring64 | 183.477 / 33,554,688 |           48.112 / 182 | 234.307 / 33,555,112 |

原始数据：[最终完整重测](src/libspu/mpc/spdzwise/benchmarks/server-64worker-2026-10-21/after/)、[相同 64-worker 配置的优化前完整重测](src/libspu/mpc/spdzwise/benchmarks/server-64worker-2026-10-21/before/)、[Laptop 历史数据](src/libspu/mpc/spdzwise/benchmarks/macbook-air-f61-2026-10-19/)。每组保存逐方逐轮 CSV、日志、统计、参数、源码/二进制 SHA256、源码补丁和资源记录。

### Binary Benchmark

最新测量：**2026-10-21**。同一最终构建、相同 C2 输入、三方 FactoryMem，固定 **64 个后台 worker（`YACL_NUM_THREADS=65`）**。ABY3 / ABFL 串行运行，各预热 1 次、测量 7 次。

| 机器   | 协议          | 电路      | Domain             | 电路计算（ms / B） | Verification（ms / B） |        总计（ms / B） |
| ------ | ------------- | --------- | ------------------ | -----------------: | ---------------------: | --------------------: |
| Server | ABY3 Boolean  | C2（AND） | GF(2)，64-bit 打包 | 32.463 / 8,388,608 |              0.000 / 0 |    32.463 / 8,388,608 |
| Server | SPDZwise-ABFL | C2（AND） | GF(2)，64-bit 打包 | 25.596 / 8,388,608 |  418.973 / 218,110,266 | 452.416 / 226,498,874 |

ABY3 不保存 triple、没有最终验证；ABFL online 包含快照保存，verification 包含随机 triple 生成、副本检查、commit–reveal、cut-and-choose 和 sacrifice。输入分享、初始化、导入检查、输出导出及重构核对不计时。ABFL verification 使用 65,536-word 工作块；两种协议每次均核对所有输出与预期通信量。

[统一实验参数](src/libspu/mpc/spdzwise/benchmarks/server-64worker-2026-10-21/after/parameters.json)、[全部原始数据](src/libspu/mpc/spdzwise/benchmarks/server-64worker-2026-10-21/after/)及[复现入口](src/libspu/mpc/spdzwise/benchmarks/README.md)。旧的 12-thread Boolean 和 Ring 结果保留在历史目录，不再用于当前服务器主表。

### Building Blocks Benchmark

#### RING Split-3 A2B Benchmark

测量日期：**2026-10-22**。相同的 **100,000 个完整 64 位输入**，三方 FactoryMem、
64 个后台 worker（`YACL_NUM_THREADS=65`）；各组串行运行，预热一次、正式测量七次，
每轮新建协议上下文。每轮取三方完整计时区间的最大值，再统计中位数及范围。

SPDZwise 在计时前完成输入认证；主计时包含转换和最后**一次完整 ABFL 验证**，
验证阶段包含现场随机材料生成。原生 ABY3 为**半诚实 baseline**，使用正常 IO、正常注册的
`aby3::A2B`，保留 resharing、内部 PRSS 和默认加法器，没有恶意验证阶段。
双方均排除初始化、输入分享、输出重构及明文核对；包括预热在内，每轮核对全部输出和副本一致性。

候选选型阶段（只在三个 SPDZwise 候选之间选赢家）：

| 协议 / 电路                            | AND 工作量 / 输入（bit） | 在线 AND 层数 | 总耗时（ms） | 转换（ms） | 验证（ms） | 三方总发送（B） |
| -------------------------------------- | -----------------------: | ------------: | -----------: | ---------: | ---------: | --------------: |
| SPDZwise CSA + bit-sliced RCA          |                      125 |            63 |      144.831 |     15.967 |    130.408 |     187,567,008 |
| SPDZwise CSA + bit-sliced Brent–Kung   |                      291 |            12 |      305.753 |     23.991 |    282.674 |     436,652,832 |
| SPDZwise CSA + word-packed Kogge–Stone |                      832 |             8 |      625.534 |     76.555 |    555.411 |     842,423,520 |
| 原生 ABY3 A2B                          |                        — |             — |      142.011 |    142.011 |      0.000 |      33,600,000 |

RCA 相对第二名的总耗时优势超过 3%，未触发追加十四轮；已删除其余候选实现和选择开关。
832 是 13 个完整 64-bit AND word 的执行工作量，含无效位，不能当作最小电路门数。
bit-sliced 路径的尾块补齐到 64 lanes 并参与验证，实际记录为 `125 * ceil(N/64)` 个 word。

清理后最终版本复测：

| 协议               | 总耗时中位数（ms） |      范围（ms） | 转换（ms） | 验证（ms） | 峰值 RSS（MiB） |
| ------------------ | -----------------: | --------------: | ---------: | ---------: | --------------: |
| SPDZwise CSA + RCA |            134.182 | 129.609–144.828 |     12.950 |    121.232 |           431.4 |
| 原生 ABY3 A2B      |            146.386 | 125.306–149.766 |    146.385 |      0.000 |           105.7 |

| 协议               | 转换发送（B） | 验证发送（B） | 总发送（B） | 总发送消息数 |
| ------------------ | ------------: | ------------: | ----------: | -----------: |
| SPDZwise CSA + RCA |     4,689,000 |   182,878,008 | 187,567,008 |          513 |
| 原生 ABY3 A2B      |    33,600,000 |             0 |  33,600,000 |           24 |

通信量和消息数均为三方合计；RSS 为承载三方的整个进程峰值，包含输入准备。
分阶段耗时独立聚合，不能直接相加重建主指标。SPDZwise / ABY3 的总耗时比为
**0.917×**，转换阶段耗时 / ABY3 完整 A2B 耗时为 **0.088×**，完整任务通信量比为 **5.582×**。
ABFL 验证约占 SPDZwise 总耗时的 90%，没有省略验证。

这些结果只代表当前服务器的本机内存通信及 100K 批量。RCA 减少 AND 工作量，但增加依赖轮数；
原生 ABY3 使用不同电路，因此该结果不能解释为相同电路下 ABFL 的 AND 比 ABY3 更快，
也不证明高延迟网络中的优劣。CSA、单 AND 全加器和 RCA 转换已有文献先例，本次为实现与选型优化。

[技术细节与相关论文](src/libspu/mpc/spdzwise/README.md#ring-split-3-a2b)、
[完整选型报告与复现命令](src/libspu/mpc/spdzwise/benchmarks/a2b-split3-100k-2026-10-22/RESULTS.md)、
[候选原始数据及源码指纹](src/libspu/mpc/spdzwise/benchmarks/a2b-split3-100k-2026-10-22/candidates/)、
[最终参数、原始数据和回归记录](src/libspu/mpc/spdzwise/benchmarks/a2b-split3-100k-2026-10-22/final/)。

#### RING Split-3 Masked B2A Benchmark

测量日期：**2026-10-22**。相同的 **100,000 个完整 64 位输入**，三方 FactoryMem、
64 个后台 worker（`YACL_NUM_THREADS=65`）；双方串行运行，预热一次、正式测量七次，
每轮新建上下文并准备新鲜 Boolean RSS。每轮取三方完整计时区间最大值，再统计中位数及范围。

SPDZwise 使用认证随机算术掩码 `r`，经 split-3 A2B 获得 Boolean 掩码，
用 bit-sliced RCA 计算 `x+r mod 2^64`；完成算术检查和**一次完整 ABFL 检查后**打开结果 `c`，
返回保留 MAC 的算术共享 `c-r`。主计时包含掩码及 MAC 生成、转置、两个转换/加法电路、
全部内部验证、打开及算术减法。原生 ABY3 为**半诚实 baseline**，调用正常注册的
`B2ASelector`，完整 64 位输入走其 PPA 路线。

双方均从 Boolean shares 开始，排除上下文初始化、外部输入准备/导入、输出导出、重构和明文核对。
SPDZwise 转换内部的掩码 A2B 与认证成本均计入；导出不重复执行非空 ABFL 验证。

| 协议                        | 总耗时中位数（ms） |      范围（ms） | 验证外阶段（ms） | 验证（ms） | 三方总发送（B） | 峰值 RSS（MiB） |
| --------------------------- | -----------------: | --------------: | ---------------: | ---------: | --------------: | --------------: |
| SPDZwise split-3 masked B2A |            250.632 | 232.225–260.404 |           61.325 |    192.303 |     296,500,854 |           447.7 |
| 原生 ABY3 B2A               |            136.924 | 123.232–166.031 |          136.924 |      0.000 |      36,800,000 |           119.1 |

SPDZwise 阶段明细：

| 阶段                             | 耗时中位数（ms） | 三方总发送（B） |
| -------------------------------- | ---------------: | --------------: |
| 随机掩码及 MAC 生成              |           13.067 |       4,800,012 |
| split-3 A2B、转置与 Boolean 加法 |           20.936 |       7,052,256 |
| 算术检查与一次 ABFL 验证         |          192.303 |     275,048,394 |
| 检查式打开与算术减法             |           31.739 |       9,600,192 |

SPDZwise / ABY3 的完整耗时比为 **1.830×**，完整通信量比为 **8.057×**。
验证是主要耗时。CSV 中的 `online_ms` 表示总耗时减去验证，包含随机材料生成，
不能解释为不含预处理的在线成本。阶段中位数独立聚合，不能相加重建主指标；
RSS 是承载三方的整个进程峰值，包含输入准备。结果仅代表本机内存通信及该批量。

电路每个输入使用 188 个 AND（掩码 A2B 为 125，后续 RCA 为 63），共 126 个在线 AND 层；
尾块补齐到 64 lanes 并验证，每方每轮记录 **18,806,016 个 AND bits**。
包括预热在内，每种协议累计核对 **800,000 个输出**；六个回归目标共 **67 个用例通过**，
覆盖窄位宽零扩展、非连续视图、MAC 关系、往返与后续算术计算，以及三方位置的 MAC/AND
篡改、填充位篡改和打开消息不一致。算术或 ABFL 检查失败时不会发送 `b2a-open`。

接口支持 1–64 位输入，窄输入零扩展；当前独立非空 B2A 至少需要 321 个元素，
或利用会话中已有的真实 AND 记录满足 ABFL 的 1024 packed-word 检查阈值。生产 factory 仍未启用。

[算法及输入契约](src/libspu/mpc/spdzwise/README.md#ring-split-3-masked-b2a)、
[完整实验报告](src/libspu/mpc/spdzwise/benchmarks/b2a-split3-100k-2026-10-22/RESULTS.md)、
[参数、原始数据与源码指纹](src/libspu/mpc/spdzwise/benchmarks/b2a-split3-100k-2026-10-22/)、
[回归验证记录](src/libspu/mpc/spdzwise/benchmarks/b2a-split3-100k-2026-10-22/validation/)。
复现：在 `src/libspu/mpc/spdzwise/benchmarks` 下执行
`bash build.sh //:conversion_bench`，再执行
`python3 run_conversion.py NEW_OUTPUT_DIRECTORY --operation b2a`。

#### RING MSB A2B / Comparison Benchmark

测量日期：**2026-10-22**，仅保留最后一次包含安全检查的完整实验。
双方调用同一 HAL `i_less`：相同的 **100,000 对 `DT_I64`/FM64 算术秘密份额输入**，
输出同形状的 **`DT_I1` 一位 Boolean 秘密份额**。输入值取自
`[-2^62, 2^62-1]`，确保 `msb_a2b(x-y)` 与有符号 `<` 一致。

三方 FactoryMem、64 个后台 worker（`YACL_NUM_THREADS=65`），两协议串行运行；
预热一次、正式测量七次，每轮使用新上下文和原生输入份额。每轮取三方完整计时
区间的最大值，再统计中位数和范围。

主计时从输入准备/认证完成后开始，到可交付的 Boolean 份额为止，包含减法、MSB、
正常输出准备。SPDZwise 的 `checkMac()`、`checkBoolean()`、验证用随机材料生成、
全部验证通信及检查后的导出/提交**均在计时内**。输入分享/认证、上下文及 PRSS
初始化、明文重构和正确性核对不计时。ABY3 使用原生半诚实实现。

| 协议          | 总耗时中位数（ms） |      范围（ms） | 电路（ms） | 最终验证及输出（ms） | 三方总发送（B） |
| ------------- | -----------------: | --------------: | ---------: | -------------------: | --------------: |
| SPDZwise RING |        **273.963** | 261.656–305.249 |     23.430 |              250.763 | **361,627,200** |
| 原生 ABY3     |        **193.877** | 189.551–199.582 |    193.875 |                0.002 |  **10,800,000** |

SPDZwise / ABY3 的总耗时比为 **1.413×**（约慢 **41.3%**），通信量比为 **33.484×**。
SPDZwise 的最终验证及输出阶段三方发送 **352,586,808 B**；电路阶段发送 **9,040,392 B**。
分阶段耗时独立聚合，不能相加重建总耗时中位数。

每方每轮调用一次 `checkMac` 和一次 `checkBoolean`。输入已认证且减法为线性运算，
没有新增非空算术 MAC 批次，但 `checkMac` 的批次计数/状态同步仍计时；
`checkBoolean` 完整验证每方 **24,107,712 个补齐后的 AND bits**。
含预热在内，两协议各核对 **800,000 个输出**，全部通过；协议实现的六个回归目标
共 **77 个用例通过**。该结果比较同一 HAL 输入/输出契约下的实现，内部电路和布局
不同，不能将全部耗时差异归因于安全模型；测量环境为本机内存通信。

[完整实验报告](src/libspu/mpc/spdzwise/benchmarks/comparison-msb-checked-100k-2026-10-22/RESULTS.md)、
[参数、原始数据与源码指纹](src/libspu/mpc/spdzwise/benchmarks/comparison-msb-checked-100k-2026-10-22/)、
[验证记录](src/libspu/mpc/spdzwise/benchmarks/comparison-msb-checked-100k-2026-10-22/validation/)。
复现：在 `src/libspu/mpc/spdzwise/benchmarks` 下执行
`bash build.sh //:comparison_bench`，再执行 `python3 run_comparison.py NEW_OUTPUT_DIRECTORY`。

#### Truncation Benchmark

统一配置：**100,000 个正负混合 FM64 输入**，`|x|<2^62`，截断 **16 位**；
三方 FactoryMem，64 个后台 worker（`YACL_NUM_THREADS=65`），预热 1 次、正式测量
7 次，每轮新上下文，各组串行运行。时间取每轮三方最大值的中位数；通信量为三方总发送字节。
**Exact = Yes 表示逐元素精确等于 `floor(x/2^16)`，包括负数。**

| Protocol                                       |  Exact  |     Total Time |       Total Comm. | Verification Time |
| ---------------------------------------------- | :-----: | -------------: | ----------------: | ----------------: |
| SPDZwise TruncNaive（默认，2026-10-23）        | **Yes** | **358.779 ms** | **484,065,462 B** |    **299.166 ms** |
| ABY3 exact（A2B → ARShiftB → B2A，2026-10-23） | **Yes** | **282.853 ms** |  **70,400,000 B** |          **0 ms** |
| ABY3 TruncAPr2（2026-10-23）                   |   No    |       8.992 ms |       2,800,000 B |              0 ms |

`TruncNaive` 是**精确截断**：A2B → 本地符号扩展右移 → B2A，当前为 `TruncASelector`
默认方案，支持完整 int64 范围、0–63 位截断并保留 MAC。其 A2B 和 B2A 的全部 AND
在 B2A 打开前统一验证，每方每轮覆盖 31,310,016 个含填充位的 AND bits。

2026-10-23 的 SPDZwise **Total Time/Total Comm. 包含现场随机材料生成、全部必要
MAC/ABFL 验证和掩码打开**；Verification Time 是其中的验证部分，不能再次加到总时间。
输入认证、调用后的空检查、输出导出和明文重构不计时。ABY3 为半诚实协议，没有恶意验证阶段。
分阶段中位数独立统计，不能相加重建主指标。

结果仅代表本机内存通信及 100K 批量。

`ENABLE_PRECISE_ABY3_TRUNCPR` 在当前源码中已开启，但注册的是概率舍入的
`TruncAPr2`，仍有 LSB 舍入误差，不能标为 Exact。表中的 **ABY3 exact** 使用正常注册的
A2B → ARShiftB(16) → B2ASelector，实现相同的精确 floor 语义；未修改 ABY3 默认
`trunc_a`。主计时包含完整转换及内部随机性，输出返回算术 shares；范围为
**253.501–298.052 ms**，三方进程峰值 RSS 为 **131.9 MiB**。
本次追加实验同样核对 **800,000 个输出**，全部精确正确。

2026-10-23 保留的 Naive / ABY3 实验各核对 **800,000 个输出**，Naive 全部严格等于 floor；
九个回归目标共 **323 个用例通过**，覆盖完整 int64、所有位移、非连续输入、尾块、
MAC 关系、后续算术及打开前故障阻断。

[ABY3 精确截断完整结果及原始数据](src/libspu/mpc/spdzwise/benchmarks/truncation-aby3-exact-100k-2026-10-23/RESULTS.md)、
[Naive / ABY3 完整结果及原始数据](src/libspu/mpc/spdzwise/benchmarks/truncation-naive-100k-2026-10-23/RESULTS.md)、
[回归验证](src/libspu/mpc/spdzwise/benchmarks/truncation-naive-100k-2026-10-23/validation/)、
[历史 ABY3 结果](src/libspu/mpc/spdzwise/benchmarks/truncation-signed-100k-optimized-2026-10-22/RESULTS.md)、
[精确截断算法契约](src/libspu/mpc/spdzwise/README.md#signed-ring-exact-truncation-default)。
复现：在 `src/libspu/mpc/spdzwise/benchmarks` 执行 `bash build.sh //:truncation_bench`，
再执行 `python3 run_truncation.py NEW_OUTPUT_DIRECTORY`；单独复现 ABY3 精确截断时添加
`--protocols aby3-exact`。

#### RING daBit MulA1B Benchmark

测量日期：**2026-10-23**。仅保留最后一次优化版 SPDZwise 测量与最新 ABY3 测量。
两组使用同一二进制、相同配置和明文，来自串行执行的不同批次；没有重新测量或混合样本。
**1M = 1,000,000 对 FM64 算术 share 与一位 Boolean share**，计算
`y = x*b mod 2^64`，不做截断；输入覆盖全宽随机 uint64、边界值和两种 bit。

三方 FactoryMem、64 个后台 worker（`YACL_NUM_THREADS=65`），关闭 context
inter/intra 并行分片。每轮新 session、新 shares，预热 1 次、正式测量 7 次；
每轮取三方完整耗时最大值，再报告中位数和范围。

| 协议 | 总耗时中位数（ms） | 范围（ms） | 三方总发送（B） | 吞吐量（对/秒） |
|---|---:|---:|---:|---:|
| SPDZwise RING daBit（含验证） | **1,031.242** | 1,011.883–1,083.038 | 321,502,535 | 969,704 |
| 原生 ABY3（半诚实） | **103.164** | 65.967–119.252 | 64,000,000 | 9,693,316 |

SPDZwise 主计时包含**新鲜 daBit 生成、bitness 与复制检查、打开前算术/Boolean
依赖验证、Bit2A 一致性打开、MulAA 及最终 `checkMac()`**。输入分享、算术输入认证
和 Boolean 输入验证在计时外；输出导出、重构和逐项核对也在计时外，并确认没有
遗漏的非空验证。ABY3 为半诚实基线，计入原生 `mul_a1b` 的完整调用成本。

每方每轮计时前完成 1 次非空输入 MAC 验证，计时内完成 2 次非空 MAC 验证；
合法输入没有上游 AND，非空 Boolean 验证批次为 0，必要同步和一致性打开仍计时。
两协议各核对 **8,000,000 个输出**，全部正确；相关 9 个测试目标共 **318 个用例通过**。
分阶段耗时独立聚合，不能相加重建总耗时。结果仅代表本机 FactoryMem 与 1M 批量。

[完整结果与阶段耗时](src/libspu/mpc/spdzwise/benchmarks/mula1b-dabit-1m-final-2026-10-23/RESULTS.md)、
[原始数据、参数与源码/二进制指纹](src/libspu/mpc/spdzwise/benchmarks/mula1b-dabit-1m-final-2026-10-23/)、
[回归验证](src/libspu/mpc/spdzwise/benchmarks/mula1b-dabit-1m-final-2026-10-23/validation/)。
复现：在 `src/libspu/mpc/spdzwise/benchmarks` 下执行 `bash build.sh //:mula1b_bench`，
再执行 `python3 run_mula1b.py NEW_OUTPUT_DIRECTORY`。

### PPML Tasks

Logistic Regression: jax_lr
DNN: stax_nn
CNN: flax_resnet
tree: SML DecisionTreeClassifier

### Statistical

按元素计数，内积按长度展开，三方不重复计数。Mult 不包含 MulA1B、公开常数乘法及截断；MulA1B 单独记录。

| Protocol    | Task                                             |     \# of Mult |   \# of Trunc | \# of A2B | \# of B2A | \# of MSBA2B | \# of MulA1B | \# of AND |
| ----------- | ------------------------------------------------ | -------------: | ------------: | --------: | --------: | -----------: | -----------: | --------: |
| ABY3 / FM64 | flax_resnet（ResNet-50 推理）                    |  4,100,484,160 |    22,416,936 |    26,560 |    79,680 |   11,214,336 |   11,214,336 |   159,360 |
| ABY3 / FM64 | stax_nn（network_a 推理）                        |        118,016 |           266 |         0 |         0 |          256 |          256 |         0 |
| ABY3 / FM64 | stax_nn（完整训练，5 epochs）                    | 76,121,132,960 | 1,023,071,818 |   300,000 |   600,000 |   80,100,000 |  156,600,000 | 1,800,000 |
| ABY3 / FM64 | stax_nn（单 batch，128 样本）                    |     32,485,760 |       436,372 |       128 |       256 |       34,176 |       66,816 |       768 |
| ABY3 / FM64 | jax_lr（完整训练，100 次更新）                   |    216,000,000 |    16,020,100 | 1,000,000 | 2,000,000 |    1,000,000 |            0 | 6,000,000 |
| ABY3 / FM64 | jax_lr（单 batch 更新）                          |      2,160,000 |       160,201 |    10,000 |    20,000 |       10,000 |            0 |    60,000 |
| ABY3 / FM64 | jax_lr（单样本推理）                             |            116 |            16 |         1 |         2 |            1 |            0 |         6 |
| ABY3 / FM64 | SML DecisionTreeClassifier（训练；数值验证失败） |      1,580,159 |     1,482,521 |     1,393 | 2,465,906 |        4,199 |            0 |   162,715 |
| ABY3 / FM64 | SML DecisionTreeClassifier（推理）               |             45 |             2 |         0 |       642 |            0 |            0 |       597 |

AND 按布尔字的张量元素计数（不展开为单比特门），仅计秘密×秘密 `and_bb`，排除公开操作数 `and_bp` 以及 A2B、B2A、MSBA2B 等算子内部的 AND。
ResNet 行的 AND 来自 BatchNorm 倒平方根计算；该行若包含转换内部调用，`and_bb` 共 **146,981,568** 次，各任务分项见统计结果的 `and_detail`。

所有行定点小数位均为 18；推理均按单样本、输入与模型参数均秘密共享统计。

- ResNet-50：输入 `1×3×224×224`，BatchNorm 统计量也秘密共享，输出 logits。
- stax_nn：默认 `network_a / secureml`，输入 `1×28×28×1`，网络为 `784→128→128→10`，输出 logits。原示例在 CPU 上做训练后预测；本表通过统计入口将相同前向网络单独在 SPU 上执行，不计训练和 argmax。[重复测量](examples/python/ml/operation_counts/2026-10-22-aby3-fm64/stax_nn_inference/recheck_2026-10-22/)与初次计数一致。 训练使用实际 MNIST 60,000 样本、5 epochs、batch size 128、Momentum SGD（学习率 0.01、momentum 0.9），实际执行全部 2,345 次更新；按原脚本保留公开初始参数，随后状态秘密共享。总计为首个公开状态 batch + 2,339 个秘密状态 128 样本 batch + 5 个 96 样本末批；单 batch 行取后续秘密状态更新。数据来自本地 Keras MNIST 缓存，顺序可能与 TFDS 存储顺序不同，计数不受影响。
- jax_lr：逻辑回归，100 个特征，推理输出 sigmoid 概率；采用 SPU 默认 Taylor exp（8 次平方）及 2 轮 Goldschmidt。完整训练使用原 `fit_manual_grad(use_cache=False)`：100,000 个样本、10 epochs、每 epoch 10 个 batch、batch size 10,000、学习率 0.1；数据与标签秘密共享，权重按原函数公开零初始化。单 batch 行使用秘密共享的输入权重，七项计数均与完整训练的 1/100 一致。
- DecisionTreeClassifier：原 SML 示例配置，深度 3、199 个二值特征、3 类、15 个树节点；源码固定为 SML 移除前的提交 `84965d67d4737a81ff7ec96d214b927b7804b5a9`。推理行使用 Boolean 输入及秘密共享 float32 树，只计单样本 `predict`。训练行使用原 Iris 实验的 150 个样本，二值特征与标签均秘密共享，完整运行 `fit(sample_weight=None)` 并输出状态 `T/F`；特征二值化和训练后评估不计入。

Stax 完整训练数值对照通过：最终 MNIST 测试准确率为 **96.45%**，CPU 对照为 **96.49%**。训练后的测试评估在 CPU 上进行，不计入表中。[完整训练、每轮评估及 batch 分项](examples/python/ml/operation_counts/2026-10-22-aby3-fm64/stax_nn_train/)、[完整训练复现入口](examples/python/ml/profile_stax_training.py)、[训练汇总与交叉核对](examples/python/ml/summarize_stax_training.py)。

决策树训练的协议执行完整，计数与运行时 profile 一致，但 **18 位小数下训练准确率为 33.33%，CPU 浮点版为 97.33%，数值验证未通过**；20 位小数复测也为 33.33%。已复现浮点节点索引更新的 1 LSB 截断偏差会使精确相等判断失败，另有 18 位下 Gini 除法归一化范围问题。训练行保留实际执行计数，诊断和失败输出见 [训练结果与正确性记录](examples/python/ml/operation_counts/2026-10-22-aby3-fm64/decision_tree_train/validation.json)。

[ResNet 统计结果与 trace](examples/python/ml/flax_resnet/operation_counts/2026-10-22-aby3-fm64/)、[ResNet 复现脚本](examples/python/ml/flax_resnet/profile_operation_counts.py)。
[新增任务统计结果、源码快照与 trace](examples/python/ml/operation_counts/2026-10-22-aby3-fm64/)、[新增任务复现脚本](examples/python/ml/profile_task_operation_counts.py)、[统一计数脚本](examples/python/ml/flax_resnet/summarize_operation_counts.py)。

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

早期验证：协议 9、IO 7、会话 7、HAL 4，共 **27 个 C++ 用例通过**；Python 域测试 1 个用例（20,162 条运算输入）、明文规格 7 个用例通过。测试范围和未完成验收见 [VALIDATION.md](src/libspu/mpc/spdzwise/VALIDATION.md)。

## 运行 benchmark

实验源码、Bazel 构建目标、数据和基本参数统一位于 [`src/libspu/mpc/spdzwise/benchmarks`](src/libspu/mpc/spdzwise/benchmarks/)。简要构建与运行入口见该目录的 [README](src/libspu/mpc/spdzwise/benchmarks/README.md)；重新测量使用新输出目录；服务器当前结果统一使用 64 个后台 worker，优化前、最终版本及历史记录分别保留。
