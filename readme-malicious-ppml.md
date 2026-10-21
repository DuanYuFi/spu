# Malicious PPML 开发进展

更新日期：2026-10-21。当前实现是三方 **SPDZwise FIELD/RING 算术原型**，尚未完成恶意安全 PPML 后端验收。`SPDZWISE` 已接入 C++/protobuf/Python 配置和 IO，生产协议工厂仍显式拒绝执行；单元测试和 benchmark 通过 C++ `registerArithmeticPrototype()` 入口运行。

## 目前代码进展

- **域与分享**：`p=2^61-1`，FM64 存储；秘密数采用中心化有符号解释，默认 16 位小数。实现域运算、输入范围检查、带 MAC 的内部分享及客户端输入/输出类型。
- **配置与路由**：对外统一使用 `SPDZWISE`，通过 `spdzwise_config.domain` 选择 `FIELD` 或 `RING`，两者都只允许 `FM64`。默认 `FIELD` 对应 Mersenne 素数域 `p=2^61-1`；`RING` 已有独立注册和 IO 路径，内部为 `Z/(2^128)`，应用/输出位宽为 64，不回退到 field。FIELD 与 RING 乘法内核分别位于 `spdzwise::field`、`spdzwise::ring` 命名空间，算术、ABFL Boolean 和 conversion 预留入口已分文件存放。
- **算术协议**：输入副本一致性检查、输入认证、加法/取负/公开乘法、秘密乘法和矩阵乘法；输入、结果进入认证队列后检查。内部 MAC 类型禁止序列化，只有检查完成并提交的客户端输出可以导出。
- **随机性与验证**：key、乘法掩码、秘密检查系数使用独立的双边 AES-CTR PRSS 流，F61 采用拒绝采样。verification 直接从队列读取记录，合并 value/MAC 点积归约及交换；每条记录本地乘法由 6 次降为 4 次，单次非空检查每方发送 5 次、共 64 B 应用 payload。
- **框架集成**：`ProtocolSession` 提供导入、最终检查、暂存输出、提交和失败后禁用导出；device/Python 路径增加相应 hook，HAL 增加协议比较分派。现有协议未安装 session 时沿用原路径。device/Python 完整集成仍未验收。
- **正确性覆盖**：C++ 协议、IO、会话和 HAL 分派测试；Python 使用真实 C++ 域头文件与大整数对照，并测试明文数值规格。篡改用例覆盖不同服务器和大批次不同位置的 MAC、失败后的会话中止和输出阻断。

ABFL/post-sacrifice 的 C++ Boolean 原型已实现：三方 replicated XOR sharing、ABY3 online AND、64-bit word 分桶以及最终验证后导出，详见 [Boolean 原型说明](src/libspu/mpc/spdzwise/README.md#abfl-boolean-development-prototype)。生产 factory 仍未启用。

当前缺口：ABFL 的安全界限及组合审查、恶意安全 daBits、密码学 A2B/B2A、精确截断、比较及非线性算子尚未完成；Python 明文规格仅为正确性参照。LR、树、CNN 等 PPML 模型未运行；客户端完整旧输出包重放防护、真实部署、协议组合安全论证及总失败概率仍待完成。域算术测试通过不等于这些验收已完成。

实现位于 [`src/libspu/mpc/spdzwise`](src/libspu/mpc/spdzwise/)，协议参考映射和数值约定见其 [README](src/libspu/mpc/spdzwise/README.md)。

### Original SPU Modification

以下以 SPU 基线 `826537d90d60c1a94286852c5ba13c8eb470a785` 为参照，记录通用框架文件的修改位置及目的，包含新增的会话接口和本次 domain 配置接入。路径均相对仓库根目录；`src/libspu/mpc/spdzwise/` 内的协议算法和路由实现另见下文。

| 框架文件（相对仓库根目录） | 修改位置与内容 | 目的 |
|---|---|---|
| [`src/libspu/spu.h`](src/libspu/spu.h)、[`src/libspu/spu.proto`](src/libspu/spu.proto) | `ProtocolKind` 增加 `SPDZWISE = 7`；新增 `SpdzWiseConfig` 及 `RuntimeConfig.spdzwise_config`（protobuf 字段号 74）。`Domain` 为 `FIELD = 0`、`RING = 1`，默认 `FIELD`。 | 在统一协议名下区分算术域，兼容未设置 domain 的旧 F61 配置；`RuntimeConfig.field` 继续表示字宽，不承担 field/ring 选择。 |
| [`src/libspu/spu.cc`](src/libspu/spu.cc) | `convertFromPB()`、`convertToPB()` 读写 domain；`DumpToString()` 打印协议和 domain。现有 `ParseFromJsonString()`、`ParseFromString()`、`SerializeAsString()` 通过这些转换支持新配置。 | 使 JSON、protobuf、C++ 配置及诊断输出保持一致；执行会话比较序列化配置时也会包含 domain。 |
| [`src/libspu/core/config.cc`](src/libspu/core/config.cc) | `populateRuntimeConfig()` 对 SPDZWISE 校验 domain 只能为 `FIELD` 或 `RING`，且字宽只能为 `FM64`；未指定小数位时仍默认 16 位。 | 在创建上下文时拒绝不支持的配置，固定当前两条路径的字宽，并保留 F61 原型的默认定点精度。生产执行仍由协议工厂门禁控制。 |
| [`spu/api.py`](spu/api.py) | `parse_protocol_kind()` 识别字符串 `SPDZWISE`。 | 让使用字符串配置的 Python 调用方能够选择统一的协议标识。 |
| [`spu/libspu.cc`](spu/libspu.cc)、[`spu/libspu.pyi`](spu/libspu.pyi)、[`spu/__init__.py`](spu/__init__.py) | 增加 `SPDZWISE` 枚举绑定；绑定并声明 `SpdzWiseConfig`、嵌套 `Domain` 和 `RuntimeConfig.spdzwise_config` 属性；包级导出 `SpdzWiseConfig`。 | 支持 Python 设置 `config.spdzwise_config.domain = spu.SpdzWiseConfig.FIELD` 或 `RING`，并提供对应的类型提示与配置序列化能力。 |
| [`src/libspu/mpc/factory.cc`](src/libspu/mpc/factory.cc)、[`src/libspu/mpc/BUILD.bazel`](src/libspu/mpc/BUILD.bazel) | `CreateIO()` 的 SPDZWISE 分支将完整 `RuntimeConfig` 传给 `makeSpdzWiseIo()`，构建目标增加协议 IO 依赖；`RegisterProtocol()` 的 SPDZWISE 执行分支仍显式抛错。 | IO 工厂必须读取 domain，避免仅按 `FM64` 将 ring 配置误接入 F61 分享与重构；同时在混合协议及安全验收完成前保留生产执行门禁。 |
| [`src/libspu/core/object.h`](src/libspu/core/object.h) | `Object` 新增 `hasState()`。 | 允许框架先判断协议是否安装 `ProtocolSession`，使会话 hook 按需启用；未安装会话的现有协议沿用原路径。 |
| [`src/libspu/core/context.h`](src/libspu/core/context.h) | 新增零参数 `bindParams(KernelEvalContext*)` 重载。 | 支持通过 `dynDispatch` 调用无参数的 `finalize_protocol` 最终检查内核。 |
| 新增 [`src/libspu/mpc/protocol_session.h`](src/libspu/mpc/protocol_session.h)、[`protocol_session.cc`](src/libspu/mpc/protocol_session.cc)；修改 [`src/libspu/mpc/BUILD.bazel`](src/libspu/mpc/BUILD.bazel) | 提供 `ProtocolSession`、`getProtocolSession()` 和 `ProtocolRunGuard`；实现 `begin()`、`prepareOutputs()`、`commit()`、`abort()`、已提交导出视图及导出失效操作，并增加库目标。 | 将检查、暂存导出、提交和失败处理纳入可复用的框架接口；`prepareOutputs()` 必须调用最终检查，即使输出列表为空；异常中止会话并清除导出。 |
| [`src/libspu/device/api.cc`](src/libspu/device/api.cc)、[`src/libspu/device/BUILD.bazel`](src/libspu/device/BUILD.bazel) | `executeImpl()` 执行前开启会话，通过已有 `mpc::import_s()` 接口导入秘密输入；发布输出前调用 `prepareOutputs()`；成功后通过 guard 提交，异常时中止；补充会话依赖。 | 将认证、延迟检查和输出提交接入实际执行生命周期，防止未验证或失败执行的结果被导出。`mpc/api.h/.cc` 中的 import/export 分派接口复用基线已有实现，未为此修改。 |
| [`spu/libspu.cc`](spu/libspu.cc)、[`spu/BUILD.bazel`](spu/BUILD.bazel) | `SetVar()` 检查会话及外部输入；`GetVar()`、`GetVarMeta()`、`GetVarChunksCount()` 经 `GetExportValue()` 读取已提交视图；覆盖、删除或清空变量时使相关导出失效；补充会话依赖。 | 避免 Python API 读取含 MAC 的内部值，或在变量更新后返回旧导出；值、元数据和分块访问遵循同一输出提交规则。 |
| [`src/libspu/core/type.h`](src/libspu/core/type.h)、[`src/libspu/core/value.cc`](src/libspu/core/value.cc) | 增加默认允许的 `isSerializable()` 类型接口；`Value::toMetaProto()` 检查实部和虚部是否允许序列化，`toProto()` 在复制数据前完成元数据检查。 | 允许内部认证类型禁止序列化，防止 MAC 数据经通用 `ValueProto` 路径离开运行时。 |
| [`src/libspu/device/io.cc`](src/libspu/device/io.cc) | `IoClient::combineShares()` 对 SPDZWISE 检查三方输出的 dtype、shape 一致，且不含复数。 | 补充分享缓冲区之外的元数据检查，避免将不一致的输出解释为同一结果。 |
| [`src/libspu/kernel/hal/ring.cc`](src/libspu/kernel/hal/ring.cc)、[`ring_test.cc`](src/libspu/kernel/hal/ring_test.cc) | `_less()` 优先分派 `less_ss`、`less_sp`、`less_ps`，保留操作数顺序；测试覆盖专用分派和原有回退路径。 | 为域协议预留专用有符号比较入口，避免中心化域边界下直接使用 `msb(x-y)`；SPDZWISE 的实际比较内核仍待实现。 |
| 新增 [`src/libspu/core/config_test.cc`](src/libspu/core/config_test.cc)，修改 [`src/libspu/core/BUILD.bazel`](src/libspu/core/BUILD.bazel) | 新增 `config_test` 目标，覆盖旧配置默认 FIELD、两种 domain 的 protobuf/binary/JSON 往返、诊断输出、FM64 限制及其他协议字宽兼容性。 | 验证 domain 在框架配置链路中不丢失，非法配置被拒绝，现有协议配置不受 SPDZWISE 专有限制影响。 |
| 新增 [`src/libspu/mpc/protocol_session_test.cc`](src/libspu/mpc/protocol_session_test.cc)，修改 [`src/libspu/mpc/BUILD.bazel`](src/libspu/mpc/BUILD.bazel)；修改 [`spu/tests/spu_runtime_test.py`](spu/tests/spu_runtime_test.py) | 增加通用会话生命周期测试目标，以及 Python domain 配置、拷贝和序列化测试。 | 覆盖最终检查、提交和失败后禁止导出等框架约束，并为 Python 配置绑定提供回归用例。C++ 测试已通过；完整 Python 扩展集成测试尚未运行。 |

开发原型的算术路由本体位于 [`src/libspu/mpc/spdzwise/protocol.cc`](src/libspu/mpc/spdzwise/protocol.cc)：`registerArithmeticPrototype()` 按 domain 调用 `registerFieldArithmeticPrototype()` 或 `registerRingArithmeticPrototype()`。field 分支注册 [`arithmetic.h/.cc`](src/libspu/mpc/spdzwise/arithmetic.cc) 中的 `field::MulAA/MulAP` 等 F61 内核、[`boolean.h/.cc`](src/libspu/mpc/spdzwise/boolean.cc) 中的 ABFL，以及 [`conversion.h/.cc`](src/libspu/mpc/spdzwise/conversion.cc) 中尚未注册实际转换的预留入口；ring 分支注册 `ring.cc` 中的独立环内核。IO 路由位于同目录 [`io.cc`](src/libspu/mpc/spdzwise/io.cc) 的 `makeSpdzWiseIo()`。这些文件属于协议实现，未将 F61/ABFL 算法放入通用框架。

本次配置与文件拆分的验证结果为 7 个 C++ 目标、共 **283 个用例通过**，另通过工厂文件的编译检查；完整记录见 [VALIDATION.md](src/libspu/mpc/spdzwise/VALIDATION.md#domain-routing-and-source-separation--2026-10-20)。生产工厂门禁未因新增配置和路由而开启。

## Benchmark

### Experimental Setup

服务器结果于 **2026-10-21** 全部重测，以下主表使用最终优化版本；Laptop 保留 2026-10-19 历史数据。各协议均以三方线程在同一进程中运行。

| 机器 | 硬件 | 系统与工具链 | 并行配置 |
|---|---|---|---|
| MacBook Air Laptop | Apple M2，8 核，24 GiB 内存 | macOS（未记录版本），Apple Clang 16，Bazel 7.7.1，`-c opt`（SPU `-O2`） | 历史配置：YACL 默认 7 个 worker，另有三方调用线程 |
| Ubuntu Server | 双路 Xeon Gold 6248R，48 物理核 / 96 逻辑 CPU，约 754 GiB 内存 | Ubuntu 24.04，GCC 13.3.0，Bazel 7.7.1，`-c opt`（SPU `-O2`） | **固定 64 个后台 worker**，`YACL_NUM_THREADS=65`；三方共享线程池，未绑定 CPU |

YACL 将调用线程计入 `YACL_NUM_THREADS`，线程池大小为该值减一，因此实际 64 worker 对应参数 65。统一脚本固定该参数，并核对启动日志中的线程池大小；小批次可能仅在调用线程执行。所有 benchmark 串行运行，测量时本任务的构建和测试已结束；共享服务器的其他负载未隔离，负载、CPU affinity 和资源记录随原始数据保存。

通信通过 `FactoryMem` 完成，属于本机内存通信，没有跨机链路或人工 RTT；结果不代表 LAN/WAN 性能。通信量统计**每方发送的应用 payload**，不含传输协议开销；三方发送合计为表中数值的 3 倍。

| 协议 | Domain | 说明 |
|---|---|---|
| ABY3-ring64 | `Z/(2^64)`，FM64 存储 | SPU 原生半诚实算术内核，无 MAC 或最终 verification。 |
| ABY3-F61 | `F_p`，`p=2^61-1`，FM64 存储 | F61 适配版，与 SPDZwise 共用域乘法和 PRSS 底层。 |
| SPDZwise-F61 | `F_p`，`p=2^61-1`，FM64 存储 | 正常输入认证并立即检查，乘法结果在电路结束后批量验证。 |
| SPDZwise-ring128 | 内部 `Z/(2^128)`，输出 `Z/(2^64)` | 应用位宽 64、扩展位宽 64；正常输入认证、最终副本检查、秘密系数聚合及 Protocol 9。 |

**FM64 不表示 F61 按模 `2^64` 运算。**算术实验均为整数电路，不包含定点截断。Binary 每个 uint64 字打包 64 个 GF(2) 比特。

每组预热 1 次、测量 7 次；每轮各阶段取三方最大耗时，再统计七轮中位数。总计先逐轮相加再取中位数，不要求等于各列中位数之和。计时排除初始分享、协议/key/PRSS 初始化、屏障、输出导出/重构及明文核对；内核运行中的 PRSS 计入电路时间。

服务器 SPDZwise FIELD/RING 共用 `ring_circuit_bench`，每轮新 session，输入副本检查和立即认证检查均计入输入阶段。ABY3 沿用 `circuit_bench` 的会话复用方式；其 F61 runner 同时输出旧 SPDZwise 对照行，主表仅取其中 ABY3 行，SPDZwise-F61 统一取正常认证入口的数据。Laptop 仍采用原先的输入延迟检查口径。

### Circuit

固定 `W×D=1,048,576`，每层输出依赖上一层；四种服务器算术协议均运行 C1–C3。

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

各协议按各自 domain 计算。服务器两个 SPDZwise 版本均先检查两个输入，最终批次包含 `W×D=1,048,576` 条乘法记录；公开展开为局部线性操作。服务器 SPDZwise 共 48 次预热/测量均核对全部输出；ABY3 的六组算术实验各核对最后一轮全部输出。

Binary 测量 C2：预先分享 `D+1=17` 个 uint64 输入向量，每层一次完整向量 `and_bb`，电路无 tiling：

```text
z[0,i]   = x[0,i]
z[l+1,i] = z[l,i] & x[l+1,i]     (l = 0,...,D-1)
```

共 1,048,576 个 uint64 AND，即 67,108,864 个单比特 AND。ABY3 / ABFL 各 8 轮均核对全部 65,536 个输出。

### Arithmetic Benchmark

每个结果为 **耗时 ms / 每方发送 B**。电路计算包含逐层加法、乘法、PRSS、通信及认证记录保存。总计包含输入、公开展开、电路和 Verification。服务器 SPDZwise-F61 输入阶段另发送 96 B，SPDZwise-ring128 另发送 242 B；Laptop F61 历史输入 MAC 为 16 B。公开展开不通信，ABY3 无认证或最终验证。

| 机器 | 协议 | 电路 | Domain | 电路计算（ms / B） | Verification（ms / B） | 总计（ms / B） |
|---|---|---|---|---:|---:|---:|
| Laptop | ABY3-F61 | C1 | F61 | 27.642 / 8,388,608 | 0.000 / 0 | 34.360 / 8,388,608 |
| Laptop | SPDZwise-F61 | C1 | F61 | 64.085 / 16,777,216 | 11.099 / 64 | 109.972 / 16,777,296 |
| Laptop | ABY3-F61 | C2 | F61 | 24.572 / 8,388,608 | 0.000 / 0 | 24.903 / 8,388,608 |
| Laptop | SPDZwise-F61 | C2 | F61 | 58.387 / 16,777,216 | 11.495 / 64 | 71.595 / 16,777,296 |
| Laptop | ABY3-F61 | C3 | F61 | 20.224 / 8,388,608 | 0.000 / 0 | 20.310 / 8,388,608 |
| Laptop | SPDZwise-F61 | C3 | F61 | 52.476 / 16,777,216 | 10.794 / 64 | 63.739 / 16,777,296 |
| Server | ABY3-F61 | C1 | F61 | 42.209 / 8,388,608 | 0.000 / 0 | 55.165 / 8,388,608 |
| Server | ABY3-ring64 | C1 | Ring64 | 48.172 / 8,388,608 | 0.000 / 0 | 53.745 / 8,388,608 |
| Server | SPDZwise-F61 | C1 | F61 | 140.952 / 16,777,216 | 20.257 / 64 | 217.920 / 16,777,376 |
| Server | SPDZwise-ring128 | C1 | Ring128 → Ring64 | 262.719 / 33,554,436 | 53.859 / 182 | 369.412 / 33,554,860 |
| Server | ABY3-F61 | C2 | F61 | 67.826 / 8,388,608 | 0.000 / 0 | 68.859 / 8,388,608 |
| Server | ABY3-ring64 | C2 | Ring64 | 42.530 / 8,388,608 | 0.000 / 0 | 43.446 / 8,388,608 |
| Server | SPDZwise-F61 | C2 | F61 | 125.499 / 16,777,216 | 26.801 / 64 | 155.211 / 16,777,376 |
| Server | SPDZwise-ring128 | C2 | Ring128 → Ring64 | 185.650 / 33,554,496 | 56.775 / 182 | 239.663 / 33,554,920 |
| Server | ABY3-F61 | C3 | F61 | 59.369 / 8,388,608 | 0.000 / 0 | 59.565 / 8,388,608 |
| Server | ABY3-ring64 | C3 | Ring64 | 62.459 / 8,388,608 | 0.000 / 0 | 62.767 / 8,388,608 |
| Server | SPDZwise-F61 | C3 | F61 | 122.224 / 16,777,216 | 26.032 / 64 | 147.488 / 16,777,376 |
| Server | SPDZwise-ring128 | C3 | Ring128 → Ring64 | 183.477 / 33,554,688 | 48.112 / 182 | 234.307 / 33,555,112 |

原始数据：[最终完整重测](src/libspu/mpc/spdzwise/benchmarks/server-64worker-2026-10-21/after/)、[相同 64-worker 配置的优化前完整重测](src/libspu/mpc/spdzwise/benchmarks/server-64worker-2026-10-21/before/)、[Laptop 历史数据](src/libspu/mpc/spdzwise/benchmarks/macbook-air-f61-2026-10-19/)。每组保存逐方逐轮 CSV、日志、统计、参数、源码/二进制 SHA256、源码补丁和资源记录。

### Binary Benchmark

最新测量：**2026-10-21**。同一最终构建、相同 C2 输入、三方 FactoryMem，固定 **64 个后台 worker（`YACL_NUM_THREADS=65`）**。ABY3 / ABFL 串行运行，各预热 1 次、测量 7 次。

| 机器 | 协议 | 电路 | Domain | 电路计算（ms / B） | Verification（ms / B） | 总计（ms / B） |
|---|---|---|---|---:|---:|---:|
| Server | ABY3 Boolean | C2（AND） | GF(2)，64-bit 打包 | 32.463 / 8,388,608 | 0.000 / 0 | 32.463 / 8,388,608 |
| Server | SPDZwise-ABFL | C2（AND） | GF(2)，64-bit 打包 | 25.596 / 8,388,608 | 418.973 / 218,110,266 | 452.416 / 226,498,874 |

ABY3 不保存 triple、没有最终验证；ABFL online 包含快照保存，verification 包含随机 triple 生成、副本检查、commit–reveal、cut-and-choose 和 sacrifice。输入分享、初始化、导入检查、输出导出及重构核对不计时。ABFL verification 使用 65,536-word 工作块；两种协议每次均核对所有输出与预期通信量。

[统一实验参数](src/libspu/mpc/spdzwise/benchmarks/server-64worker-2026-10-21/after/parameters.json)、[全部原始数据](src/libspu/mpc/spdzwise/benchmarks/server-64worker-2026-10-21/after/)及[复现入口](src/libspu/mpc/spdzwise/benchmarks/README.md)。旧的 12-thread Boolean 和 Ring 结果保留在历史目录，不再用于当前服务器主表。

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
