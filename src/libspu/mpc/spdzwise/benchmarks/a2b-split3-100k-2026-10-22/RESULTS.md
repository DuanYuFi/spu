# split-3 A2B：电路选型及原生 ABY3 baseline

最终实现选择 **CSA + bit-sliced RCA**。Brent–Kung、Kogge–Stone 及临时选择接口已从运行代码和基准中删除；这里保留其原始测量与源码指纹。

## 实验口径

100,000 个完整 64 位输入，包含固定边界值及 SplitMix64 随机值；三方 FactoryMem，共享 64 个后台 worker（`YACL_NUM_THREADS=65`）。协议串行运行，与编译、回归测试不重叠。每组预热一次、测量七次，每轮新建上下文和正常协议输入 shares。

输入分享、初始化和 SPDZwise 输入认证在计时前完成。SPDZwise 主区间包含 split、打包/转置、CSA、加法、全部 AND 记录与通信，以及最后一次 ABFL 验证（包含随机材料生成）；ABY3 主区间是正常注册的原生 A2B 完整调用，包括 resharing 和默认加法器。两者均在计时外重构并精确比较全部输出，同时核对复制分量一致。

主指标每轮取三方**完整区间**最大值，再取七轮中位数。阶段指标独立聚合，不可相加重建主指标。最终导出沿用既有流程，会再进行空算术队列的控制消息交换，但不再次执行 ABFL 验证；该输出导出过程不计入主区间。ABY3 为半诚实 baseline，不增加认证或恶意检查。

## 候选选型

| 电路 | AND 计数/输入 | 在线 AND 层数 | 总耗时中位数 ms | 转换 ms | 验证 ms | 三方总发送 B |
|---|---:|---:|---:|---:|---:|---:|
| rca | 125 | 63 | 144.831 | 15.967 | 130.408 | 187,567,008 |
| bk | 291 | 12 | 305.753 | 23.991 | 282.674 | 436,652,832 |
| ks | 832 | 8 | 625.534 | 76.555 | 555.411 | 842,423,520 |
| aby3 | — | — | 142.011 | 142.011 | 0.000 | 33,600,000 |

RCA 与第二名总耗时差距远大于 3%，无需追加十四轮。选型仅在三个 SPDZwise 候选之间进行。各候选及 ABY3 均完成八轮全部 100K 输出验证。

候选结构：

- RCA：按 64 个输入转置为位平面；63 个 CSA AND 与 62 个进位 AND，丢弃模 2^64 不需要的进位。
- Brent–Kung：相同拆分和位平面 CSA；62 个初始 generate AND，前缀网络采用向上/向下扫描，反向标记并删除未消费的 propagate，前缀部分共 166 个 AND。
- Kogge–Stone：每个输入保留 word-packed 表示；一层 CSA、一层初始 generate、六层距离 1/2/4/8/16/32 的前缀，最后一层不计算 propagate。同层两个独立乘积合批，共 13 个完整 64-bit AND word，即表中的 832 bit AND。

RCA 与 Brent–Kung 尾块按 64 lanes 补齐并验证，不忽略填充位。这里的门数与运行时间不同：转换、转置、内存分配、通信及验证均计时。候选阶段明细见 [candidates/RESULTS.md](candidates/RESULTS.md) 和逐方 CSV。

## 清理后最终复测

最终通过公开 `a2b` 接口运行，不保留电路选择开关。

| 协议 | 总耗时中位数 ms | 范围 ms | 转换 ms | 验证 ms | 三方总发送 B |
|---|---:|---:|---:|---:|---:|
| spdzwise | 134.182 | 129.609–144.828 | 12.950 | 121.232 | 187,567,008 |
| aby3 | 146.386 | 125.306–149.766 | 146.385 | 0.000 | 33,600,000 |

SPDZwise / ABY3：总耗时 **0.917×**，转换阶段 **0.088×**，完整通信量 **5.582×**。候选阶段总耗时比为 1.020×；两次测量说明本机总耗时接近，不能推断所有网络中均更快。最终数据见 [final/RESULTS.md](final/RESULTS.md)。

## 验证与复现

六个回归目标共 61 个用例通过：conversion 5、protocol 11、Boolean 16、RING 15、截断 7、session 7。最终运行执行了前五个目标，未改变的 session 目标复用通过缓存。原始 XML 与日志见 `final/validation/`。

新增测试覆盖空/单元素/63/64/65/非连续视图、随机及全宽边界、真实 128-bit 算术结果接 A2B 再接 Boolean AND、dtype/shape/会话检查、累积后一次验证、验证前导出拒绝、三方位置的副本不一致及一致性 triple 篡改拒绝。每轮基准额外核对全部 100K 输出。

从 `src/libspu/mpc/spdzwise/benchmarks` 执行：

```sh
bash build.sh //:conversion_bench
python3 run_conversion.py NEW_OUTPUT_DIRECTORY
```

测试从 `src` 主工作区运行（独立 benchmark workspace 不导入开发测试依赖）：

```sh
libspu/mpc/spdzwise/benchmarks/tools/bin/bazel-7.7.1 --batch \
  --output_user_root=/tmp/malicious-spu-bazel test -c opt \
  //libspu/mpc/spdzwise:ring_conversion_test \
  //libspu/mpc/spdzwise:protocol_test //libspu/mpc/spdzwise:boolean_test \
  //libspu/mpc/spdzwise:ring_test //libspu/mpc/spdzwise:ring_truncation_test \
  //libspu/mpc:protocol_session_test \
  --lockfile_mode=off --jobs=16 --test_env=YACL_NUM_THREADS=65 \
  --test_timeout=240 \
  --extra_toolchains=@rules_foreign_cc//toolchains:preinstalled_cmake_toolchain
```

`parameters.json` 保存构建输入与二进制 SHA-256；各协议 CSV、日志和 resources JSON 保存样本、消息数、CPU 时间和峰值 RSS。最终源代码快照差异见 `final/source.patch`。候选实现已按要求删除，其历史命令仅描述当时的实验；当前脚本仅支持保留的 SPDZwise 与 ABY3。
