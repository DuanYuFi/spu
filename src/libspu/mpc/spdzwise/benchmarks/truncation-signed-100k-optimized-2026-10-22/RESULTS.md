# Signed RING probabilistic truncation

100,000 个正负混合输入，m=16，双方输入满足 |x|<2^62。三方 FactoryMem；64 个 YACL 后台 worker；预热 1 次、测量 7 次。

主计时从已准备好的输入 shares 开始，包含现场随机材料生成与必要检查，输出 shares 形成时停止。SPDZwise 输入已正确认证，输出保留 MAC；ABY3 为正常注册的 TruncAPr2、普通 replicated shares。输入认证、最终认证检查、输出重构均不计入主耗时。所有轮次全部输出均通过验证。

| 协议 | 主耗时中位数 ms | 最小–最大 ms | 随机材料 ms | 在线 ms | 计时外检查 ms | 吞吐 elements/s |
|---|---:|---:|---:|---:|---:|---:|
| spdzwise | 3431.204 | 3243.173–3496.149 | 3419.199 | 16.493 | 1201.781 | 29144 |
| aby3 | 8.728 | 6.752–13.016 | 0.000 | 8.728 | 0.000 | 11457407 |

SPDZwise / ABY3 主耗时比：**393.13×**。各分项独立取三方最大值，不能相加重建主耗时。ABY3 的随机材料生成属于其算子内部，表中在线时间包含完整调用，并非预处理后的时间。

SPDZwise 按 MP-SPDZ 的有符号偏移、随机掩码与回绕修正实现；bitness 聚合使用现有内部 128 位零检查。主时间未包含最终延迟认证检查，不应解释为完成全部恶意安全验证的端到端时间。ABY3 保留源码中的 +1 修正，未假定与 MP-SPDZ 精确舍入分布相同。

计时内三方总发送字节：SPDZwise [1436027195]，ABY3 [2800000]。此结果为本机内存通信，不能直接推断 LAN/WAN 延迟。

复现：`bash build.sh //:truncation_bench`，随后 `python3 run_truncation.py NEW_OUTPUT_DIRECTORY`。完整命令与环境见 parameters.json 和资源记录；逐轮数据见 CSV；源码改动见 source.patch。

## 验证与归档

保留最后一次完整 100K 测量（2026-10-22 08:36:36 开始），包括同组 ABY3 对照。
每个协议包含一次预热与七次正式测量，所有输出及最终认证均验证通过。
此前基线、优化试跑和较早复测的实验归档已清理。

[validation/tests.json](validation/tests.json) 记录八组 301 个回归用例，全部通过。
对应 XML、日志及[完整命令](validation/commands.txt) 保存在 validation/。
逐方逐轮测量见 spdzwise.csv / aby3.csv，环境和源码/二进制指纹见 parameters.json，
测量时源码补丁见 source.patch，文件完整性清单见 SHA256SUMS。
指纹对应实际测量时的构建，后续选择器命名调整的验证另见协议 VALIDATION.md。
