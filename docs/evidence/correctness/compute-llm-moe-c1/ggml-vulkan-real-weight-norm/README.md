# C1-Y4：本地2B真实权重的 Vulkan 归一化片段

产品 `3c39428e`、独立 SDK source snapshot `351ec13d`；使用已通过交付检查的 SDK，未重建
HYLE 或运行 P1。原本地模型13个文件、4,571,274,023 bytes 全部重新通过冻结清单的 SHA-256
校验，没有下载模型。Git 中只保存来源、参数、哈希、测试源码和日志，不保存权重张量。

## 实际结果

Windows 与 MacBook Air 的 relocated static/shared SDK 各运行一个独立普通 C++ consumer，
四个 executable 均通过，合计72次提交。每个均覆盖 Direct/Graph、1/17/128行、宽2048、两个
invocation、三轮变化输入、旧结果保留、原 Graph/Program 销毁、输入不变与配额归零。

输入取原模型实际 BF16 embedding 行和第一层 Gemma RMS 参数；固定 prompt 的第一轮使用
原 token IDs，后两轮轮换实际 embedding 行。scale 按原公式转换为 F32(1+weight)，独立
Python scalar double/math.fsum oracle 使用固定 epsilon 和既有 BF16 误差门。

Direct 消费 BF16 RmsNorm；Graph 为 Cast(F32)→RmsNorm→Cast(BF16)，严格断言其中一个真实
ggml managed region。没有将 BF16 Direct 路径也误记为 ggml。native buffer 配额峰值为
8,398,848 bytes；这不是 provider 全部私有内存或整模型内存账单。

Windows DLL 邻置、运行 PATH unset、非交互错误模式；两平台 Context 请求 validation，实际
执行日志没有 VUID/Validation Error/SYNC-HAZARD。各 raw ZIP 保留测试源码、完整 configure/
build/execute/linkage 命令与日志、CMakeCache、编译命令、fixture 身份和 executable 哈希。
`controllers.zip` 保存原始生成/运行/归档脚本；`result.json` 保存限定 verdict。

## 未完成与外部缓存变化

这只签真实权重的归一化片段。其新 consumer 的 source delivery、Gather、gate/residual、
持续状态片段、完整 C1、整模型和性能均未由此验收；前批生成权重 FFN 的三交付不能代签这些项。

`asset-status.json` 记录：C0 原登记的 Windows B16 reference 和 Q4/Q8/FP8 派生文件路径目前
不存在。没有据此断言所有机器上的副本都丢失，也不撤销历史执行事实。C3/C6 消费前须找到
保留副本或按原冻结 recipe 恢复；当前片段直接读取已验证的原模型，不依赖这些派生文件。
