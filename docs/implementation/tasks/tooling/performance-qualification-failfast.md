# 性能资格预验证失败即停止

状态：**CLOSED，2026-10-17**。SCOPED checkpoint `7175d12df`、AUDITED `cc99356de`。独立工具任务，不属于 G09 产品 FIX；baseline
`3ed603a48`，`scripts/hyle_perf/runner.py` 原 SHA256
`df415c5bb04c1096c568cae75c84e83672d2282ef43c7965d905f3ba1d21e022`。

真实 consumer 是 [G09](../graph/inference-g09-completion-wake-correction.md) 的一次有限资格入口。
原 `_run_contract_locked` 在 pre-sampling validator FAIL 后仍调用 product collector；历史
G09 靠私有 ready guard 拒绝首个 formal child。公开工具不能把停止合同委托给每个 consumer。
stock CLI 没有已资格 resume seam，不能靠复制 collector/重复 A/A 绕过。

有限 T1：在原 artifact/source/peer 复核之后，仅 `requested_qualified` 且任一 validator
非 PASS 时抛 `ValidationError` 并列出 ID；保留实际 validator raw，不产生完成 manifest。
probe 的 diagnostic 行为、PASS 的原 n15→31、统计/contract/schema/最终复核都不改变。
只允许原 runner 八行及既有 runner unittest 的五项边界测试，不新增入口、能力或依赖。

预算十分钟应用/验证检查点；若须改统计、阈值、resume 协议、旧 evidence 或重建产品则停止。
当前 G09 普通设备阶段已失败并停止，才进入此独立工具边界；不在冻结普通 stage 中途修改工具。
隔离草稿已实际旧代码红测，五项中三项失败；相关四套 Python/mock tests 共65项通过。
单份旧 CUDA manifest strict offline verify 通过且原 SHA 未变；只证明该份可读，非所有历史兼容。
canonical 尚未应用。预声明应用后65项相关测试、docs/closure/diff及独立源码复核；不启动 native/P1。

七问：依赖/许可不变；原真实资格旅程补 stop 边界；复用原 validators/error/raw 机制；
只在既有冷 preflight 遍历结果，不改采样成本；无 C/FFI/API；纯 Python 无虚构 binary/SDK 旅程；
CLI 原错误出口给出失败 ID，无新增用户选择或统计档位。

AUDITED：root 逐行核八行 guard/五项真实 orchestrator 测试，独立 Graph 审查 SOURCE_PASS，
无 must；审查确认 probe 必须继续原 diagnostic 路线、source/artifact mutation 原 guard 优先。
隔离 patch SHA256 `2841f096a21a9dd927be8b061a8026b5b1cf84047a9b3e0a3d65df6062bbaafb`。
批准只应用上述精确两文件补丁；工具身份独立冻结，不借原工具授权签新 P1。

精确补丁已在 canonical 应用，原四套相关 Python/mock 测试 43+9+6+7=65 项全部通过。
[完整记录](../../../../evidence/correctness/performance-qualification-failfast/receipt.json)和原日志归档
保留隔离旧代码红证据、隔离及 canonical 绿日志、原历史 replay、源文件 hash 和独立复核。
scope/architecture、correctness、原 CLI 错误可读性、独立红队及冷 O(validator count) 结构账通过；
纯 Python 工具没有适用 C++ SDK/设备旅程。没有实际 compile/native/GPU/A/A/P1，也不由本任务
验收 G09。普通 selector 的本轮最终计划会纳入这些工具测试；不会因纯 Python 修改重建产品。
docs/closure/diff 检查通过，有限 T1/T2 到此停止。统计、采样和 manifest schema 均未改变。
