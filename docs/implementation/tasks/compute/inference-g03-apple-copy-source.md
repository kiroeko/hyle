# G03 Apple 原生只读来源到固定 F2 slot

状态：SCOPED，2026-10-14；owner `compute_route_apple`；baseline
`66b433cef`。完整目标和 pending 页组合仍由[总卡](../project/inference-gaps-goal-execution-2026-10.md)
与[compute_batch 活动卡](inference-g03-pending-page-consumption.md)保留。

触发是[实际公共探针](../../../../evidence/architecture/inference-g03-native-output-copy-shape/README.md)：
原 Abs Program/Invocation 的两次成功输出，经 import_readonly 后有独立的永久只读
resource owner；固定设备 ref 的 F2 program 拒绝，按导出重建 program/slot 才成功。
这是当前 ready 来源消费成本问题，不重开既有 GP01/GP02、数学或整个 native route。

主旅程：同一 native producer 生成本步数值 K/V → readonly import → 固定 F2 program
与两 slots 的精确 row copy → 原页 publish/read → 原 attention，64 步、GQA8/2 D256、
两 request、Q1/17、page16、跨页与旧结果。反偏置是 Abs 数值输出 → 固定 F2 → 完整数值
验证。请求/cache 策略留在消费者。

职责仍由原 MLX owner、ReadLease、F2 CopyBinding/LeaseUsePool、原 FIFO Job 与 copy recording
承担，不改公共 mem/compute/compute_batch、queue、Completion、工具或构建图。涉及公共头
只补现有 bind_source 的来源合同，不加新 ABI、对象字段或选项。原可写 binding、producer
qualification、成功 output coverage、取消/放弃/actual retirement 保持原机制。

预声明门：实际 source 固定 program/两 slots/64 次 readonly 重新绑定，保旧导出、partial
subview/容量和 foreign host/伪造 domain/普通异资源拒绝、可写端点原严格拒绝；元数据 allocator
计数证明 bind/prepare 无新 HYLE allocation；原 copy program/recording tests 与增量页
consumer、copy 生命周期风险直接回归；宿主 ASan/UBSan/TSan；relocated static/shared
仅 hyle::compute 公共 consumer、apple_memory 独立头。Windows 不提供此 Apple binary，
不制造 MSVC/Windows GPU journey。实际 Air 先核 Mac16,12 身份，全部 heavy 串行/最多四 jobs。

P0：热 binding 增加有界 typed qualification，零新 allocation/copy/wait/字段；移除逐步
program/slot 构造成本。不是新数学算法，不使用 P1 点值决定来源权限。opaque MLX/driver
费用仍 unknown，readonly import 本身的冷 metadata 仍明确收费；不声称整步零分配。

调查/实现/验证各 2h 检查点，按用户常设授权有限追加并保失败/累计。停止条件是本有限
FIX 实现并通过适用门，或明确 blocker；不缩减完整 G03，也不重复原 CLOSED 的 P1。

## 审计待冻结

需固定：仅可信 ImportedArrayStorage 如何匹配 nominated 原生 resource；域 spoof/异设备
及普通异资源不能借新的规则越界；实际 backing/subview 与 alias proof 仍由原 copy_span/
copy recording；程序所借 nominated resource 和每次 source owner 的寿命分别保留。
