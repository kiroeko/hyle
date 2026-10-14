# G03 Apple 原生只读来源到固定 F2 slot

状态：AUDITED，2026-10-14；owner `compute_route_apple`；baseline
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

## 能力与有限 FIX 冻结

| 能力 | 当前 | 裁决与有限动作 |
|---|---|---|
| 成功 readonly 导出原 ownership、范围与稳定保活 | COMPLETE | KEEP 原 import_readonly/ImportedArrayStorage 与独立 ReadLease；不冒充设备 resource 的 allocation/free |
| 每次导出到固定 F2 source | PARTIAL | AS01 SELF：私有 typed source qualification 接受 nominated 原 Apple MemoryResource 家族内的 ImportedArrayStorage；普通非导出 resource 仍严格相等 |
| writable/alias/device/错误/退休 | COMPLETE | KEEP 原 mutable binding、copy_span、copy recording device proof、F1 admission 与 FIFO 实际退休；直接风险回归 |
| 原 native producer → row update → ready page/attention | PARTIAL | AS02：主/反偏置公共 consumer 证明固定 program/slot 可重绑、费用与生命周期；pending 完整链仍在主卡，不能代签 |

AS01 仅允许 apple_memory.mm、私有 apple_copy_internal.h、apple_copy_program.cc 与现有
public 声明/guide 的来源合同说明。AS02 仅对应 consumer/tests 与证据；不改数学、mem
resource、公共 CopyResourceSpec、版本/reader、完成/队列或全局构建。所有现有 Apple Copy
Program/Invocation/Submission 入口分别回指固定容量准备、来源/可写权限、原依赖/退休与
成功输出；只有 bind_source 的已证明缺失来源档改变，不增公开 symbol。

类型先证明 actual 为私有 ImportedArrayStorage，nominated 为原 MemoryState，随后核原
MemoryDomainId；仅 domain 相同或 host accessible 均不成立。相同 ref 的既有路线继续。
capacity 仍按真实 view 校验；copy_span 保 native backing 范围/offset，recording 已在
prepare_replay 核两端 buffer.device 等于原 Runtime device，原 interval/root admission
处理别名。不存在调用者可伪造的 import 证书。反证包含 fake provider 复写相同 domain。

普通调用：冷 CopyBatchProgram::prepare(context, {{memory.ref(),capacity}, ...})、两
invocation；每步 producer.wait/import_readonly 后 bind_source，同一 program/slot
prepare/submit/wait；成功后原 writer 回接 page publish/read/原 attention。F2 仍保实际
source ReadLease，reset/rebind/drain 才清理原 bindings。源的 resource owner 独立保活，
原 nominated resource 仍借用至 program/slot/提交退休；consumer 不得经 MLX 别名写源。
thread-compatible、提交保留/失败不消耗来源、closed/取消/未知处理沿原合同。bind/prepare
有界资格检查、无新 metadata allocation/field/copy/wait，cold import/OOM 仍由原边界处理。

固定 peer 实际归档复核：llama.cpp cbe28bb3 的 ggml-backend.cpp:651–669 使用原 tensor/
backend copy 能力、不能将同步 fallback 当 native overlap；vLLM 98f7211 的 prefetch.py:
269–295 使用固定 buffer 与逐层 event；SGLang 7224fce 的 memory_pool.py:2825 起实际
set_kv_buffer 校验写位置、原 resident cache 与本步 K/V。不抄其请求政策或取消权限检查。
固定 SHA/出处沿[输入证据](../../../../evidence/architecture/inference-single-machine-foundation-audit/README.md)。

七问：原锁定 MLX 与许可证/升级边界不变；真实主/正交旅程及 gap 如上；复用原 F1/F2/
MLX/recording；P0 有界额外检查换去 warm program/slot 分配并显式保 import unknown；无
FFI；source/static/shared 原 hyle::compute 与独立头实验；public 声明及 Apple guide 同步。
普通门未执行前都为待验，不由 AUDITED 代写 PASS。没有新增算法，因此无需重跑数学 P1。
