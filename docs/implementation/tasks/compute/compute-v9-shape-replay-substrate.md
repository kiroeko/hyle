# Compute v9：shape 与 replay 的 Provider 基础合同

## Closure card

- component / owner：`compute` / Compute；round=`compute-v9-shape-replay-substrate`；
- baseline：`401d29698e1cd1e9d07f16c733c0c442246cd8f8`；承接未 terminal 的
  [v8](compute-v8-route-proposal-contract.md)；不是重开已 CLOSED runtime。
- trigger：[整体架构复核 A1/A2](compute-graph-route-unification-audit.md#整体架构复核与合同修正)已接受：
  proposal 必须携带可验证 shape domain，且 runtime ownership 不等于 replay 成本。
- 正常任务：Graph 把符号域投影为 provider-neutral metadata，Provider 明确 exact-only/支持域；direct
  exact Compute 保持原自然入口、执行和错误语义。
- 主纵切：动态 token/派生 reshape 的 Graph lowering；反偏置：已有 direct exact Native/oneDNN。
- scope：C90--C94 的冷路径结构；Device/Queue/Completion 热执行和算子目录不变。
- non-goals：公开 custom-kernel factory、Graph 资源域与 selector 算法、Apple adapter、全库重审。
  A3--A6 仍是 foundation 必做项，按总任务顺序后续绑定对应有限卡，本卡通过不等于基础完成。
- 依赖 / 消费者：Core/Error/Mem/ComputeOps；Graph、Native、oneDNN 及 Graph oneDNN carrier。
- 性能：P0；只改 cold metadata/能力校验，不为本卡单独启动 P1。基础 P1 在完整 candidate 统一执行。
- 预算：shape 表示/校验一轮、三个真实 Provider 的 SPI 迁移一轮、定向红队与一次纠正。
- 验证：range/equality/overflow/layout 与 exact-only 拒绝；内部版本失配；既有 direct/Graph/oneDNN
  单元与逐头；source-tree shared/static 增量验证，最终候选补三平台/sanitizer/三交付。
- stop：若需要 Graph 反向依赖、热态 shape/JIT/search、静默把独立域冒充派生域，停止对应实现并审计。

## 能力与有限改动集

| ID | 当前事实 / 动作 | 验收 |
|---|---|---|
| C90 | 旧 proposal 只有 exact TensorDesc；FIX 为带 provider-neutral shape view 的请求并版本化私有表 | 空 domain 是原 exact 路线；合法 range/equality 传递，错误域在调用 Provider 前拒绝，旧签名移除 |
| C91 | replay 成本与 ownership 尚未分开；FIX 逐 artifact 的 prepared/managed/unknown 元数据 | unknown 不推断 prepared；Native/oneDNN 保持严格路线；冷热成本不混合 |
| C92 | 派生 shape 需要可传播表示；FIX 有限 metadata program/guards 与 caller-scratch 求值，不携带 Graph ID | product/sum/exact-divide/broadcast、overflow/guard；只通过 bounds/equality 不能写 A2 完成 |
| C93 | v8 stable candidate/profile identity 未完成；FIX 保留完整 provider/build/device/domain/boundary 身份 | unknown 与 exact zero 分离；无证据不提供神奇 ns，direct 不读 profile |
| C94 | KEEP v8 事务化 outcome；迁移三个 Provider 和 Graph 直接调用 | rollback/版本/decline 回归；仅冷路径改变，无 kernel、线程或 Queue 热路径改变 |

独立 ranges/equality 是 C90 的子集，不是 bounded Graph 最终能力。当前工作区的未完成草稿只可经本卡
修改、验证，不能因草稿有 upper-point 描述就签发整个 domain proof。C92 的派生域不能伪装成彼此独立的
输出 symbol；没有声明支持的 Provider 仍 exact-only。

### API / usability

public direct `prepare()` 和 Context provider 顺序不变。私有 `ProviderOperationRequest` 持有 operation、
输入/输出 TensorDesc 与借到一次冷调用结束的 shape view；Graph-owned ID 不穿过边界。exact-only Provider
收到非 exact 域明确拒绝，不悄悄只处理上界。域校验与具体绑定使用 caller-owned spans，不创建全局表。
派生表达式具体形状在 C92 实现前完成本卡表示审计，不扩大为通用求解器；C90 不提前公开这层私有表示。

replay 成本元数据附着既有 Provider runtime facts，仅供冷准备/诊断；实际 prepared-only 筛选随 Graph
后继卡实现。它不新增 CMake 开关，不使现有 direct 用户额外配置 Graph，也不恢复 Native FFI。

C91 固定 C++ 形状：`ProviderReplayClass::{kUnknown,kPrepared,kManaged}` 与
`ProviderRuntimeProperties::replay_class`（默认 unknown），value type 提供 `valid()` 与完整 equality，
Compute/Graph 共用，不能漏比较新增字段。proposal 与最终 artifact 必须一致；该分类声明本 Provider
成功 replay 的成本合同，不替调用方提供的 executor/回调证明零成本，也不包含失败诊断的 Status 分配。
不由 shared/delegated、provider 名称或 capture flag 自动推导分类。Graph 迁移见
[v10 G100](graph-v10-reviewed-foundation-contracts.md)。

## 七问与当前裁决

没有新增依赖/许可证；需求由四引擎与 MLX 复核支撑，具体源码在总审计。复用 Error、TensorDesc 和 Mem
allocator，不为冷 shape 整数检查重开 Math。P0 检查没有热路径调用/分配增加；三交付必须验证跨 factor
私有 SPI，公开指南继续区分当前能力和未完成的 C92/Graph/Apple 路线。

本卡在既有 AUDITED 阶段重新确定有限范围；只授权 C90--C94。授权时产品验证尚未执行，状态保持
`AUDITED`，不签 Candidate、CLOSED 或性能 PASS。后续事实追加在本卡，不回写 v8。

## 2026-09-14 C90 / C94 迁移检查点

私有 `ProviderOperationRequest` 已替换旧 propose 签名，携带独立 ranges、重复轴 equality 和上界 tensor
metadata；私有表版本为 2。Context 在创建不兼容 Provider context 前拒绝版本失配，并回收此前创建的
context。Native、oneDNN direct、Graph oneDNN single-op 拒绝适配器及测试 Provider 全部迁移，无旧表分支。

shape validation/binding 的成功路径为 `O(S + (I + O) × R)`、零 heap allocation，`R ≤ 8`；具体 descriptor
写入 caller scratch。错误诊断仍使用 Error 的带消息 Status，可能分配，不能把它写成所有路径零分配。
空域保留 exact 路线，不额外遍历轴；submit、kernel、Queue、Completion 和 direct Provider 顺序未修改。

定向红队/反例覆盖：range 端点不整除与 `INT64_MAX`、相同范围的独立 symbol、重复轴绑定、零尺寸、
未知 symbol/越 rank map、错误 upper point、storage block 不完整、dynamic strided/opaque 拒绝、exact
strided/opaque 原样保留、caller scratch 和版本失配后的逆序回收。非 exact request 要求显式 entire-domain
承诺；未知 coverage 值拒绝。真实 Native/oneDNN 均仍 exact-only，测试 probe 的 entire-domain 只验证
能力传递，**不证明**任何真实 bounded artifact 已实现。

本工作树验证：

- Windows Release source-tree shared/static：Compute、CPU backend、Native、Graph 两类单元与逐头，各
  9/9 通过；两类构建分别覆盖 212 个单元用例。
- oneDNN-enabled shared：Graph/region/direct/composition/逐头初跑 7/8。新增 shape 测试把 8-row serial
  route 错当 oneDNN 支持项；源码确认该形状原本按合同留给 Native。仅把 fixture 改为已有支持测试的
  16-row 形状，失败组复跑 1/1 通过（12 个用例）；没有修改 kernel、cutoff 或路由策略。
- 首次编译还纠正草稿中 `StorageFormat` accessor 的拼写；没有留下未编译产品文件。
- closure checker（含 `--previous-ref 401d29698e1cd1e9d07f16c733c0c442246cd8f8`）、docs checker、
  `git diff --check` 通过。编译并行度限制为 2，未跑全库测试或 P1。

这些是 Windows 增量检查点；本树尚未跑 Linux/macOS、sanitizer、installed SDK journeys 和基础 P1。
C91、C92、C93 仍待实现，Graph 尚未把 bounded domain 接入实际编译/绑定；C90 substrate 不等于 A2
完成。下一步仍是本卡的 replay facts、有限派生 shape 与身份合同，然后按总任务接 Graph/真实 Provider。
组件保持 `AUDITED`，没有签发基础框架完成或任何新性能胜利。
