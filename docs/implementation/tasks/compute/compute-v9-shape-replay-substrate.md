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

## 七问与当前裁决

没有新增依赖/许可证；需求由四引擎与 MLX 复核支撑，具体源码在总审计。复用 Error、TensorDesc 和 Mem
allocator，不为冷 shape 整数检查重开 Math。P0 检查没有热路径调用/分配增加；三交付必须验证跨 factor
私有 SPI，公开指南继续区分当前能力和未完成的 C92/Graph/Apple 路线。

本卡在既有 AUDITED 阶段重新确定有限范围；只授权 C90--C94。产品验证尚未执行，状态保持 `AUDITED`，
不签 Candidate、CLOSED 或性能 PASS。记录初始范围后再提交产品代码；后续事实追加在本卡，不回写 v8。
