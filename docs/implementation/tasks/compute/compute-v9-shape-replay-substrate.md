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

独立 ranges/equality 是 C90 的子集，不是 bounded Graph 最终能力。不能因 C90 有 upper-point 描述就
签发整个 domain proof。C92 的派生域不能伪装成彼此独立的输出 symbol；没有声明支持的 Provider 仍
exact-only。

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

### C92 表示审计（实现前固定）

采用 provider-neutral、按创建顺序编号的有限维度程序；每条 dimension 是独立 input 或 add/multiply/
exact-divide/broadcast，操作数是非负常量或此前的 ordinal。独立 input 有 bounds，派生 bounds 由共用
区间算术生成并校验，不能把派生输出另造 input。附加 guards 只支持相等/小于等于，除法整除和广播
兼容性也在求值时检验。无 payload、字符串、循环、递归、动态 callback 或通用求解器。

表中 bounds 是保守 envelope，不宣称任意各轴上界组合均可执行。shape proposal 的整个域由程序和
guards 联合定义；非 exact 域不得再先用 envelope 做 concrete operation inference。无法证明支持整个域
的 Provider 仍拒绝；exact 域可保留 concrete 检查。区间过宽导致整数/描述无法表达时要求缩窄输入 bounds，
不依靠搜索/抽样去求可行子域，不伪称通用 guard tightening。

`ShapeProgramView` 借用 dimension/guard spans；每维携带 `ProviderShapeRange` 与算式，轴映射改为 value
ordinal。`evaluate_shape_program` 使用 caller 的整维度 scratch：输入槽须预填，派生槽覆写；失败弃用
scratch。验证/求值 `O(D+G)`，成功零分配，错误 Status 可能分配。shape-domain binding 复用同一求值器，
不保留旧 symbols-only 表。Graph 的公开 ID 不穿越此层；私有 SPI 同步升版。

## 2026-09-14 C91 成本分类检查点

`ProviderReplayClass`、默认 unknown 的 `replay_class`、共同 `valid()`/equality 已实现；Compute 与 Graph
删除各自复制的属性字段校验/比较。Native、oneDNN direct、oneDNN MLP 的既有成功 replay 声明 prepared；
原 kernel/Queue/Completion/线程策略和 direct Provider 顺序未改变。私有 Compute 表升级至 3、Graph
region extension 升级至 6，不保留旧布局适配分支。

反例验证三类成本均可如实传播；proposal/artifact 类别变化、非法 proposal、非法 artifact 都在接受工作前
失败，已构造 artifact 正确回收。unknown 不被 shared/delegated/capture flag 升级为 prepared，Graph report
保留原值。这个元数据实现不代表 managed Provider 已资格化，也不代表 prepared-only 筛选已经实现。

Windows Release shared/static 定向 9/9 各通过（各 217 个单元用例及四组逐头）；oneDNN-enabled shared
的 direct/composition/Graph/region/逐头 8/8 通过。沿用上一检查点三个 session-owned build，最大并行度 2。
P0：固定大小 value 增加一个 enum，校验/比较保持 `O(1)`；分类只在冷准备与诊断使用，warm 路线不增加
策略查询、循环、分配或锁。没有新计时结论，没有运行 P1 或不受影响的全库测试。

`validation-select.py plan --base 610a219f57e613ac6f1aaf79b337742122e2569f --candidate WORKTREE` 的 owner
检查为 READY；它也正确选出 CUDA 和安装交付的影响半径，本检查点未执行这些格，连同三平台/sanitizer
保留到完整候选，不能以 Windows source-tree 测试代替。closure/docs/diff 检查通过；用户指南已区分当前
分类 API、调用方 executor 自有成本和未完成的政策筛选。

C90/C91 与 C94 的当前直接迁移已实现，C92/C93 仍未完成；状态继续 `AUDITED`。下一项为有限派生
shape program 的表示审计与实现，再把完整域传入真实 Graph/Provider，不扩充算子目录。

## 2026-09-14 C92 / G101 首个真实消费检查点

C92 的有限程序已实现：独立维度、checked sum/product/exact-divide/broadcast、equal/less-equal guards；
轴映射引用程序值，Graph ID 不穿过 Compute 边界。Graph 私有记录复用相同表示与计算，不复制第二套
整数推导。私有 Provider 表升级至 4，不兼容旧 symbols-only 布局。Graph 公开形状与限制见
[使用指南](../../../user/graph.md#独立与派生维度)。

`validate_shape_program` / `evaluate_shape_program` 成功路径为 `O(D+G)`、零 heap，使用 caller scratch；
shape-domain descriptor 绑定另为 `O((I+O)×R)`，`R≤8`。Graph 的这一求值只在冷 compile 发生；原有
submit、kernel、Queue/Completion、线程策略不变。Graph-owned program/guards 仍由 Mem allocator 与
Container SmallVector 存放；没有增加第三方、CMake 开关、反向依赖或 Native FFI。错误 Status 可分配。

本地定向反例复核覆盖：前向/自引用、外来 ID、非法 enum/常量/派生 bounds、零尺寸、加乘溢出、除零/
非整除、广播冲突、scratch 数量、缺失/重复/派生 binding、metadata OOM 后逻辑原子性。小范围穷举使用
独立逐点算术 oracle，检查保守区间不会排除合法结果。guard 排除 envelope tuple 的用例验证非 exact
请求不再先做错误的 concrete schema inference；实际 Native/oneDNN 仍明确 exact-only，probe 不冒充
真实 bounded Provider。

真实 Graph 纵切运行不同具体 shape 的 concat（派生输出轴）和 normalization（batch×tokens 行数），
校验 payload 与完整 specialization identity；head split/broadcast 反例在 compile 阶段、Provider 接纳工作前
拒绝非法组合。
本轮首编译因 Graph 私有头路径拼写失败，已按既有相邻因子 include 方式修复，随后全部定向编译通过；
未修改中央 build system。

- Windows Release source-tree shared/static 定向各 9/9 通过，各含 228 个单元用例及四组逐头。
- oneDNN-enabled shared 的 direct/composition/Graph/region/逐头 8/8 通过。
- 三套既有 build 顺序执行，最大编译并行度 2；没有全库 correctness 或新 P1。
- impact selector 为 READY，保留 CUDA、三平台/sanitizer、installed SDK 与基础 P1 待完整候选执行；
  本检查点不替这些项签 PASS。closure checker（39 components/schema 4）、docs checker（288 页/1034
  relative links）与 `git diff --check` 通过。

C90--C92 与 C94 的本次 SPI 迁移完成；C93 identity/profile、Graph bounded/runtime variants、动态 View
attributes 与其余 G100--G107 仍未完成。这里只完成有限 shape substrate 和 exact Graph 消费，组件继续
`AUDITED`，不签整个 A2、foundation candidate、CLOSED 或新性能胜利。

### C93 身份审计检查点

现有 ProviderId 是逻辑家族 identity，Provider 名称/版本不证明具体 build；Device equality 是进程内资源
对象身份，也不证明硬件/driver/executor 合同。因此这些字段不能直接冒充可跨构建复用的 qualified
performance key。C93 尚待补齐明确的 build/device/domain/boundary 事实；没有可用事实时保留 unknown，
不编造 ns、不用函数地址或 POD padding 做持久 identity。这个缺项不阻止已审定的 Graph G100 replay
硬约束冷筛选；它不依赖 timing evidence，也不宣称已完成性能排序。

C93/G103 的共同成本前置迁移：现有私有 `ProviderRouteCostFacts` 移到公开 prepared-operation 合同，
不新建目标/依赖，Graph region 直接复用。共享 `valid()` 取代 leaf/region 重复检查，Compute 私有表随表示
迁移升版；Native/oneDNN 只迁移类型引用。该步骤清除不合格的 region 私有 ns 路径，但 qualified profile
的完整 identity/输入/匹配仍待实施，不宣称用结构计数替代真实性能。

## 2026-09-14 C93/G103：共同结构成本与旧估值清理

`ProviderRouteCostFacts` 已移到公开 `prepared_operation.h`，leaf/region 使用同一类型与 `valid()`。workspace
保留 size/alignment，六类事实都显式区分 unknown/known，byte/sync 的 exact zero 合法；已知 dispatch=0
和非二次幂 alignment 非法。Compute 私有表升至 5、region extension 升至 8，没有旧类型/布局适配。

Graph 不再接收或使用 Provider 相对自身 fallback 的 ns：旧字段、未使用 compile estimate 和
`kRequireEstimatedBenefit` 均删除。未伪造 full-plan profile 来保留旧打分。结构策略在所有候选都有 dispatch
事实时才使用该坐标，否则整维跳过；三个候选形成 pairwise 比较环的反例，在 interval 和一般 node-set
路径重复编译均稳定。精确零/未知、非法事实、异步 prefix→region、fallback relocation 和真实 oneDNN
全部回放。没有修改 kernel/tile、线程或 warm submit；算子目录未增长。

- Windows Release source-tree shared/static：Compute/CPU/Native/Graph 及逐头各 9/9，通过各 237 单元用例。
- oneDNN-enabled shared：direct/composition/Graph/region 与逐头 8/8（126 单元用例）通过。
- P0：共同 valid 为 `O(1)`；每次结构选择仅加一次 `O(R)` 已知性扫描，既有 interval DP 与一般 node-set
  收缩复杂度不变，warm 无新增政策/成本查询。结构计数不是速度结论，未运行 P1 或全库测试。
- impact selection READY，跳过 82 个不受影响项；closure/docs/diff 检查通过。仍欠最终 CUDA/跨平台、
  sanitizer 和完整 installed SDK journeys，不由本机 source-tree 测试代替。

C93 qualified profile identity/匹配、Graph 同池全成本 search、bounded variants 和其余 A3--A6 仍待完成。
本次消除旧实现与已定架构的一处冲突，**不**把无证据的结构 fallback 声称为自动 SOTA；保持 `AUDITED`。
