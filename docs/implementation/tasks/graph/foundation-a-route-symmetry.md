# Foundation A：四路线 Graph 语义对称性

## Closure card

- component / owner：`graph`、`graph_route_apple`、`graph_route_vulkan`、
  `graph_route_cpu_apple`、`graph_route_cpu_vulkan`；共同静态语义由 Graph 拥有，
  原生 artifact、端口、完成与资源由各路线拥有。
- baseline commit：c0792501319ad1eab482c41b054e1e63a3ca7b8a。
- trigger：用户明确要求实施四路线对称性；
  [固定源码研究](../project/foundation-f45-reference-research-2026-10.md)与
  [设计 A-01..04](../../../architecture/compute-state-paged-design.md)证明
  CPU+CUDA 复用共同 compiler，而 Apple/Vulkan 及 hybrid 拒绝全部 symbolic/control 图。
- 正常 consumer：同一有界变长张量管线，冷准备有限精确形状，独立 invocation 选择固定形状，
  显式控制前驱虽没有数据消费者也必须执行；不支持的 effect/alias 明确拒绝。
- 主纵切：F4/F5 显式状态消费者的 pure initial/final 与有限 batch shape。
  正交纵切：渲染资产变长矩阵管线的独立分支排序与旧结果保活。
- public journey：GraphBuilder 的已有 shape/control/effect 校验 → 冷 native artifact 表 →
  原路线 invocation/bind/prepare/submit → 原 Completion/ResultLease；不引入共同 native runtime。
- scope：仅 A-01、A-02、A-03、A-04。实际可支持的 effect 需由端口和 permission 证明；
  拒绝不能包装成支持，pure control 不得只删除检查。
- non-goals：F4/F5 产品代码、全引擎、自动 placer、运行时 JIT、padding/general 伪覆盖、
  新数学/copy kernel、万能 Queue、公共验证/构建/性能工具重写、重开 F1–F3。
- 依赖：既有 Graph/Compute/Mem/Sched 与锁定 MLX/ggml/IREE；不升级依赖。
  直接消费者：原 native 与三 CPU hybrid 普通 Graph consumers；CPU+CUDA 新入口沿
  [独立有限卡](foundation-a-cpu-cuda-variants.md)实施。
- 性能：P0，检查冷表与 warm 路径、复制/等待/私有成本；本卡不运行 P1 或全算子目录。
- 机器：Windows CPU/CUDA/Vulkan，WSL CPU/CUDA 与 sanitizer，Air Apple/Vulkan/CPU；
  Apple 运行前核对真实身份，仅 MacBook Air。WSL 软件 Vulkan 仅构建与包证据。
- 预算：静态审计两轮；有限四 FIX；每个 gate configure 300 秒、build 1800 秒、
  test 120 秒、consumer 180 秒。相同 gate 的失败/超时不随目录或提交重置。
  重型 gate 串行；库 build 按可用内存 j4≥12GiB / j2≥8GiB / j1≥4GiB，test1。
- 预声明验证：selector 选择 actual owners/直接消费者；共同 shape/control/effect 边界与 OOM；
  exact coverage 缺口/外来 ID/重复 binding/derived guard/overflow、独立 slots、旧结果与 close；
  control-only 活前驱与跨 fragment 排序，禁止 host wait 的拒绝；实际支持 effect 的权限/alias/失败。
  受影响 native/hybrid 普通 source 与 installed static/shared、公开头、缺路线、搬迁、runtime；
  共同静态代码适用 ASan/UBSan，新增并发共享元数据适用 TSan；无新增并发机制不重复全库 TSan。
  每轮 docs/closure/diff，registry 明确 previous-ref，生成 status，不手改生成表。
- stop：预算到达或 native 无法满足所需语义时如实 BLOCKED；新系统性 API/owner 事实退回
  SCOPED/AUDITED；原本不准入的 effect 保留明确拒绝与限制，不冒充 A 完成。

## 当前阶段

AUDITED。GraphBuilder::finish 已统一验证 shape program、控制边身份/拓扑及 effect 冲突；
共同 executable 的 bind_shape_values / specialize_bound_descriptions / make_specialized_graph
已拥有精确形状求值和原身份保留。BORROW 这些实现，禁止第二套表达式/guard 校验。
原生 lowerer 的数据剪枝不能保留 control-only 前驱，必须改成数据+控制可达性。
Apple evaluate_mlx 逐 region 等成功 outputs，hybrid 逐 fragment 等原 Completion；
控制端点冷分片并显式物化前驱输出可以复用这些顺序，不需要新 Compute runtime。
Vulkan 原生控制端点的 command barrier/managed region 仍需实现期对原录制代码定向核实。

### 能力与 API freeze

| 能力 | 当前裁决 | 有限动作 |
|---|---|---|
| GraphBuilder 形状/控制/effect 合法性 | COMPLETE | KEEP/BORROW 原规则 |
| native 精确形状 | MISSING | 共同冷 specialize 快照，保留原 ValueId/NodeId；不宣称 general |
| 有限原生 artifact 表 | MISSING | SELF 冷 `ExactNativeVariants<Artifact>`，声明表全量预检后编译；select 时校验 binding，未覆盖拒绝 |
| control-only 活性与顺序 | MISSING | SELF 冷可达性/端点分片，BORROW 原完成边界；等待/物化成本明确报告 |
| 原生 pure inputs/results/旧结果 | ROUTED | KEEP 原 native Program/Executable/Invocation/ResultLease |
| 外部 effects/resource ports | PARTIAL | 原 CPU+CUDA 规则 KEEP；native 无真实可写端口者明确 REJECT；不得假报 preserves_effects |

共同精确快照以 `specialize_exact(graph, shape, allocator)` 返回 Graph 与原
SpecializationSignature；只在冷调用分配，完整复用既有求值、输出推导和 ViewSpec 绑定。
native/hybrid CompileOptions 增加 `shape`，原无 dimension 调用保留原形状。
`compile_variants` 为各 route 的冷 typed factory，返回拥有原 artifact 的有限 family；
`select(shape)` 返回该 artifact 的共享 handle，再用原 create_invocation 管理独立 slot。
不在 submit 查表/JIT，也不增加万能执行接口；无法准备任一声明 shape 则整表拒绝，
缺省没有隐式增长，重复 signature 拒绝，完整 bounds 内的表外点仍 Unimplemented。
声明/选择都检查原 graph identity、derived guards、完整 independent bindings。

metadata 与原 native artifact 的 allocation/OOM 发生在冷准备，失败不消费调用者 tensor；
family 复制不复制 payload。共享 ResourceBudget 继续累计各 artifact 的原收费，未知
MLX/driver bytes 保留 UNKNOWN；无总预算配置不声称硬内存上限。常态每 slot 原生对象与
旧结果权限不变。control 默认允许原 managed waits，禁止等待按原 workload 拒绝。
source/static/shared 使用同一 typed entry；公开新头归 Graph；文档提供显式两形状选型。
所有新增入口回指有限 shape consumer，取消/close/producer 沿原 Invocation，family 不起任务。

七问：依赖锁与许可证沿固定研究 KEEP；功能仅按上表且外部 effects 不伪支持；复用原 Graph
求值/Compute runtime/F1 owner；P0 冷 O(V(D+N+E)+V²D)（显式表查重）、选择 O(DV)、warm 原 artifact；C++ typed
入口无新 FFI；新增公共头及四 factory 三交付需实际验证；guide 要明示 exact-only、等待与
未知 private 成本。当前均为合同冻结，正确性/交付/红队尚未裁决。

## 有限改动 ledger

| ID | 工作 | 当前结果 |
|---|---|---|
| A-01 | 共用既有静态合法性与身份，保留 native artifact/owner | 已接线；共同39项与 Win Vulkan/Air Apple source PASS；其他格待验 |
| A-02 | 冷有限 exact artifact 表与创建时 coverage 拒绝、独立 slots | 已接线；Win Vulkan/Air Apple native+hybrid PASS；CPU+CUDA新表入口/三交付待验 |
| A-03 | 真实 pure control liveness/顺序与禁止等待裁决 | 已接线；Win Vulkan/Air Apple native+hybrid PASS；完整风险/三交付待验 |
| A-04 | 真实支持 effect 的端口/权限/alias；其余明确拒绝与旧结果合同 | CPU岛已接线；Win Vulkan公开consumer/内部8项PASS；Apple/完整风险/交付待验 |

## Gate

scope、architecture、correctness、delivery、usability_docs、red_team、engineering_performance
均未裁决；本卡建立不代表产品能力或 native 验证 PASS。

## 首批实施事实

### A-04 第二次静态审计冻结

新增 CPU hybrid CompileOptions::effect_bindings，复用 Graph 的 EffectResourceBinding。
只允许显式放到 kCpuTarget 的 effects；GPU 默认或未知 carrier 仍明确拒绝。
所有 effect 节点作为活节点，即使零输出/无 tensor consumer；每个 effect 冷分片，传入
其原 resource ports/映射后的 footprints，并要求原 CPU Provider preserves_effects 资格。
单片段 graph::compile 执行与 CPU/CUDA 相同的验证，不新增操作回调或执行接口。

共同 bind_effect_resources 的全图 footprint 校验提取成 Graph-owned 静态 helper：完整 IDs、
绑定次数、缺失 footprint、不同 resource ID 的范围 alias，以及 effect 与 constants/tensors
的 alias 都复用原规则。hybrid 冷 compile 对全图 preflight，binding/prepare 对所有 caller
host inputs 再检查；不能只在单片段检查而漏掉另一片段的 readonly tensor。
native leased input 需使用原 F1 import/footprint 身份；无法证明同 backing 时拒绝 effect
组合中的 opaque native input，不把 VkBuffer/MLX array 指针当 host alias identity。

固定 footprint/backing 继续 borrowed，caller 跨 invocation 同步；无全局锁/新的 writer
权限。CPU effects 无输出时依然拥有原 Completion/失败前缀，后继在原 driver 完成边界后
执行；cancel/close/旧 result 保持原对象。没有 GPU mutable ports 的路线不假报 GPU effects。
新增接口仍是 A-04 的有限范围，不添加模型策略、页写 kernel 或 universal Queue。
本轮预声明的 effects/alias/producer/cancel/旧结果风险需要实际混合 consumer 证明。

[共同精确形状](../../../../evidence/correctness/foundation-a-shared-exact/README.md) 39/39 PASS，
包含原静态语义和新有限表、OOM、control reachability；
[Win Vulkan source](../../../../evidence/correctness/foundation-a-vulkan-source/README.md) native 与
CPU hybrid 的有限形状、两个独立 slot/旧输出、控制前驱错误与禁止等待旅程 PASS。
原 fixture 失败保留，累计预算在对应 evidence。代码仍 AUDITED，尚无 candidate/终局。

[Air Apple source](../../../../evidence/correctness/foundation-a-apple-source/README.md) native 与
CPU hybrid 的 exact/control/旧槽结果数学 PASS，真实可用内存7GiB 按规则 j1，build20秒。
这不代签 Apple 的全部失败/取消及 installed 交付。CPU+CUDA 保持原 compiler/公开路线，本轮新 finite-only table 的
自然入口仍需纳入实际 owner 审计。A-04 尤其需要 CPU 岛上的原 effect/footprint 接线资格，
不能把所有 native effects 的明确拒绝写成四路线整体对称完成。F4/F5 尚未启动产品施工。

第二批：共同 effect footprint 校验提取后保持原 SmallVector0 容量与常态分配形状；CPU+Vulkan
实际 write→GPU math→read、跨片段alias拒绝及七项原 effects/OOM 用例8/8 PASS。仅公开头的
consumer 同样通过正常/接受后失败、zero-output effect 与 close 后 old native result。
CPU+CUDA finite-only factory 两个形状/表外拒绝/真实 CPU→GPU→host math/旧 result 及错误
family-wide quota 拒绝 PASS。unused resource 声明按实际 node ports 判定，不虚构 effect。
共同最终40 tests PASS。所有记录仍为 AUDITED 实施检查点，原 failures/预算累计保留。

## 候选前红队事实

独立一遍定向源码反证发现 Apple 未裁决 mandatory-alias custom，而 Vulkan 已拒绝；
共同 NativeAnalysis 现在只对活 GPU custom 的未资格 must-alias 拒绝，CPU 交给原 artifact
alias_support。新增反例 PASS，共同41/41。纯 control MLX report 不再声称实际含 ggml model region。
公开 consumer 增加有界真实 CPU prefix cancellation：等待开始后取消、释放原 accepted callback，
原写入41保留、GPU/read后继未开始、Cancelled；Win Vulkan PASS。只用测试自己的进度 gate，
不新增产品 worker/reaper、执行器或回滚。Apple 最终候选同项待验。

## Candidate

product commit：546b0d2468e5de29031c34668f4afa4bc31b9a17。A-01..04 的有限实现与自然消费者
已接线固定；七 owner 进入 CANDIDATE，correctness/delivery/redteam 的剩余适用格尚未签。
API 包括共同 exact snapshot/family、三 hybrid 冷入口与 CPU effect footprints；原 native
程序/Queue、F1–F3、general compiler不变。cold 查重 O(V²D)、表外拒绝、各 slot 原有
private/managed 成本；warm 无新的 table 搜索/shape JIT/DAG合法性检查。
本轮共同 helper 未新增线程/全局状态；现有并发框架不变，定向 ASan/UBSan 适用，
新的 mutable global/shared 并发机制不存在，TSan 不为此重复全库。两槽/取消/旧結果仍有真实旅程门。
后续只补 immutable 候选证据与说明；产品反例须先退回 AUDITED，预算延续。

候选追加：[最终 Air 公开 consumer](../../../../evidence/correctness/foundation-a-apple-candidate/README.md)
含真实 CPU prefix cancellation、正常/失败/zero-output/close-old-result PASS；
[CPU+CUDA static SDK](../../../../evidence/correctness/foundation-a-sdk-cpu-cuda-static/README.md)
真实安装/rename迁址、public finite consumer、两个新/受影响头独立 TU、runtime与缺路线诊断 PASS。
其余格仍待固定候选资格，不能据此关闭七 owner。

## 固定候选风险矩阵（持续对证，不按测试数关闭）

| 能力 / public journey | 正常与边界 | failure / OOM / overflow | ownership / teardown | cancel / concurrency | consumer / 正交 | 平台 / delivery |
|---|---|---|---|---|---|---|
| 共同 exact snapshot / finite table | Win41项中 IDs/guard/View/表外/重复表 PASS | 配额OOM/前缀factory失败 PASS；算术沿原shape program边界 | 原Graph不变、allocator回收与原ID PASS | 表immutable；native两独立slot/旧结果 PASS；无新线程/全局 | 有限batch shape与变长资产/矩阵共享same opaque shape API | Win/Air source、Win CUDA static/shared PASS；其余SDK与WSL待验 |
| pure control / data+control reachability | shared control祖先/dead剪枝、Win/Air native+hybrid PASS | Win checked control前驱OutOfRange、cold OOM PASS；Air错误完整格待核对 | 原完成边界与旧output PASS；物化/managed等待明确 | public CPU prefix cancel Win/Air PASS；禁止hostwait拒绝 PASS | 独立张量分支顺序，无模型策略 | 候选Win/Air source PASS；Air Vulkan/其他SDK待验 |
| qualified CPU effects / 全图 footprints | public write→GPU→read、effect-only、CPU明确placement PASS | 缺/重复ID、不同ID alias、constant/input alias及原common OOM PASS；accepted DataLoss前缀 PASS | fixedborrowed backing、close-old-result与Context再用 PASS | public实际已开始prefix cancel Win/Air PASS；跨invocation冲突caller同步 | 显式sidecar候选更新与资产集合更新，schema/body由consumer声明 | Win/Air public source PASS；installed effects消费者待全部格 |
| native权限/不支持域 | 无实际GPUeffect端口明确拒绝；活mandatory-alias custom拒绝 | opaque native input+named effects alias未证者拒绝，不消费输入 | 原readonly native/F1 owner不改，效果回滚不承诺 | 原runtime退休/slot规则；本轮没有native producer实现变化 | pure initial/final保留，mutable业务动作显式分离 | 正向native pure+显式拒绝需各SKU；缺路线不得作math PASS |

Sanitizer：共同静态/metadata的新控制/shape/alias与原effect校验适用 ASan、UBSan，等待候选
定向结果；GPU math 不由sanitizer代签。TSan 无新增 mutable global/runtime机制，按上文裁决
NOT_APPLICABLE，本轮真实多slot/取消门仍须完成。producer unknown/native失联等未改原机制
只复用 tree/toolchain/配置匹配证据，若本轮反例改变退休路径则退回 AUDITED 后定向补证。
此矩阵中的“待验”是 INCONCLUSIVE，不是最终 correctness PASS。

候选定向追加：Win CPU+CUDA shared SDK、Air CPU+Apple static/shared SDK的真实迁址/public
consumer/newheads/缺路线/runtime通过；Air原错误profile缓存身份与原配置日志保留，恢复后
才签对应profile。共同 [ASan/UBSan](../../../../evidence/correctness/foundation-a-sanitizers/README.md)
各42项定向语义/Graph查询/全图effect预检通过，旧维护selector冲突FAIL及累计预算保留。
新增effect_preflight仅为候选后测试专用维护，不改产品算法/API/candidate。剩余Vulkan与WSL
CUDA实际source/package格、P0/LOC和终局红队尚待完成，状态仍CANDIDATE。
