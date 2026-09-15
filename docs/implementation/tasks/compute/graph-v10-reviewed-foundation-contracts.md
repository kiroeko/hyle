# Graph v10：整体复核后的 foundation 合同

## Closure card

- component / owner：`graph` / Graph；round=`graph-v10-reviewed-foundation-contracts`；
- baseline：`610a219f57e613ac6f1aaf79b337742122e2569f`；承接未 terminal 的
  [v9](graph-v9-foundation-routing.md)，保留已实现的 node-set/三轴资源/Completion 骨架。
- trigger：[整体复核 A1--A6](compute-graph-route-unification-audit.md#整体架构复核与合同修正)证明独立
  bounds、全局 effect chain、一律预算失败和不区分 replay 成本不足以承载既定旅程。
- 主纵切：动态 token 的 normalization→matmul→gated activation、精确 capture 变体与 general 路线；
  反偏置：单节点 exact CPU、自定义异步资源任务；四引擎的来源与取舍见总审计。
- scope：G100--G107；这是复合 planning/replay 能力，不是模型/KV/batching policy。
- non-goals：算子目录扩充、第三个 public Plan、通用符号求解器、全局自动调优/缓存、Provider 内核重写。
- 依赖：Compute/ComputeOps/Container/Mem；复用 Error、Queue/Completion；没有新增反向依赖。
- 性能：迁移 P0；全部有限 FIX 与真实 CPU/CUDA/Apple 探针稳定后，统一 one-op/MLP P1，沿用基础卡
  5% 工程门与完整边界/反向顺序，不以 managed kernel-only 成本对比 prepared 的完整调用。
- 预算：shape、selector、资源/variant ownership 各一轮表示与一轮纠正；定向红队一次。不在这些任务里
  改公共仪器；正常开发中的编译/fixture 错误不冒充最终性能裁决。
- 验证：G100--G107 的反例、原 exact/node-set 全回放、OOM/并发/完成与销毁顺序、逐头、三交付；
  Windows/Linux/两台 Apple 与适用 sanitizer 在完整候选验证，不因 docs/状态提交重跑 P1。
- stop：热态 HYLE search/JIT/分配、接纳后换路、用模型策略填基础合同缺口，或 required 约束被静默忽略。

## 已审计的有限改动集

| ID | 能力现状与动作 | 固定验收边界 |
|---|---|---|
| G100 | FIX replay 成本传播与冷筛选；复用 Compute 的属性校验/比较，不再复制字段列表 | unknown 不等于 prepared；prepared-only 排除 managed/unknown；report 原样保留，热提交不筛选 |
| G101 | FIX independent/derived shape 与 guards，复用 Compute provider-neutral program | 独立维度只绑定一次；flatten/concat/head split/broadcast、越界/overflow/非整除；完整域证明或 exact-only |
| G102 | FIX finite exact variants + 完整 general fallback | compile 后不新增 variant；concrete binding 固定路由，capture 精确绑定 shape/storage/queue |
| G103 | FIX leaf/region/placement 同池有界选择与完整成本 | 硬约束优先；copy/repack/sync/workspace/persistent/关键路径与 qualified identity；无证据稳定回退，不承诺全局最优 |
| G104 | FIX 有限事务化重选与预算终止 | 搜索预算停止扩搜而非禁止完成已有候选；有合格可行解返回并报告 optimization-limited，strict caller 可拒绝；无解失败，不吞真实错误 |
| G105 | FIX resource/effect domains，替换全局 last-effect 假串行 | schema access 来自 ComputeOps，资源 ID 属于 Graph；同域冲突有序、独立域可并行、未知保守；bind alias 与 region 边界反例 |
| G106 | FIX constant borrow/materialize 与 variant 聚合 | 成功 materialize 后 source 可释放；兼容 immutable artifact 显式共享，不能共享的副本披露；unique resident/冷峰值/并发/capture 全计预算 |
| G107 | KEEP exact replay/lifetime，并完成 report 与交付 | shortlist/拒绝/成本来源有界且可追溯；安装包普通 C++ 与 custom execution 消费同一 Compute Provider，不依赖 src SPI |

这些 FIX 来自 architecture 已接受的具体矛盾，不因本卡延长而重新审全库。Compute 的公开 owning custom
factory、ComputeOps access schema 必须先在各自有限卡完成；本卡不越过 owner 写其产品代码。

G105 的底层 schema 已在 [ComputeOps v3](compute-ops-v3-resource-access-schema.md) 绑定并实现。
Graph 的 graph-local identity/绑定、冲突顺序、backing alias 与 region 保真已实现并完成本卡 Windows
增量回归，见文末 G105 检查点；不再使用全局 last-effect 链。完整 foundation 的跨平台/交付/性能与终局
红队仍未完成，不把这一项等同于基础实现验收，也不据此扩充算子目录。

## API / owner / 成本

### G105 资源域表示审计（2026-09-14，实现前固定）

- `GraphBuilder::effect_resource()` 创建 `EffectResourceId`；`bind_resources(node, ids)` 把 ComputeOps 的有序
  资源端口绑定到这些 ID，可在 finish 前更正绑定，重复端口指向同一资源合法。Graph 提供对应只读查询。
  缺绑定/foreign ID 失败；ordinary pure 不配置资源。
- `finish()` 验证同域 RAW/WAR/WAW 必须已由 data/control path 排序，read/read 不加边；未知外部副作用
  与全部具名资源访问冲突。不会暗中按添加顺序插入边，也不会要求独立域共用一条链。
- `CompileOptions::effect_bindings` 给出实际资源 footprint：Mem buffer range 复用 resource/handle/offset/size；
  非地址化的外部对象用 owner 提供的 canonical identity。它是访问范围证明，不是新的执行参数通道，也不
  接管 owner；实际执行实现仍使用其声明的资源。用不同包装地址冒充独立 owner 是违反合同。
- Graph 在 compile 拒绝不同资源 ID 的 overlapping backing；同一 backing 的不同不重叠子范围允许。
  需要 alias 的端口应绑定同一 ID。外部 effect memory 与 Graph tensor 输入/常量/输出不得隐式重叠，否则
  会绕过 SSA 数据流；compile/bind 两处检查，建议把可变状态作为资源并显式产出 tensor value。
- footprint 在本 Executable 中固定且借用至所有 Invocation/Completion 销毁；跨 Invocation 资源互斥由
  调用方 Queue/Completion 保证。没有新增全局 registry/锁；未知资源走保守 effect，而不是伪造 footprint。
- Region 不得隐藏/改变所含 node 的副作用；请求携带按域聚合的访问声明，provider 必须明确承诺保持它。
  contracted data/control 边进入既有 lane/happens-before，反例覆盖独立与冲突的 region。
- 私有冷算法：每个实际使用域一遍 DAG writer 可达性传播，加按相邻 writer 序号分段的 backward walk
  检查所有未完成 readers；各段不重复访问节点，O(D(V+E)+A)、O(V+D+A) scratch，不制造 V×V 矩阵。
  无资源/未知 effect 的旧路线保留线性档；物理 footprint 校验 O(D²)，bind 对 D 个固定 footprint 检查。
  这些成本用于明确的外部资源访问，不增加纯图 submit 的 resource 扫描。

本表示不改变 Graph/Compute owner 或完成边界，属于 G105 的有限实现；不借它扩 custom kernel/模型目录。

保留 `GraphBuilder → Graph → Executable → Invocation`，Plan 仍私有。Builder 拥有 shape 表达式与资源
identity，compile 拥有有界候选和不可变制品，Invocation 只拥有本次 scratch/bind/capture/完成状态。
普通 exact 用户无需提供表达式、资源域或 profile；动态用户只绑定独立维度，派生维度由预构建有限程序
求值。冷编译默认允许已声明且满足硬约束的 prepared/managed 路线；显式 prepared-only 不放宽。

G100 的首步仅迁移已有 runtime facts：`ProviderReplayClass` 为 unknown/prepared/managed，字段
`replay_class` 默认 unknown，公共 value type 提供 `valid()` 与完整 equality；unknown 可被诊断/保留，
不能据此证明 prepared-only。实际政策筛选与 managed 成本预算不是这一个字段自动完成的能力。

派生 shape 不读 payload；资源访问不携带模型 KV/communicator 策略；常量共享不靠裸地址猜内容；无解/
真实 error 仍返回 Status。异步对象一直保活到真实 Completion，不能用 host 提交返回替代设备终态。
新私有存储布局和有界搜索实现可在上述合同内实验；若要改变 owner/默认政策/完成边界，先回到审计。

## 七问与当前状态

无新外部依赖/FFI；Provider 接入只复用已有因子，版本事实不按名称猜测。能力与易用性由四引擎的真实
旅程和 exact 反偏置约束；正确性/性能/三交付需按上述具体反例实证，不能以代码量、已编译或历史通过代替。
已有 AUDITED 轮按接受的架构基线绑定本后继有限卡；当前仍 `AUDITED`，尚未形成新的 Candidate/PASS。

## 2026-09-14 G100 首步

Graph 已复用 Compute `ProviderRuntimeProperties::valid()`/equality，region proposal 与 artifact 的 replay
class 变化会报错并销毁制品；CompileStepReport 原样传播 unknown/prepared/managed。region extension
版本为 6；实际 oneDNN MLP 声明 prepared，不从 shared ownership 推断。反例与 Windows shared/static
及 oneDNN 回归见 [Compute C91 检查点](compute-v9-shape-replay-substrate.md#2026-09-14-c91-成本分类检查点)。

G100 的 prepared-only 自动筛选和 managed 预算尚未实施，其余 G101--G107 未因这次元数据迁移自动
完成。仍为 `AUDITED`；未运行的 CUDA/跨平台/sanitizer/installed/P1 格不得记为 PASS。

### G101 表示审计（实现前固定）

复用 Compute C92 程序作为 Graph 私有 dimension record，不复制整数计算/guard checker。公开保留
`dimension(bounds)` 创建独立维度，新增 `dimension_sum/product/quotient/broadcast(Dimension, Dimension)`
创建派生 `DimensionId`；`require_dimension_equal/less_equal` 添加 guard。操作数必须为常量或同 Builder
已存在 ID；失败不追加半条记录。`dimension_count()` 枚举全部维度，新增 `independent_dimension_count()`
与 `dimension_is_independent(id)` 帮消费者识别需要绑定者；bounds 允许零，与 TensorDesc 一致。

compile 仅要求每个独立维度一个 ShapeBinding，拒绝用户绑定派生 ID；一次程序求值后解析 TensorSpec，
signature 保存全部解析值，因此不丢失派生维度 identity。exact-only 路线继续可用；这个步骤先以真实
Native concat 和派生 row spec 完成消费，不据此声称完整 bounded artifact、dynamic View attributes、
variant 或 runtime shape selection 已完成。后几项仍属于 G101/G102 的待办，不用固定 View 常量冒充动态
reshape。普通 concrete 图和 direct Compute 不需要构造 shape 程序。

## 2026-09-14 G101：有限派生维度接入 exact Graph

上述公开 dimension/guard API 已实现；Builder 只追加已验证的算式，Graph 私有记录直接使用 Compute
的有限程序。compile 只接受独立维度 binding，冷求值后解析 TensorSpec；signature 保留全部已解析值。
zero bounds 与 TensorDesc 对齐，未改写用户提供的固定 View attributes。两条实际执行是 concat 派生输出
轴与 batch×tokens normalization，细节和定向验证见
[Compute C92 检查点](compute-v9-shape-replay-substrate.md#2026-09-14-c92--g101-首个真实消费检查点)。

Windows shared/static 与 oneDNN 回放通过不等于 G101/G102 完成：目前 Executable 仍是一个 exact point，
缺少 bounded artifact/全域 schema proof、动态属性和 runtime variant 选择。G100 政策筛选、G103--G107
的统一成本/资源/常量/custom execution 仍待落实。保持 `AUDITED`，没有完成声明或新性能结论。

### G100 冷筛选表示审计（实现前固定）

新增 `ProviderReplayPolicy::{kAllowAny,kRequirePrepared}`，由 `CompileOptions::provider_replay` 指定，默认
allow-any。严格政策在 leaf proposal 后、资源选择与 artifact prepare 前筛除 managed/unknown；region
discovery 必须先声明 replay class，先筛选再影响 target/region 选择，不能等制品编译后才发现不合格。
region extension 升至 7；初始 known class 与后续 resource proposal 必须一致，unknown 可在默认政策下
细化，但不能通过 prepared-only。最终 artifact 继续用既有完整 runtime equality 校验。

沿用既有有限 proposal/node 预算，被政策排除的 proposal 仍计 discovery 预算；不在本步改变预算终止
策略。非法枚举/前后矛盾是合同错误，不静默 fallback。政策不把 prepared 等同于 shared/capture，也不
替调用者 executor/回调证明零分配。View/Copy 等非 Provider 步骤不受 Provider 成本类别筛选。

验证限定为：strict leaf/region 拒绝在 artifact 创建前发生；prepared delegated 可以入选；默认政策仍可
选择 managed；未知初始值不能伪装成 prepared，known 值不得改变；实际 Native/oneDNN 回放、非法政策、
fallback 与 proposal 预算反例。只改冷 compile，热提交不存储/查询这个政策，不启动单独 P1。

## 2026-09-14 G100：prepared-only 冷约束检查点

上述政策已实现，公开入口与限制见 [Graph 指南](../../../user/graph.md#何时要求-prepared-only)。leaf 在
proposal 后、resource/prepare 前筛选；region 在影响 target/partition 前筛选。discovery 的 known 类别
不得在 resource proposal 改变，artifact 继续完整比较 runtime facts；extension v7 无旧布局适配。oneDNN
discovery/resource/artifact 复用同一个 prepared 常量，未修改 MLP kernel 或调度。

定向源码/反例复核覆盖 unknown 后续细化仍不合格、managed 默认可用、prepared delegated 可执行、无
合格路线失败、跨 target 不被拒绝 region 误导、非法枚举、known 类别改变和被筛除 proposal 仍占预算。
View + 跨 storage Copy 的实际执行在严格政策下通过。新增 probe 的实际 artifact/resource 回调计数证明
拒绝发生在编译制品之前；两次 delegated submit 不重进 proposal/prepare。首次测试编译发现 fixture
未包含 CPU convenience 头，改用其已有显式 Context/Native 构造后通过，没有改产品便利入口。

- Windows Release source-tree shared/static：Graph builder/executable/逐头各 3/3，分别含 114 单元用例。
- oneDNN-enabled shared：Graph executable、真实 oneDNN MLP 与双方逐头 4/4（112 单元用例）；完整
  MLP 数值用例现明确采用 prepared-only，原普通默认路线仍由其他 region 用例覆盖。
- P0：每个 proposal 只增加 `O(1)` 冷比较，候选数/预算与复杂度不变；政策不存入 Invocation，也不由
  submit/Queue/kernel 查询。拒绝的错误 Status 属于冷路径，不承诺其零分配。
- impact selector 为 READY，只选择 Graph/Graph-oneDNN/逐头/交付及公共检查，跳过 89 个不受影响项。
  closure/docs/diff 通过；并行编译最多 2，未运行 P1 或全库测试。

这些不是完整安装 SDK、Linux/macOS、sanitizer 或真实 Apple managed 路线的资格化。G100 的分类传播与
严格冷筛选已落实，managed 完整成本、C93 identity、G101/G102 bounded/variants、G103--G107 其余范围
仍待完成。保持 `AUDITED`，不宣称基础架构实现/终局性能已经收口。

### G103 成本表示第一步审计（实现前固定）

源码仍保留 `estimated_net_savings_ns` 与 `kRequireEstimatedBenefit`，允许不同 Provider 相对各自 fallback
的数值相加/排序，违背已接受的 Compute §2.8 / Graph §6.1。DELETE 这两个字段/政策及未消费的私有 compile
ns；不保留兼容别名，不把旧 mock 数字移成常量表。需要量化裁决的真实 qualified profile 仍是 C93/G103
必须实现的后续部分，删除旧入口不等于该能力完成。

leaf/region 共用公开 `compute::ProviderRouteCostFacts`：workspace（size/alignment）、persistent/copy/repack
bytes、sync/dispatch count 都区分 unknown 与 exact zero，共同 `valid()` 检查 alignment/dispatch。runtime
仍独立携带 ownership/replay/threading，删除 region cost 中重复的 runtime bool。本步只是共同候选表示，
不冒充已完成 leaf/region/placement 同池搜索或完整资源预算。

保留已有显式 prefer-region 的确定性结构路线。dispatch 只能在本次参与比较的全部候选都声明时作为
次级比较；有 unknown 时整维不参与，避免把 unknown 当 zero，也避免 pairwise partial 比较形成非传递
sort。间隔 DP/一般 node-set 收缩和硬预算不在本步重写。功能反例需覆盖同一候选集中的 unknown/exact
dispatch、非法结构事实、重复编译稳定性及实际 oneDNN 路线；边界/异步/回退测试不再依赖虚构纳秒。

上述共同成本与旧估值清理已实现并通过定向回归，具体事实见
[C93/G103 检查点](compute-v9-shape-replay-substrate.md#2026-09-14-c93g103共同结构成本与旧估值清理)。
当前没有接受完整 qualified profile 或完成统一 selector；这个状态不能用旧 `kRequireEstimatedBenefit`
测试曾通过代替。后续仍需完成 G103 全部硬约束、完整计划成本与合法 identity 匹配。

C93 的 `ProviderInstanceId` 属于 G103/G107 冷事实传播：CompileStepReport 增加 instance 坐标，leaf 来自
PreparedOperation，region 来自已绑定 extension，Copy 无实例；metadata View 若由真实 Provider 冷准备则
保留该 ID，但不因此变成 kernel dispatch。只是固定大小诊断值，不让 report
保活 Context/artifact，不改变 Provider 家族顺序、选路算法或 warm 路径；完整 profile 匹配仍待实施。

### G104 候选发现预算审计（实现前固定）

现状 `RegionProposalWriter::add()` 把固定容量耗尽变成 ResourceExhausted，Graph 无条件丢弃已接收的
候选，违背 A6。不能捕捉任意 ResourceExhausted 当作正常终止，否则 Provider 真正的 OOM 会被吞掉。
REPLACE 为 `try_add() -> Result<bool>`：true 为已复制、false 为搜索容量不足并停止该 provider 的发现，
unexpected 仍是真实错误；callback 继续返回 Status，任何错误都原样失败。writer 记录究竟哪个容量
阻止了接收，未接收候选不进入选择，已接收候选仍通过完整 node-set/replay/cost/placement/workspace 验证。

FIX 默认允许有限发现：全局或 node 总额达到且仍有 provider 未探索时停止，per-provider 截止后可继续
下一家。`CompileReport::search()` 给出已接收的 proposal/node 数、实际 prepare 次数和触及的三个容量；
`optimization_limited()` 只说明这些预算限制过搜索，不承诺全局最优。新增 `CompileSearchPolicy` 的严格
模式可拒绝受限发现。无可行完整计划、required 约束失败、真实 OOM/Provider 错误仍失败，不能返回部分图。

这一步不把旧结构选择冒充 G103 的完整 profile/search，也不豁免尚未形成可行解时的 preparation 上限。
该上限若阻止完成唯一候选仍失败；后续同池重选须在有完整 incumbent 时保留它。各容量的实际 work/storage
界限不提高，warm 路线不读取政策/计数。oneDNN 只迁移发现循环，收到 false 正常停止，不改 pattern/kernel。

固定验证：global/per-provider/node 容量、恰好填满但未省略候选、后续 provider/target 跳过、strict 拒绝、
filtered proposal 仍计数、无覆盖/required/workspace/prepare 上限仍失败、真实错误不被受限标记吞掉、
可行受限计划实际执行与重复编译稳定、真实 oneDNN 受限发现和 source-tree shared/static/逐头。

## 2026-09-14 G104：受限发现与完整可行计划检查点

上述发现合同已实现，region extension 升至 9，无旧 `add()` 兼容入口。容量拒绝为 `Result<bool>` 的
false，真实错误仍经 Status 传播；Graph 不捕捉 OOM 来猜测是否应该回退。默认允许受限发现，严格政策
可拒绝；报告拥有实际计数与 global/per-provider/node 限制事实。填满但未跳过任何候选/provider 不标记
受限，filtered 候选仍占预算，跨 target 的总额不会重新充值，单 provider 的上限不阻止探索下一家。

定向反例验证三种容量、严格政策、未知 enum、writer 拒绝/错误分离、跨 target 跳过与 required placement、
真实 Provider OOM/Internal 传播、无完整覆盖、workspace/prepare 上限。受限的完整 region 计划连续三次
编译并执行 payload 正确。真实 oneDNN MLP 的 node 容量不足时完整 leaf 路线执行数值正确并如实报告。

- Windows Release source-tree Graph shared/static 各 3/3（各 122 单元用例及逐头）通过。
- oneDNN-enabled shared 的 Graph/direct-Graph/region 及逐头 5/5（122 单元用例）通过。
- P0：固定计数/标记只在冷 compile/报告使用；proposal 复制排序与选择复杂度不变，存储容量不扩大；
  正常容量终止不再构造错误 Status。submit、kernel、Queue/Completion 无新查询、分配、锁或搜索。
- impact READY，跳过 89 项；closure/docs/diff 通过。未运行 P1、全库、CUDA、跨平台/sanitizer 或 installed
  SDK，后四类最终义务保持待完整 foundation candidate 验证，不签全架构通过。

G104 的发现预算部分完成；当前仍没有 G103 qualified full-plan profile、同池有界重选/完整 incumbent
选择，也未完成 bounded variants 与资源/常量合同。因此组件保持 `AUDITED`，不把本次可行性回归当作
完整 selector、性能 SOTA 或基础收口。下一步继续补齐这些既定合同，不扩充算子目录。

### G103/G104 选路前置：事务化结果接入审计

当前 `prepare_placed_steps()` 调用会把 Compute 的结构化结果重新压成 Status，再据 `IsUnimplemented()`
重试；region `compile` 回调本身也只返回 Status。这既可能把 kind=Error 的 Unimplemented 当安全拒绝，
又不能正确处理明确的 candidate-local resource-budget decline。因此在接入 profile 排序之前，先完成
原有 G104/GP42 的共同结果合同；不新增并行架构决策。

MOVE Compute 已有 `ProviderPrepareOutcome/Kind` 到公开 prepared-operation 合同，并复用其合法性检查。
Region compile 返回同一类型；Graph 内部 leaf/region attempt 保留 success/decline/error 到实际选路边界，
不通过 Status code 猜 rollback。部分 artifact 必须先由共同 owner 回收，才能继续；错误与非法 outcome
立即返回，即使 Status code 是 Unimplemented/ResourceExhausted。原始 proposal 的不支持仍是冷能力查询，
不等于执行失败。当前候选拒绝后的 leaf fallback 可沿用，任意替代 region/同池重选仍是后续 G103/G104。

固定验证：两类明确 decline 可回退且部分制品先销毁；相同 Status code 的 Error 不回退；非法 kind/status、
prepared+error、strict no-leaf、真实 oneDNN 正常/失败与默认 exact 回归。P0 只改冷分类/清理，不改热提交；
沿用本轮定向 Windows shared/static/逐头，完整候选再做跨平台/SDK/sanitizer/P1。

## 2026-09-14 G103/G104：事务化结果检查点

Compute/Graph 共用公开四态 prepare outcome，region extension 升至 10、Compute 私有 SPI 升至 7；删除
压成 Status 的旧私有入口，无兼容别名。Graph 的普通 leaf、未入选 region 的 leaf，以及 region 安全拒绝后
的 leaf 三条调用链都保留分类；真正 Error 不因其错误码而重试。部分 artifact 在尝试下一路之前销毁。
direct Compute 同步修复，但只自动跳过明确 unsupported，资源拒绝仍直接返回，不扩大其选路政策。

新增反例覆盖两类安全拒绝、相同错误码的 fatal Error、非法 kind/status、清理顺序、跨 target 与两阶段
leaf fallback。初次编译发现一个未迁移的 fallback 解包点，已修复并纳入后续全部回归。

- Windows Release source-tree shared/static 各 9/9，通过各 256 个单元用例与四项逐头测试。
- oneDNN-enabled shared 定向 8/8，通过 141 个单元用例与三项逐头测试；direct、composition、Graph 和
  真实 MLP region 均包含在内。oneDNN 原始 compile 错误保持 fatal，不把 OOM 猜成可安全回退。
- P0：只改变冷准备的类型传播、分类与清理；不增加 submit/kernel/Queue 的分配、锁、搜索或间接调用。
- 构建沿用 `scripts/hyle-windows.ps1`、`--parallel 2`；ctest 精确选择 Compute/CPU/Native/Graph 及
  oneDNN provider/composition/region，不运行全库或 P1。三平台、sanitizer、installed SDK 和最终基础 P1
  仍待完整 candidate；本检查点只证明 Windows 增量路径。

这只完成统一选路所需的结果边界，不把 G103 完整成本/同池搜索、G104 全部重选、variants 或资源/常量
合同写成完成。状态保持 `AUDITED`。

### G104：安全拒绝后的候选重选审计

当前 compile 在入选 region 明确拒绝后只能拆成 leaf，不能尝试已经发现的其它 region/target；因此合法的
`operation_fallback=kReject` 也可能错失可行的另一分区。这是 G104 有限事务重选的直接实现缺口。

先固定一次 cold discovery，保存本次编译唯一的候选 ordinal/node set 和扩展 owner。一个构造尝试若因明确
region decline 或不兼容资源安全拒绝，销毁本次已建制品、移除该候选，再在原候选池重新放置/分区；不重新
询问 provider、不补充发现预算、不重置 preparation 计数。真正 Error、无匹配 destroy 的制品、公共分配失败
立即失败。每次重选永久排除一个候选，最多 R 次，仍受原 `max_placement_attempts` 硬约束。

报告记录被安全拒绝的 region 候选数，受拒绝影响的 leaf 标记 `kRegionDeclined`；正常拒绝不等于 search
limit。只有完整计划才可返回；禁止 leaf 时仍
可改选完整 region 覆盖，最终没有覆盖则返回最后一次明确拒绝。测试覆盖两个 Provider、同 Provider 不同
分区、跨 target/required placement、部分已建制品清理、全部拒绝、预算不充值、fatal 不重选及真实 oneDNN。
这一迁移仍不声称已实现 G103 同池完整成本、全局最优或已有 feasible incumbent 的完整优化循环。

## 2026-09-14 G104：有限 region 重选检查点

上述重选已实现。discovery owner 持有一次发现的 extensions/node arena/候选，构造尝试仅记录明确拒绝的
候选 ordinal；失败分区的已建制品先销毁，physical metadata 和 node→step 映射再重建。禁止 leaf 时不再
提前准备不允许使用的 leaf。成功报告新增安全拒绝数与 `kRegionDeclined`；真正 Error 仍立即传播。

- Windows Release source-tree Graph shared/static 各 3/3（各 136 单元用例及逐头）通过。
- oneDNN-enabled shared 的 Graph/region/direct-Graph 与逐头 5/5（136 单元用例）通过。
- 新增八项测试覆盖同/异 Provider、改选 node set、跨 target 与 required placement、先成功后拒绝的制品
  清理、全部拒绝、固定发现/准备预算、不兼容资源及逐次分配失败。每个故障点失败后回收全部本次分配，
  清除故障即可再次编译；成功重选的两/三/四节点路线实际执行数值正确。既有 fatal/async 回归继续通过。
- P0：正常构造不重复 discovery；最多 R 次排除/重选，每次消耗一个此前未拒绝的候选，准备总数受原硬
  上限约束。冷 scratch 增加 O(R) 拒绝标记，原 per-node coverage 计数改为一字节分类；计划/报告增加一个
  `size_t` 诊断计数。submit/Queue/kernel 无新搜索、分配、锁或调用层。
- impact READY，跳过 89 项；closure/docs/diff 通过。只运行本卡 Windows 增量验证；跨平台/sanitizer、
  installed SDK 与最终 P1 仍待完整 foundation candidate，不因本次通过签发全架构完成。

当前每个发现的 region 仍采用确定性资源选择，拒绝后不穷举该 region 的全部资源组合。G103 的
leaf/region/placement 同池完整成本与 qualified profile、G104 的完整可行 incumbent 优化循环及其余
G101/G102/G105/G106/G107 仍须继续实施；Graph 保持 `AUDITED`。

## 2026-09-14 G105：具名资源域与执行反例检查点

上述资源域表示已落地，旧全局 last-effect 实现移除。`finish()` 验证同域冲突的已有 data/control path，
不会替用户补边；未知访问与各域冲突，独立资源与同域 read/read 保持无额外边。backing 在 compile 固定
并复制借用声明，常量/input/output 的隐藏 effect alias 在 compile/bind 拒绝；exact shape 特化保留 ID。
Region extension 升至 11，请求携带聚合访问与 footprint；未承诺保留副作用的路线在创建制品前安全拒绝，
proposal/artifact 承诺矛盾则 fatal 并回收制品。oneDNN 的 pure 路线不需要虚构资源或副作用承诺。

- Windows Release source-tree shared/static：Graph builder/executable/逐头各 3/3，各 158 个单元用例。
- oneDNN-enabled shared：Graph、direct/composition/direct-Graph/MLP region 及三项逐头共 9/9，187 个单元
  用例通过。上述三个缓存的 CMake 重配置通过，中央依赖不变（Graph 已公开依赖 Mem），无新 feature 开关。
- 新反例包括 512 个确定性 DAG 与完整可达性 oracle 差分、全部 read/write 档位、writer 必须汇合所有
  readers、foreign/missing/rebind/重复端口/255-port、逐次 finish/compile 分配失败与恢复、实际缓冲区读写、
  两 target 独立/同域 read-read/有序 write-read、region 合并声明/安全拒绝/fatal 清理、不能出入重入 region、
  input/output 双绑定顺序与常量 alias、shape 特化、传入 span 及 Executable 销毁后的 Invocation 保活。
- 源码反查核对 CSR 容量受 resource-edge 总额约束、writer 链与 reader antichain、失败后的逻辑状态以及
  既有 lane/Completion 消费。测试 fixture 的一次 nodiscard 警告已修正；没有以放宽产品合同解决测试。
- P0：冲突分析 O(D(V+E)+A)、scratch O(V+D+A)。源码反查补了只读/单 node 域的无冲突短路；全部域均
  无冲突且无未知副作用时为 O(V+D+A)，不做 D 遍 DAG。footprint 校验 O(D²)，tensor bind 增 O(D)，region
  访问并集对本 region 的 Aᵣ 个端口为 O(Aᵣ log Aᵣ)。不建 V×V 可达矩阵，不在 submit 扫描资源。
- 私有 NodeRecord 新增端口范围起点，Builder/Graph 新增 packed resource edges/count，Plan 保存固定借用
  footprint；未使用时不为 resource payload 单独分配。公开 Graph/Executable/Invocation 仍为原句柄形状，
  Invocation 无新增字段；没有全局 registry/锁、HYLE 热态 search/JIT 或执行参数复制。
- `hyle-windows.ps1` Build/Test、`--parallel 2`，只选择上述 factor/直接消费者；impact READY、89 项跳过，
  closure/docs/diff 通过。没有运行 P1、全库测试、CUDA、跨平台/sanitizer 或 installed SDK；最终义务仍按
  本卡等待完整 foundation candidate，不能从 source-tree shared/static 推导 installed 通过。

footprint 是访问证明而非 custom execution 工厂；调用方仍须保证 canonical backing（不做 OS 物理 alias
探测）与跨 Invocation 同步。公开 owning custom execution、bounded/variants、统一完整成本/profile 与
constant ownership 等既定范围仍未完成；保持 `AUDITED`，尚未到用户要求暂停审视的 foundation gate。

### G107：owning Provider 的报告保活回归（实现前固定）

Compute C100--C103 在 `00abb380` 提供 owning factory 后，`build_report` 仍只复制 ProviderIdentity 的
string_view，而 CompileReport 明确不保活 Context/artifact。释放 custom Provider 和执行对象后保留
报告会产生悬空 name/version；这是 G107 直接消费者/报告合同的具体缺陷，不是新一轮架构扩张。

有限 FIX：报告按实际 ProviderInstanceId 去重名称/版本，在原 ReportState allocation 的尾部复制文本；
临时索引复用 Container FlatHashMap，全部分配使用现有 metadata allocator。报告继续不保活 factory、
Context、kernel 或代码 image；返回的文本 view 借用报告共享状态，不承诺活过最后一个报告 handle。
附加成本只在冷报告构造，禁止修改 Plan/Invocation 的 warm 状态或为修文档重跑 P1。

反例包括保留报告后先销毁全部执行 owner、复制/移动报告、同实例多 step 只复制一次、同家族不同实例的
不同文本、空文本/长文本、报告分配失败完整回滚。使用 quarantine allocator 在逻辑释放时填毒但保留
底层 allocation，确定性暴露旧的借用错误；实际 shared/static、原 Graph/oneDNN/CUDA 直接回归按本卡执行。

## 2026-09-14 G107：报告文本独立所有权检查点

上述反例在 `00abb380` 的 Graph 产品上确实失败：释放 custom factory 后，保留报告的 name/version 读到
quarantine 填入的 `!`。修复后同一测试通过，未通过保留 Context 或 factory 掩盖问题。
报告现在按配置实例去重，复制 name/version 到 ReportState 同一 allocation 的尾部；析构使用原大小/
alignment 释放。补充空文本、内嵌 NUL、长文本、同实例两 step 共享文本、同家族两 Context 不同版本、
报告复制/移动后继续观察等反例，并在原逐次 metadata OOM 测试中验证每次失败后的 allocation balance。
普通 package 示例也在全部执行 owner 释放后读取报告。

- Windows Release static：`compute-route-v8-static-20260914` 的 Graph builder/executable/逐头 3/3 通过。
- Windows Release oneDNN-enabled shared：`compute-route-onednn-v8-20260914` 的 Graph、真实 oneDNN region
  与两组逐头共 5/5 通过。只构建对应 Graph targets，未跑全库或未变化的 Provider 算子性能测试。
- `C:/hyle-scratch/custom-execution-v10-{source-consumer,shared-cuda-consumer,static-consumer}`：普通
  CPU/CUDA consumer 各 2/2 通过，包含 CUDA 真 kernel、双 Queue、exact capture 与异步错误。安装两格已
  重新构建/install Graph 并重链消费方；shared producer 与 consumer 的 `hyle_graph.dll` SHA-256 相同
  (`C31242548E9AA798FA2AD5AF1121EE7E6F683728FA7C2EABE85CBDAC5C59B498`)，不是旧 staged DLL 的假通过。
- P0：S 个 step、U 个配置实例、L 个去重文本字节时，报告新增构造成本期望 `O(S+U+L)`、临时索引
  `O(U)`；常驻增加一个 allocation-size 字段与 L bytes，无每 step 的重复文本或 factory/runtime retain。
  复用已依赖的 Container FlatHashMap、Mem allocator 与 Error；组件图、公开 handle layout、Plan/
  Invocation layout 和 warm 算法均未变。不运行 P1，不签新速度结论。
- impact READY，89 项跳过；docs/closure/diff 通过。全库代码 670,274 行、自实现不含测试 121,888 行；
  Compute 产品 16,156 行、Graph 产品 10,606 行，合计 26,762 行（统一 LOC 脚本、当前工作树）。

这是已接受 G107 的确定生命周期修复和 Windows 增量验证；Linux/macOS、sanitizer、终局独立红队与
foundation P1 仍留给完整 candidate。C93 完整证据匹配、G103 同池完整计划选择、bounded/variants 与
constant ownership 等剩余项没有被缩减或宣称完成，Graph 继续 `AUDITED`。

### G103：Provider 硬约束准入（实现前固定）

Compute 设计 §2.8 已接受全局 allow/deny 与逐 node Provider 约束。当前仅有 Context 安装顺序、target
placement 和全局 region 开关，无法要求 Q4 子图的某个节点走指定实现，同时允许其它节点继续外包。
先补这个共同候选入口，再接完整成本/profile 和有界计划搜索；不把新增约束作为性能裁决替代物。

本次有限表示：`CompileOptions` 借用 `allowed_providers/denied_providers` 家族 ID 列表和
`node_providers`。后者的 `NodeProviderConstraint` 必须匹配家族，可额外指定当前 Context slot 的
ProviderInstanceId；默认不约束，不引入全局 registry。allow 为空表示不限制，deny 优先；节点为硬约束，
不能由 region fusion、fallback 或资源重选覆盖。逐节点软偏好与完整计划成本仍属 G103 的后续实现，
不暴露尚无语义的 preferred 开关。

在 compile 冷入口拒绝无效/foreign/重复节点声明；筛选器复用在 leaf propose 和 region discovery，
全局被禁 Provider 连 extension query 都不调用。region 只有全部覆盖节点约束相容时才可进入 placement；
被拒的 region 不准备 artifact。约束不写入 Plan/Invocation 常态状态，提交无新增查表。
反例：deny-only、allow-only、冲突、未知实现、错实例、同家族不同配置、foreign/重复节点、region 跨越
两个不相容约束、资源拒绝后的重选仍守约、实际输出与回调不执行证明。只跑 Graph 与实际直接消费者、
逐头和三交付，最终跨平台/性能门保持原卡范围。

## 2026-09-14 G103：Provider 硬约束检查点

上述公开形状已实现。冷 `ProviderPolicy` 先验证家族/节点声明，排序家族列表并为显式节点建立临时
ordinal 索引；leaf 的全部准备/回退入口共用该 policy，region 在进入 placement 之前检查其全部节点。
全局排除在 extension query 前生效，资源拒绝后的同池重选不会恢复已排除的 Provider。逻辑 View/Copy
仍通过 Provider 准备，编译器插入的 storage transfer 则由 backend 拥有，不把 allow/deny 当成设备隔离。

- 11 个新增测试覆盖 allow/deny 交集与重复项、同家族不同槽位、未知/过期身份、无效/异图/重复节点、
  fusion 冲突、跨 target 与 required placement 冲突、region/leaf 资源拒绝、实际数值输出与策略借用终止。
  18-node 反例让两张家族表和节点索引都越过 inline storage，逐次注入全部 compile 分配失败并确认回滚/
  恢复；submit 没有新 metadata allocation。初轮唯一失败是新测试误用 resource-only region fixture，
  已改为具名数值 singleton fixture，未放宽产品约束。
- Windows Release static `compute-route-v8-static-20260914`：Graph builder/executable/逐头 3/3 通过，
  executable 内 147/147 用例通过。oneDNN-enabled shared `compute-route-onednn-v8-20260914`：Graph、
  真实 oneDNN region 与两组逐头共 5/5 通过。均由 `hyle-windows.ps1`、`--parallel 2` 定向执行。
- 普通 CPU/CUDA 三交付在 `C:/hyle-scratch/custom-execution-v10-{source-consumer,static-consumer,
  shared-cuda-consumer}` 各 2/2 通过，含真正 CUDA capture/replay。安装两格重新 build/install Graph
  到各自 `-static-sdk/-shared-sdk` 后 configure/relink consumer；源码格重新 configure/build。shared
  producer/consumer 的 Graph DLL SHA-256 同为
  `3EA2C15AE46AF0372D28CE05A14C27D9C84EE500EF48C2CF8C4A4A02FFF35640`，非旧 DLL 假通过。
- P0：G 个家族条目、N 个节点、K 条节点约束时，冷索引为 `O(G log G + N + K)`、空间 `O(G + N)`；
  无节点约束不建立 N 表，默认空策略无堆分配。家族匹配 `O(log G)`、逐节点身份匹配 `O(1)`，region
  检查线性于 proposal node 总量。公开 CompileOptions 增加三个借用 span；Plan/Invocation/layout、
  provider kernel、Queue/Completion 和热执行算法均不变。复用现有 Container SmallVector/Mem/Error，
  CMake 组件边与构建开关不变。
- impact READY，89 项跳过，docs/closure/diff 通过。记录前 LOC：全库 670,876、自实现不含测试 122,012；
  Compute 产品 16,156、Graph 产品 10,730，合计 26,886 行。

本检查点没有执行全库/P1；Linux/macOS、sanitizer 与终局独立红队仍按本卡等待完整 foundation candidate。
G103 的完整成本/profile、同池有界择优与软偏好仍待实施，bounded/variants 和常量所有权等既定 FIX 不缩减。
Graph 保持 `AUDITED`，没有达到用户要求暂停审视的 foundation 完成节点。

### G103 共同候选空间：普通算子发现与准备分离（实现前固定）

当前 region 已有一次性 proposal arena，但普通算子仍在 placement/fallback 中逐个 propose 后立即
prepare；因此不能比较未选 leaf，也会在 region 重选时重复能力查询。按既定同池选择合同，先把所有
允许的 `(node, target, provider slot)` leaf 冷提案与其 canonical 资源要求收集为本次 compile 的候选，
不在发现时构造重型制品。region 与 leaf 分别保留 typed payload，由同一次冷发现 owner 持有；接下来
完整 selector 消费二者，不制造 runtime registry。

固定表示：追加独立 `max_operation_proposals` 候选存储上限及报告计数/受限原因；prepare 总预算仍只
计真实构造尝试。达到提案上限停止发现，不补预算；strict 策略拒绝受限搜索，普通策略仅可返回完整可行
计划。禁止 operation fallback 时不发现不允许的 leaf。先应用 Provider/required target/replay 硬约束，
query 的明确 unsupported 不入池，其它错误直接传播。提案及借用资源只活到 compile 返回，重选复用
原提案、不重复 discovery；精确资源在 prepare 共同 validator 再检查。

反例：后排 Provider 的正常候选也能被发现但未选不 prepare、同一 leaf 跨 region 重选只 propose 一次、
发现/准备预算分离、受限但有 region 完整覆盖、受限且无覆盖、strict 停止、fatal 不吞、全部 cold OOM
回滚以及 source/static/shared/oneDNN/CUDA 直接消费者。原测试中误命名的 single_prepare_count 实为
proposal 计数，必须拆清/按真实阶段校准，不能把发现回调说成重型 prepare。P0 只增加有界冷候选存储，
不改变热执行；完整成本/profile、统一有界计划搜索仍是后续同一 G103 FIX，不因此宣称已完成。

## 2026-09-14 G103：普通算子候选池检查点

上述表示已实现：每次 compile 一次收集允许的 node/target/Provider proposal 和资源要求；region 安全
拒绝后的事务重选复用它们。只为选中的路线创建 artifact，`max_operation_proposals` 与实际 prepare
预算分开，报告区分接收数量与截断。禁止 leaf 时不发现 leaf；strict 截断在构造制品前失败，普通截断
仍须得到完整可行图。顺序为稳定的 node/target/Provider；小容量不保证找到候选空间中任何存在的解。

- 新增 6 个反例：后排候选被发现但不 prepare、两预算分离、恰好用满/真正截断、无完整覆盖时失败、
  后排 fatal 不吞、region 重选只查询一次；执行实际数值并复验全部 metadata OOM 回滚。原 fixture
  的 `single_prepare_count` 实为 propose 次数，已改名并和 artifact 次数分列；初轮 10 项旧阶段断言
  失败后按新发现合同校准。partial cleanup 的观察移到真正 prepare，并给 fixture 合法 leaf 路线，
  仍要求旧 artifact 在新准备之前销毁，没有放宽失败/所有权合同。
- Windows Release static `compute-route-v8-static-20260914`：Graph builder/executable/逐头 3/3，
  executable 153 用例、builder 25 用例通过。oneDNN-enabled shared `compute-route-onednn-v8-20260914`：
  同三项、真实 oneDNN Graph/逐头、Compute 与 CPU custom execution 共 7/7 通过。
- 普通 CPU/CUDA source/installed shared/installed static 各 2/2 通过；样例使用公开候选上限/报告，
  包含真正 CUDA kernel、双 Queue、exact capture/replay 与异步错误。缓存沿用前一检查点；安装两格
  重新 build/install Graph 并 configure/relink consumer，全部安装在显式 scratch prefix。shared
  producer/consumer Graph DLL SHA-256 同为
  `C2AE6918C80EC0B75C7D968663B9DC88F4B49C2809992045B9D25F1F7EB0952C`。
- P0：N 个 node、T 个 target、P 个配置 Provider 时，最多查询 `O(NTP)` 次（不包含 Provider 自身查询
  内部成本）；K 个已接收候选的资源端口总数为 R 时，冷存储 `O(NT+K+R)`。复用 Container SmallVector、
  Mem allocator 和 Compute 共同 proposal/资源 validator，不自建 matcher。候选只活到 compile 返回；
  Plan/Report 的 search 事实增加一个计数和一个标志，Invocation 与热执行算法不变，不新增热查询/
  JIT/分配。组件边和 CMake 开关不变。
- 用户指南区分普通/region 候选、真实准备、容量与失败；校准逻辑 View/Copy 的 Provider 筛选与编译器
  插入 backend transfer 的不同归属。impact READY，88 项跳过；docs/closure/diff 通过。当前自实现不含
  测试：Compute 16,156、Graph 10,836，合计 26,992 行。

本次没有运行全库/P1，不以 Windows 增量代替 Linux/macOS、sanitizer 或终局独立红队。G103 的统一完整
成本/profile/有界计划选择、G101/G102 bounded/variants 与 G106 常量所有权仍须完成；旧确定性 region
优先选择尚未因此变成完整成本 selector。状态保持 `AUDITED`，foundation gate 未签发。

### G103：leaf 资源规划不提前构造制品（实现前固定）

`25ad97ec` 已分离能力查询，但第一遍 target/storage 放置仍调用 leaf prepare：后续节点根本没有资源路线，
或前面的 region 最后拒绝时，已经为 tail 构造的 kernel/JIT/packing 全部浪费。完整计划成本不能建立在
这种边规划边准备的流程上；先拆这个具体耦合，再接 profile 与统一搜索，而非先加无人消费的打分类型。

有限 FIX：提取同一 leaf 资源选择器，规划阶段保存 proposal 与端口/workspace storage；只在已选 contraction
的物化阶段准备它。原每 node PreparedOperation 临时数组换成轻量的冷 placement，实际 artifact 直接归
最终 step。prepare 使用保存的资源，不重复 Graph 的 storage 搜索，Compute 的共同最终校验仍保留；
明确 candidate decline 才按已有稳定顺序
重新放置后备路线，错误仍 fatal，累计准备预算不重置。region 的完整资源 skeleton 与统一成本搜索仍属于
同一 G103 后续，不因本步没有提前 leaf prepare 就宣称全部规划都已解耦。

反例固定为：后续节点无能力/无合法资源时前节点零 prepare；region 首次拒绝不提前构造 tail；约束、跨
storage/alias、候选 decline、准备预算、实际数值与全部 OOM 回滚保留。Windows Graph shared/static、
oneDNN 以及普通 CPU/CUDA 三交付按本卡定向回放；不改公开 API/产品 CMake/backend，不跑 P1 或全库。

本轮源码消费实测暴露样例 staging 竞态：两个可执行文件的 POST_BUILD 同时写相同目录的相同 DLL，
`hyle_compute_backend_cpu.dll` copy 返回 Permission denied；没有存活的样例进程。有限修复只为该独立
sample 的 CPU/CUDA executable 设置各自 runtime 输出目录，继续使用现有 staging helper，不引入伪依赖/
全局串行化，也不修改中央构建系统。三交付复验包含这项真实消费修复。

## 2026-09-14 G103：leaf 资源规划/制品准备分离检查点

上述改动已完成：两遍初步放置只查询资源兼容性并保存候选/端口/workspace storage；制品按选中
contraction 的执行单元顺序准备并直接归最终 step。正常入选 leaf 不重新做 Graph 的 storage 搜索；
Compute 仍校验实际 proposal/绑定。明确 prepare decline 才尝试后续 Provider/target，预算跨重选累计，
真正错误不回退。删除每 node PreparedOperation 临时数组以及早期无用的描述/resource 数组填充。

- 新增后节点无算子、后节点 required storage 无解两例，均证明前节点 prepare=0；受限候选导致图不完整
  也要求零 prepare。已有 region-retry 数值用例从 5 次实际 prepare 降为 4 次：首次 region 拒绝之前不再
  准备 tail。原 alias/跨 storage/硬 Provider 约束/异步/全部 OOM 回滚用例通过，没有改宽错误或预算判据。
- Windows Release static `compute-route-v8-static-20260914`：Graph builder/executable/逐头 3/3，含 155
  executable + 25 builder 用例；oneDNN-enabled shared `compute-route-onednn-v8-20260914` 的 Graph/
  真实 oneDNN region/两组逐头 5/5 通过。产品修正后重新完成上述运行。
- 普通 CPU/CUDA source/installed static/installed shared 各 2/2 通过，包含 CUDA 双 Queue、真实 kernel、
  capture/replay 和异步失败；静态库已重新安装/relink，shared producer 与 `cpu/`、`cuda/` 两处实际
  staging Graph DLL 的 SHA-256 同为
  `C1A6B33A6D4C60563B48C78C1D3079314087C85D9A0398096E4064A0824C5B7F`。样例单独目录修复后继续
  `--parallel 2` 构建，不依赖串行重试。源码配置因样例 CMake 改动重生成 SIMD prerequisite，但没有运行
  SIMD 测试、修改 SIMD 产品或据此重签性能。
- P0：每 node 临时保存一个借用候选指针，workspace/端口继续使用已有 storage 索引；候选拥有者只在
  compile 内存在。存储搜索的渐进复杂度未变，正常最终 prepare 额外只线性恢复已选端口资源，不重做
  storage 枚举；提前重型 prepare 的浪费已由回调计数反例证明减少。公共 handle、Plan/Invocation
  layout、热提交算法、组件边与产品开关均未变。没有将少一次 prepare 外推成固定百分比或稳态胜利。
- impact READY，90 项跳过；docs/closure/diff 通过。自实现不含测试：Compute 16,156、Graph 10,847，
  合计 27,003 行（统一 LOC 脚本）。

下一段仍是 region 的冷资源 skeleton 与完整候选计划成本/qualified profile/有限搜索；这里只拆掉 leaf
提前准备耦合，未完成统一 selector。bounded/variants、常量所有权、真实 CPU/CUDA/Apple 探针与终局
跨平台/sanitizer/独立红队/P1 仍在原范围内。Graph 保持 `AUDITED`，foundation gate 未签发。

### G103：region 冷资源 skeleton（实现前固定）

`948e47ec` 的 region 仍在选中后才查询边界资源/runtime/effects，完整 selector 因而看不到未选候选的这些
事实。有限 FIX：在同一次有界发现中保存所有硬约束允许的 region 的 boundary、资源要求、效果并集与
runtime；重选和最终 compile 复用，不重新 query，也不提前准备 artifact。target required 不相容的提案
不做资源查询；明确 unsupported/不保全效果记为冷拒绝，其它错误（含未选后排）直接失败。

为避免把旧边界收集的逐 region 全图扫描扩大到全部候选，先 `O(V+E)` 统计 value 使用次数，再以带标记
的 node membership/region 内使用计数，在候选自身 node/port 上找边界；重复输入去重，但计数按实际
端口累加。结果只存在 compile 的临时平坦存储中，容量由已有 proposal/node/arity 上限约束，所有增长
检查溢出/OOM。资源要求复用 Compute 原 validator，不新增公共 API/语义平行实现。

验证：后排被查询但不 prepare、跨两次 partition 重选只 query 一次、required target 在 query 前生效、
晚发现 fatal 不吞、malformed requirement 提前失败、复杂/非连续 region 的边界与全扫描 oracle 一致、
重复输入/外部多消费者/只发布中间值/effects/零端口，以及原实际数值/OOM/预算回放。执行 Graph 与
Compute 受影响直接消费者、逐头及 CPU/CUDA 三交付，不运行 P1/全库；统一资源放置与完整成本搜索
仍属于随后 G103，不以存下 skeleton 就宣称已实现。

本段结果（2026-09-14～15，Windows 本地时间）：

- 已保存每个硬约束允许候选的 boundary/resource/runtime/effects；全局 proposal ordinal 作为缓存索引，
  target 过滤产生的空位不误用其它候选。冷 unsupported 进入同一 rejected 集；其它查询失败在任何
  artifact 准备前传播，重选不重复 query。具体 storage/lane 的完整计划搜索仍未完成。
- 新增 6 个用例；其中 32 个确定性 DAG 的全部长度 1–3 连续窗口，与独立全图扫描 oracle 对照并实际
  执行 leaf fallback；覆盖重复输入、外部多消费者、发布中间值和只发布中间值。原非连续/重排/effects/
  零端口用例也使用该 oracle。输入、输出、workspace 各检查坏 alignment、矛盾 flags、无 authority 的
  exact domain；均在 prepare=0 时失败。
- 两次 partition 的回退用例证明每个资源提案只 query 一次，旧制品先销毁；逐分配失败回放扩到 18 节点，
  越过 boundary index、resource skeleton、flat values/requirements 的内联容量，逐项 OOM 后均可重试且
  live allocations 归零。没有放宽错误、数值或预算断言。
- Windows Release static `compute-route-v8-static-20260914` 4/4（含 161 个 executable 用例）；shared
  `compute-route-onednn-v8-20260914` 的 Compute/CPU/Native/oneDNN/Graph 与逐头 13/13；
  `compute-graph-v7-cuda` 的 backend/late-failure/Graph/逐头 4/4 通过。最后扩大 OOM/仅中间输出案例后，
  static 定向 4/4 再通过；没有为最后的测试数据与说明修改重跑整套交付。
- CPU/CUDA 普通 source、installed static、installed shared 各 2/2 通过，使用本卡前述 v10 custom
  consumer 缓存；三种交付均重建 Compute/Graph，安装始终使用显式 scratch prefix。shared producer 与
  consumer 的 `cpu/`、`cuda/` 两处 Graph DLL SHA-256 均为
  `FEE9294B5438FF13376BB78E1AC0BB3E000663F7F20A418CAA8BC4769E964186`。
- P0：边界收集单独从逐 region 全图扫描改为一次 `O(N+V+E)` 索引、每候选自身 node/port 遍历；效果
  并集仍为该候选访问项排序。缓存增加的是有界 compile 临时存储，不进入 Plan/Invocation 或 warm submit。
  为完整裁决会查询更多未选候选，不把局部复杂度改善等同于整个 compile 的计时胜利。没有运行 P1，
  没有新稳态性能结论；公共 API/layout、组件边、产品开关不变，header 仅补充查询阶段合同说明。
- guide/closure/docs/diff 校验通过。统一脚本自实现不含测试：Compute 16,162、Graph 10,952，合计
  27,114 行。两个组件保持 `AUDITED`；完整 selector/profile/有限搜索、bounded/variants、常量所有权与
  终局跨平台验证仍未签发，不代表 foundation gate 完成。

### G103/G104：完整物理 skeleton 与统一 prepare 拒绝（实现前固定）

`1c955b3c` 仍边发出 physical step 边 prepare：后排 region 无合法 storage 或输出 copy 无承载 target 时，
前排 artifact 已创建；leaf prepare decline 还会原地换 Provider/target/storage，不适合完整计划成本比较。
本段固定改为先建立全部 leaf/region/copy 的物理步骤、端口与 storage，再只准备这份确定的路线。
不伪造尚需 artifact 才能获得的精确 workspace/persistent 事实；lane/内存与实际 Queue transfer 校验仍在
既有后续阶段，不能把 physical skeleton 当成完整可执行资格。

leaf 复用已有 node/target/Provider 候选，增加 compile-local rejected 位；明确 prepare decline 与 region 一样
销毁本次制品后排除对应候选、整图重规划。禁止在最终 prepare 循环改变路线。新增
`CompileSearchReport::operation_route_rejections` 披露该事实；候选、累计 prepare 上限仍独立且不重置。
真正错误（包括 kind=Error 的 Unimplemented/ResourceExhausted）立即终止。空 prepared handle 只存在
compile 私有事务内；仅 Plan/Report 已有 search facts 增加冷计数，不增加每 step/Invocation 字段，
不允许未准备步骤进入 schedule/report/replay。

验证：后排 region 资源无解与最终 copy 无 target 时前排 prepare=0；region/leaf 在同一重选中互不重复
发现；leaf 拒绝后 target/storage 变化重新生成 copy/依赖，已有前缀先释放；所有预算、hard Provider/
target、错误与 OOM 回滚反例、实际数值及 CUDA 多 lane 继续通过。只做相关 Graph/oneDNN/CUDA/
逐头/三交付及公共检查，P0 审查新增冷位与循环边界，不运行 P1；统一 qualified 成本/state search 仍继续。

本段结果（2026-09-15）：已按上述顺序拆开构造/准备，移除 leaf 原地换路线的 prepare 循环及单节点 span
中转。空 handle 只留在私有构造期；成功返回前每个非 copy 步骤都取得合法 owning artifact，后续调度与
replay 仍使用原结构。新增 report 的 leaf 拒绝计数，默认 0；没有修改已 CLOSED 依赖或 CMake 组件边。

- 新增 4 个反例，executable 共 165 例：后排 region 资源拒绝不提前准备前缀；最终输出 copy 无承载
  target 时 leaf/region 均零 prepare；leaf 拒绝先释放 region 前缀；region+leaf 混合拒绝共用发现结果和
  累计预算。原跨 target 反例扩成跨 storage 实际计算，证明替代路线不残留旧 copy；6 次 prepare 的完整
  路线在预算 5 时正确失败，没有扩大预算掩盖退化。原 OOM/错误/硬约束/并发用例通过。
- Windows Release static Graph/逐头 3/3，shared + oneDNN Graph/两组逐头 5/5，CUDA Graph 1/1 通过，
  缓存分别为前述 `compute-route-v8-static-20260914`、`compute-route-onednn-v8-20260914`、
  `compute-graph-v7-cuda`。CPU/CUDA 普通 source/installed static/installed shared 各 2/2 通过；三个交付
  旅程均重建 Graph，安装前缀仍显式指定 scratch。
- shared producer 与 consumer 的 CPU/CUDA Graph DLL SHA-256 均为
  `8ED2039F7C7308F4C3437919B439B27B80C2F7BCDBD17B8DC5E12A23F92A8F4F`。公共 report 字段变化后安装
  consumer 已重新编译/relink，不用旧客户端二进制冒充交付验证。
- P0：每 leaf 候选增加一个冷 rejected 标志及其对齐，每 Plan/Report search facts 增加一个计数；不增加
  per-step/Invocation 字段或热提交逻辑。每次安全拒绝至少排除一个此前未拒绝的有限候选，累计 prepare
  上限仍生效；准备循环是已选步骤/端口的线性遍历，不搜索路线。重选可能重建成功前缀，属于显式冷
  事务成本，并未签成全局性能胜利。没有运行 P1 或全库测试。
- impact READY，89 项跳过；closure/docs/diff 通过。Graph 自实现不含测试 10,863 行，Compute 未改仍为
  16,162 行，合计 27,025 行。保持 `AUDITED`；下一项仍为完整候选计划的 qualified 成本、同池选择和
  state 预算，不将本段当成 G103/G104 或 foundation 整体完成。

### G103/G104：统一无 profile 候选搜索（实现前固定）

`7f552012` 的 prepare 已与 physical skeleton 分离，但初始 target 仍优先有 region 者，随后只让 region
互相竞争；即使配置中更靠前的 leaf 可用，也可能被高覆盖 region 绕过。当前没有完整 build/device/driver/
workload 身份的 profile，不能把 `ProviderInstanceId` 单独当计时资格。本段先落实无 profile 的确定性
完整路线搜索，C93 的真实 qualification 与成本比较仍必须随后接入，不能据本段签完 G103/G104。

固定实现：所有已接收 leaf 与 region 共用按最小 node 建索引的冷候选表；required Provider/target 先过滤，
region 的重叠/联合成环不合法。按用户 preferred target、配置 target/Provider 顺序枚举，coverage/dispatch
只在此前顺序等价时 tie-break。用非递归、可恢复的有限深度搜索产生完整 node 覆盖，交给已有 physical
构造/prepare；无资源/copy 路线排除此组合后继续，明确 prepare decline 排除对应候选，真实错误终止。
首个完整且全部后续硬校验通过的计划才可返回；不得返回部分图或在 submit 扩搜。

新增独立 `max_route_search_states` 和相应 report 计数/limit；候选存储、搜索、重型 prepare 三种预算不得
混用或重置。无完整路线时耗尽失败；exact 用满但未省略必要搜索不标记受限。公共 region policy 改为
allow/disabled，不保留误导性的高覆盖优先名字或兼容别名。target/storage/lane 完整成本和 qualified
incumbent 的继续优化仍未实现，不声称 DFS 找到全局最快方案。

验证限定：前排 leaf 不被后排 region 抢占；前排 region 可胜出；先选组合资源不通而后备可行；非连续
region 联合环与回溯；hard target/Provider 不突破；搜索/prepare 上限独立且累计，错误不吞，原 OOM、
数值/生命周期、CPU/CUDA/oneDNN 和三交付回放。P0 审查最坏状态预算与 compile 临时空间，P1 仍待终局。

资源联合约束反例：`View` 必须 alias input 0，但 required output storage 可以通过先复制 input 0
满足；不能先贪心选 input 所在域，再把不同的 required output 当矛盾。此段同时修正该确定的冷路径
约束传播，新增真实跨 storage View 的数值/ownership 回归；不改变 View 的零拷贝别名语义。

### 2026-09-15 统一无 profile 搜索检查点

- 已移除 region-only DP/greedy 与先按 region 支持选择 target 的两段逻辑。leaf/region/target 共用一次
  建立的冷候选索引，按最小未覆盖 node 做可恢复非递归搜索；已拒绝候选的子树跳过，联合成环或无
  copy 路线排除组合。prepare 仍在完整 physical skeleton 之后，submit/Invocation 没有新增搜索层。
- 无 profile 的顺序是 preferred target、配置 target/Provider，然后覆盖/dispatch/提案序号；旧的
  `kPreferProviderRegions` 已删除，公开名字是 `kAllowProviderRegions`。已有五个测试依赖旧的 suffix/
  dispatch 优先次序：按新合同改断言或把 decline 注入真正首选 prefix，仍保留数值、回退和销毁断言。
- 新增六个回归：两种配置顺序下 leaf/region 同池；同 Provider 等价候选的 dispatch tie；累计 state
  上限与 exact-capacity；部分覆盖不得 prepare；隔离 storage 的复制失败可回溯而真实验证错误终止；
  required View 域先复制 input 再建立 alias。原联合环测试增加组合拒绝计数，原逐分配失败回放仍通过。
- P0：冷索引额外空间 `O(C + N)`，排序 `O(C log C)`；最多 `S=max_route_search_states` 次候选访问，
  每次 node-set 检查/回溯最坏 `O(N)`，不再递归。完整物理构造仍有独立资源成本，最多随已访问完整
  路线重建；实际制品准备另受 `max_placement_attempts` 约束。一般 DAG 的组合数可能指数增长，
  不能把有限搜索写成多项式全局最优；没有完整可行解时预算耗尽失败。
- Windows：static 的 Graph/逐头 3/3（执行套件 171 用例）；shared + oneDNN 5/5；CUDA Graph 1/1。
  公开 owning custom-kernel sample 的 source/shared/static 各 CPU+CUDA 2/2，均用本轮 public header
  重编；不消耗 GitHub CI。shared producer 与 CPU/CUDA consumer 实际加载目录的 Graph DLL SHA-256
  一致：`DD087FDD8697BB2C3EED10675A75DC5E54E5135F976091CA5921656F83AA93BD`。
- selector（基线 `7f552012`）READY，89 项有据跳过；docs、closure、diff 检查通过。没有修改 registry、
  Backend、CMake、公共性能仪器或已 CLOSED 依赖。Linux/Apple、sanitizer、独立终局红队与 P1 仍待完整
  foundation 候选，不把本轮 Windows 结果冒充那些资格。
- 自实现不含测试：Compute 16,162，Graph 10,758，合计 **26,920**，比上一检查点净减 105。Graph 口径
  为 public headers + graph + graph_provider_onednn；Compute 口径沿用前轮且本轮未改。

本段仅完成无 profile 的有限可行路线搜索，不签完 G103/G104：qualified identity/cost、storage/lane
联合优化与有 incumbent 的优化截断仍待接入；G101/G102 的完整 shape domain/variants、G106 常量物化与
聚合预算、真实 CPU/CUDA/Apple 架构探针及统一终局验证也未完成。下一步仍在这些基础合同内，不扩算子。

G103/C93 下一段固定为[稳定实现键与临时 token 分离](compute-v10-owning-custom-execution.md#c93稳定实现键与临时准备-token-分离实现前固定)：
Graph 只迁移 region 提案与冷 report，复用 Compute `ProviderRouteKey`；实际选择仍按本轮无 profile
合同，不因 Provider 填了非零 key 就授予性能优先级。keys 临时数组活到 build_report，不进入 PlanStep/Invocation。

### G103/G104：最终硬约束进入同一重选事务（实现前固定）

接完整成本/profile 前复核发现：当前搜索只包住 physical/prepare，最终 schedule/workspace 在搜索外；
首选制品虽合法但完整 workspace 超预算时，后备更省内存的路线不会被尝试。不能把尚未通过这些硬约束
的 physical skeleton 当作可比较计划，也不能将真实 OOM 当成候选不合适。

有限 FIX：把 schedule/segment/workspace 收入同一次有界冷事务。只有本地检查明确发现缺可用 lane 或
完整 workspace 无法满足显式预算/允许的有界序列化时，才拒绝当前**组合**并继续已有搜索；不永久排除
其中每个 Provider。其余错误原样终止，特别是 metadata OOM、Provider 错误和别名合同损坏。候选发现、
state/prepare 预算继续累计；丢弃制品、合成 delegated lanes、序列化边与 workspace/alias 决策后再重选。
只有完成最终校验才写入胜出 route key/report，不放宽 hard Provider/target/storage 约束。

验证固定为：大 workspace region→小 region/leaf 的真实数值回放、delegated lane/旧 key/制品不泄漏、
全部无解、硬约束与 state/prepare 上限不突破、逐 allocation fault 保持 fatal/可重试，以及既有 Graph/
oneDNN/CUDA/逐头和三交付。只移动既有冷流程并增加明确的拒绝标志，warm 路径不改；不跑 P1，不据此
签完 qualified profile、storage/lane 联合优化或有 incumbent 的继续优化。这仍在 G103/G104 原 FIX 内。

### 2026-09-15 最终约束重选检查点

- 已将最终 schedule/segments/workspace 放进同一候选事务；只有明确的本地 lane/预算拒绝触发重选，
  不根据 ResourceExhausted/InvalidArgument 名称回退。新组合复用原候选池与累计预算；旧制品、合成
  lane、串行化边和 alias/workspace 决策先释放/清除，全部校验成功后才生成胜出 report/key。
- 两个新增数值反例在 `d0552110` 产品上稳定失败（大 workspace region 挡住更小 region/leaf）；修复后
  通过。新增共 6 个用例，还覆盖缺 lane 改选 target、required 不放宽、state/prepare 上限、全部无解、
  真正 Provider OOM 不吞、旧串行化边不污染后备并行计划。原 metadata allocator 逐次失败回放保留
  18 节点 prepare-decline 模式，另增完整 workspace 拒绝与 delegated lane 的模式；每次失败清理后可重试。
- Windows static Graph/逐头 3/3，最终执行套件 180/180；shared + oneDNN Graph/region/逐头 5/5；
  CUDA Graph 1/1。缓存沿用前述 v8 static、v8 oneDNN、v7 CUDA；没有修改 Backend 或 Provider 产品。
- 公开 custom-kernel sample 的 CPU+CUDA source/installed shared/installed static 各 2/2，沿用 v10
  consumer 缓存。安装前缀均为显式 scratch；shared producer、SDK、CPU/CUDA 加载目录的 Graph DLL
  SHA-256 同为 `68C09ABA41EC1C40D25F292334EEE686A79E0BB87C9FAC4C1748154EA4E3D053`。
- P0：正常首选成功只移动既有冷校验，没有增加每 step/Invocation 常驻字段或 warm 逻辑；重选才清理
  旧决定并重新 schedule/plan。最坏冷成本受原候选 state/prepare 与每候选序列化上限限制，仍不保证全局
  最优；为取得真实 workspace 而准备的失败制品有实际成本，不把修复签成计时胜利。
- selector 基线 `d0552110` READY，89 项有据跳过；docs/closure/diff 通过，registry 不变。未运行 P1、
  Linux/Apple、sanitizer 或独立终局红队；均仍属完整 foundation candidate 的义务。
- 自实现不含测试：Compute **16,181**（未改），Graph **10,800**，合计 **26,981**；全库自实现
  **122,107** 行，统一脚本 self-check 通过。本段不是完整 profile/成本选择的交付，foundation 尚未完成。

### G106：常量 source release（实现前固定）

现有 ConstantBinding 一律借用，不能满足模型加载 staging/mapping 在 compile 返回后释放的明确需求。
本段添加 `ConstantPolicy::{kBorrow,kMaterialize}`（默认 borrow），不改变逻辑 tensor 描述和 replay 用法。
materialize 的目标是该 constant 已声明的 external storage 域（默认 0；沿用 value/external placement），
可以与 source 不同；使用 Mem Buffer 和 Compute 明确支持的复制/Completion，不暗中 staging/算子求值。
copy 在完整物理路线成立后、入选 Provider prepare 前发生；每个 constant 每次 compile 只物化一次，
后续事务重选共用该不可变副本。只能在 cold compile 等待自有复制，绝不执行用户 graph；source 必须已
ready 且在调用期间不变。失败不接管 source，accepted copy 必须到真实终态后才清理副本。

Graph 给 Provider 的 prepare 常量改为有效的物化副本；resource query 不读取 payload，也不能把旧 source
地址缓存为制品输入。Executable 拥有 buffer 到全部 invocation/completion 释放；先销毁制品再释放它可能
借用的常量，最后释放 Context。borrow 不因为 aggregate packed count 而缩短寿命。逐 constant report
给 policy、storage、description 与 owned bytes；published owned constant/view 返回 Executable ownership。
新 ConstantBinding 布局对应 region extension 升版，不解释旧版或保留兼容头。

限定验证：CPU 数值与 source 释放、混合/重复/错误 policy、zero/strided 常量、跨 storage、copy 失败与
OOM 清理、重选不重复复制、制品先于副本销毁、report 不保活；真实 oneDNN MLP、CUDA upload/capture/
Completion 及源码/static/shared。P0 检查默认 borrow 零 payload copy、物化按字节一次冷复制，warm 不变。
本段交付 source-release；variant 间共享、opaque packed artifact 的身份/唯一 resident 与冷峰值预算仍是
G106 后续，不用 canonical 副本字节冒充全部 Provider 内存，profile/variants 也仍须继续。

### 2026-09-15 G106 source-release 检查点

- 已实现显式 `ConstantPolicy`、逐 constant report 与 `OutputOwnership::kExecutable`。Graph 在完整
  physical skeleton 后创建 canonical Buffer，并以真实 Completion 等待复制；Provider prepare 使用
  替换后的 view。默认 borrow 不复制 payload，region extension 升到 13，不保留旧布局解释。
- 新增 6 个 CPU/region 用例：混合/乱序 bindings、源释放与 Executable/Invocation 共享、zero/strided/View、
  allocation/map/metadata 逐次失败、重复/错误 policy、最终 workspace 重选只复制一次、制品先于常量
  销毁、报告不保活、无 transfer 路线与 accepted late failure。已有形状/外来 ID/effect overlap 反例扩到
  materialize；没有放宽数值、错误或预算断言。执行套件 186/186。
- oneDNN 新用例覆盖两种实际 gated MLP 拓扑，释放 gate/up/down 源后数值仍正确；CUDA 新用例覆盖
  pinned-host→device 与 device→device。公共 CPU 示例先释放源再 replay；CUDA 示例同时覆盖动态输入
  和物化常量，源在 capture 前释放，三次 replay 不增加 proposal/prepare。
- Windows Release static `compute-route-v8-static-20260914` Graph/逐头 4/4；shared + oneDNN
  `compute-route-onednn-v8-20260914` Graph/Provider/逐头 5/5；`compute-graph-v7-cuda` Graph 1/1。
  末尾仅扩充 malformed-binding 测试后只重建并回放 static，不为测试断言再次重跑三交付。
- 沿用 v10 custom sample 缓存，source/installed static/installed shared 各 CPU+CUDA 2/2，均按当前头
  重编；显式 scratch install prefix，未触发 CI。shared producer、SDK 与 consumer 的 CPU/CUDA 实际
  DLL 目录 Graph SHA-256 一致：
  `D0F4ED844CDBBEC2B16EFD0156E835A1832002ACAFA09995A2033E2C5B5CFA50`。
- P0：物化最多按每个 constant 的必要字节冷复制一次，route 重选不重复；选择 transfer 的上界是每常量
  遍历配置 targets，Queue 在一次物化中按 target 复用。只有请求物化才分配 canonical ownership 表；
  report 新增每常量冷记录，默认借用也需这份报告元数据。PlanStep/Invocation/warm submit 没有新增字段、
  搜索或 payload copy；不将冷路径复杂度审查签成计时胜利。
- guide/架构/示例说明已更新。selector 基线 `5869e8ce` READY、89 项有据跳过；closure/docs/diff 通过。
  registry、Backend、CMake、公共仪器与已 CLOSED 依赖均未改。P1、Linux/Apple、sanitizer 和独立终局
  红队仍待完整 foundation candidate，不能由本轮 Windows 回放外推。
- 自实现不含测试：Compute **16,181**（未改）、Graph **10,978**，合计 **27,159**，净增 178；全库
  自实现 **122,285** 行，统一脚本 self-check 通过。组件保持 `AUDITED`。本段完成 canonical source
  release，不签完 G106 的跨 variant 共享/packed identity/唯一 resident 与冷峰值预算；C93/G103/G104
  完整 profile/成本/有 incumbent 的优化，以及 G101/G102 bounded/variants 仍须继续。

### G103/G104：完整候选 profile 与保留可行解（实现前固定）

- 公开冷输入 `CompileProfileContext` 由 profile 的 producing owner 提供真实 build、device/runtime/config、
  storage/replay 环境、语义实例及 workload identity；target/Provider 绑定须与本次 Context 的真实坐标匹配。
  Graph 不从地址、显示版本、POD padding 或临时 route token 伪造可持久身份。缺事实保留 unknown。
- Graph 从本次完整计划生成结构身份：内建属性、图边/副作用、原 shape program、全部 TensorDesc、
  选中 Provider 实现键、target/storage/lane、copy、alias/workspace placement、segment/dependency、常量策略。
  自定义实例语义、外部 build/config 与未来实际 replay 环境仍由 owner 证明；报告的 key 不签发测量资格。
- `CompileOptions` 接收只读完整 Graph-terminal profile；普通无 profile 仍止于第一个完整可行方案。
  首个方案成本未知时也保留确定性选择。已知基准才继续有限搜索；只有同合同成本区间严格分离且达到
  调用方显式 improvement 要求才替换，不相加单 kernel 时间、不把 unknown 当零、不做隐式测速。
- 保留一个完整 incumbent 和一个 trial，二者共享唯一 canonical 常量 backing；只移动冷计划制品/元数据。
  state/prepare 容量有独立的明确终止标志；allow-limited 返回已验证 incumbent，strict 拒绝。真实 OOM/
  Provider 错误一律失败，不因存在 incumbent 被吞掉。无可行解仍失败，warm submit 不存 profile 或再搜索。
- 固定反例：完整成本择优/区间重叠/未知/失配、首选与后选实际数值、build/config/shape/属性/resource
  变化、state/prepare 截断、真实错误、逐 allocation fault、constant source release/制品销毁、实际 Native/
  oneDNN/CUDA 与三交付。本步不宣称已完成 storage/lane 全组合优化、bounded variants 或终局 P1。

### 2026-09-15 完整候选 profile 与 incumbent 检查点

- 上述 G103/G104 已接入真实 compile：Graph 从实际逻辑/物理计划生成身份，复用 Compute matcher；
  首个完整可行解有合格成本后才继续搜索。未知/失配/重叠不替换，调用方经济门与测量区间分别判断。
  同一次编译最多保留 incumbent 与 trial 两套路线制品，共享唯一 canonical 常量；搜索或 prepare 容量
  耗尽只在 allow-limited 下返回可行解，strict 和真正 OOM/Provider error 仍失败。
- 新增 12 个 GraphProfiles 用例，覆盖完整成本、重复记录保守包络、冷成本摊销、错误完成边界、硬
  Provider/并发约束、换 Context 后稳定身份、属性/bounds/stride、预算截断、metadata 逐 allocation
  fault、workspace/lane/alias 还原、常量源释放与销毁顺序。真实 Native 与真实 oneDNN gated MLP 均可
  在完整 leaf/region 计划间选择并验证数值；计时区间及外部身份是明确标注的 synthetic fixtures，
  不构成 qualified measurement、自动硬件身份 producer 或性能胜负。
- 最后身份测试的 fixture 曾漏给 symbolic op 的输出规格，并误要求 dense-only Native 支持非标准
  stride；前者按公开合同补规格，后者用 published-input 检查同字节数、不同 stride 的边界身份。
  没有为测试放宽产品。最终 Windows static 执行套件 198/198，Graph/逐头 4/4；shared + oneDNN 5/5；
  CUDA Graph/逐头 4/4。缓存分别为 `compute-route-v8-static-20260914`、
  `compute-route-onednn-v8-20260914`、`compute-graph-v7-cuda`，均在 `C:/hyle-scratch/`。
- 沿用 v10 public custom-kernel sample，source/installed static/installed shared 的 CPU+CUDA 各 2/2；
  新增 profile-context/report 的公开链接消费，无 `src/` 或伪资格化 key。shared producer、SDK 和
  consumer 的 CPU/CUDA 实际 Graph DLL SHA-256 均为
  `84424AF08E4EAC54B2E21181E88C232F35A5416051B99B4D98193C297172D22B`。
- P0：无 profile 的默认路径仍停止于首个可行解；报告增加固定冷 metadata，profile 的变长向量只在
  opt-in 时分配。身份编码按图/shape/物理计划大小线性扫描，Provider binding 关联为每 step 扫描显式
  binding span；每候选匹配遍历调用方 profile span。最坏冷搜索继续受 state/prepare 预算限制，额外
  峰值包含两套路线制品，尚不宣称完整 resident/cold-peak 硬预算已实现。PlanStep/Invocation 与 warm
  submit 未改，没有热态 profile、搜索、JIT 或 payload copy。
- impact selector 基线 `090cda28` READY，89 项有据跳过；closure/docs/diff 检查通过，registry 未改。
  最大编译并行度 2、CUDA 测试串行，未启动 CI、全库测试或 P1。Linux/两台 Apple、sanitizer、独立
  终局红队与真实性能资料资格仍待完整 foundation candidate，不由本轮 Windows 增量通过代签。
- 自实现不含测试：Compute **16,404**、Graph **11,673**，合计 **28,077**；全库自实现 **123,203**，
  全库代码 **675,429**；统一脚本 self-check 通过。
  仍为 `AUDITED`；G101/G102 bounded/variants、G106 跨变体共享/聚合预算、storage/lane 联合选路及
  CPU/CUDA/Apple foundation 探针继续按既定卡推进，不扩算子目录。

### G101/G102：共享计划的完整域实例化（实现前固定）

接入 [Compute v11](compute-v11-bounded-instantiation.md) 的已编译域，不为每个具体尺寸复制整个 Plan。
`CompileOptions::shape_policy` 默认 exact；显式 declared-domain 模式消费 GraphBuilder 已声明的完整
bounds/guards，compile 不接受单点 shape binding。`Executable::create_invocation(shape)` 只绑定独立
维度；Invocation 暴露实际 specialization，重复 submit 沿用原执行路径。普通 exact 用法不变。

Plan 保有原 Graph 的只读 shape program/spec，调度、workspace 上界、常量和非并发制品的互斥 gate 仍为
同一份。仅 domain Invocation 分配 concrete logical descriptions 并从域根冷实例化普通 Compute 制品；
不复制整份 DAG、重新发现 Provider、prepare/JIT 或重新规划 workspace。actual TensorView 截取实际字节，
copy 与 alias 绑定使用实际描述，不能按 envelope 多复制。根与 source metadata 可释放，存活 Invocation
保活其计划和域制品；具体实例的 workspace 不超过 compile 上界。

只有真正使用非 exact 轴的 node 才请求 EntireDomain；静态 node 继续 exact prepare。当前 region SPI
没有域实例化合同，因此含任一动态 node 的 region 不准入 general 路线，静态 region 仍可组合。常量
在 Executable 生命周期内保持固定描述；动态尺寸的外部值应为 input。compile 上界不是合法单点：只用于
安全内存包络，不能对可能违反 guards 的上界组合调用 exact schema 后宣称完整域正确。domain/exact
进入不同 profile identity，不复用“恰好上界”的 exact 计时记录。

固定反例：guard 排除 envelope、派生维度、完整小域数值、zero extent、多节点内部 workspace、实际
copy 字节、静态/动态混合、exact-only Provider/region 拒绝、错误/重复/foreign binding、源与根释放、
冷分配失败与恢复、不同尺寸并发和非并发 gate、重复 submit 无新增 metadata allocation；原 exact、
oneDNN、CUDA、逐头与 source/static/shared 回放。只执行 P0 与相关正确性，不运行 P1/全库/CI。

本段是 G101/G102 的 general 路线，不冒充全部完成：dynamic View attributes、有限 exact variants、
跨变体共享和总预算、真实架构探针与终局验证仍保留在原 FIX 集内。

### 2026-09-15 G101/G102 general 路线检查点

- 上述表示已实现：公开 declared-domain compile 与 concrete `create_invocation(shape)`，冷投影完整
  program/axes/guards 给 Compute；静态 node 沿用 exact。Plan 只保有一份原 Graph，不按 shape 复制
  DAG、schedule、workspace 或互斥 gate。具体绑定和内部/复制 TensorView 使用实际描述，submit 路线未改。
- 新增 12 个 Graph/region 反例：完整小域数值（含 guard 排除 envelope、派生乘积和 zero）、内部 workspace、
  输出尾部哨兵、跨 storage 实际复制、固定节点与动态节点组合、动态常量拒绝、非法/重复/foreign/derived
  binding、exact-only 与不等价输出 symbols 拒绝、逐次 compile/create OOM 回滚恢复、原 Graph/常量源/
  公开 Executable 释放、不同尺寸并发、异步非并发制品的共享 gate、domain/exact profile 身份隔离。
  成功 warm replay 将 metadata allocator 设为拒绝分配，仍完整执行。
- 两处 fixture 用法被纠正而未放宽产品：有 caller Queue 的图不能使用 queue-free submit；带哨兵的较大
  backing 读取有效部分必须用实际 TensorView，不能给整块 Buffer 的 exact read 传较短 span。另一次
  install 使用了 CMake 不支持的 `--message=NEVER`，在安装前失败，随后去掉该选项完成真实安装与验证。

| Windows Release 验证 | cache（`C:/hyle-scratch/` 下） | 结果 |
|---|---|---|
| Graph static / 逐头 | `compute-route-v8-static-20260914` | 3/3 |
| Graph shared、oneDNN Graph/region / 两组逐头 | `compute-route-onednn-v8-20260914` | 6/6；Graph execution 210/210 |
| CUDA Graph / 逐头 | `compute-graph-v7-cuda` | 3/3；包含真实 CUDA/capture 既有回归 |
| source CPU/CUDA 公开 sample | `custom-execution-v10-source-consumer` | 2/2 |
| installed shared CPU/CUDA sample | `custom-execution-v10-shared-cuda-consumer` | 2/2 |
| installed static CPU/CUDA sample | `custom-execution-v10-static-consumer` | 2/2 |

命令沿用 Windows wrapper 的 Build/Test，构建至多 `--parallel 2`、测试 `--parallel 1`；只重建对应 target。
CPU 公开 sample 额外用 Native RMSNorm 演示 `rows=0..4` 域的 0/1/4 行实例、源 Graph 释放与重复执行，
没有私有 include 或另一套 FFI。安装显式使用 `custom-execution-v10-{shared,static}-sdk` scratch prefix。
Graph DLL 在 shared producer、SDK、CPU/CUDA 实际加载目录的 SHA-256 一致：
`B92684FD6403D40E5BA2974414530377D8915B7F674D17697A3A5A9064BD8CD4`。

P0：compile 沿用有限候选搜索；每个动态节点投影和准备其完整域 metadata，普通 exact 节点不分配域。
domain Plan 保有一次原 Graph IR；每个 domain Invocation 增加 `O(V+D)` concrete descriptions/signature，
求值及 schema 检查为 `O(D+G+V+E)` 加各 schema 成本；每个动态 step 再经 Compute 的域绑定/alias 检查，
因此不是整个多节点实例化只做一次 `O(D)`。这全部在冷创建，且沿用已有 allocator/shape evaluator。
不复制 `O(V+E)` 的物理计划，不重新选路/prepare/JIT。workspace 预留完整 envelope，不能声称为小实例
自动压紧；需要精确预算可选 exact 编译。exact Invocation 只增加空冷 metadata 字段，无额外分配次数；
warm callback 层数、步骤与 completion 算法未改，不据此签性能胜负。

用户/架构和公开 sample 已同步；selector 基线 `a0b210c4` READY，89 项有据跳过。registry、backend、
Provider 产品、公共 CMake/治理/性能仪器未改。现有 shared + oneDNN graph 产品缓存的 CMake 重配置成功；
oneDNN 自身仍输出既有 CMP0146 OLD 弃用提示，未为本模块改写上游构建。Linux/两台 Apple、sanitizer、独立终局红队和统一 P1
尚未执行，仍等待完整 foundation candidate；本段不是它们的 PASS。
Graph 自实现不含测试 **11,862**，Compute **16,670**，合计 **28,532**；全库自实现 **123,658**。
保持 `AUDITED`，继续 G101 动态属性、G102 finite exact variants、G106 跨变体共享/聚合预算及其余既定
基础 FIX。没有达到用户要求暂停审视的 foundation 完成节点，也未扩充算子目录。

### G102/G106：有限 exact 表与共同常量 owner（实现前固定）

`CompileOptions::exact_variants` 是 declared-domain 编译的有限 shape binding 表，普通 exact 编译不接受。
每项必须满足原完整域，重复 concrete shape 拒绝；通用路线必须先编译成功。随后在同一次冷 compile
准备全部请求的 exact 路线，某项失败则整次失败，不偷偷返回缺项的产品。所有路线沿用同一 Graph、
environment、硬 Provider/resource 约束与 profile 输入；不引入业务 bucket/cache 或 warm compilation。

Executable 保有 general Plan 和 exact 子 Plan；子 Plan 可独立保活，因此 materialized 常量改为一次
独立 `util::Ref` owner，持有 Mem Buffer。所有同 Graph constant slot、同描述和 external storage 的
路线借用同一不可变副本；不能靠地址碰巧相同猜权重相同。owner 不反向持有 Plan，不形成引用环；
每条存活路线均保有 Context lease，制品先于最后一份常量副本销毁。borrow 策略不改变。

创建 Invocation 先用既有 shape evaluator 验证独立值/guards，以全部维度的确定性有序索引查 exact 表，
命中后只创建该已编译 Plan 的 Invocation，否则走既有 general。查找在冷路径，热 submit 不变。
报告可枚举各 exact route/signature，Invocation 可查询命中序号；默认报告仍描述 general，累计编译工作
单独查询。prepare/state/proposal/node 总预算跨全部路线累计，不能每个变体重新充值；per-provider 单次
discovery cap 仍由总预算约束。常量、workspace 等逐路线报告不能简单求和当作 unique resident。

验证：乱序绑定/变体表、重复及域外拒绝、exact 命中/general fallback 数值和生命周期、全表预算耗尽、
常量仅复制一次、根释放而子路线/Invocation 存活、OOM/部分变体销毁、报告不保活 payload、无 warm
搜索/新增 allocation，及原 exact/domain/oneDNN/CUDA/三交付。没有新的 kernel 或 Provider 产品修改。

本段的范围是 finite table 和 canonical sharing，不签完 G106：opaque packed identity/共享、全部 unique
resident/冷构建峰值/并发与 capture 预留的整体硬预算仍是同一 FIX 的后续，必须单独实现、不能以每个
variant 的 workspace 合格代替。P1 和终局跨平台仍只在完整 foundation candidate 执行。

### 2026-09-15 G102/G106 有限变体检查点

- 已实现上述有限表：`CompileOptions::exact_variants` 只在 declared-domain 模式生效，先准备完整 general，
  验证全部点与重复后准备所有 exact 子计划。任意错误清理整张表；没有懒编译、缺项成功或业务 cache。
  `Invocation::exact_variant_index/executable` 披露实际路线；父/子报告分开，报告不持有 payload。
- canonical materialization 使用现有 Util Ref 和 Mem Buffer 共享，子路线可独立保活；prepared artifact
  先于常量、Context lease 销毁。borrow 不改，默认无变体路径不创建 shared owner。source release 与
  多次 compile 重选不重复复制的原合同保留；Provider 自有 packing 尚未共享。
- discovery/state/prepare/node 上限跨 general/exact 累计扣除，`total_search()` 汇总真实计数与限制标志，
  不相加不同尺寸的执行时间。源码复核纠正了耗尽预算继承为零时误报 `InvalidArgument` 的路径，现对
  prepare/state 返回 `ResourceExhausted`；显式非法零参数仍保留原拒绝合同。region 剩余额度为零时可在
  allow-limited 下由合法 leaf 完成，strict 拒绝；没有候选覆盖完整图仍失败。
- 新增 8 个测试（7 个 GraphVariants、1 个 GraphRegions），另扩充 profile 身份反例。覆盖全小域的
  exact 命中/general fallback/zero/哨兵尾部、乱序与重复/非法/外来/派生 binding、exact-only 不能代替
  general、共享常量只分配一次及父/源释放、报告不保活 payload、各级累计预算、每次冷分配失败回滚与
  恢复、并发、空句柄/零维图与无 warm metadata 分配。身份 fixture 不含真实性能资格。

| Windows Release 验证 | `C:/hyle-scratch/` 下缓存 | 结果 |
|---|---|---|
| Graph static / 逐头 | `compute-route-v8-static-20260914` | 3/3；诊断修正后执行套件 1/1、218/218 |
| Graph shared + oneDNN Graph/region / 逐头 | `compute-route-onednn-v8-20260914` | 6/6；诊断修正后受影响三组 3/3 |
| CUDA Graph / 逐头 | `compute-graph-v7-cuda` | 3/3，包含既有真实 CUDA/capture 回归 |
| source CPU/CUDA 公开 sample | `custom-execution-v10-source-consumer` | 2/2 |
| installed shared CPU/CUDA sample | `custom-execution-v10-shared-cuda-consumer` | 2/2 |
| installed static CPU/CUDA sample | `custom-execution-v10-static-consumer` | 2/2 |

Build/Test 沿用 Windows wrapper，target 仅为上述测试/样例，编译至多 2、测试串行。sample 在同一公开
CPU journey 中同时消费 exact 表、actual route、各报告与 general fallback，不从源码 SPI 获得能力。
两个 SDK 从同名 `custom-execution-v10-{shared,static}-build` 的 `hyle_graph` target 构建，安装显式指定
对应 `...-sdk`，未写入 Program Files。shared producer、SDK、CPU/CUDA 装载目录的 Graph DLL SHA-256
均为 `80A2E6214217433090965A1E8D72CD53E21FEE8BF8C170FE69B3B00BB7B1CBE9`。
现有 shared + oneDNN graph 闭包重新 configure 成功，保留已有 oneDNN CMP0146 OLD 弃用提示，未修改上游。

P0：K 个变体、D 个维度、G 个 guard 的表验证为 `O(K*(D+G) + D*K*log K)`，临时点表 `O(K*D)`，
排序后的常驻索引 `O(K)`；随后确实构建 K 个 exact 计划，不把冷编译成本说成免费。具体创建先绑定域，
再作最坏 `O(D*log K)` 查找；命中直接创建普通 exact Invocation，回退沿用已编译域。canonical payload
只复制一次，每个子 Plan 持有一个共享引用；默认 Plan/Report 增加空冷字段及计数，无新增 heap 分配次数。
Invocation 布局、warm callback 层数与 submit 算法未改，不由这些结构事实签性能胜负。

用户/架构及直接 Compute/Native 记录已校准。selector 基线 `09b3a990`；registry、Backend、Provider 产品、
CMake、公共治理/性能仪器均未改。未启动 CI、全库/P1；Linux/两台 Apple、sanitizer 和独立终局红队仍待
完整 foundation candidate，不由本段代签。保持 `AUDITED`：G101 动态属性、G103 storage/lane 联合优化、
G106 packing/唯一 resident/冷峰值/并发/capture 总预算、真实平台探针与共同终局资格仍未完成。

统一 LOC self-check 通过：全库代码 **678,254**、全库自实现不含测试 **123,938**；Compute **16,670**、
 Graph **12,142**，合计 **28,812**。本次没有完成整个 foundation gate，不扩充算子目录。

### G106：编译制品的累计存储准入（实现前固定）

当前只有逐路线 workspace 和累计搜索预算，不能限制保留整张变体表的常量/Provider 制品。第一步固定为
`CompiledStorageReport` 与 `CompileOptions::compiled_storage`：统计 canonical materialization 的请求字节、
各独立制品声明的 persistent bytes、region 额外 packing，以及未知 persistent 的制品数。普通 operation
必须传播已有 `PreparedOperation::persistent_bytes()`，不能像当前 step report 一样丢弃它。

- 报告区分单路线与整个 family；已共享的 canonical allocation 在 family 只计一次，各独立 Provider
  制品/packing 逐份计费。报告不持有 payload，未知量不记成零，也不通过 Provider 名称猜共享。
- 有限 `max_bytes` 下默认拒绝未知量；显式 allow 只约束已知部分，报告仍无完整总量保证。默认无限上限
  保留现有路线。checked 累加溢出拒绝，不使用饱和值冒充合格。
- 每个完整候选在同一冷事务内接受此门，超预算或未知量可安全销毁并尝试其它路线；后继 exact 子计划
  使用 family 已消耗额度扣除后的余量，不重新充值。固定点表任意一项失败则整体不发布，不隐式删项。
- 这是 **compiled storage 的一项预算**，不是进程 RSS/物理显存、完整编译峰值或运行期总内存：Graph/
  Compute 控制 metadata、allocator rounding、borrowed 外部 backing、Invocation workspace/capture 和
  Provider 内部临时内存不冒充已覆盖。完整 unique resident/冷峰值/并发/capture 与 packing 身份共享仍是
  G106 剩余实现，不缩减 foundation gate。不得为此在热提交加计数、锁或扫描。
- P0 为每个冷候选 `O(constants + steps)`、常量附加空间；family 累计 `O(variants)`。预声明测试：
  精确边界/零/overflow、unknown/explicit allow、region→region/leaf 重选、共享常量只计一次、父/子报告
  独立保活、跨变体耗尽及完整 rollback；沿用本卡 Windows Graph/oneDNN/CUDA 与三交付增量验证。

本段只改变已 AUDITED 的 Graph；Provider/Mem/backend 合同不变，也不签完整内存或性能终验。

### 2026-09-15 G106 编译制品存储准入检查点

已实现上述有限存储门：每个候选在 workspace 后、成本比较前累计真实 canonical 请求字节及 Provider
声明，失败参与已有整图重选；family 向子路线传剩余额度，最终累计时共享 canonical 只计一次。默认
无上限不排除 unknown；有限上限默认拒绝，显式 allow 仍返回空 `total_bytes()` 与未知数。普通 operation
的 persistent facts 已补入 step report；插入的 Backend copy 没有另一个 compiled artifact，报告为零。

新增 9 个 GraphStorage 测试，并扩充既有共享常量及真实 oneDNN 旅程。覆盖 native 实际制品报告、精确
边界/零/overflow、borrow 不计新增编译存储但仍保留 ownership 报告、unknown 与已知部分限制、region
重选/leaf fallback、不放宽 hard limit、跨变体累计/溢出、失败整表回滚和报告独立生命周期。oneDNN
完整未知量拒绝与显式 opt-in 在真实 BRGEMM region 上验证，不以 fixture 的已知数字冒充第三方事实。

Windows Release 增量验证：

| 验证 | `C:/hyle-scratch/` 下缓存 | 结果 |
|---|---|---|
| Graph static / 逐头 | `compute-route-v8-static-20260914` | 3/3；执行套件 227/227 |
| shared + oneDNN Graph/region / 逐头 | `compute-route-onednn-v8-20260914` | 6/6 |
| CUDA Graph / 逐头 | `compute-graph-v7-cuda` | 3/3，含既有真 CUDA/capture 回归 |
| source CPU/CUDA sample | `custom-execution-v10-source-consumer` | 2/2 |
| installed shared CPU/CUDA sample | `custom-execution-v10-shared-cuda-consumer` | 2/2 |
| installed static CPU/CUDA sample | `custom-execution-v10-static-consumer` | 2/2 |

Build/Test 沿用 Windows wrapper，build 至多 2、测试串行；shared/oneDNN cache 重新 configure 成功，
保留上游 CMP0146 OLD 警告。该 configure 更新 SIMD 生成文件时间戳并触发依赖重编，不是 SIMD 产品
改动。SDK 从同名 `custom-execution-v10-{shared,static}-build` 构建并显式安装到 `...-sdk`，没有安装到
Program Files。公开 sample 使用有限预算并读取单路线/总报告，不依赖私有 SPI。

shared producer/SDK/CPU/CUDA 装载目录的 Graph DLL 均为
`3D9C44EDEBAE44CA88A761BE01B0DE542419717E6822767C068F072CD00813AA`；源码与两个 SDK 的
`executable.h` 均为 `972BE1A4E7EE8D743705C81B5618D8445984418FD474C367EAAE87E444FFAFA6`。

P0/源码复核：每个候选一次 `O(constants + steps)`、附加 scratch `O(1)`；每个 family 子项 `O(1)`
累计。ReportState 增加两份 4-word 冷事实；未增加 Plan/Invocation 热字段、heap 分配次数、submit
回调层次、原子或扫描。该结构结论不是计时胜负，也不是独立终局红队签名。

selector 基线 `2337618d`，选 Graph/Graph-oneDNN/逐头/交付与 docs/closure/configure，89 项明确跳过。
registry/Provider/Backend/Mem/CMake/公共仪器未改；closure 39/schema 4、docs 292/1064 links、diff check
通过。Linux/两台 Apple、sanitizer、独立终局红队和一次 foundation P1 仍待完整 candidate，未运行不签 PASS。

统一 LOC self-check：全库 **678,778**，自实现不含测试 **124,069**；Compute **16,670**，Graph **12,273**，
合计 **28,943**。仍为 `AUDITED`：本项只覆盖已声明的 compiled storage，不代替 G106 的完整 unique
resident/冷峰值/并发/capture 或 packing 身份共享；G101 动态 View、G103 storage/lane 联合优化、真实
平台探针及共同终验继续保留，不扩算子目录。

G106 后续的运行期/冷峰值检查发现 Mem LifetimePlanner/PlannedBuffers 的全部 metadata 硬编码 system
allocator，无法进入 Context 配额与错误注入。这是具体依赖缺口，已按
[Mem v9 M90--M92](../mem/mem-v9-lifetime-plan-allocator.md) 定向重开；Graph 的统一 planner build
调用传入既有 allocator，instance 继承同一 owner。先补此 seam，再实现整体 reservation；不在 Graph
复制 Mem planner 或猜测其内部 allocation bytes。

### 2026-09-15 G106：planner metadata 依赖回放

上述 Mem v9 前置已在候选 `8e4f096e` 完成：四环境 Mem 正确性/适用 sanitizer/三交付通过，Graph
Windows shared+oneDNN 10/10 闭包回归、CUDA 3/3（执行 229 用例）及 CPU/CUDA source/static/shared
各 2/2 通过。具体命令、DLL 身份、范围与 P0 见 Mem v9；Mem 已关闭这次定向重开，Graph 不复制其
allocator bookkeeping。Graph 这一修改只向原有调用传 allocator，热 submit 无变化。

本项解决 metadata 绕过 Context quota/fault injection 的真实缺口，但**尚未**提供 Graph 的完整
unique resident/compile peak/并发 Invocation/capture 预算或 Provider packing 共享；这些仍是 G106
活动 FIX，不把前置依赖通过当作基础 gate 完成。当前 Compute/Graph 与此前检查点的实现 LOC 不变：
**16,670 / 12,273**，合计 **28,943**，不含测试与第三方。

后续 owner 追踪确认 Compute 的 leaf propose/prepare/domain instance/invocation 仍固定使用 Context/root
allocator，会绕过独立的 Graph metadata 预算。该下层有限修改已绑定
[Compute v12 C120--C122](compute-v12-metadata-ownership.md)；Graph 本卡只在现有冷调用传递自己的
plan/instance allocator。Region prepare 已使用 Graph owner；完整 resident/峰值/并发/capture 与私有
packing 仍是 G106 待实现范围，不以 allocator seam 冒充整体预算。

该 seam 已实现并通过 [Compute v12 检查点](compute-v12-metadata-ownership.md#2026-09-15metadata-owner-检查点)
的 Windows static/shared、oneDNN/CUDA 和三交付回放。新增 Graph public journey 将全部 node required
放到第二个 Context，禁止该 target 新增 metadata 分配后仍可完成 family compile 与 exact/general
Invocation；Graph 的第一个 Context allocator 统一承担该控制存储。没有新增公开 allocator 开关或
把 Context/Queue/Provider 私有 heap 纳入虚构的总量；G106 整体预算仍未完成。

### G106：共享运行存储准入（实现前固定）

- `CompileOptions::invocation_storage` 是可选的 family-wide 运行存储限额；未配置不创建 tracker，
  普通 submit 不加计数、锁或查找。配置后 general/exact 子 Executable 共享同一 owner 和额度。
- 计费集合为 Invocation 的所有 supplied-allocator metadata、实际 LifetimePlan backing 请求字节、
  Provider 声明的额外 per-invocation 上界。后者不包含前两项；通过 Compute v13/Region properties
  披露，未声明为 unknown。限额约束预留请求，不冒充物理 RSS/driver allocator rounding。
- 复用 Mem v10 QuotaAllocator Reservation：先预留独立 resource 请求，再创建 Buffer；失败完整归还。
  创建实例后的 retained Completion/公开 signature 仍保活其 allocator owner；直到最后实际对象释放才
  归还相应额度。并发创建共享严格额度，不能按 shape/variant 重新充值。
- `Executable::invocation_storage_usage()` 返回可选 live/peak/limit 与完整性事实；未配置为无值。
  snapshot 可与并发创建/释放交错，不是多个原子字段的线性化全局事务；硬 admission 本身是严格的。
- 有限额度默认拒绝未知 private 上界，并进入已有冷候选拒绝/重选；显式 allow-unknown 只限制已知部分。
  Native exact/domain 的额外 private 为 0；typed/外部 Provider 默认 unknown。不拿 persistent 字节抵扣。
- command capture 目前没有完整 backend 私有存储上界；严格有限额度下 require-capture 在接纳前拒绝，
  prefer-capture 走已允许的 commands 路线；显式 allow-unknown 捕获必须使 usage 披露不完整，不能假称
  capture 已计全。Context/Queue、外部绑定和 Executable 常驻不属于此运行限额。
- Region invocation 独立使用运行 allocator，create/failure/destroy 同 owner；复用已有 callback 参数，
  不增加另一 runtime。Region contract 因 prepared facts 布局变化递增，不保留兼容字段。
- 验证：公开 Native exact/domain/finite variants、两线程共享限额、失败逐点回滚、workspace 与 metadata
  双来源、最后 Completion/子对象释放、unknown/zero/overflow、Region 和 capture policy；原 CPU/Native/
  oneDNN/Graph/CUDA 定向回放、逐头和三交付。完整候选的跨平台/sanitizer/P1 義务不变。

此项落实运行集合，不代签其余 G106：完整编译 resident/冷峰值、packing identity/共享与 capture 私有上界
仍须完成；已有 compiled-storage 分类门不改名为总量。能力、生命周期和预算口径在产品实现前固定。

### 2026-09-15 G106：共享运行存储检查点

上述共享限额已实现：general/exact 计划共享一个 opt-in owner；Mem QuotaAllocator 同时接管 allocator
请求和独立 workspace/private 上界的 Reservation，没有伪分配或第二套 quota 算法。Region create、
部分失败与 destroy 改为同一运行 allocator；Compute 域实例不能把根制品已声明的 private 上界扩大或
改成 unknown。Native 声明额外 private 为零；其它 Provider 未声明仍为 unknown，不能由名称猜零。

额度随真实对象生命周期归还：Invocation 自身、Provider 状态和 workspace 销毁后才释放 reservation，
保留的 Completion/signature 不会悬空使用 allocator。strict finite budget 不接纳未知的 capture 私有
内存；prefer-capture 可走普通命令，require-capture 在开始捕获前失败。allow-unknown 计数只约束已知
部分，usage 的 unknown 标志是保守粘性事实，不会在一次失败/结束后伪装成完整总量。

新增 16 个用例：Mem 4、Compute typed facts 1、Native 域实例上界 1、Graph 10。覆盖精确计费、
两线程共享硬限额、逐次 allocation/OOM 回滚、unknown/zero/overflow/重选、general/exact 共享、
异步放弃后真实完成才回收，以及全部 Plan 已销毁但独立 signature 仍保活 allocator 的反例。公开 CPU
sample 使用有限运行预算，并验证根/子路线观察到同一份非零用量；没有新增 kernel 或 benchmark。

| Windows Release 定向验证 | `C:/hyle-scratch/` 下缓存 | 结果 |
|---|---|---|
| Mem/Compute/CPU/Native/Graph static + 逐头 | `compute-route-v8-static-20260914` | 11/11；后增 Native 反例复跑 1/1 |
| 上述 shared + oneDNN direct/composition/region | `compute-route-onednn-v8-20260914` | 16/16；强化 signature 反例后 Graph 238/238 |
| CUDA runtime/Graph/capture/晚到失败 + 逐头 | `compute-graph-v7-cuda` | 8/8；Graph 执行 240 用例 |
| source CPU/CUDA sample | `custom-execution-v10-source-consumer` | 2/2 |
| installed shared CPU/CUDA sample | `custom-execution-v10-shared-cuda-consumer` | 2/2 |
| installed static CPU/CUDA sample | `custom-execution-v10-static-consumer` | 2/2 |

Windows wrapper 初始化 MSVC 19.50.35728，build `--parallel 2`、CTest `--parallel 1 --timeout 120`。
shared/oneDNN 的 target 是表内各 factor 的普通测试与 Graph executable 测试；CTest 精确匹配这七个
factor 的普通/逐头/executable/composition 名称。CUDA 用已有五个产品测试 target；三交付复用
`samples/package/graph/custom_kernel`，两个 SDK 从对应 `custom-execution-v10-{shared,static}-build`
安装到显式 `...-sdk`。producer/SDK/CPU/CUDA 装载目录的 shared Graph DLL SHA-256 均为
`7D43990011357947AC92916D0014B8E7C2A19A682D0012F37ECADADAB8489F0A`；同步更新 Compute/Native DLL，
不从旧装载副本取得假通过。SDK 回放后只移动一段错位注释、加强测试，没有产品语义变化。

初次 MSVC 编译暴露私有 `Ref<Self>` 在类型未完成时触发 concept；工厂改为返回待立即 adopt 的 +1
裸指针，不降低公开 Ref 约束。另修正 Native 测试把 shape 求值 scratch 错写为 const；这两项是实现期
编译纠正，不是性能失败，也没有改变已固定 API。源码复核补强了最后 Plan 释放后的 signature 反例。

P0：opt-in 每次冷创建额外 `O(steps)` 汇总和 Mem 原有 CAS admission；tracker 为一次控制分配，各成功
metadata allocation 保活 tracker。普通 Plan 增加一个空 Ref，prepared facts 增加一个 optional 字段，
RegionInvocation 增加一个 allocator 借用指针；不把这些常驻成本写成零。普通 InvocationState 本体
不增字段，仅 opt-in 在同一 allocation 的尾部存两字 Reservation。未配置不建 tracker，不执行 quota
计数；warm submit 没有新增预算遍历、查表、锁或原子。释放 opt-in 对象会执行归还与引用计数，不是免费。

selector 基线 `dd85cab8`。它因 Mem public header 保守展开全部反向闭包；本卡增量只执行表内预声明
风险，不把未执行的 Container/IO/Sched/Process/Util/MemVM 门写成 PASS。源码 include 检查确认产品里
只有新 Graph owner 消费 `quota_allocator.h`，其它直接引用是既有 Mem/CPU/MemVM 测试；QuotaAllocator
原字段及 allocate/reallocate/free 算法不变。广泛无关消费者未因这个添加式入口重建，完整候选的适用
风险仍须最终裁决。CMake/公共仪器/CI 未改，未启动全库测试或 P1。

这些是实现期 Windows 检查点；Linux/两台 Apple、sanitizer、独立终局红队及一次 foundation P1 仍待
完整候选，未运行不签 PASS。Mem/Compute/Graph 保持 AUDITED，不签整体基础完成。剩余 G106 编译
resident/冷峰值、packing identity/共享、capture 私有上界，G101 动态 View、G103 storage/lane 联合优化
和 CPU/CUDA/Apple 真实架构探针继续保留；没有横向扩算子目录。

统一 LOC self-check：全库代码 **680,831**，自实现不含测试 **124,428**；Compute **16,748**、Graph
**12,495**，合计 **29,243**。检查器通过 39/schema 4、296 Markdown/1072 links，diff check 通过。

### G106：编译 resident/peak 表示审计（实现前固定）

- `CompileOptions::compilation_storage` 为可选 `CompilationStorageLimits`：family-wide
  `max_resident_bytes/max_peak_bytes/unknown`。既有 compiled_storage 保留为特定 artifact 分类限额，
  不改成另一口径；新限额约束整个 compile 新取得的 requested storage，而不是进程 RSS。
- 集合是 Graph/Compute supplied-allocator 控制 metadata、实际 canonical materialization 请求，以及
  Compute v14/Region resource proposal 的额外 private resident/peak 上界。不重复加原 persistent_bytes。
  输入 Graph/外部 backing、已有 Context/Queue 与 backend/driver 控制开销、Status 诊断和 allocator
  rounding 不在此集合；不为观测 OS/driver 私有 malloc 改 runtime。Invoke/capture 仍是独立集合。
- cold scratch、incumbent/trial、全部 exact 子计划和报告共享一个峰值额度；在任何受管请求前准入。
  私有 peak 在 prepare 前预留，成功后只保留 resident；失败先销毁部分制品再归还。未知量在有限 strict
  模式下于 prepare 前拒绝，可参与既有路线重选；allow 只约束已知部分并披露不完整。
- 根 compile 返回前、临时对象已销毁后检查 resident 上限；所有指定 variants 原子发布，不删点求成功。
  `CompileReport::compilation_storage()` 给 family 发布时 resident/peak 与 unknown 快照；报告不靠
  保活 payload 维持计数。未知尝试影响 peak 的事实保守保留，不能失败后写成已知总量。
- 复用 Graph 已有引用保活 allocator adapter 与 Mem QuotaAllocator，不另写通用 quota；常量和制品
  reservation 必须与真实 owner 同步销毁、随 constant sharing 和 route transaction 一起移动。
  Invocation 使用基础 runtime allocator，不误消耗 compiler quota；运行 tracker 的控制块算编译 metadata。
- 普通未启用路线不分配 compiler tracker、不加热态计数；新增冷 Plan/Report 字段及 opt-in side arrays
  记录 P0，不给每个热 PlanStep 增加一份 token。不会用此项假称 opaque packing 兼容共享已经完成。
- 验证：metadata 对账、精确 resident/peak 边界、未知/非法/overflow、编译前拒绝、private peak 归还、
  失败逐点清理/重选、constant/variant 共享、独立报告/子路线/运行实例生命周期，原 Graph/Native/
  oneDNN/CUDA 与三交付。跨平台/终局红队/P1 仍由完整 foundation candidate 统一执行。

这是既定 G106 的完整编译 requested-storage 集合，不扩 kernel、packing 算法或后端目录；未知 vendor
上界允许明确拒绝，不能改成虚构的有限值。现有 invocation 限额和所有其它 foundation FIX 继续保留。

### G106：region discovery scratch owner 纠正（实现前固定）

编译配额源码复核发现真实 oneDNN region 的 `propose_regions()` 仍用 Context allocator 构造 use-table
和 contractibility scratch。它发生在 prepare 之前，不能用事后 private bounds 补记过去的峰值。
因此本轮有限纠正把 `RegionProviderVTable::propose/propose_resources` 都显式传入本次 Graph compiler
allocator；发现/资源查询只用该 allocator 的 scratch 与 caller writer，不在这些回调里创建私有 heap/
artifact。错误 Status 仍按既有合同排除。Compute leaf proposal 已有 success allocation-free 合同，不改。
Graph extension 升版、真实 oneDNN 直接消费者迁移与 allocator/OOM 反例属于 G106/GP41，不扩 pattern/
kernel 或公共运行时。验证只回放 Graph、oneDNN、逐头、CUDA/三交付受影响闭包；不再重跑 Compute/
Native 无语义变化的用例或 P1。此纠正未通过前，不把整个编译集合签为完成。
