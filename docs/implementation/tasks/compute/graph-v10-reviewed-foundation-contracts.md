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
