# HYLE Graph：计算 DAG、编译与可复用执行计划

> 决策状态：`ACCEPTED`。foundation-v1 架构基线已经裁决；这不表示当前实现已经越过 foundation
> implementation gate，也不新增一种模块状态。本文定义 Graph 的长期产品合同；当前实现状态只查
> [`module-status.json`](../implementation/module-status.json)。每轮纵切只证明其 closure card 登记的
> DAG/Executable/Invocation 合同。旧 `TaskGraph`、parallel/data-parallel API 与 benchmark 已退出产品树；
> 被删除的旧字节只可从 Git 历史取得调查事实，历史 task/evidence 不是当前设计输入、实现模板或用户承诺。
> 本页不签发用户可用性，也不要求兼容旧头。

2026-09-14 的整体复核保留上述骨架，修正 replay 成本、派生 shape、custom execution、资源副作用域、
variant 内存与搜索受限合同；依据见[复核记录](../implementation/tasks/compute/compute-graph-route-unification-audit.md#整体架构复核与合同修正)。

## 1. Graph 解决什么问题

Graph 把**模型无关的逻辑计算 DAG**编译成可重复、可检查、低开销提交的执行物。它拥有：

- logical tensor/value 与 operation node 的拓扑；
- shape/layout/resource/side-effect 约束及图合法性；
- provider/backend capability matching、region partition 和 placement；
- fusion、boundary transformation、copy、同步、liveness、alias 与 workspace 规划；
- compiled artifact 的组合、重复执行和每次调用状态分离。

Graph 不是通用 task executor，也不是模型 runtime。Sched 拥有 task/fiber/coroutine、取消、等待和线程；
Compute 拥有 device/context/queue/kernel/provider 与单个 region 的执行；Mem 拥有 storage 与通用 lifetime
packing。推理引擎仍拥有模型构图、KV/prefix-cache、batch/admission、sampling、循环停止和请求策略。

## 2. 三个概念层，两个公开产品

系统需要 Compute、Graph 和 compiled plan 三层概念，但不新增一个含义模糊的 `HYLE::plan` 产品：

```text
HYLE::compute                         直接 op/kernel 与 backend/provider runtime
HYLE::graph                           逻辑图、编译和执行 façade
  graph_ir                            逻辑计算 DAG
  graph_compile                       冷路径编译/partition/placement/planning
  graph_plan                          内部执行物与 invocation runtime
```

`graph_plan` 是内部实现职责，不是当前独立 component identity；对 C++ 用户公开的结果类型是
`graph::Executable`。只有它以后形成不同依赖、测试 owner 和独立 consumer，才按质因数规则拆 target。
用户始终不需要在“graph 还是 plan”之间选择。

## 3. 四个生命周期对象

概念上的正常路径固定为：

```text
GraphBuilder -> Graph
Graph + CompileEnvironment/Options(ShapeDomain) -> Executable + CompileReport
Executable + concrete ShapeBinding -> Invocation
Invocation.bind(...) -> prepare_replay(...) -> submit(...) -> compute::Completion
```

### `GraphBuilder`

- 单 owner、可修改；添加输入、常量声明、operation、输出与显式依赖。
- 返回 typed `ValueId/NodeId`，不把裸下标、地址或第三方 node 泄漏为身份。
- `finish()` 做结构验证并产生不可变 Graph；失败不留下半可用图。

### `Graph`

- 不可变、可共享、与设备/provider 无关；持有逻辑描述而不持有运行时 payload。
- value 至少描述 dtype、rank/shape constraint、stride/layout、storage class 和可变/常量意图。
- node 使用 Compute 的公共 operation schema；side effect、resource write 和 control dependency 必须显式。
- hash/fingerprint 只在完整 canonical contract 成立时提供，不能把进程地址或不稳定类型布局当缓存键。

### `Executable`

- 编译后不可变、可共享；绑定 graph fingerprint、closed shape domain、device/backend/provider identity
  和 compile options。
- 拥有 provider compiled artifacts、预打包常量及不可变 plan metadata；清楚报告其 storage owner。
- 暴露输入/输出合同、workspace 要求、所选 regions、copies/sync、fallback 与并发能力。
- 不承诺跨 provider/version/device 可移植，也不在首轮承诺序列化格式。

每个 Executable 对应一个封闭 `ShapeDomain`：可以是 exact point，也可以是有界/guarded domain。它可以
包含一个真正接受范围的 artifact，或 compile 时已经生成的有限 exact variant 表与显式 general fallback；
必须在 report 中声明自己覆盖的集合，HYLE steady submit 不因新 shape 触发 JIT、建图、计时或 cache mutation；
Provider 的 managed replay 内部成本按 Compute §3.4 另行声明，不扩大已编译 shape domain。
同一 Graph 的跨 Executable specialization cache 由推理引擎/消费方拥有，HYLE 不建立进程全局 cache。
`ConstantBinding` 必须显式选择 source lifetime：borrowed immutable 表示 source 活到 Executable 与全部
Completion 终态；materialize 表示 compile 成功前 Graph/Provider 已拥有 replay 所需的全部 canonical/
packed bytes，调用返回后 source 可释放。选择 materialize 的候选不能因内部实现方便而静默退回 borrow，
CompileReport 要逐 constant 报告最终 owner/release point，不能再用 aggregate packed count 猜生命周期。
若消费方要在 Executable 内原位替换权重，应把该 value 建模为普通 input；否则重新 compile。

canonical materialize 沿用 constant 的 external storage placement；Graph 复用 Mem Buffer 和 Backend
明确的 transfer/Completion 合同，不暗中 staging 或执行用户 Graph。复制在完整物理路线成立之后、
Provider prepare 之前进行，同一 compile 的路线重选共用已经完成的副本。resource proposal 只读取
metadata/policy，不读取 payload 或缓存 source 地址；prepare 接收可借用到制品销毁的 canonical view。
失败须先等已接纳复制到终态；常量 allocation 后于制品、先于 Context 释放。报告独立保存 policy、目标域
与 canonical bytes，不保活 payload，也不能将这些 bytes 冒充全部 packed/resident 内存。

同一 Executable 的 variants 引用兼容的共享不可变 constant materialization，不因 shape bucket 不同就
自动复制整套权重。共享键包含 constant 的稳定绑定/内容版本、provider/build/device、packing/precision/
layout 和 storage 身份；裸地址不足以证明内容相同。Provider 不支持借入同一 packed artifact 时，可以
保留已披露的重复副本，但全部计入预算，不能假称已共享。跨 Executable 的共享池由消费方显式拥有，
不为此建立进程全局 cache。

### `Invocation`

- 每次并发执行独占的可变对象；在创建或 replay preparation 前绑定落在 Executable domain 内的 concrete
  shape，再绑定 inputs/outputs、workspace、execution-lane route、completion/capture state 和 provider run
  state。一次 prepared Invocation 的 concrete shape 不在 warm submit 中变化。
- 默认非线程安全，但可在上次 Completion 结束后 reset/reuse；并发请求各用一个 Invocation。
- submit 后所有 borrow 和 workspace 活到 Completion 终态；in-flight 时 bind/reset/resubmit 明确失败，取消
  不允许提前复用。发布的零拷贝 view 若根在私有 workspace，由 Invocation 持有并在成功终态后借出；需要
  caller destination 时用显式 Copy。
- Provider 若需要串行 context，Graph 必须在 report 中声明并由 queue/resource admission 表达，不能藏在
  process-global mutex 中。

把 Executable 与 Invocation 分开不是 API 装饰：TensorRT 的 engine/context、PJRT 的 executable/buffer、
oneDNN 的 compiled partition/scratchpad 和真实推理引擎的 graph capture/replay 都说明“可共享编译物”与
“每次运行状态”有不同并发生命周期。

## 4. IR 的最小完整语义

HYLE Graph 是有限计算 DAG，不是任意控制流 IR。完整产品合同必须能表达：

- 多输入/输出 operation，显式常量与 external binding；
- static shape、带上下界/倍数约束的独立 symbolic dimension 和有限派生维度（§4.3），共享 `DimensionId` 表示相等；编译结果声明 exact 或 bounded/guarded shape
  domain，实际 concrete shape 在运行前验证；无界、数据依赖的动态输出不属于 foundation v1；
- dense/strided/layout-tagged 与量化 storage 描述，但不绑定模型文件格式；
- in-place/alias 候选与禁止 alias 的输出；最终合法性由 operation/provider 合同决定；
- canonical tensor copy 和 host/device boundary；provider-private layout 只能留在 region 内。跨 region
  repack 必须先有有版本的 boundary-layout producer/consumer 合同和真实执行 owner，不能用枚举冒充实现；
- 有副作用或外部资源访问的 node、资源域与顺序边（§4.4）；纯 node 默认可重排，独立资源不强制串成一条链；
- typed custom operation extension，供 HYLE/消费者 kernel 接入，不使用字符串属性袋或裸 callable。

每个 operation schema 必须区分“验证一个 concrete shape”和“证明一个 bounded domain”。只有 schema 与
入选 Provider 都能对完整 subdomain 给出可检查合同，Graph 才生成 bounded route；否则该 operation 明确
exact-only，只能进入 exact variant。抽样几个 shape 通过不能冒充对整个范围的证明。

首轮不提供任意 `if/while`、递归图或模型脚本解释器。recurrent-depth、speculative 和 adaptive compute
先由引擎重复提交一个或多个 Executable，并负责 active-set、停止与重聚批。Graph 必须让这种 replay
低成本且 state binding 显式。内部 IR 从首轮起使用可嵌套的 typed region 边界，而不是把所有 node 永久
焊死为只能存在于顶层：v1 region 只承载 provider partition，不签发公开控制流。只有第二个独立通用消费者
或真实 recurrent/device-loop 纵切证明必要时，才以新版本增加 counted loop/conditional region；不得为此
把 Graph 变成脚本 VM，也不得要求已有有限 DAG 改写。

### 4.1 Execution target、storage domain 与 execution lane

Graph 分别规划“谁执行”“值在哪里”和“step 沿哪条时间线执行”。概念上的编译环境固定为：

```cpp
struct StorageDomain {
  mem::MemoryResourceRef resource;
  std::size_t workspace_budget_bytes = SIZE_MAX;
};
struct ExecutionTarget {
  const compute::Context* context;
  StorageDomainId default_storage;
};
enum class ExecutionLaneRole { kCompute, kTransfer, kCommunication };
struct ExecutionLane {
  ExecutionTargetId target;
  ExecutionLaneRole role;
};
struct CompileEnvironment {
  std::span<const StorageDomain> storage_domains;
  std::span<const ExecutionTarget> targets;
  std::span<const ExecutionLane> lanes;
};
```

- node placement 使用 `ExecutionTargetId`；value、physical instance、workspace 与 external resource contract
  使用 `StorageDomainId`；step/segment 使用 `ExecutionLaneId`；
- `StorageDomain` 指向一个 exact canonical resource 和独立预算。其 `MemoryResourceRef::domain()` 是 Mem
  提供的 coherence/accessibility identity，不与 Graph ordinal 混为一谈；
- `ExecutionTarget::default_storage` 只为普通输出/workspace 提供自然默认值，不参与 target identity；同一
  target 可访问多个 storage domain，同一 storage domain 也可被多个 target 访问；
- shared-runtime lane 在 `prepare_replay` 时绑定 exact Queue；delegated provider 的 endpoint 由其 artifact/
  invocation 保活，并以合成 lane 进入 report，但不要求调用方伪造 QueueBinding；
- lane role 是调度意图而非 vendor capability。capture 由 `ReplayMode` 和 exact Queue capability 表达；
  communication role 给 collective/TP 等 typed operation 留出真实时间线，但不把 NCCL 等 API塞进 Graph core；
- Graph 不创建全局 stream、线程池或 Queue。空 lanes 时，每个 active target 只有一条默认 compute lane；
  空显式环境的单 target convenience 仍可由 `Context + MemoryResourceRef` 构造。

target 不同并不自动产生 copy。Graph 先检查 consumer provider 的 binding requirement 与 target 对 source
resource 的访问能力；只有不兼容或调用方 required placement 明确要求另一 storage domain 时才物化 copy。
因此不能把两个 Queue 表达成两个假的 storage domain，也不能从 Queue 地址反推 tensor residence。跨 lane
依赖优先成为 device wait/timeline edge；只有显式 policy 允许时才退到 report 中可见的 host wait。

### 4.2 并发调度与 workspace 是一个联合问题

多 lane 会让原本在某个拓扑序上不重叠的两个值真实并发。最终 workspace 不能在 lane scheduling 之前按
线性 node interval 生成。Graph 必须先形成由 data/control edge、同 lane 顺序和跨 lane dependency 组成的
happens-before 图，再裁决 alias/reuse：只有旧值的全部使用都 happens-before 新值的首次写入（或反向成立）
时，两者才能共享 backing；无法证明就保守地不复用。

默认 `WorkspaceReusePolicy::kExistingOrderOnly` 保留已有并发，不为了省内存增加依赖；如果某个 storage
domain 的硬预算不足，compile 返回 `ResourceExhausted`。显式 `kAllowSerialization` 才允许 planner 添加可见的
ordering edge、重新 schedule 并换取复用；新增 edge、失去的并行机会和最终 peak bytes 都写入
CompileReport。该搜索必须确定性、有迭代/内存上界并有保守 fallback，不能把历史 benchmark 常数变成
调度 oracle。

Graph 只规划 canonical boundary value、HYLE/shared-runtime workspace 与 copy staging。Delegated island 的
内部 tensor/workspace 由 provider 自己规划，Graph 只记录其 exact/unknown persistent、workspace 与 boundary
cost，不能与 HYLE workspace 重叠复用。

预算还必须覆盖同一 Executable **全部常驻 variants** 的唯一 constant/packed/artifact allocation，而不是
逐 variant 都合格就认为整体合格；compile 临时峰值与 steady resident 分开报告。Invocation workspace
按预计并发数和实际预留计费；不同并发 Invocation 不可因形状相同就共享可变 backing。互斥 variants 仅在
lifetime、alignment、storage 和 capture 稳定地址均可证明兼容时复用 workspace，不能仅因“每次只选一个”
就覆盖仍被 capture 持有的 allocation。同物理内存上的 host/device resource 不重复计费；外部已常驻
backing 与本次新增预留分列，不能把 borrowed 误当成零物理内存。Provider 不能给出预算所需上界时，
unknown 只可在调用方明确允许未知量的路线中出现，不得宣称满足该硬上限。

### 4.3 有限派生 shape，而非图外手工等式

正常 reshape/flatten、concat、head/group split 和 broadcast 需要 `B*T`、`S1+S2`、整除与广播相容性；
不能让输出维度成为独立区间后要求引擎在图外重复维持关系。Graph 持有只读 metadata shape program：
独立 bounded symbols、固定常量、checked add/multiply/exact-divide 和 broadcast 派生式，以及显式 typed
guards。输出维度由 schema/表达式派生，不要求消费者把每个中间维度再绑定一次。

封闭 domain 是独立 bounds 与 guards 的交集，不是包含非法输出组合的笛卡尔积。compile 验证表达式无环、
rank/type 合法及域覆盖；保守区间与 guards 不足以证明 schema/provider 合同时，缩窄域或 exact-only，
不能用若干端点样本代替 proof。Invocation 在 bind/prepare、用户工作接纳前一次求值并检查 overflow、
除零/非整除、元素数、broadcast、layout 与 workspace；不读取 tensor payload。计算成本为 shape-program
大小的有界线性遍历，HYLE warm submit 不再求值。Compute 的直接 TensorDesc 仍描述 concrete tensor，
provider-neutral domain 投影不引入 Graph ID 或反向依赖。

表达式按创建顺序引用此前维度或非负常量，因而不需要递归图解释器。派生 bounds 是保守 envelope，
guards 可能排除各轴上界的组合；不能把这个组合当作合法 concrete shape 先做单点 schema inference，
再声称覆盖整个 domain。整数或 descriptor 的保守上界无法表示时要求缩窄输入范围，不靠通用约束求解
或抽样自动收紧。具体表示与当前落地范围分别见活动卡和用户指南，不把设计中的 bounded replay 写成已交付。

这不是任意方程求解器、脚本 VM 或数据依赖输出支持。最小反证用 `[B,T,H] -> [B*T,H]`、concat 后
reshape、整除失败和 overflow，不能只用输入输出共用一个 symbol 的 RMSNorm 证明自然动态图已完整。

### 4.4 按资源描述副作用

除 tensor 数据依赖外，node 可以声明 graph-local 资源/副作用域及 read/write/read-write 访问。相同域的
冲突访问必须有 data/control 顺序，纯 read 可并行；不同且可证明独立的域不强制全局排序。未知外部
副作用默认进入保守域，与其它未证明独立的外部访问保持顺序，不鼓励谎称 pure 来绕过检查。

编译先验证冲突顺序，再形成 region、lane 与 happens-before；Provider 也必须披露 region 的资源访问，
不能因内部不透明而漏掉依赖。绑定时若两个声称独立的资源实际 alias，应拒绝或要求重新准备有序计划，
不能靠不同 ID 忽略同一 backing 的写冲突；子范围并行必须有明确不重叠合同。跨 Invocation 的共享可变
资源顺序由调用方的 Queue/Completion 或显式 admission 约束，不偷偷建立 Graph 全局锁。

逻辑资源由 `GraphBuilder::effect_resource()` 创建、按 node 的 schema 端口绑定；`finish()` 验证已有的
data/control 顺序，不按插入顺序补边。`CompileOptions::effect_bindings` 固定该 Executable 的借用访问
footprint：地址化存储使用 Mem 的 canonical allocation/subrange，非地址化对象使用 canonical owner。
它不是另一个执行参数通道；custom 执行的资源必须与声明相符。不同 ID 的 backing 重叠直接拒绝，需 alias
时复用 ID 或给出不重叠范围。effect memory 与 tensor input/constant/output 隐式 alias 也拒绝，避免绕过
SSA；可变状态的数据结果应显式产出 value。数组/span 在冷 compile 复制，backing 则由 caller 保活。

访问 schema 属于 ComputeOps 的计算语义，不持有 Graph ID；图内资源 identity、绑定与冲突检查属于 Graph。
KV eviction、communicator 创建、请求顺序等仍是引擎策略。测试同时覆盖独立两域可并行、同域写冲突、
未知副作用与跨 region 顺序。

## 5. 编译流水线

`graph_compile` 按可检查阶段工作：

1. **validate/canonicalize**：检查拓扑、shape、dtype、alias、side effect 和输出；
2. **infer/normalize**：传播 shape/layout 约束，把等价 operation 变成稳定形式；
3. **capability/proposal**：询问每个 provider 对 op/region 的支持、binding/storage requirement、runtime mode、
   结构成本和可选量化估值；
4. **partition/target/storage placement**：联合选择 region、execution target 与 storage domain，不把不可移植的
   计时常数冒充通用性能 oracle；
5. **prepare selected artifacts/materialize boundaries**：让入选 provider 对 exact resource route 冷验证并
   给出精确或 unknown workspace/persistent facts；在有限 fallback 预算内处理拒绝，为真正不兼容的 canonical
   boundary 插入 copy。同步表示为 dependency，只有版本化 layout 合同才允许真实 repack；
6. **lane schedule/happens-before**：选择 execution lanes，建立同 lane order、跨 lane/device wait 与 delegated
   island dependency；capture 仍只是后续 replay mode；
7. **liveness/alias/workspace**：按最终 happens-before 与 storage domain 生成 Mem planner 请求；只有显式
   memory policy 才增加 serialization edge，并在增加后重新 schedule/plan；
8. **lower/replay recipe**：固化预绑定 callable、backend command、dependency route 与 capture recipe；
9. **emit plan/report**：生成不可变 graph plan 与可诊断 compile report。

Partition 成本必须覆盖跨 region copy/sync、layout conversion、thread handoff、workspace、首次编译和常量
packing。Provider 的 `supports(op)` 不是“选它一定更快”。无法覆盖的 node 必须回到已登记 fallback 或使
编译失败；不能在运行时突然下载依赖、切 backend 或用 host scalar 静默求值。

provider-neutral region 用 canonical `NodeId` 集合和显式 boundary inputs/outputs 表达，不把 begin/count
当成公共语义。集合必须非空、identity 唯一，且不存在一条路径从集合内离开后又进入
集合；Graph 验证它可被收缩成一个节点而不改变依赖或制造环。多个入选 region 必须互不重叠，收缩后的
segment graph 仍须是 DAG。Provider 经 Compute Context 的对象局部、精确版本 extension 提案，Graph 不
枚举任意 node 子集，只在 candidate/state 硬预算内验证并选择显式 proposal。连续拓扑区间可以作为发现、
验证和 replay 的 fast path，但不是能力上限。单-op 提案就是单节点 route candidate，必须与 region 一起
进入计划，而不是先由 Context 顺序固化、再让 region 只和 region 竞争。

选择采用约束优先而不是把 ns、bytes 和线程数随意加成一个分数：先满足语义/精度、resource/ownership、
host-wait 与内存预算等硬约束，再比较绑定到相同设备/shape/layout/完成边界的合格关键路径估值；compile
成本和常驻内存按显式 workload policy 约束或摊销。无合格估值、差异不可分或低于 material 门时，按已
配置的成熟 Provider 优先级作确定性裁决，覆盖数、dispatch 和 ordinal 只用于等价候选的最后 tie-break。
Provider 相对“自己 fallback”的估值不是跨 Provider 货币，不能直接相加。

Region proposal/artifact 同时声明 shared-runtime 或 delegated-runtime 模式。前者必须能在所选 HYLE Queue
sequence 中录入；后者必须给出 provider-owned invocation/timeline、边界 tensor residence、输入依赖消费和
完成导出合同。Graph 把 delegated region 视为显式 island segment，在两端物化必要 copy/interop 和同步；
不得让 adapter 忽略 HYLE Queue 后在未知 stream 执行，也不得把一次 host API 返回误作设备完成。

自己 host 算子图的第三方 runtime 也沿这条 region 合同接入，而不是反过来接管 HYLE Graph。adapter 在
compile 冷路径把一个可收缩 HYLE region 翻译为 provider-native graph/engine；一个 region 可以只含一个节点，
也可以覆盖足以保留融合和内部 layout 的多节点片段。`Executable` 保有该制品，steady replay 只重绑本次
Invocation 的 canonical boundary tensors 并提交。prepared artifact 不重建第三方图；managed artifact
允许已声明的内部执行节点实例化，仍不得重新选择 HYLE region 或隐含 JIT，详见
[Compute replay 合同](compute-design.md#34-replay-成本合同独立于-runtime-ownership)。第三方不公开稳定 leaf kernel 时，
HYLE 只能在 region 边界与其他 provider 竞争，不能钻进不透明制品逐节点替换。

ggml 是这类路线的具名例子，但“链接 ggml”不等于 Graph 自动获得其算子。CPU、CUDA、Metal 与 Vulkan
拥有不同 buffer、stream/thread、completion 和失败合同，必须由不同 backend-qualified carrier factor
分别审计。某个 CPU carrier 的 blocker 不应自动否决 CUDA carrier；反过来，CUDA graph 能执行也不能证明
CPU threadpool 或 Metal storage 合同成立。只有实际 adapter 能把 HYLE storage/operation/region 映射到锁定
版本的公开 ggml API，并导出真实 Completion 后，该候选才进入 partition。

全局 route selector 必须把单-op fallback 与 provider region 放进同一计划空间，比较最终 placement 后的
关键路径、边界 copy/repack、同步、dispatch、workspace/persistent 和 compile amortization。最大覆盖节点数、
最少 dispatch 或 provider 顺序只能在没有可比较性能证据时作为确定性 fallback/tie-break，不能冒充
SOTA cost model；provider 各自相对自身 fallback 的估值也不能跨 provider 直接排序。长期稳定的推理
specialization 可以消费显式 qualified route profile；选中结果必须绑定完整 artifact/device/shape/layout/
workload identity，且 warm replay 不再探测。

Graph `compile()` 本身不以用户 binding 执行 graph，也不产生随机/合成 payload 来做跨 Provider tune。
Provider 可以像成熟编译器/厂商 builder 一样，在自己拥有的隔离 buffer 上做有预算的局部 tactic/JIT
选择，但必须披露时间、cache identity 与确定性政策，不能读写调用方 runtime input/output 或外部可变
resource；immutable constant 只按显式 policy 读取。未来可选 `graph_tune` 工具
只有在调用方提供代表性 bindings、明确允许执行的纯/确定性 graph 范围和独立时间/内存预算后，才能运行
结构筛选后的少量完整候选计划并产出 profile。这样 side-effect/custom/external-resource graph 不会因一次
普通 compile 被执行；Graph core 也不需要把 tuner 或持久 cache 变成第三个公开 runtime。

ggml/llama.cpp 的 backend split/reserve、ONNX Runtime 的 execution-provider partition、oneDNN 的
partition/compiled partition、TensorRT 的 network/engine/context、vLLM/SGLang 的 symbolic + exact
specialization/piecewise capture，以及 ds4 的 decode islands 都作为该流水线的独立参照；HYLE 不复制它们
的 public API，也不把其中任何一个强行定为唯一执行引擎。

## 6. Plan 的热路径合同

内部 graph plan 是已经排好序的 region/copy/dependency/workspace 程序。每个 step 持有预绑定 callable
或 backend command；Invocation 只填充运行时 slots。Plan 冷生成使用确定性 Kahn 调度：每个 lane 保持
ready FIFO，并尽量延续当前 lane，以减少不必要的 segment/finalization；storage domain 负责 residence，
execution target 负责执行资格，lane 负责执行顺序；这是有界结构启发式，不宣称
求解一般 DAG 的全局最优设备并行调度。

- `prepare_replay` 完成后，HYLE steady submit 不遍历原始 DAG、不重新推 shape、不搜索 provider、不 JIT、
  不打包常量，也不再查询 transfer/dependency capability；
- exact/captured variant 的 workspace layout 固定；bounded artifact 必须在 compile 时给出 domain worst-case，
  并可提供只依赖 concrete shape 的有界 requirement query。`prepare_replay` 固化本次 layout，正常 replay
  HYLE plan 层零 heap allocation；Provider 是否同样零分配由 prepared/managed 合同决定，不能从 shared/
  delegated ownership 推断。prepared-only 是冷路径约束，不能在 submit 中偷偷改用 managed；
- per-region 的一次预绑定 dispatch 可以存在，per-node/per-element 的通用虚调用或字符串 lookup 不可接受；
- plan 在同一 execution lane 的 ordered Queue 上以一个不中断 submission sequence 录入；provider/region step 只返回接纳事实，
  sequence 在 segment 边界一次 finalizes 为 prefix Completion，不按 node 创建再覆盖 event。多 Queue plan
  保存各 segment 前缀。`prepare_replay` 按 exact Queue identity 把 dependency 固定为 ordered Queue、同步
  source、device wait 或 host wait；同一 target 的不同 lane 也必须走该依赖模型，不能因共享 storage 而假设
  已同步。host wait 只作为 `ReplayReport` 明示且可由 policy 拒绝的不兼容
  fallback，并在取得 destination Queue admission 前完成。可由 queue/event 表达的 dependency 不退化成
  同步 round-trip；
- CompileReport 独立拥有轻量 metadata，并逐 step 枚举 kind、fallback reason、canonical node set/fingerprint、
  boundary ports、shape domain、device/provider/version、constant
  borrow/额外 packed ownership、跨 variant 共享/重复字节、copy/sync/dispatch、workspace、persistent bytes、
  replay 成本类别和 threading/blocking/concurrent-invocation fact。
  opaque provider 无法报告的 compile/persistent/workspace estimate 及 artifact persistent bytes 必须为
  `nullopt`，不能伪装成 0；显式 0 才是精确 none。dispatch 是 backend execution episode：一次同步 CPU
  executor fan-out 或一次 device command launch 计一次，worker/task/chunk/leaf call 不分别计数；
  Graph 不因 aggregate packed count 缩短 source borrow，复制 report 也不保活重型 Executable artifact。

复杂度基线：图结构验证和拓扑为 `O(V + E)`；单 lane liveness 保持线性。多 lane 的安全复用必须对
happens-before 使用明确的有界分析，不能再谎称任意 DAG 下仍是简单线性 interval；预算耗尽时保守地不做
跨 lane alias，而不是冒险复用。在 `P` 个已登记 provider 上的 capability matching 也必须有明确上界，
不得随运行次数重复。更复杂的 partition/cost/reuse search 若采用启发式，必须登记最坏预算与 deterministic
fallback。

### 6.1 Workload intent、route profile 与有限搜索

`CompileOptions` 的长期语义必须携带 workload intent：steady latency/throughput objective、预期并发
Invocation 数、确定性要求，以及可选的预计 replay 次数。默认是长寿命单 Invocation steady latency；预计 replay 次数
缺失时不把 compile/packing 时间伪装成可与 steady execution 相加的标量。workspace、persistent、host wait、
compile time 和是否允许为内存序列化分别由硬预算/policy 约束。

可选 route profile 是调用方显式传入的 immutable borrowed input，不是 Provider 私有相对估值、全局
registry 或隐式磁盘文件。每条 estimate 必须绑定完整 provider/build、backend/device/driver、graph/region、
shape、dtype/format/layout、precision、完成边界和 workload identity，并以相同 HYLE boundary 计费；不满足
这些条件就是 unknown。无合格 estimate 或 material 不可分时，成熟 Provider 顺序决定稳定 fallback。
CompileOptions 还必须允许全局 allow/deny Provider 和 per-node preferred/required Provider constraint；region
必须满足其覆盖的全部 constraint。精度、确定性和 required constraint 都是硬门，不参与时间/内存评分。

v1 route search 接纳通过 contractibility 验证的显式 node-set region，并把每个 node 的 leaf 放进同一空间；
连续区间走 fast path。target/storage/lane 与 boundary transformation 在同一有界 state search 中裁决。
实现可以使用确定性 DP/beam 等启发式，但必须有 candidate、state、prepare-retry 三类独立硬预算，不能
宣称求解一般 DAG 全局最优。预算区分**求得可行计划**与**继续优化**：搜索耗尽只停止新增候选/状态，
仍可在独立 prepare 预算内完成已选计划。最终没有满足全部硬约束且 prepare 成功的完整计划就失败；
已有可行计划时，默认返回最佳已知可行解，并在 report 中明确
`optimization_limited`、耗尽的预算与停止原因，调用方可要求预算不足即失败。不得把截断的 node set、
未验证 proposal 或未覆盖 shape 当成 fallback；结构非法、合同违约与真正错误也不能被这项策略吞掉。
搜索次序须确定，所有存储/迭代仍有硬上界。每次只为当前选中的完整 trial 构造重型 artifact，而不是
预先准备全部独立 proposal；继续优化时保留一个已验证 incumbent，冷峰值包含二者。selected-route prepare 通过
结构化 outcome 返回 success、无副作用 candidate decline 或真正 error。只有 unsupported 或明确
candidate-local resource-budget decline 才允许排除后重选；裸 Status 不能被猜成 rollback 证明，其它错误
立即终止。

完整候选 profile 的身份由 owner 和 Graph 分工：owner 提供真实外部 build/device/runtime/config、
workload/replay 环境和 custom 实例语义；Graph 编码实际内建属性、图/shape 程序和物理路线，校验当前
target/Provider 绑定，并复用 Compute matcher。指纹非零不代表已资格化；不从名称/地址推断外部身份。
首个可行方案成本 unknown 时按配置顺序结束；有合格基准才继续有界比较，只有成本区间严格分离且满足
调用方 improvement policy 才替换。profile 是同一完整 terminal 边界的测量，不能将 leaf/region 私有计时
相加作替代。具体已实现输入、寿命和限制见 [Graph 用户指南](../user/graph.md#按完整计划的性能资料选路)；
此身份机制不等于自动调优、全局最优、所有 storage/lane 组合或 bounded variants 已实现。

### 6.2 Shape domain、capture 与运行时选择

编译和 capture 是正交轴。一个 bounded/guarded Executable 可以走接受范围的普通 artifact，也可以内含
compile 阶段已经完成的有限 exact plan variants；Invocation 用 concrete shape 和所有会影响合法性的显式
operation/execution facts，在用户工作被接纳前确定一个已有 route。某个 Invocation 的显式冷
`prepare_replay` 可以再捕获所选 exact variant；可共享的 immutable capture recipe 可留在 Executable，
但 address/workspace/run state 仍由 Invocation 拥有。CUDA Graph 等 capture state 必须绑定 exact shape、
storage address/extent、Queue/device 以及 Provider 声明的其它稳定 key；不匹配时只能选择已编译的 general/
piecewise/eager route 或在接纳前明确失败，不能在 submit 中临时 capture/JIT。

prefill、mixed、uniform decode、LoRA 或 speculative batch 等业务分类由推理引擎拥有。若其中某个事实会
改变 Graph 结构、Provider 支持或 capture 合法性，引擎必须把它表达为不同 Graph/Executable，或通过已
登记的 typed operation/Invocation execution fact 显式传入；Graph 不扫描 tensor 内容猜测 batch 模式。
消费方也拥有多个 Executable/shape domain 的 cache 与选择策略，HYLE 只保证每个不可变 Executable 内部
的有限 dispatch 可解释、有界，且不会变成隐藏的模型调度器。

一旦某个 Queue/Provider 已接纳用户工作，后续错误只经 Completion 报告，不能静默在另一 Provider 重跑；
否则 side effect、资源写入和部分完成会被重复。只有 compile/prepare 或 submit 的 pre-acceptance decline
可以采用已冻结 fallback。

## 7. Compute、Mem、Sched 与 Obs 的边界

```text
graph_ir      -> compute_ops
graph_compile -> graph_ir + compute provider contract + mem
graph_plan    -> compute + mem                            # 当前为 Graph 内部源码边界
graph_sched   -> graph_plan + compute_sched + sched       # 未来可选组合
```

- Graph 向 Mem 交付 lifetime/alias/resource facts；Mem 不理解 Tensor、Node 或 provider policy。
- Graph 向 Compute 交付 region 和 bindings；Compute 不反向读取全图做偷偷融合。
- Sched 决定 CPU work、blocking provider、请求取消和公平；Graph 只声明 dependency/resource readiness。
- 分布式引擎拥有 rank/topology、communicator 创建、全局 batching 与故障恢复。Graph 可调度一个使用
  external communicator resource 的 typed **本地 collective participation**，并把其 Queue/Completion 接进
  本地 plan；它不在 foundation v1 内变成跨进程控制面。一个进程多 GPU 仍可由引擎为每个 rank 持有
  Executable/Invocation，并通过 communicator runtime 协调。
- Obs 接收 compile/submit/region/completion seam；Obs OFF 时不改变 plan layout 或热路径分支。
- Error 保存 node/region/provider/backend context；不能另造 graph 私有 TLS error。

Compute 不依赖 Graph。直接 kernel/op 用户因此不取得 graph compiler、container-heavy IR 或 Sched runtime。

## 8. 旧 Graph 资产的处置

当前源码中的三族能力不能因目录叫 `graph` 就继续留在新产品：

| 旧能力 | 新裁决 |
|---|---|
| incremental dirty/cached `TaskGraph` | `DEFER` 为未来 `graph_incremental` 候选；只有 CAD/build 等真实消费者出现才恢复并重新审计 |
| `parallel_for/reduce/scan/sort` | 不属于计算 DAG；若 Sched 原语不足且有独立用户，另审 `parallel` factor/product |
| data-parallel partition helper | 作为 Graph compiler 的私有启发式或未来 `parallel` 候选逐入口裁决，不保留旧公开面 |

迁移时逐入口标记 KEEP、MOVE、REPLACE、DELETE；没有消费者和自然 owner 的旧 API 直接删除，Git 历史
负责恢复。不能为了兼容旧 benchmark，把 callable task DAG 与 tensor compute DAG 焊成一个万能类型。

## 9. 性能与正确性资格

Graph 至少要用三种不同层级验证：

1. **one-op 税**：Graph Executable 对同一 PreparedOperation 的额外 steady 成本；同 buffer/layout/queue；
2. **multi-op 收益**：真实 region 的 submission、fusion、workspace reuse、copy/sync 与 direct sequence 对比；
3. **消费者纵切**：量化 transformer block 的 prefill/decode、CUDA 同 storage domain 多 lane
   copy/compute 与 capture，以及 Vulkan compute 的资源流水线。

每项都记录 compile/cold 与 execution/warm 两种边界，不能把 compile amortize 后与 peer 每次 prepare 对比，
也不能把 HYLE external storage 与 peer 内部 cache 混为同合同。正确性覆盖 shape/stride/zero-size、alias、
unsupported op、OOM、provider failure、device loss、取消、迟到 completion、并发 Invocation、重复 reset、
partial compile failure 和销毁顺序。

prepared 与 managed 可在同一用户完成边界上比较，但必须把 managed 的执行节点实例化、元数据 allocation
及内部 runtime 工作全部计费；不能拿其 kernel-only 数字比较 prepared 的完整 submit，也不能把首次编译
混入一方 warm 样本。HYLE 抽象税用同一个 Provider/同一 replay 合同作 direct 对照，避免把 Provider
内部成本误归因于 Graph。

Graph 的价值不要求每个 node 都由 HYLE kernel 获胜。基础 Graph factor 必须先证明选择和组合没有
material 税。减少 submission、copy、repack、allocation 或同步的收益，只能在 provider region、fusion
或跨设备 copy planning 已进入相应 closure card 后测量；没有这些变换的 CPU direct-composition 纵切不得
为了制造 `WIN` 改变接口或测试边界。长期 Graph 产品在进入完整发布裁决前，仍须在至少一个真实 multi-op
provider/backend workload 上形成可解释收益。

## 10. 首轮纵切和停止点

开工与 [Compute 的顺序](compute-design.md#10-开工顺序)共同执行：

1. 用 CPU reference provider 建 `GraphBuilder -> Graph -> Executable -> Invocation -> Completion` 的 one-op
   完整旅程；
2. 加 HYLE SIMD/native provider，证明同一 Graph 无需改写即可选择另一实现；
3. 扩为 normalization → matmul → gated activation，接 Mem liveness/workspace；
4. 先统一单-op 与 region candidate、完整边界成本和资格化 route profile seam；用 oneDNN 真实 region
   验证，不能让 blocked 的 CPU ggml 或只按覆盖节点数的 selector 成为前置；
5. 用 NVIDIA CUDA 验证同一 storage domain 的 copy lane → compute lane → copy lane、设备依赖、
   happens-before workspace 与 capture；
6. 用 CUDA Q4 MatMul/MLP 验证 leaf/region、HYLE/成熟 Provider 与不同 storage residence 的完整路由；
7. 再完成两条主平台栈：NVIDIA 的 CUDA backend/providers；Apple 先资格化 delegated MLX compiled region，
   同时保留原生 Metal custom-kernel/interop 和 MPSGraph/MPS 候选；Core ML 留在模型级 routed runtime
   边界；
8. 再落到正式支持的 Vulkan SPIR-V，验证消费级设备与 renderer 的 external memory/semaphore handoff；
9. 扩 attention、quantized layout 和重复 decode；HIP/ROCm 依 Compute 的无限期 defer 决策，不进入当前
   Graph 产品资格。

上述步骤中的算子只作为真实纵切的最小架构探针：先用 gated MLP 检验常量、分支、region、workspace 与
liveness，再用 attention/decode 检验动态 binding、resource state、异步 completion 与重复执行。它们不
授权在 Graph 合同稳定前横向填充整个 CPU kernel catalog。反过来，Graph 也不能只靠 fake provider 或
单节点测试宣称完整；至少一个成熟 provider 的真实多算子 region 和一条真实异步 backend 纵切通过后，
才算具备承载后续算子扩展的稳定底座。

每一步都有独立 closure card、固定 consumer slice 和停止预算。one-op lifecycle、P0 成本或三种 C++ 交付
未通过时，不得用“后面完整 graph 会摊薄”进入下一步；provider region 不成立时也不得扩大 operator
catalog 掩盖根因。

## 11. 架构稳定条件

Graph 架构只在四条探针上接受或反证：CPU one-op 抽象税、CPU multi-op region/workspace、CUDA 同 storage
domain 多 lane/happens-before/capture、Apple delegated MLX region 与原生 Metal interop。探针只允许修正它
直接反证的 seam，
不能借机横向增加 operation catalog 或模型 runtime。

foundation-v1 架构基线不替代实现证据。只有上述四条探针再加 CUDA Q4 MLP 的 leaf/region/
storage 路由、source/static/shared 和终局红队全部通过，Compute/Graph foundation implementation 才能签发
不可变 candidate；在此之前禁止把 provider/算子广度当作“基础架构完成百分比”。Vulkan、更多算子和
完整 Transformer 是锁后扩展，不作为反复改变对象生命周期与调度权的借口。

这些探针还须覆盖有限派生 shape、安装包 custom execution、独立/冲突资源域、多 variant 常量共享与
并发/capture 内存、可行解存在/不存在两种预算耗尽，以及 Apple managed/Metal prepared 的真实成本
分类。它们是本轮发现的基础合同缺口，不另起一套泛化 runtime 或无限算子任务。

形成候选后，只有真实纵切与合同矛盾、第二个独立消费者需要现有 ownership/concurrency/control-flow
机制无法表达的能力、正式 backend/provider 无法实现合同，或合格证据证明 material 热路径税来自 Graph
抽象，才允许重开架构。新的 op、shape bucket 或 provider 若能表示为既有 typed node/region/specialization，
只新增 route，不重开 Graph 生命周期和资源模型。
