# HYLE Graph：计算 DAG、编译与可复用执行计划

> 决策阶段：**四路线有限调研完成，重构基线待用户审视**。共同合同与旧代码处置已形成研究结论，
> 不是实现或性能资格通过；不得从本文自动恢复正式实现，也不新增模块状态。本文定义 Graph 的长期产品合同；当前实现状态只查
> [`module-status.json`](../implementation/module-status.json)。每轮纵切只证明其 closure card 登记的
> DAG/Executable/Invocation 合同。旧 `TaskGraph`、parallel/data-parallel API 与 benchmark 已退出产品树；
> 被删除的旧字节只可从 Git 历史取得调查事实，历史 task/evidence 不是当前设计输入、实现模板或用户承诺。
> 本页不签发用户可用性，也不要求兼容旧头。

2026-09-14 的整体复核保留上述骨架，修正 replay 成本、派生 shape、custom execution、资源副作用域、
variant 内存与搜索受限合同；依据见[复核记录](../implementation/tasks/compute/compute-graph-route-unification-audit.md#整体架构复核与合同修正)。
当前范围与研究停止点见[架构定案调研](../implementation/tasks/compute/compute-graph-route-unification-audit.md#当前任务四条独立路线的架构定案调研)。
2026-09-15 的[研究裁决](../implementation/tasks/compute/four-route-architecture-decision.md)要求重构
Provider 生成结果的所有权和固定执行入口；这些新合同不能当作当前用户接口已交付。

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

模型推理在 CPU、CUDA、MLX/Metal、Vulkan 中选择一条计算路线。CPU+GPU 分担同一模型计算的联合
计划不在 HYLE 功能范围内；host 控制/输入输出传输与同一 GPU 上跨 Provider 的执行不在此排除项内。
精确边界见 [Compute §1.3](compute-design.md#13-四条独立路线与明确非目标)。

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
- Provider 在运行时生成的结果由真实 result owner 保活，不强制先分配 Graph output 再复制。固定结果槽
  与实际 payload 分开；借用、导出、复用、预算遵守 [Compute §3.6](compute-design.md#36-输出-binding-与生成结果的-owner)。
  这不是允许无界/数据依赖输出 shape，也不是把 Provider 私有 heap 归入 Graph workspace 后忽略它。
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

三条资源轴仍必要：即使只有一个 GPU 计算设备，也有 host/device storage、transfer/compute lanes
和 Provider 自有资源。它们不是模型 CPU+GPU 联合放置的承诺。推理路线先限制计算候选，再规划资源；
host storage、host callback 或同步 endpoint 不自动增加 CPU 模型计算 target。

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

各预算必须说明所约束的集合，不能用一项局部计数冒充全部内存。编译制品存储准入按 canonical
materialization 的请求字节、Provider 声明的 persistent 和额外 packing 聚合；有限上限下未知项默认
拒绝，显式接受未知量只能约束已知部分。该门进入完整候选的冷重选事务，family 共享项只扣一次，
后继变体只取得余量。它不等于 allocator/驱动的实际物理占用，也不包含控制 metadata、外部借入
backing、cold scratch 或 Invocation/capture；这些仍需各自 owner 的完整账目和峰值/预留准入。

整个编译过程另有 family-wide requested-storage 配额：supplied-allocator metadata、canonical 请求与
Provider 在 prepare 前声明的额外 private resident/peak。它不与上述 persistent 分类重复相加；峰值在
受管请求前准入，resident 在全部编译 scratch 释放、完整变体表发布前检查。私有 resident 上界覆盖保留
到最终 teardown，peak 另含 prepare/失败清理；成功后归还临时差额，失败则先销毁部分制品再归还。
Region discovery/resource-query 显式借用同一个 compiler allocator，scratch 在回调返回前释放；这些
尚未取得私有上界的阶段禁止创建 private heap/artifact，不能回用 Context allocator 绕过准入。
输入 Graph/借入 backing、Context/Queue/backend/driver 控制与物理 rounding 分开，不冒充进程 RSS。
报告保存发布快照而不保活 payload；运行实例使用独立的 base/runtime allocator，不误扣编译额度。

运行存储配额是另一明确集合：同一 family 的所有 general/exact Invocation 共用严格 requested-byte
admission，包含各实例的 supplied-allocator metadata、真实 workspace backing 请求以及 Provider 的额外
private 上界。Graph 用 Mem 的 quota/reservation 组合生命周期，不实现第二套通用配额算法。独立资源先
预留后分配，晚到 Completion/escaped metadata owner 不提前归还。Provider 上界明确排除前两项，防止
重复计费；未声明不能猜零。Context/Queue、外部绑定、编译常驻与物理 allocator/driver overhead 分开。
未选择配额时不创建 tracker，submit 不做预算搜索/逐项计费；有限完整配额不能偷偷允许无上界 capture。

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

Graph 的 `ViewSpec` 用同一组维度表达式描述 reshape/broadcast 的尺寸以及 slice 的 start/count，axis、
step 和 permutation 在编图时固定。逻辑 node 区分未绑定 ViewSpec 与 concrete Compute Operation，
不能用一个假固定尺寸描述冒充动态操作。输出可声明 `TensorSpec::inferred` 的 shape/type/storage-format
约束，让唯一 producer schema 推导实际 strides 和 block axis；这种 output-only spec 不能作为独立
input/constant 的存储描述。具体绑定按拓扑复用 ComputeOps 的 View 语义，不默认把转置或广播变成 dense，
也不隐式 materialize。全域路线还需证明布局与 alias offset 的上界及实例化合同，精确点的成功不构成全域证明。

View 的完整域证明只使用有限 shape program 的关系：相同表达式、显式 guards、乘除因子消去和保守区间，
不枚举尺寸或引入通用符号求解器。无法证明时拒绝该域而保留 exact 路线。包络描述允许把不同合法点的
stride/span 上界组合在一起，因此它不是实际 slice/reshape 的一个输入样本。共享 lifetime plan 对 alias
只延长 backing 根的寿命；实际 BufferView 范围在冷创建/绑定时从父 view 推导并检查，避免把互斥的最大
offset 与最大 span 相加。父 view 按生产顺序复用，长 alias 链不在每个后代重新回溯。

未融合的零拷贝 View 是 Graph 自有 metadata step：复用 ComputeOps schema/alias 计算，没有 Provider
制品或 kernel dispatch，不能仅为读懂描述而要求 host-accessible storage。它仍拥有正常的 DAG/control
边与 alias lifetime。真正跨 storage 的搬运是显式 Copy；如果 region 吸收 View，则由该 Provider 完整
保持其语义。Provider 硬约束不能被 metadata 路线假冒满足，报告也不得给无 Provider 的步骤伪造 identity。

表达式按创建顺序引用此前维度或非负常量，因而不需要递归图解释器。派生 bounds 是保守 envelope，
guards 可能排除各轴上界的组合；不能把这个组合当作合法 concrete shape 先做单点 schema inference，
再声称覆盖整个 domain。整数或 descriptor 的保守上界无法表示时要求缩窄输入范围，不靠通用约束求解
或抽样自动收紧。具体表示与当前落地范围分别见活动卡和用户指南，不把设计中的 bounded replay 写成已交付。

完整域的一般路线共享同一个不可变物理计划、调度、workspace 上界及非并发制品的互斥边界；具体
Invocation 只持有实际描述与从已编译根实例化的执行状态，不能通过复制整个 Plan 意外创建独立互斥锁。
静态节点仍可使用 exact 制品；含动态节点的 region 必须另有完整域合同，不能因其边界形状恰好固定就
忽略内部动态节点。常量描述在该 Executable 中固定，动态外部张量属于 input。报告必须区分 envelope
与 exact signature；域内 workspace 保留上界，不能宣称实例化时已经做了紧凑重规划。

有限 exact 表是 compile 的显式输入，不是热态扩充的缓存。general 必须完整覆盖声明域，指定 exact 点
全部冷准备成功后才发布 Executable；点失败不能默默缺项。Invocation 在接纳工作前求值 guards，并用
全部维度的确定性有序索引选择现有 exact 子计划或 general；submit 不做此选择。每个 exact 制品可独立
保活，因此 canonical 常量使用不反向持有 Plan 的共享 owner，artifact 先于最后一份常量和 Context lease
销毁。Provider 私有 packing 不能未经身份/兼容性证明就合并。

Region 的 cold `ConstantArtifactAccess` 把兼容性固定为同一次 family、具体 Provider instance、logical
constant/descriptor、Provider 声明的 packing-format/configuration key 和已登记 storage。factory 只在缺失
时创建完整不可变 payload；opaque owner 由 Graph 保活，mutable invocation/capture 仍独占。采用此入口的
packing 单独声明 packed bytes、optional persistent metadata 及 allocator 之外的 private resident/peak，
不再进入 region 自身的重复计数；unknown persistent 在 family 也按 unique owner 保留，不能当零。每个候选 Plan
保有实际使用的 owner refs；拒绝先毁 region，再归还无引用 payload。冷 flat hash lookup 在 compile 返回前
销毁，发布后只剩 immutable refs，submit 无查表。现有不支持该入口的 Provider 仍持有独立 packing，不能
因 format 看似相同就强行合并。完整 profile 包含 packing key、资源与 family 复用量；跨 compile 内容身份
仍由 workload-data 契约描述，不是对裸 source 指针做缓存。这是渐进式 Provider 接入，不重写 vendor kernel。

所有路线共用一次 compile 的累计 discovery/state/prepare 预算，不能按变体重复充值；per-provider cap
只是每次发现的局部上限。普通查询描述 general，子路线有自己的 exact signature、队列、workspace 和
并发合同；聚合工作计数与逐路线执行成本分开报告，不把不同 shape 的运行时间求和成一个 workload。
逐路线 workspace 限额不替代 §4.2 的完整常驻/冷峰值/并发/capture 预算。

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
HYLE 不能钻进不透明制品逐节点替换；若它公开 custom lowering/Primitive 扩展，则可在 **Provider 编译前**
形成“成熟算子 + HYLE lowering”的完整 region 候选，让该 runtime 保留内部资源与依赖。MLX 的这条
路径已有有限 GPU FFN 证据；扩展造成的 metadata/layout/fusion 成本仍计入完整比较，不能外推给无扩展的 ggml。

Region boundary 分别声明预绑定 storage 与 Provider-generated result，不统一强制 out-buffer ABI。
编译期验证逻辑 descriptor/owner/导出能力，运行时保活真实 result lease；同 Provider 的内部中间值不
强制物化为 HYLE buffer。Graph 只规划自身持有的 backing 与必要边界转换，不重复规划 opaque 内部 heap。

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
- 每个已准备 op/region 边界可以有一次固定 dispatch，单节点 region 同样适用；不得逐节点重新筛选
  backend/Provider 或逐元素做类型擦除。连续片段可 lower 成原生程序，避免逐层通用 wrapper；
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

v1 route search 先限制在用户所选计算路线及其允许的设备/Provider/资源内，不遍历全部已构建 backend，
不搜索 CPU+GPU 模型联合计划。接纳通过 contractibility 验证的显式 node-set region，并把每个 node 的
leaf 放进同一空间；连续区间走 fast path。唯一可行路线可以直接准备，有真实竞争才搜索；
target/storage/lane 与 boundary transformation 在所选范围内联合裁决。
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
node-cover 内的资源选择使用依赖前缀的有限回溯：只有真实分支保存 choice ordinal/count，共用累计 state
预算；不物化全部组合。资源局部拒绝不等于该 Provider 在其它资源上也不可用。lane 固定后才做 happens-before/
workspace，显式内存序列化的再排程保留该 lane。身份机制与此搜索都不等于自动调优、穷尽所有组合或全局最优。

完整计划的性能不能仅靠 cost interface 和人工 timing fixture 签发。基础验收必须有一条具名真实消费：
分别约束两条可行路线，按相同输入、精度、resident/cold/warm 范围与真实终态测量，形成身份完整的
profile，再重新 compile 检查选路；不要求故意制造 HYLE 赢家。缺失、过期、不可分与硬预算拒绝继续
使用已有反例验证，不再扩一套 tuner。选择后 warm 路径不读取这些资料。

量化的 weight encoding、activation 舍入/量化、accumulation 与 output dtype 是不同事实。比如一方先将
F32 activation 量化成 Q8，另一方保持 F32，不能仅凭输入输出类型相同认定数值合同相同。输入量化不能
由 accumulation_type 推断授权；必须显式进入 schema/转换或经审计的数值策略，再进入 profile identity。
相同数值合同不要求使用相同算法，但双方都须满足同一允许舍入/误差边界和正确性判据。
不相容路线只能作自然产品取舍，不能通过放宽测试容差冒充同合同替换；这同样约束基础 CUDA Q4 纵切。

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

### 6.3 路线内编译与特化执行

先不考虑旧 Graph 类型和实现，从用户固定部署、原生 runtime 资源/执行合同及完整性能成本设计。
下面的 compiler、Program 和 per-run state 是职责，不规定旧类名或布局；现有搜索器、segment loop、
capture 接口和 CMake factor 均可改写/删除。普通 C++ 旅程不要求用户组装三轴 ID 数组：选路线/设备、
提供图与数据、显式准备和运行即可；专家资源/lane 控制才显式展开。旧测试不得成为形状约束。

部署选定后不是持续做全局设备调度，而是「固定配置 → 有限计划族 → 重复执行」。prefill/decode、
batch/shape 或其它显式事实变化时选择已准备的适用 variant；输入值、KV position、专家 ID 等变化不
天然要求重新选 Provider。确实影响程序/资源/合法性的事实必须进入 key/domain；新配置或域外 shape
走显式冷准备，不在 submit 中重规划。CMake 只能锁定实现集合，不能提前知道所有模型运行参数。

Graph 编译可以共用语义、合法性检查与候选/边界协议，但执行物允许分别实现：CPU 的 prepared calls/
持久 executor；CUDA 的 native/vendor commands 与可选 capture；MLX 的 GPU managed island 与必要
Metal 片段；Vulkan 的原生录制/提交。原生 pipeline、pool、handle、array 和回收状态留在 owner 内。
不要求四条路线经过相同 segment 解释循环；单片段可直接绑定完整执行入口。多 Provider 或多 lane
确实需要的依赖协调才保留，不能为了通用性让每个 kernel 回 host 等待或创建无关 runtime。

旧 Graph 的 `submit` convenience 会在未准备或 Queue 改变时调用 `prepare_replay`，通用循环还保存了
逐次 Queue/segment 协调。重构的低延迟入口必须显式完成 prepare/bind，再执行已绑定 native program，
不能自动准备；如保留方便入口，明确其冷行为。把不变验证移出热路径，不删除必要的 in-flight、alias、
接纳/失败检查。先限定路线再做 region 选择，唯一可行 route 走快路，不无条件跑全局组合搜索。

具体[旧代码处置](../implementation/tasks/compute/four-route-architecture-decision.md#4-旧代码的处置依据)已经形成，
但尚未实施；以实际依赖、生命周期和完整性能成本取舍，不为旧 API 保留多层适配。

HYLE 仍拥有逻辑图、同路线内的 partition、边界资源/同步和最终完成责任。Provider 可以接管已选择
region 的局部计划，但不得跨边界改放置。原生片段一次固定调用不是零指令；低开销必须用同 Provider、
同资源/完成边界的 direct 对照证明，不能靠宏/模板名称或“全部 GPU”推导。

模型 CPU+GPU 联合路线不是待补第五类执行器。已有通用多 target 数据结构不必为缩小产品范围立即删除，
但不能据此签发模型联合推理能力；`capture + commands` 的 hybrid replay 也不表示 CPU+GPU 联合计算。
后续产品修改须按当前小卡审计实际入口、约束与特化，不在文档轮伪造已实施。

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
- 最小可选 Obs 接入属于基础验收：compile/submit/region/completion 事件须有明确 owner，且必须区分
  host enqueue 与设备真实终态；Obs OFF 时不改变 plan layout 或热路径分支。`CompileReport` 是冷计划
  诊断，不替代这些运行事件。当前是否实现由活动卡取证，不从本条设计承诺推导已可用。
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

prepared 与 managed 可在同一用户完成边界上比较，但必须把 managed 的执行节点实例化、元数据/结果/临时 allocation
及内部 runtime 工作全部计费；不能拿其 kernel-only 数字比较 prepared 的完整 submit，也不能把首次编译
混入一方 warm 样本。HYLE 抽象税用同一个 Provider/同一 replay 合同作 direct 对照，避免把 Provider
内部成本误归因于 Graph。

Graph 的价值不要求每个 node 都由 HYLE kernel 获胜。基础 Graph factor 必须先证明选择和组合没有
material 税。减少 submission、copy、repack、allocation 或同步的收益，只能在 provider region、fusion
或跨设备 copy planning 已进入相应 closure card 后测量；没有这些变换的 CPU direct-composition 纵切不得
为了制造 `WIN` 改变接口或测试边界。长期 Graph 产品在进入完整发布裁决前，仍须在至少一个真实 multi-op
provider/backend workload 上形成可解释收益。

## 10. 首轮纵切和停止点

施工顺序、已完成证据和下一项只维护在
[当前任务](../implementation/tasks/compute/compute-graph-route-unification-audit.md#当前任务四条独立路线的架构定案调研)，
不与 Compute 再各维护一套队列。Graph 的基础完成边界保持如下：

- CPU direct/one-op、真实 multi-op region 与持久 executor 组合；
- NVIDIA 的真实浮点 vendor 调用、保留压缩权重的 Q4 decode/prefill、不同候选与完整资源成本；
- Apple MLX 计算与受支持的原生 Metal kernel 组合，以及 Vulkan 的真实资源/计算/同步流水线；
- 以上路线共用 DAG/Executable/Invocation、冷选路与 shape/capture 合同，并通过实测 profile 回灌、
  生命周期/失败、最小 Obs、三交付和终局性能/红队资格。

每条纵切只补证明该边界必须的算子，不要求完整模型、全部量化格式或全部 Provider；也不能以 mock、
只有 CPU 计算的 GPU copy、或 Q4 整体展开样例代替相应真实路线。到基础 gate 完成时暂停共同审视，
之后才进入 attention/完整 Transformer 与目录扩充。HIP/ROCm 的无限期 defer 保持。

上面是正式基础实现的终点，不是当前架构研究的结束条件。当前只允许有限原型解决四条独立路线的
系统性未知量；输入输出传输、权重常驻、外部状态、同 GPU 跨 Provider 与最终完成仍必须能表达。
CPU 层段/FFN/专家与 GPU 联合推理的纵切已从范围移除。研究定案后先报告，不为完成本节全部产品
资格而继续施工，也不因为裁剪范围把未实现的其余路线写成通过。

纵切的单位是**一次完整公开用户旅程及其待裁决问题**，不是新增几个 API 的展示。每条记录正常输入、
至少一个会推翻对应假设的反例、资源/线程 owner、最终完成点，以及失败时应改哪个 seam。低层借入探针
只证明局部机制，不能把 Graph 的跨 Provider 调度、失败回收或完整成本一并签为通过。CUDA 主纵切需
贯通压缩 FFN 的第三方叶实现与可替换 custom 计算；Apple 需贯通 MLX region 与原生 Metal kernel 组合；Vulkan
专门反证跨资源访问和内存可见性。具体形状、证据与先后顺序只维护在上面的总计划中。

性能用同一程序分两步隔离问题：先对同实现 direct 与 HYLE 路线检查抽象税，再把成熟完整 region 与
同路线内跨 Provider 替换计划作同精度/格式/完成边界比较，包含复制、repack、融合损失和线程转交。若局部替换更慢，
保留成熟完整路线；若固定额外成本来自 Graph 边界，记录具体根因，不能因为 kernel-only 赢或差距落入
某个百分比就宣称计划更优。仅构造快慢标签或伪造 profile 的测试只验证 selector，不签性能胜负。

## 11. 架构稳定条件

Graph 架构在有限探针上接受或反证：CPU one-op 抽象税、CPU multi-op region/workspace、CUDA 同 storage
domain 多 lane/happens-before/capture、Apple MLX region 与受支持的原生 Metal kernel 组合，以及 Vulkan
SPIR-V 的执行与内存可见性。探针只允许修正它直接反证的 seam，
不能借机横向增加 operation catalog 或模型 runtime。
Apple 同 runtime 的公开扩展是有效组合路线，不强制为测试拆成外部 Queue；真正公开的外部
buffer/Queue 互操作仍须独立资格，不能由该探针外推。

foundation-v1 架构基线不替代实现证据。只有上述探针再加 CUDA Q4 MLP 的 leaf/region/
storage 路由、source/static/shared 和终局红队全部通过，Compute/Graph foundation implementation 才能签发
不可变 candidate；在此之前禁止把 provider/算子广度当作“基础架构完成百分比”。Vulkan 的算子/provider
广度、更多算子和完整 Transformer 是锁后扩展，不作为反复改变对象生命周期与调度权的借口。

这些探针还须覆盖有限派生 shape、安装包 custom execution、独立/冲突资源域、多 variant 常量共享与
并发/capture 内存、可行解存在/不存在两种预算耗尽，以及 Apple managed/Metal prepared 的真实成本
分类。真实持久 CPU executor、完整测量 profile 的回灌、跨运行时晚到失败/回收、可选 Obs 事件和每个
backend 的三交付也须有具名验收；复用已存在的单测，不机械做全部平台乘全部错误的笛卡尔积。
它们是有限的基础实现/资格任务，不另起一套泛化 runtime、通用 tuner 或无限算子任务。

形成候选后，只有真实纵切与合同矛盾、第二个独立消费者需要现有 ownership/concurrency/control-flow
机制无法表达的能力、正式 backend/provider 无法实现合同，或合格证据证明 material 热路径税来自 Graph
抽象，才允许重开架构。新的 op、shape bucket 或 provider 若能表示为既有 typed node/region/specialization，
只新增 route，不重开 Graph 生命周期和资源模型。
