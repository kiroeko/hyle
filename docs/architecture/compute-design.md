# HYLE Compute：异构执行与算子提供者架构

> 决策状态：`ACCEPTED`。foundation-v1 架构基线已经裁决；这不表示对应实现或平台纵切已经完成，
> 也不新增一种模块状态。实现状态与施工资格只查当前 closure task 和
> [`module-status.json`](../implementation/module-status.json)，当前可用能力查[用户指南](../user/compute.md)。
> 本文只定义 Compute 的长期产品合同，不复制阶段性完成状态。下文平台栈和开工顺序是目标边界，不表示
> 每个 provider/backend 已经交付。被删除的旧实现只可从 Git 历史取得调查事实；历史 task/evidence
> 不是当前设计输入、实现模板或用户承诺。

2026-09-14 整体复核后的合同修正及源码依据见
[架构复核记录](../implementation/tasks/compute/compute-graph-route-unification-audit.md#整体架构复核与合同修正)。
这些决策不代表新增接口、MLX adapter 或性能资格已经交付。

## 1. 产品承诺

Compute 是 HYLE 的**异构执行层**。它让 C++ 消费者可以：

1. 发现并选择 CPU、CUDA、Metal、Vulkan 等计算设备，建立显式执行上下文；
2. 在不经过 Graph 的情况下直接准备并执行一个算子或原生 kernel；
3. 让 Graph 编译器把一段逻辑计算交给最合适的 HYLE 或第三方 provider；
4. 显式管理 queue、storage、copy、同步、完成和 native interop，而不隐藏昂贵工作；
5. 在 provider 不支持某种形状时得到可解释的能力/失败结果，而不是静默复制、同步或退回慢路径。

Compute 不是推理引擎。模型格式、tokenizer、KV/prefix-cache 策略、continuous batching、sampling、
HTTP serving 和模型循环仍由消费方拥有。Compute 也不是 Graph：它执行已经决定的 op、kernel 或
compiled region，不拥有跨节点拓扑、全图 placement、liveness 和融合决策。

## 2. 两条同等重要的用户路径

### 2.1 直接执行

工程计算、渲染 compute pass、codec/图像处理和自定义推理 kernel 不应为了运行一个操作先造完整图。
概念上的正常路径是：

```text
Device -> Context -> Queue
TensorDesc + Mem BufferView -> TensorView
Operation -> PreparedOperation -> per-lane OperationInvocation
OperationInvocation -> submit(bindings) -> Completion

typed backend Module/Kernel + Queue -> submit() -> Completion
```

- `Device` 是可复制的能力描述，不拥有 runtime；`Context` 拥有 backend/provider 会话。
- `Queue` 表达一个有序、单 producer 的**执行 lane**；同一 Context 和同一 storage domain 可以建立多个
  Queue，用于计算、copy、通信或独立并发请求。capture 是一组 Queue/命令的录制与 replay 模式，不是另一种
  工作 lane。不能用全局默认 stream 隐藏 affinity 或同步。Graph 等组合者
  可取得一个短生命周期 submission-sequence guard，使一组命令连续录入；guard 只覆盖 host enqueue，
  绝不持有到设备执行结束。
- `TensorDesc` 把 logical dtype/shape 与 physical storage format/byte layout 分开；fixed block encoding
  使用开放数值 identity，无法通用解释的 packed layout 使用 opaque identity + exact extent。
  scale/zero/bias 和量化公式属于 typed operation/provider inputs，不进入 storage descriptor；
  `TensorView` 只把描述与 Mem storage 绑定，不拥有 payload。
- `PreparedOperation` 是可共享的不可变 artifact，把能力检查、provider 选择、layout/repack 决策和可能的
  编译移出热调用；`OperationInvocation` 拥有一次并发 lane 的 stream、binding metadata 与 workspace。
- 原生或 HYLE kernel 的低层路线由 typed backend façade 提供；Vulkan SPIR-V、CUDA module/kernel、
  Metal pipeline 和 CPU function target 都通过有类型的 backend extension 录入 Queue。普通 Compute 不公开
  一个抹平 vendor 语义的万能 `Module/Kernel/CommandList` 层。
- `Completion` 是**公开 direct submit 或组合 sequence 边界**的生命周期对象。等待、轮询、callback 和
  coroutine bridge 必须共享同一个完成状态，不能制造互不相认的 event。Provider/region 在内部只报告命令
  是否已接纳，不为 Graph 每个 step 生产用户 Completion；一个成功的 sequence finish 返回覆盖完整 ordered
  Queue prefix 的 Completion。销毁 pending handle 不等待，但 backend 必须把附着的 execution owner 保活到
  真实终态。

一次性 convenience 调用将来可以组合 prepare + invocation + submit，但必须在名称和文档中暴露它包含冷路径工作；
循环执行默认使用 prepared 形状。

### 2.2 Graph lowering

Graph 不逐节点调用一个通用 `execute(op)` 并在热路径重新路由。Compute provider 接收单 op 或一段
候选 region，返回绑定到设备、layout、shape signature 和 provider identity 的 compiled artifact。
Graph 的 `Executable` 只保存这些 artifact 与已经决定的 copy/sync 步骤，调用时直接提交。

因此 HYLE 既能借成熟算子库获得广度，也能保留 HYLE 自实现的胜出 kernel。二者竞争发生在
prepare/compile 阶段，不在 element、lane 或每次节点执行中反复裁决。

### 2.3 Provider 是冷路径翻译器，不是另一套 Operation

`Operation`/`TensorDesc` 是 HYLE 面向用户的唯一语义描述；它们既不执行，也不因接入一个 vendor 增加
`GgmlMatMul`、`CublasMatMul` 一类平行公共枚举。Provider 是显式装入 `Context` 的小型类型擦除配置，
在冷路径把这份中立描述翻译为自己真正拥有的制品：

```text
HYLE Operation + concrete Tensor/storage contract
  -> provider proposal / prepared route
  -> immutable provider artifact
  -> per-lane invocation + prebound submit

HYLE Graph region + shape domain + boundary contract
  -> provider region proposal / compile
  -> immutable provider-native graph or engine artifact
  -> delegated or shared-runtime region submit
```

叶函数型实现可以产出一个 `PreparedOperation`，自己 host 图的实现则产出 compiled region。后者不需要被
伪装成公开叶函数：例如 ggml adapter 可以在 compile 时建立一个单节点或多节点 `ggml_cgraph`，随后把它
当成不可变制品重复提交。单节点 mini-graph 在 HYLE 热路径上可以具有函数式调用成本，但仍由 ggml
拥有其 tensor metadata、graph/runtime 和内部融合语义。这样的 **prepared replay** 必须证明制品可直接
重放；不能从第三方 API 名字里有 `compile` 推断这一性质。只缓存编译程序、每次仍实例化执行节点的
runtime 使用 §3.4 的 **managed replay** 合同，不伪装成零分配制品。

把第三方源码下载、编译或链接进制品本身不产生 HYLE 能力。只有一个已审计 adapter factor 能完成
HYLE 语义到第三方制品的翻译，并兑现 storage、failure、threading、workspace 与 Completion 合同后，
相应 route 才存在。若第三方只公开 graph API 而叶 kernel 是私有符号，合法选择只有：

- 以 shared/delegated compiled region 借用公开 API；或
- 另立版本锁定、维护责任明确的 source-patch/kernel factor，并独立通过升级、错误、交付和性能审计。

包含私有头、调用未导出 kernel 或复制实现片段都不能在普通 provider 内偷渡成稳定路线。

渐进式 best-of 的裁决边界也是 HYLE op/region，而不是第三方黑盒内部的某个隐藏 kernel。同一段计算可
同时收到成熟 provider、HYLE native/vendor provider 和 custom kernel 的候选；Graph 按完整 region 成本
选择。未证明 material 优势时保留成熟路线，HYLE 明显胜出时在 canonical tensor boundary 切出相应
op/region。不能在不透明第三方 region 内局部替换；拆分造成的 copy、repack、同步与 fusion 损失必须计入
选择，必要时扩大第三方 island，而不是为了“逐算子可换”制造更慢的计划。

### 2.4 HYLE 保留全局调度权，Provider 只拥有局部制品

HYLE 的性能责任不要求它亲自调度每个底层 kernel。cuBLASLt、oneDNN、TensorRT、MLX 和 ggml 都可能在
一次公开调用内选择或提交多个 kernel；强行拆开反而会丢失 fusion、私有 layout 和厂商调优。必须唯一
保留在 HYLE 的是**全局决策权**：

- HYLE Graph 拥有逻辑 DAG、region 边界、execution target、跨 region storage/copy/repack、lane、依赖、
  全局 liveness/workspace 和最终 Completion；
- 入选 Provider 只拥有一个 region 内的 kernel/engine、局部临时 storage、局部顺序与真实执行终态；
- Provider 不得越过已选边界读取其它 HYLE node、偷偷改变全局 placement，或启动未声明的并行/runtime；
- HYLE 可以把整个模型、一个 block、一个融合片段或单 op 交给 Provider，但粒度本身必须参与成本裁决。

因此 Graph 执行的是由 HYLE 排好的异步 segment DAG；每个 segment 可以是 HYLE 叶操作，也可以是第三方
compiled region。这样的层级调度仍由 HYLE 控制全局关键路径，同时允许成熟实现保留其最擅长的局部计划。

Provider region 的语义是**可收缩 node set + 显式 boundary ports**，不是某个拓扑序中的 begin/count。
候选集合必须非空、node identity 唯一，并且不存在一条路径从集合内离开后又进入集合；
这样 Graph 才能把它收缩为单个 segment 而不改变 DAG 依赖或制造环。多个入选 region 必须互不重叠。
连续拓扑区间是常见且便宜的发现/执行 fast path，但不是 Provider SPI 的能力上限；Graph 不枚举任意子集，
只在显式 candidate/state 预算内验证 Provider 提案并做有限选择。

### 2.5 Route 选择必须比较完整计划，而不是“覆盖更多节点”

所有单 op 与 region 实现最终都要归一成同一类冷路径 route candidate，至少携带：覆盖的 HYLE nodes、
适用的 shape subdomain/typed guard、target/provider/version、输入输出 storage/layout、shared/delegated runtime、workspace/persistent/packing、
copy/repack、同步、dispatch、线程/host blocking、replay 成本类别、精度/确定性/alias/failure 与完成边界。选择目标是满足硬约束后
降低预期关键路径和资源成本，不是机械最大化第三方覆盖率，也不是看到一个 HYLE kernel 的局部点值更快
就强行切开 region。

性能信息按以下规则使用：

1. 结构事实先排除会增加不可接受 copy、host wait、内存、线程嵌套或不兼容 layout 的候选；
2. 资格化 evidence profile 只在绑定 provider/version、device/ISA、dtype/layout 和 shape domain 时用于
   冷路由；不同 provider 各自声明、基准边界不同的纳秒估值不能直接相加或排序；
3. Graph `compile()` 只消费显式传入、身份完整且同完成边界的 evidence profile，绝不执行用户 graph、
   分配测试 payload 或在编译中偷偷计时。Provider 可以在自己的冷 prepare 内用隔离的私有 scratch 做
   有预算的 tactic/JIT 调优，但不得把调用方 runtime input/output 当调优 payload；immutable constant 只按
   显式 policy 读取。调优须报告成本、cache identity 与确定性政策；
4. 未来可选的 `graph_tune` 工具可以用调用方提供的代表性 binding 生成跨 Provider profile，但它是独立的
   显式任务，不是 Graph core 的隐藏阶段。

无合格性能证据、结果不可分或小于预声明 material 门时，优先成熟实现；HYLE 自实现只在精确适用域内
成为首选。route 在 prepare/compile 后冻结，HYLE steady replay 不查 profile、不计时、不搜索、不重新建图；
managed Provider 的内部执行准备按 §3.4 单独披露并计入完整成本。
profile/cache identity 至少绑定 HYLE/provider 版本、backend/device/driver、operation/region fingerprint、
shape domain、layout/precision、workload objective 和构建 feature；任一相关身份变化都重新裁决，不能
继承旧机器的赢家。Graph core 不读写进程全局 cache 或隐式配置文件；跨 Executable 的 specialization
cache 由消费方拥有。

一个 bounded/guarded Executable 若需要多条 route，compile 必须产出有限、互斥且对声明 domain 完整覆盖的
plan variants，或保留一条覆盖剩余范围的 general fallback。guard 只能读取已验证的 concrete symbolic
dimensions 和显式 typed execution facts，不能扫描 tensor payload；prepare 阶段按有界表选择，submit 不再
询问 Provider。无法覆盖整个声明 domain 时 compile 失败或由调用方显式缩窄 domain，不能让未覆盖 shape
留到执行中突然 `Unimplemented`。

### 2.6 “借能力广度”与“保留全局性能权”怎样同时成立

这两个目标不是二选一。固定分工如下：

| 决策 | 唯一 owner | 说明 |
|---|---|---|
| 公共算子语义、逻辑 DAG、shape/precision | HYLE | 用户不写 `GgmlMatMul`、`CublasMatMul` 等平行图 |
| region 边界、target/storage/lane、跨 region copy/sync、全局 liveness/workspace | HYLE Graph | 这是 HYLE 不可外包的全局计划权 |
| region 内 kernel、fusion、私有 layout、局部 workspace 与局部顺序 | 入选 Provider | HYLE 不拆开厂商或成熟 runtime 已经优化好的黑盒 |
| 最终 route | HYLE 冷路径 selector | 比较完整可执行计划；Provider 不能用名称或自报局部点值直接获胜 |
| steady replay | 已冻结的 `Executable` | HYLE 不搜索、不计时、不重新建图；Provider 内部执行成本由 replay 合同声明，不能隐藏 |

因此“使用第三方”不是让 ggml、MLX 或 TensorRT 接管 HYLE Graph，而是把一个已经由 HYLE 划定边界的
segment 外包给它；“HYLE 自实现胜出”也不是强行替换第三方黑盒里的私有 kernel，而是在相同 HYLE
boundary 上提交一个竞争候选。若拆开第三方 region 会丢失 fusion 或增加 copy/repack/sync，局部更快的
HYLE kernel 仍可以在完整计划上落败。反过来，只要 HYLE 单 op 或融合 region 在精确适用域内形成合格的
material 胜利，selector 就能选择它，而无需改变用户 Graph。

Provider SPI 是 HYLE 各可选 factor 之间的源码级内部合同，不承诺稳定的动态插件 ABI。普通用户只看到
`Operation`、`Context`、`Executable` 与公开 provider factory；新增第三方实现由独立 factor 完成版本、
许可证、三交付和 backend-specific runtime 审计。需要用户自定义计算时使用 typed custom operation/kernel
入口（§6.1），不是要求安装包用户包含 `src/` 私有头。数值 schema identity 必须跨 source/static/shared
成立，不能依赖某个 DSO 内静态对象的地址；识别 schema 和提供执行实现是两项分别验收的能力。

### 2.7 规范例：CUDA 上的 Q4 MatMul 怎样路由

以 `RMSNorm -> Q4 gate/up MatMul -> SwiGLU -> Q4 down MatMul` 为例；单独一个 Q4 MatMul 是它的退化情形。
正确流程是：

1. 模型 loader 用普通 `MatMul` 表达逻辑计算。权重 `TensorDesc` 仍声明逻辑 scalar/shape，同时用对应
   Provider 发布的 `StorageFormatId`、block elements/bytes、alignment 与 byte strides 表达 Q4 物理编码；
   scale/zero 等若不是块内编码，则是显式 typed tensor input，不能藏进 `void *` metadata；
2. 用户或上层引擎明确把 activation、weight 与 output 放在 CUDA storage，或允许 Graph 选择其它 target。
   “CUDA backend 已存在”只证明能建立 context/queue/memory/event/copy，不证明任何 Q4 算子已经存在；
3. 冷编译枚举精确候选。例如 HYLE/CUTLASS 自实现的 Q4 kernel、确实接受该格式的 NVIDIA vendor route、
   ggml CUDA 单节点 artifact，以及覆盖整个 MLP 的 ggml compiled region。不能理解该 `StorageFormatId` 的
   Provider 必须返回 `Unimplemented`，不得把压缩 bytes 当普通标量或偷偷反量化；
4. 对每个候选先检查 dtype/precision、适用 shape subdomain/guard、alignment、alias、CUDA device、buffer
   import、stream/event 与 completion 可行性。无法证明 bounded domain 的 kernel 明确 exact-only。ggml 若只能使用自有 allocation/stream，就成为 delegated island，并显式产生
   ingress/egress、常驻显存和同步成本；若公开合同允许 caller storage/stream，才可成为 shared-runtime；
5. selector 比较的是完整候选计划：kernel/region 执行、constant packing、copy/repack、launch/event、
   host/thread handoff、丢失或保留的 fusion、workspace/persistent budget 与 compile amortization。不能只拿
   `q4_matmul` 私有 kernel 的一次耗时决定是否切开四节点 region；
6. 若 HYLE Q4 MatMul 的合格收益大于拆分边界成本，可只切出该节点；若权重已在 ggml-owned storage、相邻
   SwiGLU/MatMul 能融合，完整 ggml region 可能仍胜出；若 HYLE 的整段融合实现后来胜出，再把整个 region
   替换为 HYLE artifact。没有可比较证据时选择已配置的成熟路线，不用微基准猜测；
7. 编译结果固定 provider、artifact、packing、storage、lane、workspace 与 Completion 路线，并在 report
   中说明为什么选中/拒绝每类候选。一个 bounded decode Executable 可以同时保留 general route 和若干
   exact plan variants；某个 Invocation 的显式冷 `prepare_replay` 再按 concrete shape、address 与 Queue key
   捕获所选 variant。推理引擎把自己的 prefill/decode/batch class 映射到明确 Graph/Invocation facts，Graph
   只按 concrete shape 与已编译 guard 选择。后续 token replay 只重绑合同允许变化的 activation/KV storage，
   不再询问 Provider 或重新构造 ggml graph。

若当前构建只有 CUDA runtime、没有任何接受该 Q4 格式的 operation/region Provider，唯一正确结果就是
`Unimplemented`。这不是“自动 fallback 还没写好”，而是防止错误计算、隐藏 CPU offload 和伪零拷贝的
必要合同。

### 2.8 冷编译目标、证据与回退边界

Graph compile 必须解析出显式输入或默认的 workload intent，至少区分 steady latency 与 steady throughput、
预期并发 Invocation 数、确定性要求，以及 compile/packing 是否需要按预期 replay 次数摊销。默认是**长寿命、单 Invocation 的
steady latency**：在没有 replay 次数时不把 compile 纳秒与执行纳秒任意相加；workspace/persistent、host
wait 和 compile latency 仍受各自硬预算或 policy 约束。内存不是和时间混成一个魔法分数的第三种目标，
而是由 storage budget、persistent budget 和允许序列化策略形成硬约束。

可比较的执行估值只能来自 Graph 收到的只读 qualified route profile。它以同一 HYLE boundary 的完整
候选计划为计费范围，并绑定 provider/build、device/driver、shape、dtype/format/layout、完成边界与
workload intent；Provider 自报的“相对自己 fallback 节省若干 ns”不是这种 profile。没有合格 profile 时，
Graph 先按结构成本和硬约束排除劣路线，再按调用方配置的成熟 Provider 顺序作确定性裁决；覆盖节点数和
dispatch 数只能作最后 tie-break，不能冒充自动 SOTA。

调用方可以全局 allow/deny Provider，也可以对具名 node 设置 preferred/required Provider；region 只有在
覆盖的全部 node constraint 相容时才可入选。没有约束时 Context 中的 Provider 顺序就是成熟 fallback
顺序，而不是性能证据。Provider 必须声明候选是否满足 operation 的精度和调用方的确定性要求；这些是
硬约束，不允许 selector 为了速度静默放宽。

`compile()` 是对用户 graph 无执行副作用的冷转换：它可以 validate、枚举候选、编译/JIT 入选 artifact、
按 `ConstantBinding` 的显式 borrow/materialize policy 处理常量，但不能用用户 binding 运行 operation 或以真实 output 改变全局
路由。TensorRT/编译器一类 Provider 可以在自己拥有的隔离 buffer 上做有预算的局部 tactic search；它不
取得 Graph 全局候选的裁决权，也不能产生外部 side effect。未来 `graph_tune` 若准入，必须由调用方显式
提供代表性 bindings、允许执行的纯/确定性范围和独立预算，产出 profile 后再调用正常 compile；有 side
effect、外部资源写入或无稳定 instance fingerprint 的 custom operation 默认不可调优。

候选 prepare/compile 必须事务化，并以结构化 outcome 区分 success、无副作用的 candidate decline 与
真正 Status error。`Unsupported` 或明确 candidate-local resource-budget decline 可以在有界预算内排除后
重选；裸 `ResourceExhausted`/`Unimplemented` Status 不能靠调用方猜成回滚证明。数据损坏、内部错误、
device loss、违反 schema、已接纳工作或未知失败立即终止 compile，不能借 fallback 掩盖。重选次数是显式
硬预算；耗尽时没有已验证的完整可执行计划就失败，已有可行计划则按 Graph §6.1 的显式搜索受限策略
处理。优化搜索耗尽不是正确性放宽，也不能把尚未 prepare 的候选当成可交付 fallback。

## 3. Backend 与 Provider 是两条轴

这两个概念不得再混成一个“大后端”枚举：

| 角色 | 回答的问题 | 例子 |
|---|---|---|
| backend | 设备、context、queue、memory、module、同步和 native handle 怎样工作 | CPU、CUDA、Metal、Vulkan；合同允许以后加入新的 backend |
| provider | 哪些 op/region 能以何种 layout、精度和成本在该 backend 上执行 | HYLE native、ggml、oneDNN、MLX、MPSGraph、vendor library |

Provider 必须声明而不能让调用方猜测：

- 支持的 op、dtype、shape/stride/layout、量化格式、in-place/alias 与精度；
- 编译和执行所需 workspace、常量 packing、storage domain 与 alignment；
- 是否启动或占用自有线程、是否阻塞 host、是否支持并发 invocation；
- copy、同步、host round-trip、动态分配和首次 JIT 等可观察成本；
- cancellation、device loss、迟到 completion 和销毁顺序；
- provider/version/device/ISA identity，供 plan cache 和诊断使用。

Provider 可以自行拥有 device allocation 或 graph allocator。HYLE 必须如实保留这个 owner；不能把
一个伪造的 `BufferView` 当成零拷贝。ggml CUDA 的历史调查只证明这种风险真实存在，见
[ggml provider 的 storage 约束](borrowing-ggml-memory-model.md)；adapter 开工时仍须对锁定版本复验。

### 3.1 Execution target、storage domain 与 execution lane 是三条正交轴

异构运行时不能用一个 `domain` 同时表示“由谁执行”“在哪里分配”和“沿哪条时间线提交”。长期合同明确
区分：

| 轴 | 身份与 owner | 典型实例 |
|---|---|---|
| execution target | 一个 HYLE Compute Context 及其 backend/device 执行会话；provider 仍按 op/region 独立选择 | CPU context、CUDA context、Metal context |
| storage domain | 一个具名 `MemoryResourceRef`、分配属性和可选硬预算 | system host、pinned host、device-local、managed/unified |
| execution lane | 绑定 target 的 ordered Queue/timeline | compute stream、copy stream、communication stream、CPU caller lane |

Graph 的 `StorageDomainId` 是其编译环境中的 resource ordinal；Mem 的 `MemoryDomainId` 是 provider 给出的
accessibility/coherence identity。两者相关但不互换：前者决定从哪个 resource 分配和记账，后者参与判断
两个 resource 是否处于同一可见/一致性域。`ExecutionTarget` 可以有一个 `default_storage` 作为普通输出和
workspace 的默认选择，但这个默认值绝不成为 target identity。

因此一个 target 可以访问多个 storage domain，一个 storage domain 也可能被多个 target 访问；同一 target/
storage 组合可以有多条 lane。Graph 的 node placement 选择 execution target，value placement 选择 storage
domain，step scheduling 最后选择 lane。target 不同不自动产生 copy；只有 exact resource、backend access、
provider binding requirement 或调用方 required placement 不兼容时才物化 copy。不能通过复制 target 或
storage domain 来伪造多 stream，因为那会错误复制 workspace、liveness 和物理值身份。

lane 的 `compute/transfer/communication` 只是冷路径调度意图，exact Queue 的真实能力在 replay preparation
时验证；capture 由 `ReplayMode` 与 backend capture capability 表达，不进入 lane role。普通用户仍只需一个
target、一个 storage domain 和一条 compute lane。多 lane 不得让单 Queue 路线增加每 step 搜索、动态分配
或隐藏同步。vendor-specific priority、engine、queue-family 或 native stream 参数继续放在 typed backend
extension 中。

### 3.2 Provider 的两种诚实运行模式

Provider 在同一 capability/compile 协议下必须声明其执行资源属于哪一种模式：

1. **shared-runtime provider**：使用 HYLE Context、MemoryResource 和 Queue 录入命令，例如 HYLE native、
   共享 CPU executor 的 oneDNN，以及确实能借入 HYLE stream 的 provider；
2. **delegated-runtime island**：provider 自己拥有 allocator、stream、graph/context 或线程 runtime，例如
   不能自然借入 HYLE Queue 的 MLX/第三方完整 region runtime。Graph 只通过显式 boundary binding、copy/
   interop、依赖和 Completion bridge 调用该 island。

两种模式共享 operation/region、layout、precision、failure 和诊断语言，但不能共享一份虚假的资源所有权。
compiled artifact 必须保活真实 runtime owner，并报告输入输出 residence、可否零拷贝、host blocking、内部
线程、并发 invocation、capture、取消和 drain。delegated island 不得忽略传入 HYLE Queue 后在另一条未知
stream 偷跑，也不得把 provider completion 当作已经加入 HYLE timeline；无法建立明确 bridge 时该 route
就是不支持。

公开 direct `prepare()` 只枚举 shared-runtime provider，因为它返回的 invocation 必须能提交到调用方 Queue。
delegated proposal 只由 Graph 的内部 exact-route compile seam 选择；Graph 拥有该 invocation，并以 provider
返回的真实 Completion 连接 island 边界。这样 direct 用户不会拿到一个构造成功却无法执行的 artifact，
shared steady path 也不需要每次为 delegated mode 做动态路由。

### 3.3 Provider 的资源合同必须先于 steady submit

“Context 能访问某个 `MemoryResourceRef`”只证明 backend 可以触碰它，不证明某个 provider/kernel 接受该
layout、alignment、storage kind 或 owner。provider 合同因此分成两个冷阶段：

1. capability/proposal 根据 operation、shape domain/layout、execution target 给出适用 subdomain/typed guard、
   每个 boundary binding、workspace 和逐 constant borrow/materialize 要求，以及 shared/delegated runtime mode；
2. route 被选中后，prepare/compile 对 Graph 选定的 exact resource 或显式 provider-private bridge 做最终
   验证，并把结果固化进 immutable artifact。

Direct Compute 的 resource-aware prepare 也走同一内部合同；普通 convenience 可以从实际 TensorView 推导
resource。资源不匹配必须在 prepare/bind 或 Graph compile 的冷路径返回，使另一个 provider/copy route 仍可
被选择。已经开始 Queue admission 后，provider 不得第一次发现 storage 不兼容，也不得隐藏 allocation、copy
或 fallback。Delegated island 只向 Graph 暴露 canonical boundary storage；island 内部 tensor、allocator、
stream 和 workspace 仍由 provider 规划并在 report 中如实计量。

bounded 制品在 compile 时必须证明完整域，而不是只证明 envelope 的一个点。Graph 创建具体 Invocation
时，可经 Compute 私有冷接口把已编译域实例化为普通 exact PreparedOperation：复用 checked shape program
绑定尺寸并重验 schema，Provider 只生成既定代码的参数/执行状态，不重跑 proposal、prepare、JIT 或 tactic
选择，不访问 tensor payload。域根拥有 dimensions/guards/axis maps，具体子制品保活它至自身销毁之后；
runtime/alias 合同不得变化，workspace 不得超过已编译上界。这样重复 submit 仍走原来的普通执行回调，
不增加动态 shape wrapper 或第二套 Queue/Completion。实例化的冷 metadata 分配与跨变体总内存预算须
单独计量；它不等于每次 replay 可以重新分配，也不是新的公开产品入口。

动态维度与动态布局是不同合同。默认布局继续保留 exact strides/opaque，变化的尺寸重建 dense 或
blocked-dense。需要消费 Graph View 推导的具体 strides 时，Provider 必须显式接受 plain、非负 strides
和有限可触及字节上界的布局族；其 entire-domain 承诺包括零步幅等合法成员，不能由 dense envelope
推断任意 strided 支持。固定尺寸的这种布局族也不是 exact point。Graph 冷推导描述后，Compute 重验
维度程序/guards、dtype/format/rank、布局族/字节上界和原 operation schema，再调用已编译制品的冷
实例化；既有默认布局仍须完整描述相等。此布局承诺不替 Graph 证明 View 关系、alias offset 或并行写入
安全，不放宽已编译 alias/资源/完成边界，也不新增公开 stride 求解器或热路径解释器。

### 3.4 Replay 成本合同独立于 runtime ownership

shared/delegated 回答“谁拥有 Queue、allocator 与完成状态”，不回答“每次调用是否重新分配”。同一
Provider 的不同 artifact 也可能具有不同成本，必须逐 route 声明，而不是按库名给资格：

| Replay 合同 | 准备完成后的每次执行 | 用法与约束 |
|---|---|---|
| prepared | HYLE 与 Provider 均不重建图、不推导 shape、不进行 heap allocation、JIT 或常量 packing；只录入已准备命令 | Native、厂商 primitive 或 capture 的严格路线，必须实证而不能按名称推断 |
| managed | HYLE 固定计划不变；Provider 可为已编译程序实例化执行节点、分配元数据、做已声明的内部 cache lookup/shape 计算 | 完整开销、owner、失败、线程、内存及完成边界可审计；不能标成 prepared，也不能隐含新编译 |

两类共用 Operation、Executable 与 Invocation，不制造两套算子 API。默认可选择已资格化且满足调用方
约束的两类路线，按完整成本与成熟 fallback 顺序裁决；调用方可在冷路径要求 **prepared-only**，此时
managed/unknown 一律排除。该策略不是 CMake 开关，也不是允许 managed 绕过精度、内存或同步硬门。

两类都禁止 HYLE 热路径搜索/建图/分配、submit-time JIT/tactic search/首次常量 packing、未声明 copy/sync 和
接纳后静默 rerun。Provider 的首次设备编译及合法 warmup 在显式 prepare 内完成，不执行调用方的副作用
或可变资源；无法证明整个 bounded domain 不再首次编译时只能缩窄域或 exact-only。managed 内部成本
按实际实现报告 exact、上界或 unknown；unknown 不能证明零成本或满足有限硬预算。内部状态和临时
allocation 若被异步工作引用，必须活到其真实完成点；仅用于 host 录入的临时 metadata 可按合同提前
释放。OOM/部分接纳仍须处理，不能靠名称豁免生命周期审计。

MLX v0.32.2 的公开 compiled function 在缓存命中后仍调用 `compile_replace`，创建执行数组和容器；这是
不能一律承诺 prepared 的具体依据，而不是 MLX 性能失败的结论。
[MLX 实现](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/compile.cpp#L985)
Apple adapter 先验证这种 managed 路线的真实边界；若后来证实某条公开 API/capture 满足 prepared，可
单独升级该 route。原生 Metal 保留严格 kernel 路线；不为满足纸面零分配而默认维护 MLX 私有补丁。

## 4. 平台栈与优先级

“正式支持”与“首要优化目标”是两件事，设备 backend 与算子/图 provider 也不是同一层。CUDA 是一套
比 Metal 更宽的生态，但在 HYLE 分类中，二者首先都承担设备、queue、module、memory 和 completion
这一执行轴；CUDA libraries、MLX 等再作为上层 provider 审计。当前长期路线是：

| 优先级 | 设备 backend | 可审计的 provider/runtime | 产品要求 |
|---|---|---|---|
| 主要 | Apple 原生 Metal | **MLX（首个资格化 provider）**、HYLE Metal kernel、MPS/MPSGraph、Accelerate/BNNS | 推理纵切优先证明 MLX region；原生 Metal 保留 custom/best-of kernel、资源互操作和缺口路线；不同层不得混成一个 backend |
| 主要 | NVIDIA CUDA | HYLE CUDA kernel、cuBLAS/cuDNN/TensorRT、ggml 等 | 覆盖消费级 RTX 与可用数据中心形状，按完整 region 成本裁决 provider |
| 正式保底、次要优化 | 原生 Vulkan；Apple 上经 MoltenVK | HYLE SPIR-V kernel、ggml Vulkan 等 | 给广泛消费级 GPU 和 renderer interop 一条共同路线；MoltenVK 只签 Apple portability，不替代原生 Metal 性能路线 |
| 基线 | CPU | HYLE SIMD/native、oneDNN、ggml；Apple 可含 Accelerate/BNNS | 三平台 correctness、fallback、provider contract 和可重复性能基线；不是“只剩 CPU 时才用”的 stub |

AMD HIP/ROCm **无限期不进入当前产品线**：没有 component、CMake gate、空 target 或恒失败 stub，Vulkan
只提供跨厂商保底而不冒充 ROCm。只有同时具备可持续使用的 AMD 真机、明确消费者和维护预算，才以新的
closure card 恢复 HIP backend/provider 调研；通用 backend 扩展合同为这次恢复保留空间，但不构成承诺。

### 4.1 施工包不能隐藏 SIMD

进度清单必须把产品纵切与实现底座分开。`simd` 是已经独立关闭的 CPU 叶组件；Compute 不重新实现它，
但 **HYLE-native CPU provider** 仍是一项显式施工包：它负责把 Tensor/Operation 的 shape、layout、alias、
precision 和 completion 合同映射到 SIMD whole-row/fixed-kernel 路线，并拥有 tiling、tail、fallback 与
prepare-time route。把它只写成“Compute 算子目录”会错误隐藏 HYLE 自有算力路线。

| 层 | 施工包 | 直接实现底座 | 边界 |
|---|---|---|---|
| CPU 执行 | CPU backend + HYLE-native provider | Platform + Mem + **HYLE SIMD** | reference/fallback/best-of；backend 与 Native provider 分别拥有 runtime 和算子实现，身份以中央组件图为准 |
| CPU 借入 | oneDNN provider | oneDNN x64 BRGEMM ukernel | 可选 MatMul provider；不是 CPU backend，也不替代 native/SIMD |
| 跨后端借入 | ggml provider islands | ggml CPU/CUDA/Metal/Vulkan 等已准入 backend | 每种 backend 分别审计，不能把“ggml provider”写成 CPU-only 单项 |
| Apple | Metal backend + MLX/MPS/HYLE/ggml providers | Apple runtime 与 HYLE custom kernels | MLX-first，但保留 native Metal escape hatch |
| NVIDIA | CUDA backend + vendor/HYLE/ggml providers | CUDA runtime、vendor libraries、custom kernels | provider 选择按完整 region 成本 |
| 可移植 GPU | Vulkan backend + HYLE SPIR-V/ggml provider | Vulkan；Apple portability 使用 MoltenVK | 正式保底，不冒充原生 Metal 性能 |
| 图编译 | Graph IR/compiler/plan | 上述 prepared op/region | 横跨所有 backend/provider，拥有 partition、placement、copy/sync 和 workspace |

因此阶段清单应写成“CPU backend/HYLE-native-SIMD → oneDNN/ggml CPU region → Apple/CUDA/Vulkan
backend-provider 组合 → 跨 provider Graph”，而不是把 oneDNN、ggml 与整个 GPU backend 放在同一级。
当前状态仍只从 registry 和各 closure card 读取，本表不复制完成百分比。

### 4.2 Apple 各层的边界

- **Metal** 是 Apple GPU 的底层图形与计算 API。HYLE 只拥有其中的 compute pipeline、resource、queue、
  synchronization、profiling 和 native interop；raster、swapchain 和 present 不进入 Compute。
- **Metal Performance Shaders / MPSGraph** 分别提供优化 primitive 与符号计算图，更接近 operator/graph
  provider。若借入，artifact、workspace、stream 与 graph ownership 留在隔离的 provider island。
- **MLX** 是面向 Apple Silicon 的机器学习数组与计算图框架，具有 lazy evaluation、compile、stream、
  unified-memory 模型和完整 C++ API。对 HYLE 的 Apple 推理纵切，它是首个要资格化的 region provider、
  interop 对象和独立性能参考；它不是 Metal 的别名，也不是设备 backend。
- **Core ML** 是加载、编译和运行模型的更高层部署 runtime，并可调度 CPU、GPU 与 Neural Engine。
  HYLE v1 不把它伪装成低层 kernel provider；只有真实纵切证明模型级委托仍保留 HYLE 所需的 storage、
  scheduling、diagnostics 和 specialization 控制时，才登记 routed runtime adapter。

Apple 主路线因此是“**MLX-first provider + 原生 Metal backend/escape hatch**”，不是“MLX 对 CUDA”。
HYLE 优先把 MLX 能完整、高效承载的 region 交给 MLX；缺失算子、已证明胜出的 custom kernel、原生
resource/queue interop 和 provider 边界由 HYLE Metal 路线承接。选择发生在 prepare/compile 冷路径，
并计算 ownership、copy、graph capture、线程和 source/static/shared 交付成本；“MLX-first”不是未经
测量就强制所有 op 经 MLX。MLX 的 replay 成本按 §3.4 实证分类，compiled function 不自动等同于 Metal
command replay；managed route 必须把执行元数据和 runtime 开销计入比较。

### 4.3 NVIDIA 的默认不是“裸 CUDA kernel”

CUDA backend 只拥有 device、context、stream、event、module、memory 与 native interop；它不是与
cuBLAS/cuDNN 竞争的另一套算子库。NVIDIA 正式构建的默认算子路线采用 **vendor-first、HYLE
best-of/escape-hatch**：

- 普通 GEMM、batched GEMM 与受支持 epilogue 首先资格化 cuBLASLt/cuBLAS。cuBLASLt 的 algorithm
  heuristic 只在 prepare/compile 冷路径查询并缓存到 artifact，不允许每次 submit 重新搜索；
- 受支持的 SDPA/normalization 等完整 primitive 优先资格化 cuDNN，而不是把几个 HYLE 小 kernel
  拼成一个必然增加 launch 和中间流量的默认实现；
- HYLE CUDA kernel 承接 vendor primitive 不表达的量化格式、融合、layout、极小 decode shape、interop
  约束，以及同合同证据明确胜出的路线。CUTLASS/CuTe 可以作为这些 kernel 的私有实现底座，不成为
  另一套公开 Tensor/Operation API；
- ggml 等 provider 仍可竞争完整 region。选择必须计入 repack、workspace、launch、同步、capture 与
  中间 tensor 流量，不能因为 provider 名称或单 kernel 峰值硬编码胜者；
- direct `prepare` 与 Graph compile 完成选择后，热路径只执行预绑定 artifact。用户可以要求特定
  provider、确定性或禁止 fallback，但普通用户不应被迫手选 cuBLAS handle 或 CUDA kernel。

因此“CUDA 为默认”表示默认使用 CUDA **平台栈**；在这条栈内，成熟 NVIDIA library 是常规首选，HYLE
只在功能缺口、跨算子收益或可复现的同合同胜利上替换它。相关能力边界以
[cuBLASLt](https://docs.nvidia.com/cuda/cublas/) 和
[cuDNN Frontend Attention](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/operations/Attention.html)
的锁定版本重新审计，不能把当前厂商支持矩阵当永久事实。

Vulkan compute 不是只为 benchmark 保留的“可选图形 API”。它服务：

- Windows/Linux 上的跨厂商 compute；
- 与已有 Vulkan renderer 共用 device、queue、buffer 和 timeline semaphore 的资源管线；
- SPIR-V compute kernel、工程计算和不要求 CUDA 的部署；
- Apple 上经 MoltenVK 把 Vulkan 调用翻译到 Metal 的 portability 资格路径，但不能冒充原生 Metal 性能、
  Apple 专有能力或完整原生 Vulkan 支持。

Compute 的 Vulkan extension 必须覆盖 device/queue-family 选择、pipeline/module、descriptor/binding、
dispatch、barrier、timestamp 和 device-lost 语义。与 renderer 或其他 API 共享资源时，memory 与
semaphore/fence 必须分别导入/导出并查询 handle capability；共享地址或 native handle 不自动意味着
同步、coherence 或 ownership 正确。raster render pass、swapchain、窗口和 present policy 属渲染产品，
不进入 HYLE Compute。

## 5. Storage、同步与生命周期

Mem 是 storage-semantics owner；Compute 是 device/provider artifact 与异步使用 owner：

- Mem 提供 `MemoryResourceRef`、`Buffer`、views、mapping、range allocator 和 lifetime planner；
- Compute context/provider 暴露具名 storage resource，并解释 native allocation identity；
- Graph 推导 resource、liveness 和 alias facts，再交给 Mem planner；
- Queue/Completion 决定 Buffer、workspace、packed constants 和 staging slot 何时可复用。

控制 metadata 的 owner 与 execution target 分开。direct 默认继承 Context allocator；Graph 的私有冷
prepare/instantiate/create seam 可以指定编译制品或运行实例的 allocator，不能因为 node 路由到另一个
Context 就绕过 Graph 配额。成功分配和错误回滚使用同一 owner/size/alignment；该借用必须保活到所有
相关子对象和 Completion 释放。Provider 必须使用本次 callback 收到的 allocator，不能假定 prepare
与 create-invocation 的 allocator 相同。Context/Queue、自有 Provider heap、payload resource 与驱动
内存仍由各自 owner 负责；这条注入边界不自动证明完整物理内存预算，Graph 必须另行对账。

任何异步 submit 的输入、输出、kernel argument backing 和 provider state 都必须活到 Completion 终态。
direct `OperationInvocation` 和 Graph Invocation 都由最终 Completion 保活到真实终态，并在此期间拒绝复用；
Tensor payload、外借 workspace 与 Queue 仍由调用方保活，不能因为 C++ handle 被移动或销毁就推断设备工作结束。
取消是“请求停止”，不是“立即可回收”；不支持抢占的设备仍需 drain 迟到完成。Context 的关闭是显式
`close/drain` 旅程，析构不能偷偷做无界等待，也不能在 work in flight 时释放 backend state。

组合 plan 不以逐命令 `wait()` 建依赖。同一 Queue 的命令在一个短生命周期 sequence 中按录入顺序建立
设备依赖，只在 queue segment 结束时记录一次 event/终态并返回 prefix Completion；不能先为每个 step
制造 Completion 再覆盖句柄。若后一命令在已有前缀之后同步拒绝，sequence 仍 finalizes 已接纳前缀并把该
拒绝附着到终态，而不是丢失工作或提前释放 workspace。不同 Queue 的兼容依赖优先录入 device wait；不兼容
路线只能成为 CompileReport 中显式 host barrier 或编译失败，不能偷偷同步。

Storage domain 至少区分普通 host、backend-visible/pinned host、device-local 和 managed/unified。
unsupported 必须返回明确能力错误；不能把普通 host RAM 冒充 pinned，把 unified address 冒充零复制，
也不能把共享物理内存的 host/device budget 重复相加。

Tensor 的 physical-format identity 不等于 backend memory identity。Plain Tensor 以 logical scalar width 和
byte stride 表达；单轴 fixed-block Tensor 记录 elements/block、bytes/block、alignment 与 block axis；更复杂
格式只暴露 opaque identity 和 exact bytes。通用层不得为了识别格式引入字符串查找、全局 registry、owner
或虚调用。格式专属 factory、alignment 终检与 scale/zero 等关联 tensor 由 typed provider extension 拥有。
未知但结构合法的 encoding 是 provider fallback 的 `Unimplemented`，不是 schema corruption。

## 6. Operator 面与可扩展性

公共 operator 合同以真实消费者反复出现的模型无关计算族为核心：

- copy/fill/cast、layout transform、gather/scatter 与 quantize/dequantize/repack；
- unary/binary、reduction、softmax、top-k 等张量运算；
- GEMV/GEMM/batched matmul 与必要的低精度 accumulation；
- normalization、activation、residual/gated 组合；
- embedding、RoPE/position transform、attention 的可复用计算 primitive；
- MoE dispatch/combine 所需的模型无关 tensor primitive。

这不是“看到 peer 有一个 op 就复制一个枚举”。精确准入集必须由 ggml/llama.cpp、ds4、vLLM、SGLang
的真实调用与至少一个独立成熟 provider 对账，并逐项登记精度、layout、alias 和 workspace。KV cell
管理、expert policy、mask 业务语义、sampling policy 与完整模型 layer 仍留在引擎。

未进入公共 catalog 的计算走 typed custom-kernel/provider extension。扩展不能依赖字符串属性袋、裸
`void*` 或 process-global 注册副作用；其 schema、owner 和 backend identity 必须在编译期或显式对象中
成立。这样后续可以逐个用 HYLE 胜出 kernel 替换 provider region，而不推翻 Graph 或普通用户 API。

### 6.1 安装包消费者的 typed custom-kernel 路线

`Operation::custom` 只定义计算语义，不把“可以识别 descriptor”当作“可以执行 kernel”。执行扩展使用
公开、对象局部的 typed factory：由显式 Context/编译环境拥有实现配置，冷 prepare 创建不可变 kernel
artifact；并发运行状态属于各 Invocation。实现代码及其资源 owner 保活到 artifact 和全部 Completion
终态；借用 native handle 必须明确 lifetime，不能因注册函数返回就释放捕获状态或卸载其代码。

factory 声明 backend、ports/layout/alias、资源访问、副作用、workspace、replay 与失败合同，使用既有
typed CPU/CUDA/Metal/Vulkan kernel/Queue 能力提交。内部可在每个 prepared op/region 边界类型擦除，
不在 element 上分发；不公开整套私有 Provider SPI，不增加全局 registry、稳定插件 ABI 或 Native FFI。
Compute 永远不依赖 Graph；Graph 通过同一 prepared proposal 取得执行实现，而不是要求消费者手写
第二套图。用户实现与成熟 Provider 在同一 boundary 上竞争，不因是 custom 就默认优先。

三交付验收必须用**仅安装头与库**的独立 C++ 消费工程：定义 schema、绑定 owning implementation、冷
prepare、放进 Graph 并取得正确 Completion；CPU 和 CUDA 各一条最小路线，检查失败/销毁/并发与跨
shared identity。只通过 schema 的跨 DSO 测试不算执行扩展通过；类型名和 helper 拼写在该能力卡中固定，
不把这里的设计描述冒充已有 API。

## 7. 内部因子与依赖方向

长期分层如下；名称是中央 component graph 的目标形状，不表示当前已经登记或实现：

```text
HYLE::compute                         # 用户产品 façade
  compute                             # 现有 runtime factor：Device/Context/Queue/Completion/interop
  compute_ops                         # Tensor/Operation 描述与公共算子合同
  compute_sched                       # 可选 Scheduler/coroutine completion bridge
  compute_backend_cpu|cuda|metal|vulkan
  compute_provider_native|ggml|onednn|mlx|mpsgraph|vendor

graph_ir -> compute_ops
graph_compile -> graph_ir + compute provider contracts + mem
graph_plan -> compute + mem                            # 当前为 Graph 内部源码边界
graph_sched -> graph_plan + compute_sched + sched      # 未来可选运行时组合
```

`execution lane` 是 Compute core 的资源语义、也是 Graph plan 的调度维度，不满足独立用户产品、依赖 owner
或测试 owner，当前不拆成 `compute_lane` factor。每个 delegated provider 仍按其外部依赖、runtime owner、
构建开关和测试资格建立独立 provider factor，不能做一个聚合所有第三方 runtime 的万能 adapter target。

`compute` runtime factor 不因 coroutine convenience 强制公开依赖 Sched；需要 `co_await`/task 组合的用户取得窄的
bridge factor。可选 provider/backend 各自拥有外部依赖和 feature 开关，未选择时不下载、不编译、
不导出恒定失败 stub。Compute 永远不依赖 Graph。

Compute/Graph 的真实纵切也是底层合同的消费者验收。若它证明 Mem、Sched、Platform 或其它已关闭
factor 缺少必要原子能力、所有权语义或自然调用形状，必须以该可复现 consumer journey 建立精确
reopen trigger，先修正真正的 owner，再回放受影响的直接消费者；不得为了保住旧状态而在
Compute/Graph 私造 allocator、scheduler、platform probe 或同义 lifetime 层。反过来，尚未被具体纵切
需要的便利功能不构成重开理由，也不能借 Compute/Graph 施工顺手扩张底层模块。

内部 factor 是否按上图拆成独立状态 owner，要在开工 baseline 审计中逐项满足
[质因数判据](capability-factorization.md)；用户始终只选择 `hyle::compute`，不会被迫理解 provider target。

## 8. 性能合同

Compute 的目标是**端到端可组合的最佳路线**，不是包装层在一个空调用 benchmark 中少几纳秒：

1. HYLE 的 prepared submit 不做 provider 搜索、shape inference、JIT、常量 packing 或 allocation；
   Provider 按 §3.4 声明 replay 类别，managed 内部执行成本不能隐藏或冒充零分配；
2. direct static kernel 可编译期路由，prepared op/region 至多支付一次预绑定调用边界，不按 element 分发；
3. provider 选择同时计入 kernel、layout/repack、copy、同步、thread handoff 和 workspace，不能只比 kernel；
4. host/device transfer 可批处理并与计算重叠时应以独立 execution lane 和显式依赖表达，不通过隐藏
   staging 或复制 storage domain 制造假零拷贝；
5. provider 自有线程必须可配置或在 region 级隔离，禁止与 HYLE workers 无界嵌套造成 oversubscription；
6. compile latency、warm execution、steady throughput、tail、peak/resident memory、copy bytes 与能耗分别报告。

同合同对拍必须锁定 op 语义、dtype/精度、layout、shape、线程/queue、warmup、storage residence 和计费
边界。HYLE 原生只有在资格化 workload 上形成可复现 material advantage 才取代成熟 provider；无法区分、
汇编等价或结果随顺序/地址翻转时优先成熟实现。Graph 路线还必须与直接 Compute 路线做 one-op 税基线，
防止编排层吞掉 kernel 收益。

## 9. 交付、诊断与负空间

- source、installed static、installed shared 提供同一 C++ 语义；当前不设计 Native FFI。
- feature 集合必须确定产物 identity；正式构建不能依赖 `AUTO` 让机器环境悄悄改变 provider。
- capability query、compile report 和 tracing 能解释选了谁、为何 fallback、插入了哪些 copy/sync、需要
  多少 workspace；关闭 Obs 时热路径不支付诊断成本。
- native interop 使用 backend-specific typed extension；普通 API 不泄漏 CUDA/Vulkan/Metal/MLX/ggml 类型。
- 不提供训练、自动微分、分布式 collective runtime、模型 zoo、render graph、窗口/present 或 serving。

## 10. 开工顺序

Compute 与 Graph 采用共同纵切，但按独立 factor 小步验收：

1. **资产与身份审计**：固定现有源码/头/CMake baseline；把每个入口裁决为 KEEP、MOVE、REPLACE 或
   DELETE；按质因数判据登记必要 component ID。旧 API 不需要兼容。
2. **Compute core**：先完成 CPU Device/Context/Queue/Completion、Mem resource bridge、typed native
   kernel 和 explicit shutdown；清除对 Sched 的无条件公开依赖。
3. **Compute ops**：冻结最小 Tensor/Operation/PreparedOperation 合同，以一个 CPU reference op 和一个
   HYLE SIMD/native op 证明 direct path。这里的产物是 typed schema 和少量**架构探针**，不是提前填满
   CPU kernel catalog；只实现随后 runtime/Graph 纵切实际需要的最小算子集合。
4. **立即接 Graph 纵切**：Graph 只做一个节点也能编译为 `Executable` 并通过 `Invocation` 运行；此时
   验出抽象税，而不是等 Compute 大全套完成后才发现形状不合。
5. **统一 route selection**：把单-op proposal 与 compiled-region proposal 放入同一有界计划搜索，先完成
   resource/ownership 筛选，再使用显式 workload intent 与资格化 route profile 比较完整边界成本；修复 typed custom
   operation 的 source/static/shared identity 和安装包自定义执行入口。此前按最大覆盖选择 region 的实现
   不能作为本阶段完成品；派生 shape、资源副作用域、variant 内存与搜索受限行为按 Graph 合同一并落实。
6. **真实 CPU region**：以量化模型中的 normalization → matmul → gated activation 为首个多节点片段，
   加 liveness/alias/workspace，并用 oneDNN 等成熟 Provider 验证 leaf/region 竞争。CPU ggml 当前 blocker
   不在关键路径上，也不能用空 factor 冒充纵切。
7. **异步资源探针**：先在 CUDA 走通同一 storage domain 上的 copy lane → compute lane → copy lane，
   覆盖 device dependency、Completion 聚合和 capture/replay；它是 execution-lane 合同的验收，不是提前
   扩大 CUDA 算子目录。
8. **CUDA Q4 路由纵切**：用 Q4 MatMul/MLP 同时验收 HYLE/vendor leaf、第三方 compiled region、量化
   storage residence、完整成本裁决与 warm fixed-plan；没有合法 Provider 时保持 `Unimplemented`。
9. **主要平台 slice**：NVIDIA 继续扩 CUDA providers；Apple 先资格化 MLX delegated
   region，验证 provider-owned runtime 的 boundary/completion bridge；同轮建立原生 Metal custom-kernel/
   interop 路线并审计 MPSGraph/MPS；随后扩 attention、quantized matmul、repack 和异构 partition。Core ML
   不进入首轮低层路线。
10. **Vulkan slice**：同一 graph/op 合同落到 SPIR-V，验证 native device、buffer、timeline completion、
   external memory/sync 与 Windows/Linux 消费级设备；MoltenVK 只签 portability 证据。
11. **Transformer block 与重复执行**：冻结后运行真实 prefill/decode、动态 shape specialization、并发
   invocation、取消/drain 和 source/static/shared 交付门。
12. **算子广度扩展**：只有 runtime、Graph、至少一个真实多算子 region，以及一条真实异步设备纵切共同
   证明上述对象与执行合同不再需要改写后，才按推理热路径与正交工程 workload 系统补齐 CPU 算子；
   每条新 route 复用既有 plan/provider seam，不借“补算子”重新设计 runtime。

因此本顺序不是“先造所有算子，再看它们能否组成图”，也不是脱离执行负载追求抽象意义上的完整 Graph。
在第 3--11 步，每条纵切只补充用于验证 ownership、layout、partition、workspace、completion 或异步语义的
最少算子；目录式 CPU kernel 填充必须等待这些结构经真实消费链压实。进度报告也必须把“schema 已定义”、
“已有 executable route”和“系统性 catalog 完成”分开，不能用前两者冒充第三者。

每一步都先创建合法 closure card 并完成 `SCOPED -> AUDITED`，只实现冻结的有限 `FIX/BORROW/DELETE`
集合；未达到下一步所需的 lifecycle、correctness 或 P0 成本结构就停止为 `BLOCKED`，不靠扩 benchmark
或继续研究掩盖状态。

## 11. 架构稳定条件

本轮跨 ggml/llama.cpp、vLLM、SGLang、ds4、oneDNN、ONNX Runtime、TensorRT、IREE、PJRT 与
ExecuTorch 的调查只签发上述边界，不把任何 peer 的完整 runtime 复制为 HYLE 产品。已接受的基线是：

- execution target、storage domain/resource 与 execution lane 三轴分离；默认 storage 只是 target 的属性；
- provider 先公开 binding/resource requirement，再对 exact route prepare，不能到 submit 才拒绝；
- region 用可收缩 node set 与显式 boundary 表达，连续拓扑区间只是一条 fast path；
- shared Queue 和 delegated island 两种 ownership 都保留，不能用一套伪 stream/allocator 抹平；
- prepared/managed 是独立的成本合同；HYLE 固定热计划，Provider 的内部执行开销如实声明、完整计费；
- 公共 custom schema 与安装包可用的 custom execution 入口分别验收，不能以身份测试代替执行旅程；
- Executable 的 shape domain 可以是 exact 或 bounded/guarded；capture 使用 exact key，steady submit 不 JIT；
- capture 是与编译正交的 replay capability，communication 才是独立 lane 用途；
- Graph 只规划 HYLE/shared-runtime storage 与 island boundary，delegated provider 自己规划内部 storage；
- request/KV/batching/sampling/offload 策略仍归推理引擎，Compiled Graph 不扩成模型 runtime。

实现必须先通过四个架构探针：CPU direct/one-op 同合同、CPU multi-op region/workspace、CUDA 同 storage
domain 多 lane/capture、Apple delegated MLX region + 原生 Metal interop。正式代码与这些结论冲突时停止并
回到审计；operator 数量、单次 benchmark 或为旧 API 保兼容都不能覆盖该反证。

foundation-v1 架构基线表示上述设计已接受，不代表没有反证可能。实现还必须补充 CUDA Q4 MLP 的 leaf/region/storage
路由探针，并让这些探针共同通过 source/static/shared、正确性、one-op 抽象税和终局红队，才能签发
Compute/Graph foundation candidate。该 gate 以前不横向扩展算子目录、Vulkan 或完整模型；gate 以后新增
provider/operation 若能进入既有 seam，也不得重开 Device/Context/Queue、Graph/Executable/Invocation 或
三轴资源模型。

这些探针通过并形成候选后，只有下列新事实允许重开 Compute 架构：真实探针与合同矛盾；第二个独立
消费者需要现有 ownership/concurrency/control seam 无法表达的机制；正式 backend API 使合同不可实现；
或合格同合同证据证明 material 热路径成本根因位于公共抽象。新增 operation、模型或 provider 若能进入
现有 typed operation/region/backend extension，不构成架构重开理由。
