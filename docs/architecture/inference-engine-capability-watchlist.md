# 已收口单元的推理引擎能力观察清单

> 本页记录推理引擎**可能需要、但尚不足以形成产品承诺**的基础能力。当前产品状态只看
> [`module-status.json`](../implementation/module-status.json)；本页不是 task、backlog、`MUST` 清单或
> 重开授权。推理需求与性能口径仍以
> [推理引擎北极星](inference-engine-north-star.md)和
> [能力完整性决策](capability-completeness.md)为准。

## 为什么需要观察清单

`CLOSED` 表示模块已覆盖本轮所有**已准入**的常用旅程，不表示它穷尽未来推理引擎可能提出的能力。
如果把不确定设想提前塞进产品，基础类型会被假想模型、peer API 数量或 benchmark 扭曲；如果完全不记录，
真正实现引擎时又容易忘记此前已识别的边界、重复调研，或把能力放进错误模块。

因此本页使用 `WATCH`，其含义固定为：

- 有一个可描述的潜在推理旅程或成熟实现信号；
- 当前没有足够 consumer、生命周期、性能或 owner 证据把它提升为 `ENGINE_BLOCKER` / `COMPETITIVE` /
  通用基础 `MUST`；
- 不允许据此新增公开 API、layout、常态 metadata、依赖、benchmark 或组件边；
- 它可能最终被准入、由标准库/第三方/相邻模块承载，留在引擎私有 seam，或被证明不需要。

本页只枚举已经取得 `CLOSED` receipt、且存在尚未准入推理观察项的产品组件，以及不进入组件
registry、但拥有独立 receipt 的 build/package 开发单元；没有单独小节不表示模块状态。仍在施工的
`simd`、`sched`、`io`、`compute`、`graph` 的需求应进入各自当前/未来 closure card，不能借本页把它们
伪装成已关闭模块。
观察项被合法提升后，本页只保留尚未准入的剩余边界；当前施工范围和状态改由对应 closure task/
registry 管理，不能继续把已经提升的部分写成“未承诺”。

## 从 WATCH 到施工

一项观察能力只有依次满足下列条件才可施工：

1. **具名旅程：**绑定版本化模型/workload、对象、操作序列、规模和失败路径；“以后推理可能用”不够。
2. **语义擦除：**去掉 token/model/backend 名称后仍是稳定的通用机制；否则留在 compute 或引擎。
3. **owner 裁决：**分别确定 semantic、artifact/mechanism、decision authority 和 delivery owner，避免两个
   模块控制同一件事。
4. **路线比较：**先证明标准库、成熟依赖、现有 HYLE 原语组合或私有 seam 不足；peer 只有 API 不算证据。
5. **成本合同：**登记复杂度、常态 allocation/copy/lock/syscall/metadata、正确性和 source/static/shared
   影响；性能项必须有正常 consumer 与同合同 peer。
6. **合法进入：**若 semantic owner 是 `CLOSED` 模块，形成具名 reopen trigger 并走
   `REOPENED -> SCOPED -> ...`；若 owner 是未收口模块或引擎，不得重开相邻模块。

## Core

Core 当前没有已知推理阻断项；下列观察多数只有在多个底层消费者重复出现时才可能属于 Core。

| ID | 可能的推理旅程 | 暂定 owner / route | 当前不承诺的原因与准入证据 |
|---|---|---|---|
| `CORE-WATCH-01` | 新编译器/ISA hint、target attribute 或 optimizer contract 被多个 hot kernel 共同需要 | 先由 `simd`/`compute` 私有 seam 验证；只有模型无关且跨两个消费者时才考虑 Core | 单个 kernel 的 codegen 需求不能扩成全库宏；需两个消费者、三编译器语义和真实 codegen/footprint 证据 |
| `CORE-WATCH-02` | fatal invariant 需要进程级 crash ID、stack/backtrace、signal/minidump 或持久化策略 | crash/obs owner；Core 已提供有界 typed fields 与 failure-only observer seam | Core observer 仍只在 failing thread 同步借用、返回后终止，且不承诺跨 DSO 或 async-signal-safe；更高层能力需具名生产崩溃收集旅程并证明不污染成功热路 |
| `CORE-WATCH-03` | tensor/queue layout 想按 runtime cache/core domain 动态选择 padding | `platform` 提供事实，拥有该对象的 `container`/`sched`/引擎选择策略 | runtime topology 不能进入编译期 ABI 常量；需真实 false-sharing trace，且不能通过现有 interference constants + owner policy 解决 |

## Container Intrusive

原 `INTR-WATCH-01` counted membership 与 `INTR-WATCH-05` one-pointer forward chain 已由
`container-intrusive-v4-capability-extension` 合法提升并冻结为独立 opt-in 类型，不再是“尚未承诺”项；
它们没有给默认双向 list 增加 metadata。剩余观察如下：

| ID | 可能的推理旅程 | 暂定 owner / route | 当前不承诺的原因与准入证据 |
|---|---|---|---|
| `INTR-WATCH-02` | safe-link 节点需要在保持业务对象身份时 O(1) replace/move | list-level `replace`，不能是绕过 owner 状态机的 hook swap | 当前节点要求 stable address；需真实 movable-safe 生命周期和异常/owner 转移合同 |
| `INTR-WATCH-03` | 一个对象同时进入多种索引、使用 base/offset/function hook 或复杂 intrusive algorithm | 优先直接使用 Boost.Intrusive；稳定重复形状才考虑独立 HYLE 层 | peer 的完整 policy matrix 不是基础需求；需证明依赖路线不能满足交付/性能，且 HYLE 能保持窄类型 |
| `INTR-WATCH-04` | 并发 free-list/ready-list 需要 reclamation、ABA 或多 waiter 进展保证 | `container`/`sched` owning concurrent structure | 当前组件有意不保存 owner、lock、epoch 或 reclamation metadata；不得把并发策略下沉到普通 link primitive |

## Util Core

| ID | 可能的推理旅程 | 暂定 owner / route | 当前不承诺的原因与准入证据 |
|---|---|---|---|
| `UTILC-WATCH-01` | sampling、随机初始化或并行算法需要 bulk RNG、丰富 seed material、可复现子流 | 引擎先消费现有显式 `Rng`/jump；批量向量化可能归 `simd` | 当前 scalar draw 已满足已知任务；需真实批量规模、确定性合同和对 token sampling 端到端 materiality |
| `UTILC-WATCH-02` | backend/session 对象需要 weak observer、aliasing owner 或打破引用环 | 首选 `std::shared_ptr/weak_ptr` 或新的领域 handle；不扩一指针 `Ref` | 这些语义需要 control block/lifetime identity，会破坏当前 Ref 形状；需反复出现的通用 ownership 任务 |
| `UTILC-WATCH-03` | 高争用或 fiber-aware 临界区需要公平、parking 或自适应 mutex | `sched`/`container`/`platform` 对应同步 owner | `SpinLock` 只服务极短低争用区；需 contention trace 与进展/取消合同，不能按微基准给它加状态 |

原 `UTILC-WATCH-04` 已由 `util-core-v3-foundation-utility-superset` 和
`util-core-v4-resource-move-ctad` 提升：条件 scope cleanup、通用 resource pair、move-only deleter 与自然
CTAD 已进入当前 `ScopeFail`、`ScopeSuccess` 和 `UniqueResource`。剩余 typed Platform/Mem/IO owner 仍由
对应领域模块负责，不能再把已经完成的通用 RAII 写成未承诺能力。

## Error

| ID | 可能的推理旅程 | 暂定 owner / route | 当前不承诺的原因与准入证据 |
|---|---|---|---|
| `ERROR-WATCH-01` | CUDA/HIP/Metal/Vulkan、模型格式或远端 backend 需要稳定 typed detail/domain | backend owner 先映射到现有 code/native detail；稳定第二字段再由 Error 审核 | 当前一个 native detail 足够；需至少一个真实 backend 无法保真恢复，并定义跨 DLL/序列化边界 |
| `ERROR-WATCH-02` | 模型 shard、多 device 初始化或 shutdown 同时产生多个错误 | 调用方先用 caller-owned aggregate；重复通用形状才考虑公共 helper | `MultiError` 会引入顺序、截断、OOM 和展示策略；需正常恢复流程证明“只保留首错/附 context”不足 |
| `ERROR-WATCH-04` | 生产崩溃需要 source location、stack/backtrace、trace correlation | `obs`/crash reporter 拥有采集与符号化；Error 只携带必要关联值 | 每层自动 location/payload 会增加传播噪声和常态成本；需实际诊断收益、采样/OOM/隐私和失败路径合同 |
| `ERROR-WATCH-05` | 第三方 C++ callback 抛异常，需要转为值错误 | 拥有 callback/backend 的 adapter 在边界转换 | Error 不承诺跨模块 exception ABI/RTTI；只有两个重复 adapter 证明需要通用 helper 时才复审 |

原 `ERROR-WATCH-03` 已由 `error-v5-capability-completion` 提升为 `Result<T&>`：它提供非空 borrowed
success、deep-const、临时对象拒绝和完整 monadic surface，同时明确不延长 owner lifetime。

## Math

| ID | 可能的推理旅程 | 暂定 owner / route | 当前不承诺的原因与准入证据 |
|---|---|---|---|
| `MATH-WATCH-01` | tensor extent、stride、quant block bytes 反复需要 checked product/sum/rounding chain | 若模型无关则可能扩 scalar math；模型/layout 语义仍归 `compute`/引擎 | 当前只准入同类型、非负整数 ceil-div/round-up；需重复易错 boilerplate、完整 signed/overflow 合同和普通 consumer |
| `MATH-WATCH-02` | shape API 混用 `size_t`、signed dimension 与窄 backend integer | 调用方先显式转换；稳定 checked conversion 可能归 math/core helper | 自动 common-type 会隐藏负数与截断 policy；需具名边界错误和统一失败语义 |
| `MATH-WATCH-03` | low-precision、quant/dequant、activation、向量近似或矩阵运算 | `simd`/`compute`/成熟领域库 | scalar Math 不复制 lane/buffer/operator surface；只有独立的模型无关 scalar 任务才能重开 Math |

## Time

| ID | 可能的推理旅程 | 暂定 owner / route | 当前不承诺的原因与准入证据 |
|---|---|---|---|
| `TIME-WATCH-02` | request deadline、batch window 或 maintenance 需要 periodic/cross-thread timer service | `sched`/`io` 拥有 callback、routing、cancel 和 missed-tick policy | Time 只提供 clock 与 caller-owned timer containers；不得造第二个 event loop |
| `TIME-WATCH-03` | GPU/accelerator kernel timing 与 CPU timeline 对齐 | `compute` backend event/timestamp owner，必要时 obs 做 correlation | device clock domain、同步和频率不等于 host monotonic clock；需真实 backend 与可校准误差合同 |
| `TIME-WATCH-04` | serving 需要 histogram、SLO、TTFT/ITL window 或采样 clock policy | `obs`/引擎策略 | 指标命名和聚合策略不属于 Time；只有底层 clock primitive 缺失才重开本模块 |
| `TIME-WATCH-05` | 先以 monotonic clock 记录跨线程 request stage，导出 trace 时再关联 Unix epoch | Time 可提供不可混淆的 paired clock snapshot；重校准频率、跨进程传播和 trace policy 归 Obs/引擎 | SGLang `req_time_stats.py` 是当前唯一具名消费者；需第二消费者或成熟 peer、采样误差/NTP 跳变合同，并证明直接 realtime read 或 owner 私有 calibration 不足 |

原 `TIME-WATCH-01` 的底层部分已由 `time-v3-host-clock-factor` 与 `time-v5-interoperability` 提升：
`ManualClock`、`MonotonicClockRef`、显式 clock-source deadline/remaining 和 caller-owned timer queue 已完成。
把同一 clock 注入完整 scheduler runtime 仍归 Sched；它不再是 Time 缺失能力。

## Platform

| ID | 可能的推理旅程 | 暂定 owner / route | 当前不承诺的原因与准入证据 |
|---|---|---|---|
| `PLATFORM-WATCH-01` | worker 每 physical core 一个线程、cache-domain shard、P/E/SMT 精确 per-logical-ID placement | Platform 提供 class/cache/topology 事实，`sched`/引擎拥有放置策略 | class-level cache 容量/共享宽度已完成；macOS 仍无公开硬 pin 和可信 class-to-logical-ID 映射。需 engine placement trace、跨 OS 可映射 ID 和相对默认策略的 material 收益 |
| `PLATFORM-WATCH-02` | AMX、SME/SME2 kernel 需要探测、per-thread permission/enable 和 dispatch | Platform + `simd` backend 同轮；具体 kernel dispatch 只有一个 authority | 只有 capability bool 不能完成旅程；需已准入真 kernel、真 ISA 机器和 thread enable/fallback/codegen 门 |
| `PLATFORM-WATCH-03` | 大模型在多 socket/多 die 主机上做权重、KV 或 worker locality | Platform mechanism + mem/sched/engine policy | 当前只签单节点 API/rollback/复杂度；需多节点真机、local/remote bandwidth、migration 和 pressure workload |
| `PLATFORM-WATCH-04` | specialized model loader 可能还需要 THP collapse 或更强的 load-to-ready 控制 | Platform 保留 VM mechanism；file owner 与 hole safety 归 `mem_vm`；direct/async route 归 IO | progressive prefix residency、同步 `populate` 与 POSIX/macOS partial release 已进入 Mem-VM；Windows whole-view 无安全同合同并诚实返回 `Unimplemented`。显式 collapse 仍需 load-to-ready/RSS/page-fault material 证据 |
| `PLATFORM-WATCH-05` | direct/scatter-gather/并发或 async positional 模型读取 | `io::File` 或专用 loader | caller-synchronized blocking explicit-offset 已由 Platform `RandomAccessFile` 完成；把 OVERLAPPED/proactor 状态塞进该同步一字 owner 会破坏形状。剩余路线需真实 loader、alignment/fallback、并发和 cancellation 合同 |
| `PLATFORM-WATCH-06` | JIT kernel、named shared memory、process sandbox 或 plugin 隔离装载 | 分别归 compute/JIT、io protocol、tooling/process；Platform 仅提供必要 OS mechanism | executable permission、跨进程 ownership 和 loader search policy 都有独立安全边界；需具名产品旅程再单独设计 |
| `PLATFORM-WATCH-07` | 动态 backend 需要 lazy/global/deep-bind/nodelete 或隔离 search path | Platform DynLib + backend owner | 当前固定 eager+local；需实际依赖解析/冲突复现，并先定义相对路径信任边界 |

`PLATFORM-WATCH-08` 已由 `platform-v6-file-control-and-process-policy` 提升：Platform 现在提供不含
realtime 的显式五档 startup policy，并诚实保留 Windows process class/QoS、Linux 调用线程继承与
macOS process nice 的作用域差异。它不承诺保存任意 native policy 后精确恢复；有该需求的宿主仍在
自己的 OS adapter 中拥有 snapshot/restore。

`PLATFORM-WATCH-09` 已由 `platform-v7-cache-class-and-random-access` 提升：Platform 现在保留异构
cache class 的容量/共享宽度，提供 opened-object `duplicate/info` 与 caller-synchronized blocking
explicit-offset owner。class-to-logical-ID 硬 placement、direct/vectored/concurrent/async I/O 仍分别
留在上面的 WATCH-01/WATCH-05，不把已完成机制和仍缺策略混成一项。

## Mem

| ID | 可能的推理旅程 | 暂定 owner / route | 当前不承诺的原因与准入证据 |
|---|---|---|---|
| `MEM-WATCH-01` | worker/request phase 希望每线程 arena、thread-exit 合并和全局 reclaim | `sched`/runtime 拥有 worker registry，Mem 提供 Arena/backing | 隐藏 TLS 会引入 zombie、exit 和全局 teardown 状态；需 runtime 生命周期、内存压力和跨线程回收 trace |
| `MEM-WATCH-02` | 多 producer 共享 caller-backed variable heap 或 pool | caller lock、标准 synchronized PMR，或具名 HYLE wrapper | 当前 FixedHeap/PMR 组合已有路线；需第二消费者和 contention/ownership 证据，不能让单 owner 路径付锁成本 |
| `MEM-WATCH-03` | 多 socket/统一内存机器需要 host placement、迁移或跨 die 带宽策略 | Platform 提供机制，Mem 提供 owner，engine/sched 决策 | 当前只签单节点 NUMA；需真硬件、page residency/pressure 和 end-to-end tokens/s、tail、joules/token 证据 |
| `MEM-WATCH-04` | GPU/NPU 需要 pinned/registered/coherent/unified buffer 与 staging lifetime | `compute` backend 拥有 device handle、mapping、completion；Mem 只在通用 host backing 成立时参与 | host allocator 不能猜 device coherence；需真实 backend、异步 completion、fallback 和 copy/overlap workload |
| `MEM-WATCH-05` | KV page/cell、prefix sharing、fork/copy、evict/defrag 或 allocation planner | 推理引擎/compute/graph 策略消费 Mem 原语 | KV/token/sequence 是模型状态，不进入 Host Mem API；只有语义擦除后的通用 pool/backing 缺口才重开 Mem |
| `MEM-WATCH-06` | 内存压力触发模型/KV eviction、tiering 或 OOM recovery | 引擎策略读取 `system_memory_snapshot`/quota/stats；Platform 提供 pressure fact | Mem 不决定丢哪个模型/sequence；需真实压力控制环证明缺少通用机制，而非策略函数 |

## Container

| ID | 可能的推理旅程 | 暂定 owner / route | 当前不承诺的原因与准入证据 |
|---|---|---|---|
| `CONTAINER-WATCH-01` | 极大稀疏 token/sequence ID 空间需要 compressed bitmap、rank/select 或 fast set-bit merge | 先组合 `DynamicBitSet`/`SparseSet` 或成熟 bitmap；稳定通用 owner 才重开 Container | 当前没有规模、density、mutation/iteration trace，也不能让普通 mask 支付压缩索引 metadata；需端到端 active-set workload 与同合同 peer |
| `CONTAINER-WATCH-02` | BPE、graph frontier 或 scheduling 需要 indexed/decrease-key heap、stable handle 或 move-pop | tokenizer/graph/sched 先用私有 policy + `std` heap；语义擦除后再评估通用 indexed heap | 当前 llama.cpp 的 priority journey 可由标准 heap 完整承载；需两个消费者、handle invalidation/OOM 合同和 material profile |
| `CONTAINER-WATCH-03` | tokenizer、grammar 或 protocol parser 需要 trie、Aho-Corasick、double-array trie 或 compact automaton | text/tokenizer/grammar owner；Container 只在 alphabet/policy 被擦除后拥有通用 storage primitive | alphabet、normalization、failure output、build/persist policy 都是算法语义；单个 llama.cpp trie 不能自动形成基础容器 MUST |
| `CONTAINER-WATCH-04` | 权重/KV/backend buffer 复用需要 LRU、segmented cache、quota-aware eviction 或 concurrent cache | runtime/compute 拥有 budget、device lifetime 与 eviction；Container 可提供被证明缺失的索引/storage mechanism | cache 命中、回收、异步 completion 和内存压力是产品策略；需具名引擎生命周期并证明 map/list/queue 组合不完整或有 material 成本 |
| `CONTAINER-WATCH-05` | ggml 风格 graph traversal 需要 stable compact slot、SoA side arrays 或 generation handle | graph owner；当前 `FlatHashMap<pointer, State>`/owning vector route | slot identity 会泄漏 graph pass 与 C side-array 布局；需第二个模型无关消费者和更好复杂度/内存证据，不能只对齐 ggml C API |

## Util Runtime

| ID | 可能的推理旅程 | 暂定 owner / route | 当前不承诺的原因与准入证据 |
|---|---|---|---|
| `UTIL-WATCH-01` | backend/plugin 动态发现后支持 unload/reload | 独立 module manager；Platform 拥有 `DynLib`，Util Registry 只拥有名字表 | 必须排空对象、函数指针和并发 reader；需具名 plugin lifecycle，不能只加 `unregister()` |
| `UTIL-WATCH-02` | backend registry 需要依赖求解、版本选择或初始化/关闭顺序 | backend/module owner，Registry 可作为底层表 | 这些是模块图和业务策略；需两个 backend 复现相同通用生命周期后再提炼 |
| `UTIL-WATCH-03` | factory 必须携带捕获状态或任意 callable | 先通过显式 `Args...`/owner 传递；必要时由 owning module type erase | 当前 function pointer 避免隐藏 allocation/lifetime；需真实 consumer 证明显式状态路线不自然且成本可接受 |
| `UTIL-WATCH-04` | crash path 需要 async-signal-safe symbolization/backtrace | crash/obs/tooling owner，可能使用离线 symbolizer | 当前 Itanium demangle 可分配，Windows DbgHelp 需全局锁；caller buffer 不等于 signal-safe |
| `UTIL-WATCH-05` | 只需要 demangle 的小 consumer 认为完整 Util closure 过重 | build/package + Util，可独立拆 component | 当前没有 material demangle-only consumer；需 artifact/dependency/启动成本证明，而不是 symbol 数偏好 |

## Build / Package

Build/package 是贯穿开发单元，不是可链接组件；它的观察项也不能制造产品组件状态。

| ID | 可能的推理旅程 | 暂定 owner / route | 当前不承诺的原因与准入证据 |
|---|---|---|---|
| `BUILD-WATCH-01` | 同一引擎发布 baseline/native/exact ISA 与 CUDA/Vulkan/其他 backend 组合 | build/package 交付；`simd`/`compute` 定义 artifact 和唯一 dispatch authority | 当前 compute/simd 尚未冻结最终形状；需真实 selected roots、安装后运行和无设备 fallback 矩阵 |
| `BUILD-WATCH-02` | shader、kernel、repack table 或生成代码需要 build-time 生成、缓存和安装 | compute/simd semantic owner，build/package 承载可复现生成 | 生成物身份、tool version、交叉编译和 source bundle 尚无稳定 consumer；不得先造全局开关 |
| `BUILD-WATCH-03` | Bazel/Meson/pkg-config/CPack 或系统发行包消费 HYLE | installed CMake package 或薄 bridge 优先 | 现有 relocatable install-prefix 可归档；需具名生态消费者和 source/static/shared 全闭包 smoke 才新增入口 |
| `BUILD-WATCH-04` | Windows/Linux/macOS 之外的交叉编译、移动端或专用 appliance | 标准 CMake toolchain + 对应模块 backend | “平台可能存在”不够；需真 toolchain/runner、支持矩阵和依赖锁/许可证路线 |
| `BUILD-WATCH-05` | 客户需要签名 binary/source bundle、SBOM、provenance attestation 或可复现发布 | release/deployment owner + build/package metadata | 当前 lock/offline/license/source bundle 已 fail closed，但没有具名签名/审计格式和验收方 |
| `BUILD-WATCH-06` | 模型权重、tokenizer、backend plugin 随应用一起发布 | 推理引擎/deployment owner；HYLE 只交付库与必要 kernel artifact | 模型资产版本、授权、下载、缓存和服务配置不是基础库 package；只有 HYLE artifact 缺口才修改本单元 |
| `BUILD-WATCH-07` | Java/C# 等已命名 foreign-runtime 消费者需要动态入口 | 按 Native FFI policy 设计窄 runtime type erasure | 当前没有 consumer，C++ template/header 能力也不能机械导出；必须先有 create/call/error/cancel/destroy/unload 完整旅程 |

## 维护规则

- 每当一个模块新取得 `CLOSED` receipt，或已有模块因推理 consumer 合法重开时，应核对本页相关 ID；
  只更新观察事实，不在 architecture 中复制当前 gate、commit、LOC 或性能点值。
- 观察项被准入后，从本页链接到其需求/closure card，并标明最终 owner；产品状态仍只由 registry 表达。
- 观察项被现有组合、标准库或成熟依赖完整承载时，改写为明确 route；被证明是模型策略时移到引擎设计；
  被证明无需求时可删除，不保留“也许有用”的永久 backlog。
- 推理引擎尚未提供版本化 workload 前，本页任何一行都不得作为修改 `CLOSED` 模块的唯一依据。
