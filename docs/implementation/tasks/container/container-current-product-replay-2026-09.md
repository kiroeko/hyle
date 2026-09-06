# Container 当前全产品能力回放 — 2026-09

<!-- READ-ONLY CAPABILITY REPLAY: PRODUCT_TRIGGER -->

> 本页是 product-non-mutating replay，不是第二套模块状态。机械状态仍只看
> [`module-status.json`](../../module-status.json)：Container 当前是旧方法下的 `CLOSED`，且 v4 只验证
> CPU feature selector 迁移。形成满足协议的具体 `PRODUCT_TRIGGER` 前，本页不授权修改 Container 的
> 公开 API、算法、布局、依赖或正常 workload。

## 回放合同

- component / replay：`container` / `container-current-product-replay-2026-09`；
- reviewed rounds：`container-v2-rebuild`、`container-v3-node-table-reclamation`、
  `container-v4-cpu-feature-migration`；历史 task 只提供事实和候选边界，不提供当前授权；
- current registry identity：v4 candidate
  `d015b0de1cf6d5899da8336fef94104974bcd90e` / tree
  `1456eea7575fa72ce48626a9db13a80f0aa31c09`；
- replay baseline：HYLE `a9b5b6523867601dd2c708a2bedc8fc4d81469f2` / tree
  `fadf478497ce9a873a31ea984ed4ba165765530f`；
- owner request：按新的能力超集、自然接口、真实推理消费、性能等价、红队、文档和构建标准重新审查
  Container；不能把 v4 的窄 `CLOSED` 冒充完整产品冻结；
- 范围：`include/hyle/container/**` 中非 `container_intrusive` owner 的公开面、`src/container/**`、
  Container tests/bench、直接依赖和直接消费者、source/static/shared selected-component 交付，以及
  architecture/user/implementation/performance 文档；
- 只读依赖：已经 `FROZEN` 的 `core`、`mem`、`mem_reclaim`、`platform` 和
  `container_intrusive`。只有本回放发现击中其自身 reopen trigger 的直接事实时，才另行登记，不能
  顺手修改；
- 非目标：Compute/Graph 产品设计，tokenizer、KV-cache、scheduler、request-routing 或 backend policy，
  Native FFI/C API，稳定跨工具链 C++ ABI，复制标准库或 peer 的全部 API，修改公共 closure/performance
  仪器；
- tier：先做 P0 capability/complexity/resource/API/build 审计。只有 API/usability 和有限 FIX 集冻结后，
  才为已证明的重复正常路径建立 P1；旧 v2 五项只作历史输入，不自动继承；
- 调查预算：只读 replay 最多 4 engineering hours，独立红队最多 30 分钟，计时预算为 0。若得到
  `PRODUCT_TRIGGER`，新 closure round 单独冻结实现和性能预算；
- 停止 verdict：`CONFIRMED`、`METHOD_GAP` 或 `PRODUCT_TRIGGER`，不得以“继续研究”结束。

## 锁定真实消费者

| Consumer | Clean identity | 本轮用途 |
|---|---|---|
| llama.cpp | `3ad1ba7336986d98592d3e28cafd1a406715351f`；随仓 ggml tree `7674bdaa9279cfc46abde8ce489652f508c0db09` | 本地模型加载、metadata、tokenizer/sampling、graph/backend、scheduler 的 C++ 数据结构形状 |
| standalone ggml | `e91ded11bdcd78c42f9c8d3978ff6686eb4c1226` | 基础 tensor/graph/backend/allocator/scheduler 的低层机制和资源合同 |
| ds4 | `9ab705347c1775e7599ede7eb81a6255ec7dccb5` | 面向量化推理的紧凑 C/C++ runtime 数据结构与热路径 |
| vLLM | `52358e6e192aeae73dd3764046c98fc4b156ea83` | production serving、prefix/KV、batch/request、distributed metadata 与 native-extension 边界 |
| SGLang | `28457f0dcab4ccf748d60f2387a1a4a7fdb5a110` | radix/prefix cache、scheduler、request/token 与 native kernel 数据结构形状 |

这些仓库是 demand 与 normal-journey 北极星，不自动成为性能 peer。Python service policy、模型算法和
device-runtime owner 不能只因源码中出现 `dict/list/deque/cache` 就下沉到 Container。每项候选能力必须
记录精确文件/符号、结果语义、ownership/lifetime、failure、concurrency、复杂度与唯一 owner。

## 必须回答的审计面

1. **功能超集：**按 sequence、associative、concurrent registry、handoff/work-sharing、ring、dense/
   sparse membership、hash/text/identity 及新发现 family，逐项对照四组推理实现、标准合同和每族最多
   两个成熟 peer。只有通用、可复用且 owner 属 Container 的能力才准入；不是按类名或 API 数求并集。
2. **自然 C++ 形状：**反向追踪每个公开入口，查 benchmark-only 类型/参数、`fast/light` 命名、半项
   生命周期、非标准失败形状、不自然默认值、重复标准/第三方能力和用户必须拼 internal seam 的路径。
3. **质因数和复用：**核对 `container -> PUBLIC core,mem,mem_reclaim; PRIVATE platform` 与真实 target；
   `container_intrusive` 通过产品 façade 聚合而不制造环。每项已有能力写 `REUSE/BORROW/LOCAL/
   NOT_APPLICABLE`，不得为了“尽量互相引用”建立没有语义收益的依赖。
4. **实现与红队：**检查复杂度、allocation/copy/hash scan、常态 metadata、iterator/reference/lifetime、
   OOM/overflow、exception/no-exception、并发 progress/quiescence、ISA dispatch 和第三方所有权；独立红队
   只读给出可复现 file:line finding。
5. **性能：**先判定真实热路径和合同。任何比较先标为 `NATURAL_ROUTE`、`SAME_CONTRACT` 或
   `MECHANISM_DIAGNOSTIC`，并对齐 ownership、failure、capacity、thread-safety、setup/teardown 和
   allocator。不得用苹果对橘子、亚纳秒点值、地址/顺序翻转或未资格化结果改 API/layout/metadata。
6. **构建与交付：**核对中央 component graph 与 target direct links、Highway build-only route、xxHash/
   wyhash/simdutf private mechanism、依赖锁/离线/license、公开头 owner、source/static/shared selected
   journeys，以及没有伪 C ABI/Native FFI 残留。
7. **文档：**architecture 只保存长期 owner/边界/决策，user guide 按场景选择并写 ownership、failure、
   invalidation、concurrency 和坑，implementation 保存本轮事实，performance 只发布候选绑定的有限结论。

## 初始产品面与待证问题

- sequence：`SmallVector`、`StaticVector/InplaceVector`、`PodVector`；
- ring/queue：`RingDeque`、`CircularBuffer`、`ByteRingBuffer`、`SpscQueue`、`BoundedQueue`、
  `ShardedQueue`、`WorkStealingDeque`、`WsFifoQueue`；
- associative：flat hash map/set、flat sorted map/set、B-tree map/set、concurrent flat/node map；
- append/membership：`ConcurrentVector`、`DynamicBitSet`、`SparseSet`；
- text/identity/hash：`StringView`、string/path algorithms、`StringInterner/StringId` 与 semantic hash；
- separately frozen factor：intrusive list/forward-list/hlist 只检查聚合与依赖，不重复审其产品实现。

待证问题包括：四组推理 consumer 是否暴露旧 v2 未见的通用 family；现有 broad family 是否存在半项 API
或不正确复杂度；是否把 domain policy 错放入 Container；现有五项历史 benchmark 是否仍代表正常路径；
以及 direct Highway/vendor/build route 是否与已经冻结的 Platform/SIMD/Build 合同一致。

## 停止与下一状态

- 没有未处置 `MUST`，公开形状自然，复杂度/资源/依赖/交付没有当前产品事实击穿：`CONFIRMED`，
  Container 保持旧 `CLOSED`，另决定是否具备签发新 `FROZEN` 标记所需证据；
- 未发现产品缺陷，但当前证据或公共工具无法裁决适用 gate：`METHOD_GAP`，产品状态不变，登记有限
  evidence/tool task；
- 存在可复现缺失 journey、错误 owner/生命周期/复杂度/交付或同合同正常路径 material regression：
  `PRODUCT_TRIGGER`，记录 trigger ID、证据和最小影响面，然后才依法进入
  `CLOSED -> REOPENED -> SCOPED -> AUDITED` 并冻结有限 FIX 集。

当前只允许 public-surface trace、consumer/peer differential、依赖/交付静态审计、既有测试的只读复现和
红队；不得先修改 Container 产品代码，也不得因 benchmark 某一行输赢改变产品形状。

## 只读回放结果

### 当前能力与四组推理消费

现有 sequence、flat/sorted/B-tree、ring、bounded/SPSC/work-stealing queue、dense/sparse membership、
concurrent map/vector、runtime hash、borrowed text、path 和 interner 仍有真实通用合同；没有发现只由旧
benchmark 使用、因而应直接删除的公开类型。四组推理实现暴露的差异按语义 owner 裁决如下：

| 候选 | 当前证据 | 回放裁决 |
|---|---|---|
| bounded hash + recency + eviction storage | ggml CANN graph cache、vLLM `utils/cache.py` 与 LoRA manager、SGLang expert/KV cache 都重复组合 lookup、touch、capacity 和 eviction；Folly 有 `EvictingCacheMap` | `MUST` candidate；通用 entry/weight budget、peek/touch、erase/clear 与 eviction callback 属 Container，device/KV/in-flight/pinning policy 留调用者 |
| unbounded typed handoff storage | llama.cpp server 以 mutex + unbounded deque 承载 burst；SGLang tokenizer/runtime 使用 unbounded async queues；Folly/oneTBB 提供 segmented unbounded queue | `MUST` candidate；Container 只拥有 try-only storage、OOM 与近似 size，wait/close/cancel/backpressure 属 Sched/IO |
| concurrent set | runtime registry/membership 是通用并发形状，oneTBB 提供独立 set；当前只能让用户伪造 mapped value | `GENERAL -> MUST` candidate；必须证明 flat/node 两种 set 不额外保存 mapped payload，并保持 map 同等级 lifetime/reclamation |
| Unicode diagnostics/transcoding | ggml Windows path 和 backend adapter、llama tokenizer/text、simdutf 已锁定机制都需要 UTF-8/16/32 边界；当前只有 bool validate 与 count | `MUST` candidate；typed error offset、caller-owned output 与 UTF-8/16/32 conversion 属 Text factor，不把 filesystem/tokenizer policy塞入 codec |
| mutable/indexed heap | vLLM priority request removal 当前是线性 search + heapify；llama BPE 和多个 backend 使用普通 heap | `GENERAL`；先在审计中裁决 stable handle/update/erase 是否已有第二个通用 owner，普通 push/pop 继续路由标准 heap |
| concurrent priority queue | oneTBB 提供，但当前四组消费者的 priority policy 多由单 scheduler owner 串行管理 | `DEFER`；没有证据要求所有调用者支付 concurrent heap metadata |
| trie/radix/Aho/token automaton | llama `common/trie` 与 SGLang radix/ngram 真实存在，但 alphabet、match、frequency、device blocks、eviction 和 serialization 合同明显不同 | `ROUTED`；Container 提供 storage，matcher/tokenizer/KV-cache algorithm 由后续 owning factor 设计，不制造一个虚假的通用 trie 超集 |
| dense graph ID、stable checksum、ordinary vector/string/node containers | ggml/llama 有真实使用 | `ROUTED`；分别由 Graph 组合、IO/format 具名算法或 C++ 标准库承载，不复制 API 数量 |

上述 `MUST candidate` 仍须在新轮 `AUDITED` 状态冻结 public 调用草图、failure/lifetime/concurrency、复杂度、
正常 metadata 和有限验证；本回放只证明旧结论不能维持，不提前授权具体实现。

### 可复现产品触发

1. `CONT5-R1 PUBLIC-ISA-FLOOR`（high）：`probe.h` 宣称 x86-64 public inline probe 使用 SSE2 baseline，
   但 MSVC x64 无条件启用 `_mm_sign_epi8`，而 `src/container/CMakeLists.txt` 又把 SSSE3 编译选项作为
   `PUBLIC` usage requirement 传播到任意 consumer TU。合法的 SSE2-only x86-64 可执行文件存在非法指令
   风险；较新 CPU 上也不应由 Container 扩大消费者整个 TU 的 ISA floor。该事实直接命中 illegal-
   instruction、public codegen 和 delivery reopen trigger。
2. `CONT5-R2 PRIVATE-VENDOR-VISIBILITY`（high）：vendored `simdutf.cpp` 直接编入 shared
   `hyle_container`，Unix 路线没有 hidden visibility；simdutf 的非 Windows export macro 为空，因此其
   external namespace symbols 会进入 ELF/Mach-O 默认动态导出面，形成 interposition 与同进程双版本冲突。
   该事实命中 source/static/shared delivery trigger。
3. `CONT5-R3 CAPABILITY-SUPERSET`（high）：generic eviction storage 已由三个独立推理实现和成熟 peer
   证明为重复通用机制；unbounded handoff、concurrent membership 与 Unicode codec 也有跨 consumer/peer
   证据。旧 v4 明确排除了新能力，不能代表当前产品完整性。
4. `CONT5-R4 FACTOR-CLOSURE`（medium）：长期架构已把 Container 分成 sequential、concurrent、queue、
   Text 四个内部质因数，真实 CMake 却仍只有一个 13k-line compiled target；Sched 只用 queue 也会取得
   hash/text/simdutf/concurrent reclamation 闭包。owner 本轮明确要求模块独立/并行开发与最小依赖闭包，
   因而该偏差成为本轮 component/delivery fix，而不再只是历史候选。

### 红队与方法缺口

独立只读红队没有发现新的确定性 ownership/concurrency correctness bug，也没有发现公开 C/Native FFI。
它另外确认三项必须在新轮处理的方法缺口：

- 旧 v2 五行性能数据不是当前推理 consumer 证据，v4 又只签 selector 的 P0；当前不能声称 Container
  性能 SOTA。新 P1 必须先冻结真实任务和 SAME_CONTRACT/NATURAL_ROUTE 分类；
- `try_emplace` 在标准 associative vocabulary 中表示“存在时不覆盖”，并不自动表示 recoverable OOM，
  但 user guide 把所有 `try_*` 都描述成 OOM 路线。新 API freeze 必须消除这一心智冲突；
- `SmallVector`/`RingDeque` 对 potentially-throwing move 的 class-wide 排除、hash 的 OS/toolchain 固定
  crossover、以及若干直接引用 benchmark/旧稳定 ABI 的实现注释，需要在 API/成本审计中逐项裁决；
  点值或措辞本身不授权改 layout。

## 回放终态

本页 terminal verdict 为 **`PRODUCT_TRIGGER`**。旧 v4 `CLOSED` 只对私有 CPU selector 迁移成立；
`CONT5-R1..R4` 已具体击穿 public ISA、shared delivery、能力完整性和组件闭包。下一轮只允许先登记
`REOPENED`，随后冻结 closure card 进入 `SCOPED`，完成全部候选的 owner/API/成本审计后进入
`AUDITED` 并固定有限 `FIX` 集；在此之前仍不得修改产品代码或运行新的计时矩阵。
