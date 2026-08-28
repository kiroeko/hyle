# container-v1 重构与收口记录

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 本页是完整 `container` 组件的当前 closure card。既有
> [`container-module-closure.md`](container-module-closure.md)、architecture、guide 与 benchmark
> 只作审计输入；其中旧 C ABI、旧点值和“最终证书”不能继承为本轮结论。独立的
> `container_intrusive` 已有自己的产品组件和冻结记录，不属于本轮产品面。机械状态只看
> [`module-status.json`](../../module-status.json)。在 capability map、public-surface trace 与有限
> `FIX` 集进入 `AUDITED` 前，不修改 container 产品实现。

## Closure card（`SCOPED`）

- component / owner / round ID：`container` / data-structure owner / `container-v1`
- task kind：`REFACTOR`
- baseline commit / tree：`2866c7df73c6bd36cc4808226d370ff0ea338da2` /
  `5e31a8cabed513c1f5d6c4c7328ef00ed4f67612`
- 进入原因与 demand/defect ID：
  - `CONTAINER-CAP-01`：按当前能力完整性标准，从真实任务而不是已有类名或 benchmark 重新枚举完整
    container 能力族，并反查全部公开入口是否只有半项生命周期或错误 owner；
  - `CONTAINER-NORTHSTAR-01`：本地量化推理需要 token/metadata 索引、小批量连续 storage、并发 registry、
    stage handoff、工作窃取、活跃集合与 byte/text staging；工程计算留出使用 ordered/range index、
    sparse active IDs 与同样的 queue/ring 机制；
  - `CONTAINER-CONSUMER-01`：当前 `util` registry、`sched` queues、`io` text/hash 与 `graph` small sequence
    已直接消费 container，需要核对语义、依赖和失败路线；
  - `CONTAINER-API-01`：旧 C 产品已经删除，但现有 `carray`、`string_ref`、buffer-fill 与若干 pointer/out
    形状仍需证明是自然 C++ primitive 或重塑，而不是保留旧 ABI 外壳；
  - `CONTAINER-PERF-01`：2026-07 的性能总账早于当前 `closure-perf-v1`，混有不同合同、平台点值与
    benchmark-oriented 叙事，不能证明当前 source-primary 产品候选；
  - `CONTAINER-DEP-01`：当前组件依赖 `core/mem/platform`、build-private SIMD header seam 以及 vendored
    xxhash/wyhash/simdutf；升级后需重新核对算法、ISA fallback、license 与三种交付；
  - `CONTAINER-DOC-01`：architecture 仍标 `NOT_REVIEWED`，旧 task/user/performance 文档仍包含已删除
    Native C surface 和过期证书，需要按四类文档职责重新收束。
- 真实 consumer task / normal workload：
  - 推理引擎用 flat hash 保存 token/张量/算子 metadata，用 small/fixed sequence 组装短 shape、batch 与
    dependency list，用 stable-node/concurrent map 保存跨 grow 仍需 identity 的 registry；
  - worker pipeline 使用有界 SPSC/MPMC/sharded/work-stealing 机制传递 typed payload，控制面使用 bitset/
    sparse active set，模型与 I/O loader 使用 byte ring、borrowed text、hash/UTF-8/path primitive；
  - 当前 HYLE `util/sched/io/graph` 的直接调用序列作为仓库内 consumer replay；未关闭下游只验证它们
    实际依赖的 container 合同，不借此宣称下游模块已经收口；
  - 工程计算留出以 read-mostly sorted index、持续 mutation 的 ordered/range index、active-ID set 与
    streaming ring 验证 API 没有被模型、token 或单一 benchmark 过拟合。
- 范围：`include/hyle/container/**` 中除 `intrusive_list.h` 外的完整公开 C++ 面、`src/container/**`、
  container-owned external adapters、全部 public-surface trace、实际依赖和直接消费者；source/static/shared
  交付；architecture/user/performance 文档；有限的 correctness、P1 consumer replay 与独立红队。
- 非目标：已关闭的 `container_intrusive`；模型格式、tensor/KV layout、batching/serving policy；scheduler
  wait/cancel/backpressure service；通用 cache/LRU、阻塞或持久化队列；device memory；复制 std/Boost 全部
  容器或单库独有 feature；Native FFI/C API、稳定 C++ ABI；下游模块整体收口与 Formal 发布证书。
- 直接依赖 / 直接消费者：产品边为 `PUBLIC core mem`、`PRIVATE platform`，实现 header seam 为
  `PRIVATE_BUILD simd_vec_headers`；外部源码 owner 为 xxhash、wyhash、simdutf。中央图的直接消费者为
  `util`、`sched`、`io`、`graph`；当前具体语法消费者还包括 package/sample。
- tier 与理由：`P1`。lookup、append、queue、bitset、ring、hash/text 等可在推理 worker 热路径重复发生；
  reserve/rehash/grow、bulk build、clear、path normalization 和 teardown 等控制/冷路径先按 P0 结构门审查，
  只有能传播到正常 journey 的操作进入计时。
- material margin / resource budget：W1-WIN 同合同 peer 或 immutable baseline 的工程 NI margin 为 5%。每族先
  达到成熟复杂度档；不得为点值增加无消费者承担的常态 metadata、padding、allocation、lock、dynamic
  dispatch 或重复 hash/copy。队列容量、引用稳定性、allocator/OOM 与 synchronization 语义必须等价后才
  比较；吞吐不能抵偿丢失 lifetime、错误 progress 或额外驻留内存。
- 总调查预算 / 性能预算：从 `SCOPED` 起能力/实现/依赖调查最多 6 小时，其中统一性能取证最多 2.5 小时、
  最终独立红队最多 30 分钟。预计产品修复超过一个工作日时，先把冻结 gap 拆为可独立验证的有限批次；
  到预算只能按协议裁决、具名 `BLOCKED` 或显式重新 scope，不能继续发明实验。
- 最多复现、机制诊断、产品修改次数：每个 material 异常最多一次反序/复现、一次 codegen/profile/
  allocation/contention 机制检查、一个同时满足机制与 consumer 两把钥匙的产品修改及一次重跑；不在本轮
  改中央统计规则或扩建公共仪器。
- machine profile：`W1-WIN` 为 P1 engineering 裁决机；`W1-WSL` 与 `M1-MAC` 承担适用 correctness、
  sanitizer、ISA fallback 和交付复核，不跨不同机器池化点值。Mac 大小核只使用登记的 QoS 合同。
- instrument profile：P1 为 `closure-perf-v1`；correctness/delivery 为 `closure-correctness-v1` 与当前
  component/package contracts。
- strong peers（当前本地 clean source，按能力族最多两个）：
  - general/inline sequence：Abseil `81c993af4fb88b3a51829a796df02251fa27102f` 与 Boost
    `df8eeb90846648464e3c8e378e312f0cec772e83`；strict fixed-capacity 另以 CCCL
    `132670eba92ca5bbddb2d479e76045184ed885a5` 代替 Abseil；
  - flat/sorted/ordered associative：Abseil `81c993af4fb88b3a51829a796df02251fa27102f` 与 Boost
    `df8eeb90846648464e3c8e378e312f0cec772e83`；
  - concurrent associative/vector：oneTBB `e349c15263bbb1d6b48f2b514c06201e1b90eea3` 与 Folly
    `423fd7f8d97fd7bf52fc4c986dfd20320c4de8f9`；
  - bounded FIFO/ring/work sharing：Folly `423fd7f8d97fd7bf52fc4c986dfd20320c4de8f9`，并按具体
    MPMC 合同使用 rigtorp `b9808ede08f26fa9df4df4e081d19cace8f6c6ea` 或 moodycamel
    `c227e283a9edd9d15393733223235527af6101e9` 二选一，不把不同 progress/capacity 合同池化；
  - bitset/sparse-set family：Boost `df8eeb90846648464e3c8e378e312f0cec772e83` 与 Folly
    `423fd7f8d97fd7bf52fc4c986dfd20320c4de8f9` 只用于能力/复杂度审计；
  - byte hash 与 UTF-8：xxHash `e573d4d2aaeaba0f3e5a0a9a54144a1f2b4b56e7` 与 simdutf
    `282107c23196ddc6c7b8ef06567f81bbd056fc3c` 各自只比较对应算法合同。
- 无同合同 peer 时的 consumer budget 及禁止声明：StringInterner、path/view、SparseSet、byte staging 等
  对 frozen baseline 做 5% source-primary NI，并守住零额外常态 allocation/hash scan/lock/syscall 与既定
  渐进复杂度；若没有合格可比 peer，只能声明该冻结 workload 无 material regression，不能写成
  “container 全面 SOTA”、跨平台固定纳秒或由另一种语义的 peer 推导胜负。
- correctness gates：normal；empty/zero/N=0；capacity/wrap/overflow；allocator OOM/fault；invalid input、
  hash/equality/comparator contracts；copy/move/destroy/self-alias/iterator/reference invalidation；non-trivial/
  move-only lifetime；producer/consumer/accessor/rehash/quiescence progress；property/differential/random stress；
  ASan/UBSan/TSan；x86 scalar/SSE/AVX 与 arm64 scalar/NEON runtime dispatch；public-header self-containment。
- delivery gates：中央依赖/include-owner 对账；xxhash/wyhash/simdutf pin、license、offline 与 package
  projection；container-only source embedding、relocated static/shared C++23 consumer；不安装 intrusive
  product header 为 full-container ownership 假象；无旧 C schema/bridge/fixture/Native FFI symbol。
- performance gates：冻结五组正常 family，不把不同语义池化：
  1. `sequence-and-inline-batch`：短 append/iterate、spill/grow、strict bounded batch 与非平凡 lifetime；
  2. `associative-index`：integral/string flat hit/miss/build/erase、read-mostly sorted lookup/bulk、ordered range
     mutation/expiry；
  3. `concurrent-handoff-and-registry`：SPSC/MPMC/sharded/work-steal、parallel append、flat/stable-node map 的
     read/mixed/rebuild，并同时报告 progress、tail 与驻留资源；
  4. `active-set-and-stream`：bitset algebra/count/iteration、sparse active-ID lifecycle、object/byte ring staging；
  5. `hash-text-intern`：短/中/长 byte hash、UTF-8 validate/count、borrowed search/case、interner
     acquire/find/release 的单线程与正常并发 journey。
- 预期输出：capability-family map、全部 public-surface trace、有限 `FIX` 集、owner/复杂度/资源表、直接
  consumer 与依赖裁决、correctness/sanitizer/ISA 证据、统一 P1 bundle、三种交付、四类文档校准、独立
  红队与四项 LOC。
- 停止条件 / 可接受限制：全部 `MUST` 为 `COMPLETE` 或完整 `ROUTED`、五组适用工程门 `PASS` 且无
  correctness/security/delivery blocker 即停止。业务 cache/eviction、blocking service、ordered multi、
  node extraction/merge、Native FFI 与 Formal 可在证据不足时保持具名负空间；任何 material `MUST`、
  correctness、交付或统一性能门不能在预算内裁决时进入具名 `BLOCKED`，不无限研究。

## Capability-family coverage map（待 `AUDITED`）

| Family | admitted user journeys / evidence | required semantic variants | lifecycle / boundary / ownership dimensions | current route | coverage | missing edge / negative space |
|---|---|---|---|---|---|---|

## Capability / gap ledger（待 `AUDITED`）

| ID | requirement class | demand kind | exact source | normal shape | semantic / artifact / delivery owner | ownership/failure/concurrency | complexity/resource target | coverage | action | severity | verification / negative space |
|---|---|---|---|---|---|---|---|---|---|---|---|

- 现有公开入口反向追踪：待完成。
- 已有能力完整性检查：待完成。
- 语法消费者 / 语义重复扫描：待完成。
- public/internal/higher-layer 裁决：待完成。
- 本轮冻结 `FIX` 集：只有完成两遍审计后填写；当前不得改产品实现。
- 相邻模块 backlog：待审计；只记录，不顺手修改未关闭下游。

## Candidate

- product commit / tree：尚未形成。
- 公开合同与 user journey 变化：待审计与有限修复。
- ownership / lifetime / concurrency / failure 变化：待审计。
- 算法 / 复杂度 / 常态 metadata 与 layout 变化：待审计。
- 直接消费者迁移：待审计。
- candidate tests：待冻结。
- harness / tool identity：与 product identity 分开，待冻结。

## Verification manifest

尚未执行。未运行项不得写为 `PASS`。

## 独立红队

尚未开始；只在 candidate 冻结后，以 30 分钟预算攻击已冻结合同。

## Closure decision

本轮当前只完成 `SCOPED`，没有 candidate 或 terminal verdict。
