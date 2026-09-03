# Platform v3：推理 host 能力补全与质因数迁移

<!-- MODULE CLOSURE RECORD: ACTIVE -->

## Reopen evidence（`REOPENED`）

- component / owner / round / kind：`platform` / Platform /
  `platform-v3-capability-factorization` / `REOPEN`；
- baseline：`6bf023c90b6ec3d62e9892028e172a385671de1e`；前轮：`platform-v2`；
- trigger：`PLATFORM-FACTORIZATION-2026-09-03`；
- `PLAT3-R1`：已接受的全局能力账本把 physical-core/SMT/cache-domain topology
  (`PLAT-A-05`)、host memory budget (`PLAT-A-09`) 与 caller affinity save/restore (`PLAT-A-17`)
  判为当前 llama.cpp/ggml 推理旅程的 `MISSING/MUST`；旧 platform-v2 明确没有审计这些能力；
- `PLAT3-R2`：`PLAT-A-02` 已准入 scalable-vector width 与 per-thread AMX/SME enablement。当前只有
  coarse feature bool，不能完成 ggml kernel 选择、线程进入和恢复的完整旅程；
- `PLAT3-R3`：Util-core 的统计 RNG 明示不安全，而两个成熟基础库都提供 OS entropy；全局 owner
  裁决已把 secure seed mechanism (`PLAT-A-33`) 放入 Platform host factor；
- `PLAT3-R4`：接受的三因子边界证明 host facts/raw VM/NUMA/DynLib、native thread/wait 与同步
  filesystem 具有不同直接消费者和依赖成本。现有单一 factor 让只需 CPU facts 的 SIMD/Container
  传递取得 Time/fs/thread artifact，已满足 component dependency/delivery 变化的重开条件；
- `PLAT3-R5`：既有 MappedFile/residency 与 SharedMemory storage 最终 owner 已裁决为未来 `mem_vm`。
  本轮只冻结迁移 seam，不在 Mem 未依法重开前修改其产品 ownership；不能为赶进度留下双 owner；
- 2026-09-03 clean reference identities：llama.cpp
  `67a17c17caa95742186f8b1ecadd1b5abd6d5ebb`（随仓 ggml tree
  `b91bd1c0eadf24e7d56143f36942e1a838c1dd8b`）、standalone ggml
  `d4716378882593333721eb33f153144b6885caf2`、Folly
  `eeab4246827eae9a11c0fa22529bff1a258aa83a`、oneTBB
  `843162db8cf4b2531a4a6147b6ea3fd860d44668`、Boost
  `cb94f33fcb13e90922905c2d8105508e71fa6db4`；
- 初步受影响面：Platform capability/API audit；`platform`、`platform_thread`、`platform_fs` 的头、
  source、tests、direct edges 与 product façade；SIMD/Mem/Sched/IO 的直接消费；source/static/shared；
  correctness、P1、architecture/user/performance 文档；
- 初步 tier：`P1`。CPU facts 是初始化/重配控制面，thread park 可处于 runtime 热路，filesystem 是
  loader 控制面；API freeze 后按语义族分别使用结构成本或已有正常 consumer，不把 syscall 微点值
  混成一个“Platform 更快”结论；
- 暂定非目标：在 Mem round 前搬迁 mapping/shared-storage owner；async/direct/scatter I/O、scheduler
  placement policy、device facts/kernel dispatch、JIT/process/plugin policy、Native FFI/C API、稳定
  C++ ABI、benchmark-driven 入口、公共仪器改造与 Formal certification。

当前只登记足以击穿旧结论的新事实和新 component ID。closure card、完整双向能力审计、自然 C++
API freeze 与有限 `FIX` 集尚未完成，因此不授权产品实现。

## Closure card（`SCOPED`）

- component / owner / round / kind：`platform` / Platform /
  `platform-v3-capability-factorization` / `REOPEN`；baseline 与 trigger 见上；
- 真实消费者：SIMD/Container 的 CPU facts，Mem 的 raw VM/NUMA 与 loader storage，Sched 的 worker
  placement，IO 的 host budget/backend loading；北极星是最新锁定 llama.cpp/ggml 的 physical-core
  sizing、NUMA allocation、host-memory planning、dynamic backend 与 model-loading journey；
- 本 factor 范围：CPU/ISA/cache/topology、host memory、NUMA、raw anonymous VM、DynLib、process
  environment 与 secure OS entropy；现有 `platform` component ID 保留为该 host factor，避免伪兼容
  target 或删除 registry 历史；
- 产品级范围：与 `platform_thread`、`platform_fs` 协同完成三因子迁移，最终
  `hyle::platform` façade 聚合三者；MappedFile/SharedMemory 的 storage artifact 只在 Mem factor
  round 依法 `SCOPED/AUDITED` 后迁走，本轮不得留下双 owner；
- 非目标：device inventory/kernel dispatch、scheduler placement/stealing、async/direct I/O、lexical
  path、JIT/process/sandbox/plugin ABI、Native FFI/C API、稳定 C++ ABI 与 Formal；
- 预期直接依赖：host factor 只 `PUBLIC core,error`；OS API/compiler builtin 是机制合同。预期直接
  consumer 是 `platform_thread`、`platform_fs`、Mem VM/pool、SIMD、Container、Sched 与 IO；
- tier / normal performance：`P1`。primary 是推理初始化中的 process-available topology + host-memory
  snapshot，对同语义 direct-OS/locked peer route；holdout 是已缓存 CPU/ISA facts 的重复读取。5%
  non-inferiority；不得把多 syscall 的完整 snapshot 与单个 cached scalar 作苹果橘子对比；
- 常态成本：稳定 facts 一次探测后 O(1)，动态 availability/topology 为 O(available CPUs + reported
  relations)，buffer-fill core 不分配；host-memory snapshot 为 O(1) 有界 OS/file queries；secure entropy
  为 O(bytes/native chunk)，无 TLS PRNG state；raw VM/NUMA 保持既有 complexity；
- 总预算：三个 Platform factor 协同调查与实现最多六小时，性能取证合计九十分钟；每个 family 最多
  一次机制诊断和两次产品修正。达到预算仍有适用 gate 未裁决即 `BLOCKED`，不扩建公共仪器；
- machine / instrument：`W1-WIN`、`W1-WSL`、`M1-MAC`；P1 primary 仅 W1-WIN +
  `closure-perf-v1`，Formal `PENDING/formal-release-v1`；
- strong peers：Folly `eeab4246827eae9a11c0fa22529bff1a258aa83a` 的 portability/Random/
  MemoryMapping，oneTBB `843162db8cf4b2531a4a6147b6ea3fd860d44668` 的 info/NUMA；CPU/VM/
  entropy 还受各 OS 官方 ABI 约束；
- correctness gates：normal、empty/zero、capacity/overflow、unsupported/permission/OOM/system failure、
  sparse IDs、snapshot churn、move/destroy、concurrency/cache init、property/stress、ASan/UBSan/TSan、
  可用的真 ISA/NUMA；不适用项必须逐项说明；
- delivery gates：三个 factor 精确构建和 header ownership；Platform product source embedding、relocated
  static/shared；SIMD/Mem/Container/Sched/IO direct consumers；三平台；旧 C/Native FFI 负向扫描；
- 停止条件：双向审计冻结有限 `FIX`，固定 API/guide 后实现；新 process/JIT/device/async journey 只
  路由。正常 P1 无资格则停止为 `BLOCKED`，不依据微观点值改变类型或常态 metadata。

`SCOPED` 只冻结问题、预算、机器、性能等级和交付边界；完整 capability/public-surface trace、所有
`MUST` 的 owner 与有限 `FIX` 集完成前，仍不授权产品实现。

## 双向能力审计（`AUDITED`）

审计绑定 SCOPED baseline 与 2026-09-03 clean references。正向检查了最新 llama.cpp 的物理核
计数/线程亲和恢复、ggml 的 host-memory/device planning、SVE/RVV vector width 与 Linux AMX
permission；peer 只使用 Folly CacheLocality/Random/File 与 oneTBB info/NUMA 的同能力部分。反向追踪
覆盖当前十个 Platform public headers、全部 Platform sources/tests、package sample，以及
SIMD/Container/Mem/Sched/IO 的直接 include/link 边。结论不是按 peer API 求并集：只有下表 `MUST`
进入本轮，成熟库中没有当前消费或定义性基础合同的 process/JIT/file-lock 能力保持触发器。

### Capability-family coverage map

| Family | normal journey | baseline | audit verdict / owner |
|---|---|---|---|
| CPU/ISA/cache facts | fat binary 安全选择、cache hint | feature/tier/单份 cache 完整 | `PARTIAL`：补 AMX/SME 静态事实；per-thread state 归 `platform_thread` |
| process-available topology | 物理核 sizing、避开 SMT、按 LLC/NUMA/强弱核放 worker | sparse available ids 与 rank 有，缺 core/cache domain | `PARTIAL`：host factor 补 process-view location 与 aggregate counts |
| host memory budget | 分配/层拟合前取得当前可用预算 | `MISSING` | host factor 补 fallible snapshot；device/UMA 聚合仍归 Compute/引擎 |
| NUMA/raw VM/guard | preferred/bind/interleave、reserve/commit/protect | `COMPLETE` | `KEEP` host mechanism；storage owner 归 Mem |
| native thread/wait | configured worker、address wait、Parker | lifecycle/wait 完整，affinity set-only | `PARTIAL`：查询/恢复与 extended ISA 归 `platform_thread` |
| synchronous filesystem | model/config/cache cursor I/O、namespace/temp/replace | 当前 portable journey 完整 | `COMPLETE`；迁到 `platform_fs`，不加 benchmark-only operation |
| mapped/shared storage | model mapping/residency、named shared pages | Platform mechanism + Mem wrapper 双层公开 | `PARTIAL`：唯一 public owner 迁 `mem_vm`；prefault/fragment 由 Mem round 裁决 |
| DynLib/environment/entropy | backend discovery、startup env、secure seed | loader/env 完整，entropy 缺失 | `PARTIAL`：host 补 OS entropy；env 从 fs source/header 拆出 |
| advanced process/JIT/fs/device | subprocess、W^X、lock/preallocate、GPU facts | 无或上层已有 | `ROUTED/OUT_OF_SCOPE`；不进入本轮 |
| C++ delivery | source/static/shared 与 baseline probe isolation | 单 target 已通过，精确 factor 不存在 | `PARTIAL`：三 factor、facade、direct edge 和 relocation 必须重签 |

### 冻结 capability / gap ledger

| ID | class | frozen normal shape | owner / boundary | complexity/resource target | action / verification |
|---|---|---|---|---|---|
| `PLAT3-H01` | `MUST` | process-available `CpuInfo` 同时给 package/core/SMT/NUMA/L2/L3 domain；aggregate 给 physical-core/package/cache-domain count | `platform` host facts；Sched 选 policy | snapshot O(available CPUs + OS relations)，buffer-fill core；unknown 不猜 | `FIX` 三平台 topology tests + llama physical-core journey |
| `PLAT3-H02` | `MUST` | `HostMemoryInfo` 给 physical total/available、可用时的 commit/process limit 与保守 effective available | `platform` host；Compute 聚合 device budget | O(1) 有界 native/file queries，无 payload allocation | `FIX` synthetic parser/limit 边界 + real-host smoke |
| `PLAT3-H03` | `MUST` | `secure_random(span<byte>) -> Status`；empty 成功，partial/EINTR 完整填满或失败 | `platform` host mechanism | O(bytes/native chunk)，无 TLS PRNG/hidden fallback | `FIX` OS path、failure taxonomy、temp-file reuse |
| `PLAT3-H04` | `MUST` | CPU/NUMA/raw VM/DynLib/env/entropy 组成可独立链接 host factor | `platform` ID 保留为 host | CPU probe 仍 baseline/LTO-off；不携带 Time/thread/fs | `FIX` source/header ownership、`hardware_concurrency` 迁移、exact-factor build |
| `PLAT3-T01` | `MUST` | caller affinity buffer-fill + owning convenience，返回值可原样交给 set 完成 save/set/restore | `platform_thread` | O(mask words)，只有 convenience 分配 | `FIX` sparse/large mask、restore、unsupported tests |
| `PLAT3-T02` | `MUST` | current-thread scalable/streaming vector bytes 与 AMX tile permission 是窄机制；SIMD 仍唯一选 kernel | host static fact + `platform_thread` state | O(1) syscall/register query；enable 只显式调用 | `FIX` Linux SVE/SME/AMX 与 RVV route、其他平台诚实 unsupported/zero |
| `PLAT3-T03` | `MUST` | Thread/wait/Parker 独立 factor 并由 Platform facade 聚合 | `platform_thread -> PUBLIC core,error,time; PRIVATE platform` | wait 仍一次 native primitive；Parker 一个 atomic word | `FIX` target/headers/direct Sched edge + existing P1 replay |
| `PLAT3-F01` | `MUST` | FileHandle/namespace/temp/replace 独立 factor；env 不再藏在 fs | `platform_fs -> core,error,time`，private host entropy | FileHandle 一字；普通 read/write 至多一次 syscall | `FIX` source/header/convenience split、exact-factor build |
| `PLAT3-F02` | `MUST` | warmed regular file open + complete sequential read + close | `platform_fs` | 同语义 5% non-inferiority；setup/bytes/teardown全计费 | `FIX` frozen consumer P1 + holdout temp/sync/rename budget |
| `PLAT3-X01` | `MUST` | mapping/residency/shared storage 只有一个 Mem public owner，不留 Platform facade 双 owner | `mem_vm` over host/fs mechanisms | 不增加 pimpl、copy 或常态 dispatch | `FIX` 先登记并审计 Mem factor，再迁 owner和 package journey |
| `PLAT3-D01` | `MUST` | architecture/user/performance docs 与 source/static/shared/direct consumers 跟随最终图 | 各文档/Build owner | 不制造 C/Native FFI 或稳定 C++ ABI | `FIX` scans、三平台、package relocation、六问与红队 |

以上十一票是本轮完整 `FIX` 集。`PLAT-A-16` detach、`A-18` process priority、`A-26` lock/preallocate
保持 `GENERAL/DEFER`；`A-13/A-35` 保持 `OUT_OF_SCOPE`；没有这些功能不能被描述为当前推理闭包缺陷。

### API / usability freeze

- 当前状态：`FROZEN`（2026-09-03）。公开头自含、自然 compile/use journeys、类型与默认值、失败边界，
  以及 Windows/WSL 的三个精确 factor 构建均已通过；从此只有正确性缺陷可以令本轮退回 `AUDITED`，
  P1 不能再反向改 API、layout 或常态 metadata。
- host 日常路线保持 `cpu_features/available_cpu_topology/available_cpu_info/host_memory_info/secure_random`；动态
  topology 与 memory 是显式 snapshot，不把 syscall 偷藏进已缓存的 scalar getter。
- thread 日常路线是 `this_thread::affinity()` owning snapshot 或 buffer-fill core，配合现有
  `set_affinity`；extended state 的名字描述事实或显式 request，不叫“fast mode”。
- fs 日常路线仍为 `FileHandle` 与 buffer-fill core；owning convenience 拆成 factor-owned opt-in headers，
  不再用一个 `convenience.h` 把 CPU、env 与 filesystem 强耦合。
- public namespace 仍为 `hyle::platform`，内部 factor target 不成为用户必须理解的新产品。HYLE 尚未发布，
  不保留旧 convenience/mapping alias；迁移在同一 candidate 完成。

### Owner 六问（AUDITED）

1. **第三方路径：**产品实现没有第三方库；只有 C++23/compiler builtin/Windows、Linux、macOS ABI。
   Folly/oneTBB/llama.cpp/ggml 是锁定需求与 peer，不链接进产品。
2. **推理能力：**现有 36 行账本中本轮准入的未处置 `MUST` 正是 H01/H02/T01/T02/X01；完成前答案为
   `NO`，其余 device/kernel/async/policy 均已路由。
3. **性能：**当前只有旧单 target 证据，不能续签新 factor。API freeze 后按 host snapshot、Parker、
   warmed file 三个不同合同裁决；禁止给出“Platform 全部 SOTA”的混合结论。
4. **旧 ABI：**扫描无 C header/schema/bridge/未装饰 C export/managed sample/FFI option；decorated C++
   export 与 OS-private boundary 不构成 Native FFI，当前 verdict `CLEAN`。
5. **交付：**旧 Platform product 三形状曾通过；三个精确 factor、facade 聚合与迁移后的 relocated
   source/static/shared 尚未运行，当前 `BLOCKED-UNTIL-CANDIDATE` 而不是伪 `PASS`。
6. **易用性/文档：**既有 owner/Result/buffer-fill + opt-in owning 方向自然；缺口在 topology/memory/
   affinity/state 和混合 convenience。architecture/user/sample/performance 必须随 candidate 更新。

`AUDITED` 授权且只授权十一票产品修改。`PLAT3-X01` 要求先给 `mem_vm` 建立合法 closure card；在此
之前不得通过临时 cycle、重复 owner 或把 mapping 塞进 fs 来完成 target 拆分。

## 候选前实现检查点

- H01--H04 已落地：process-available topology、host-memory budget、OS entropy 与 host-only factor；
- T01--T03、F01--F02 已落地：affinity save/restore、current ISA state、独立 thread/wait 与同步 fs factor；
- X01 已通过合法 Mem v3/mem_vm round 完成唯一 owner 迁移，Platform 不再公开 storage owner；
- D01 的 architecture/user/build graph 已随最终边界更新，性能证据与 terminal record 仍待 candidate；
- W1-WIN 最新单元结果为 host `89/89`、thread `42 PASS + 1 hardware skip`、fs `51/51`；GCC 14
  `-Werror` 为 host `88/88`、thread `44/44`、fs `58/58`，公开头自含均通过；
- 以上只记录进入 candidate 前的事实；该检查点尚未裁决 M1-MAC、sanitizer、三种产品交付、
  direct-consumer replay、P1 qualification 与红队，后续进度以紧接的 Candidate 节为准。

## Candidate（`CANDIDATE`）

- product commit/tree：`e9b9a43d4782c60d84badc02b1a41fbe21207a00` /
  `c6429316279aaac1e0f5d7a7a71872a60cf967b6`；
- 十一票均已实现，公开合同、算法、layout、三 factor 图与 host snapshot / Parker handoff / warmed
  regular-file 三个彼此独立的 normal P1 workload 同时冻结；
- candidate 前已通过 W1-WIN、GCC 14 `-Werror`、ASan、UBSan、公开头自含、三个 Windows exact-factor
  roots，以及 Platform product 的 source/relocated shared/static journeys；
- 尚未裁决 M1-MAC、direct consumers、正式 P1、FFI/symbol 扫描与最终红队，故不能写 `VERIFIED` 或
  `CLOSED`。任何产品改动都必须依法退回 `AUDITED`。
