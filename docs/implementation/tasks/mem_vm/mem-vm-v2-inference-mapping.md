# Mem VM v2：推理 mapping 与 OS storage 合同重构

<!-- MODULE CLOSURE RECORD: HISTORICAL -->
<!-- MODULE CLOSURE RECEIPT: CLOSED -->

> 本轮终局裁决为 `CLOSED`。下方 API、成本、七项有限 `FIX`、全部适用工程门与接受限制已绑定
> 不可变候选；本记录只作历史证据，不再授权产品修改。

## Reopen evidence（`REOPENED`）

- component / owner / round / kind：`mem_vm` / Mem mapping and residency semantics /
  `mem-vm-v2-inference-mapping` / `REOPEN`；baseline
  `5aaeaa770be72cd7a58465659b2abc7d98a2d89b`，tree
  `2035afb59e4a658526bc3f39e458b305a36f1a1d`；
- trigger IDs：`MEMVM-INFERENCE-SPARSE-MAP-01`、`MEMVM-PLATFORM-V8-REPLAY-02`；
- 当前锁定 llama.cpp `74a7c897f049c17e7080423aa2111776eff6ebbf` 的 model loader 直接使用
  lazy file ranges，并在 POSIX 上通过 `llama_mmap::unmap_fragment(first,last)` 释放已经不再需要的
  weight pages；同一 owner 还按 lazy range 分别施加 sequential/random advice。`mem-vm-v1` 明确把
  holey/segmented mapping 排除并要求用户创建多个 owner，不能自然表达这条已命名北极星旅程，命中
  “正式准入的共同 storage journey 不能由当前 owner 表达”的 reopen trigger；
- Mem-VM v1 候选后，直接依赖 Platform 的 NUMA implementation 与 VM range error contract 已进入
  `platform-v8` 候选 `c39df6f94c636ab8fa94cb700dd263763a42d680`。现有 `VirtualMapping`、
  `HugeMapping` 与 guarded stack 路线需要重新验证唯一 decision authority、错误传播、平台差异和
  source/static/shared 导入；
- 初步 red-team 输入还包括：`VirtualMapping::advise` 目前依赖两个 enum 的数值偶然一致；
  `VirtualMapping`/`HugeMapping` 与统一 Buffer/View 组合是否自然；shared mapping 的 read-only、size、
  alignment 与 name lifecycle 是否形成半项能力；file mapping 的 offset、prefetch/populate/lock/flush/
  partial-release 在 Windows、Linux、macOS 上是否诚实使用系统接口。这些只是待裁决 finding，未在
  `AUDITED` 前预判为全部要实现；
- strong inputs 固定为当前 llama.cpp、standalone ggml、Folly
  `65749da412d92fe27d87ae45a8ef670720712858` 的 `MemoryMapping` 与 Boost Interprocess 当前本地
  checkout。peer 用于发现稳定任务与语义档，不按 symbol 数量复制；
- `CLOSED -> REOPENED` 前 registry 快照位于 canonical checkout 外：
  `C:\public\project-hyle\_validation\registry-before-mem-v6-mem-vm-v2-reopened-5aaeaa77.json`，
  SHA-256 `036760D948DFF33F9600A7B926F9F7E7CCB3304A328842EB2C7706E9383E9D6B`。

## 初步停止边界

- 本轮只拥有 anonymous/file/shared mapping、range release/advice/residency、large/huge-page placement
  与 guarded stack storage；base allocator/pool/planner 归协同 `mem-v6`，reclamation 无新事实不重开；
- async/direct IO、backend/device mapping、Tensor/KV/model policy、IPC synchronization protocol、
  Native FFI/C API、稳定 C++ ABI、公共性能仪器和 Compute/Graph 均不进入；
- 非便携 OS 能力可以形成 capability query、明确 `Unimplemented` 或具名 platform-specific route，
  不能静默假装成功，也不能为了表面统一复制 Platform 已拥有的机制；
- API freeze 后只允许内部 OS adapter、owner metadata/algorithm 与 build wiring 修正。性能只比较相同
  mapping/residency/release 语义和相同 charged pages/syscalls；苹果对橘子只保留 natural-route 或
  diagnostic 结论。

## Closure card（`SCOPED`）

- component / owner / round / kind：`mem_vm` / Mem mapping and residency semantics /
  `mem-vm-v2-inference-mapping` / `REOPEN`；baseline
  `5aaeaa770be72cd7a58465659b2abc7d98a2d89b`；
- demands：`MEMVM-INFERENCE-SPARSE-MAP-01` 与 `MEMVM-PLATFORM-V8-REPLAY-02`。主 journey 是模型
  loader 对同一文件建立长期只读 mapping，按 lazy tensor ranges 提示访问、逐步锁定可选前缀，并在
  POSIX 上释放不再需要的页区间，同时 owner 仍正确清理剩余 fragments；
- 范围：anonymous reserve/commit/decommit/protect/advice/lock；file RO/RW/COW mapping、offset/
  length、range advice/prefetch/populate/residency/flush 与 partial release；named shared-memory
  create/open/access/size/name/unlink/flush；large/huge-page hint 与 NUMA placement；guarded fiber stack
  及 bounded cache；与 Mem Buffer/View 的安全组合；
- 非目标：base allocator/pool/planner、reclamation、async/direct IO、device/backend mapping、Tensor/KV/
  model policy、IPC synchronization protocol、C/Native FFI、稳定 C++ ABI、公共性能仪器和 Compute/Graph；
- 直接依赖：`core`、`error`、`mem`、`platform`、`platform_fs`、`util_core`；直接消费者为 Sched 的
  fiber stack、模型加载/IO/Compute 的 mapping journey，以及 Mem product facade；
- 待复用设施：Platform `VmRegion`、NUMA/host facts、`FileHandle` 与 OS error translation，Mem
  Buffer/View，Util-core `SpinLock`；direct OS file/shared mapping 只在 Platform 没有同合同 owner 时保留，
  但错误、range、page/alignment 和 decision authority 不能复制或依赖 enum 数值偶合；
- tier：`P1`。主要 workload 是 `NORTH_STAR_JOURNEY` 的 model-loader RO map + deterministic lazy
  ranges + range advice + POSIX partial release + remaining-owner teardown；在 `W1-WSL` 原生 ext4 上与
  当前 llama.cpp 同语义机制 adapter 作 `SAME_CONTRACT`，计费相同 mmap/advice/munmap/剩余 unmap/close
  及实际 mapped pages/syscalls。W1-WIN 的 whole-map lifecycle/progressive residency 仅作一个冻结
  holdout，不与 POSIX partial-unmap 数字池化；
- 5% material/non-inferiority margin；无相同 OS 语义的平台只签 correctness/fallback，禁止跨 OS
  排名或以 Windows no-op partial release 冒充 POSIX 同合同；
- 总调查预算：6 小时；其中性能最多 90 分钟。最多两轮产品修正；性能异常只允许一次反序复现、
  一次 syscall/codegen/profile 机制检查、一个产品修改和一次重跑；
- correctness machines：`W1-WIN`、`W1-WSL`、`M1-MAC`；主要 P1 machine：`W1-WSL`；instrument：
  `closure-perf-v1`，P0 辅助为 `closure-correctness-v1`；WSL 只签明确的 Linux guest engineering
  contract，不冒充 native-Linux 或独立硬件 Formal；
- strong inputs：llama.cpp `74a7c897f049c17e7080423aa2111776eff6ebbf` 的 `llama_mmap`/
  `llama_mlock`，Folly `65749da412d92fe27d87ae45a8ef670720712858` 的 `MemoryMapping`；shared
  memory family 以当前本地 Boost.Interprocess `cb94f33fcb13e90922905c2d8105508e71fa6db4` 取代
  Folly，仍保证每 family 最多两个；
- API 语义档：reserve-only 与 committed anonymous owner 分明；file mapping access/lifetime 显式；
  partial release 后不得继续声称完整连续 `bytes()/view()`；shared access 与 payload size/alignment 明确；
  THP hint、explicit large page、NUMA bind/interleave 不混为一种保证；guarded stack cache 只缓存完整 owner；
- 适用门：全部 public surface reverse trace；正常、空/零、page/range/offset/overflow、OS failure/
  invalid、move/destroy/partial teardown、共享 name 生命周期、并发 cache、property/stress、ASan/UBSan/
  TSan；Windows/WSL/macOS 系统交互，source/static/shared delivery，直接消费者、FFI 清洁、文档/易用性、
  内部复用、红队和 P1 工程性能；真实 ISA/backend 仅在确实不适用后记 `NOT_APPLICABLE`；
- 输出：能力 family map、七问、冻结 API、有限 `FIX` ledger、candidate、验证 manifest、终局 receipt
  和同候选 LOC；
- 停止：所有 `MUST` 为 `COMPLETE/ROUTED`、适用门通过且无 blocker 即 `CLOSED`；平台缺失而无法
  诚实实现的路线必须明确 capability/`Unimplemented` 并裁决，固定预算内 P1 或 correctness 未决则
  `BLOCKED`，不得扩建仪器或不断换 benchmark。

## Capability-family audit 与 API/成本 freeze（`AUDITED`）

### 当前超集与 owner 裁决

| family | 真实任务 | 当前差距 | 裁决 |
|---|---|---|---|
| anonymous VM | reserve/commit/decommit/protect/advice/lock/NUMA 与 whole release | `VirtualMapping` 只是 `platform::VmRegion` 包装并复制 enums | `ROUTED/FIX`：Platform 唯一 owner，Mem 只给 type alias/discovery |
| file mapping | RO/RW/COW offset/length、checked ranges、lazy advice/populate、progressive residency、flush、POSIX partial release | 连续 raw view 无法表示 holes；缺 inference loader 的 release；flush 对 RO/COW 静默 OK | `MUST/FIX` |
| named shared storage | create/open size/access、page-aligned payload、unlink、flush、name lifecycle | 私有 16B header 破坏对齐并有 publication race；open 只能 RW；Windows 固定 260 wchar | `MUST/FIX` |
| large host region | THP hint、prefault、NUMA preferred/bind/selected interleave、accepted/applied 可见 | `HugeMapping` 名字把 THP hint 容易误读成 explicit huge-page guarantee | `MUST/API FIX`；hard 1 GiB allocator reserve 仍由 base Mem |
| guarded stack | guard pages、move owner、bounded concurrent cache、remote return/trim | Sched 真实消费且机制完整 | `COMPLETE` |
| Buffer/View combination | mapped range 进入统一 const/mutable view vocabulary | 连续 mapping 完整；hole 后必须拒绝跨 hole view | `MUST/FIX` checked subrange |

Folly/Boost 与当前 llama loader 共同支持上述稳定任务。explicit per-owner hugetlb/MEM_LARGE_PAGES、async/direct
IO、persistent-memory flush 和 IPC synchronization 没有当前具名通用 consumer，分别 `PENDING/ROUTED`，
本轮不借“功能越多越好”越过权限/设备/IO owner。

### 冻结 C++ API

- `VirtualMapping`、`MappingProtection`、`MappingAdvice`、`Residency*` 不再是第二套类型；Mem facade 用
  `using` 指向 `platform::VmRegion/Protect/VmAdvice/VmLock*`，匿名 VM 方法名和语义以 Platform 为准；
- `MappingRange { offset, size }` 是普通 logical byte range；控制函数默认 length 用
  `std::dynamic_extent`，显式 `0` 是空范围；
- `FileMapping` 保留 move-only owner/open/map/access/size，删除不安全的 whole raw `data()/bytes()/view()`；
  新增 checked `address/bytes/view(offset,length)` 和 writable 对应项；`mapped_size()` 与
  `fragment_count()` 可观察当前 live logical storage；
- `release_range`/`release_ranges` 返回实际释放的完整页字节数。POSIX/macOS inward-round 到完整页并
  O(R log R + F) 合并 range/fragment；Windows 当前 whole-view primitive 无安全等价，返回
  `Unimplemented` 且不改变 owner；Windows 在 O(R) bounds/page 校验后早退，不分配临时表；普通连续
  owner 不分配 fragment metadata；
- partial release 后 owner 仍持有 logical extent；checked access 跨 hole 失败，teardown 只释放剩余
  fragments。重复释放 fully-hole owner 是无分配的幂等 0；有序 fragments 让 checked access 为
  O(log F)、局部控制为 O(log F + K)，而非对每个 tensor 线扫 O(F)；release 不得穿过已锁 prefix；
- `FileMapping::flush` 只对 RW shared mapping 成功，RO/COW 返回 `FailedPrecondition`；`reset()` 返回
  teardown Status，析构只 best-effort；offset/native cast/overflow 显式检查；
- `SharedMapping::open(name,size,access=kReadOnly)` 显式映射请求大小，底层对象至少要有这些字节；移除
  header，payload 从 mapping base 开始并保持 OS page alignment。create 返回 RW；只读 owner 不暴露 mutable
  pointer/span/view；native capacity 小于请求时返回 retryable `Unavailable`。Linux shm 保留精确 size，
  Darwin/Windows page-round capacity；初始化同步仍归 IPC 协议；
- `HugeMapping/HugeFlags` 改名 `LargeMapping/LargeMappingFlags`；accepted THP 表示 OS 接受 hint，不表示
  已获得物理巨页，NUMA applied 仍遵守 Platform 定义。

### 冻结 FIX ledger

| FIX | 产品修改 | 复杂度/常态成本 | 验证 |
|---|---|---|---|
| `MEMVM2-F01` | 用 Platform aliases/机制删除 VirtualMapping wrapper、enum cast 与重复 advice/lock/unlock | anonymous 路线零中间层；file control syscall 数不增 | API compile + three-OS range tests |
| `MEMVM2-F02` | FileMapping checked ranges、lazy ordered fragment metadata、single/batch partial release、hole-aware teardown/control | 连续 owner 零额外分配；首个 release 才分配；batch O(R log R + F + syscalls)，lookup O(log F) | property/range/repeat/partial failure/move/destroy + llama-shaped replay |
| `MEMVM2-F03` | 修复 flush/access/native size/cast、lock/release 与 backing-specific advice；Windows file mapping 禁止危险 `DiscardVirtualMemory` | 无热路常态税；Windows kDontNeed 一次分支后诚实失败 | boundary/OOM/OS-failure/access-mode tests |
| `MEMVM2-F04` | SharedMapping 去 header、显式 size/access、动态 Windows UTF-8 name、native name/capacity/race 校验 | payload page aligned；Linux exact size，Darwin/Windows page-round capacity；create/open 各一 view | two-process/read-only/name/size/alignment tests |
| `MEMVM2-F05` | `LargeMapping` taxonomy、THP/NUMA observability 文档和 tests；不新增 explicit hard owner | header-only 路线仍直接 Platform call | capability-aware three-platform tests |
| `MEMVM2-F06` | CMake/selected headers、Mem product aggregation、Sched/sample、architecture/user/performance 文档 | 依赖图不新增边 | source/static/shared + direct consumer |
| `MEMVM2-F07` | 冻结 WSL model-loader mapping/advice/release/teardown same-contract P1 和 Windows whole-map holdout | 5% NI；相同 ranges/pages/syscalls；不跨 OS 池化 | `closure-perf-v1` |

以上七票是全部授权。Folly 的 chunked `mlock/munmap` 解决大 VMA kernel semaphore 停顿，但会改变
syscall 数；只有冻结 model-loader replay 复现 material tail/throughput 问题时，才允许在 F02 内采用固定
内部 chunk，不开放 knob、不做孤立 syscall benchmark。

### 七问预裁决

1. 超集：六族在上述 FIX 后 `COMPLETE/ROUTED`，推理 sparse-loader 缺口被直接实现；
2. 依赖/复用：Platform 是 OS/NUMA/anonymous authority，Mem 是 Buffer/metadata allocator，Error 统一失败；
3. API：连续/有 hole、只读/可写、hint/guarantee 都由类型和 Result 明说，不由 benchmark case 塑形；
4. 性能：normal whole-map 不加 metadata allocation；batch 避免 llama 当前逐次 vector rebuild 的 O(k²)；
5. FFI：无已准入 C/Native FFI，只有 C++ exports；
6. build：`mem_vm` 独立 factor，Mem product 聚合；三种交付与 Sched direct consumer 必须实跑；
7. 文档/红队：系统交互按 Windows/Linux/macOS 分开写，unsupported 不伪装成功，五项红队 finding 逐项关闭。

`SCOPED -> AUDITED` 前 registry snapshot：
`C:\public\project-hyle\_validation\mem-v6-mem-vm-v2\module-status-before-audited.json`，SHA-256
`CB157D90B34C09F95BF62DE4F62821286E09E49321B15BBA3992FC14575F096B`。

## Candidate

- product commit：`9604bf4a33f13bd5f84ef78fedce55f305ac9d50`；tree：
  `57adb69f185cf02d42ab7f220ebeccbb91170b1b`；
- `FileMapping` 用 checked range/view 与 lazy ordered fragments 表达 partial release；连续 owner 无
  fragment allocation，查询 `O(log F)`，局部控制 `O(log F + K)`，batch release
  `O(R log R + F + syscalls)`；
- `SharedMapping` 去掉私有 header，create/open 显式 size/access，payload 原生页对齐；Linux 保留精确
  native size，Darwin/Windows 的 native capacity 可按页取整，初始化同步属于外部 IPC 协议；
- `VirtualMapping` 直接别名到 Platform authority；`LargeMapping` 明确 THP request/accepted 与 NUMA
  applied，不把 hint 命名成 hard huge-page guarantee；
- Windows 对 file-backed `kDontNeed` 与 partial release 诚实返回 `Unimplemented`，不调用语义不等价的
  `DiscardVirtualMemory`，空 flush 不升级成 whole-file flush；
- `AUDITED -> CANDIDATE` 前 registry snapshot：
  `C:\public\project-hyle\_validation\mem-v6-mem-vm-v2\module-status-before-candidate.json`，SHA-256
  `173814FFC86DB38FC7587C43D64F2433E7CFC4AB6DC09A2E6C33B75CDB37553C`。

## Owner 七问终裁

1. **第三方产品路径：**Mem-VM 没有第三方 implementation dependency；它直接使用支持 OS 的 mapping/
   shm primitives，并复用 HYLE Platform/FileHandle、Mem Buffer/View 与 Error。Folly、Boost 与锁定
   llama.cpp 只作 capability/consumer peer，不进入链接或 public type。base Mem 的 mimalloc 是传递
   product artifact，但不参与 file/shared/native mapping mechanism。
2. **推理引擎能力完整度：**anonymous reserve/commit、file RO/RW/COW、checked range、advice/populate/
   lock/flush、POSIX sparse release、named shared bytes、large/THP/NUMA owner 与 guarded stack 六族均
   `COMPLETE/ROUTED`。async/direct IO、device mapping、persistent-memory flush、IPC synchronization、
   Tensor/KV/model policy 都有别的 semantic owner 或缺少当前具名基础 consumer；未处置 `MUST=0`。
3. **内部复用与加速：**`VirtualMapping` 直接 alias Platform `VmRegion`，advice/lock/unlock 复用唯一
   Platform range authority；FileMapping 消费 `platform::FileHandle`，所有失败走 Error，views 复用 Mem，
   fragment metadata 用 base Mem system allocator。file/shared owner-specific mmap/shm lifecycle 与
   fragment table 因 Platform 无 owner state、且下沉会反向制造依赖而保留 `LOCAL`。verdict `PASS`。
4. **拟真公平性能：**Linux 任务严格等同一次 file lifecycle、64 advice、64 partial munmap、相同 survivor
   state 与 65 次 teardown；只比较 HYLE batch `O(R log R + F)` 与锁定 llama vector rebuild。4/4 5% NI、
   all controls `PASS`；2 `PARITY` + 2 `INCONCLUSIVE`，点估计有利 3.07%--4.67%，不宣称 material win。
   Windows 没有 partial-unmap 同合同，明确不参与苹果对橘子排名。
5. **旧 C ABI / Native FFI：**没有 C header/schema/bridge、undecorated HYLE export、FFI option/fixture 或
   FFI-paid layout；Windows shared DLL 导出仅为 decorated C++ symbols。verdict `CLEAN`。
6. **Source/static/shared：**Mem product 聚合 `mem_vm`，内部 factor 仍可独立选择；Windows exact candidate
   source embedding、relocated static/shared SDK consumer 全部运行，selected install 的新 headers 与
   runtime DLL 正确。fresh Sched selected build/test 证明 FiberStack consumer。verdict `PASS`。
7. **C++ API / 文档：**move-only owner、checked view、read-only mutable denial、explicit shared size/access、
   dynamic-extent 与 explicit-empty、requested/accepted/applied 和 reset Status 都由类型表达；没有为
   benchmark 暴露 fragment scan/threshold knob。guide 按 OS 写清 hole、borrow invalidation、hint、lock、
   flush、shared readiness 和 Windows unsupported。verdict `PASS`。

## Verification manifest

### Correctness risk matrix

| Risk | Verdict | 证据 / 限制 |
|---|---|---|
| normal；empty / zero | `PASS` | anonymous/file/shared/large/stack suites；empty range、explicit zero 与 to-end 分离 |
| capacity / boundary / overflow | `PASS` | unaligned offset、EOF、native cast、page rounding、shared capacity/name/range overflow |
| OOM / system failure / invalid input | `PASS` | invalid path/name/access/node、lock limit、unsupported advice/release；metadata failure keeps owner coherent |
| move / destroy / lifetime | `PASS` | move/reset、open-handle consumption、hole teardown、shared unlink 和 subprocess peer |
| concurrency / reentrancy | `PASS` | FiberStackPool concurrent recycle；SharedMapping real subprocess visibility |
| cancellation / close | `PASS` | `reset()` 返回 teardown Status；析构 best-effort；partial release 保留已完成 state |
| property / differential / fuzz / stress | `PASS` | overlapping/repeated/batch ranges、large multi-page/fragment invariants、cross-process shared route |
| ASan / UBSan / TSan | `PASS` | exact candidate ASan/UBSan Mem-VM suites各 `1/1`；TSan FiberStackPool `1/1` |
| real ISA / backend | `NOT_APPLICABLE` | 没有 SIMD/device backend；三个原生 OS mapping backend 均运行 |

### Platform 与交付

| Gate | W1-WIN / MSVC 19.50 | W1-WSL / GCC 14.2 | M1-MAC / AppleClang 21 | Verdict |
|---|---|---|---|---|
| exact-candidate selected correctness closure | `35/35`（含 Mem-VM） | `34/34` | `33/33` | `PASS` |
| native partial release | 正确 `Unimplemented` 且 owner 不变 | full tests `PASS` | full tests `PASS` | `PASS` |
| SharedMapping subprocess/capacity | read-only peer、page-rounded capacity | read-only peer、exact capacity | read-only peer、page-rounded capacity | `PASS` |
| source/static/shared + Sched consumer | 三种均 `PASS`；Sched `1/1` | source/correctness | source/correctness | `PASS` |

### 性能与同合同前置门

冻结合同为
[`mem-vm-model-loader-p1-v2.json`](../../../../bench/contracts/mem-vm-model-loader-p1-v2.json)，完整结果见
[`benchmarks-mem-vm-v2.md`](../../evidence/performance/benchmarks-mem-vm-v2.md)。W1-WSL ext4 evidence 位于
`/tmp/hyle-mem-vm-perf-evidence-r1`，Windows 归档 SHA-256 为
`EFD00C22451D0E2D6E9CA500FCCA2ABE343E454EE2ED0F28DDA4234D57C81A0E`；contract-lock / manifest /
summary SHA-256 为
`3F2FCD4EFA42406E4EB370BB76AAA1A45B23797A4955EF5107CB0427AA2D1AC0` /
`14C90027CBE0715C855E20D9382B73EAB3EE614FF708F090FE2FF02B764191F3` /
`E338BFD642F5254E315769457DA0E748EC7A35B7B6E5354315BDA33AFE2F4878`。deep verifier 为
`QUALIFIED/CURRENT/EXACT`。相同页、syscall、failure、ownership、setup/teardown 已逐项对齐；没有把
Windows whole view、无 advice 或不 teardown 的弱路线混入。

### 独立红队

独立 reviewer `mem_vm_red_team` 于 2026-09-06 对冻结候选做 hostile review。

| Finding | severity | 处置 | 证据 |
|---|---|---|---|
| fully released repeat、zero/subpage batch 曾在判空前分配 | high | `fixed` | O(R) pre-scan 后无分配早退；tests |
| hole-aware access/control 曾线扫全部 fragments | high | `fixed` | ordered table + binary search：O(log F) / O(log F + K) |
| Windows partial release 曾在确定 unsupported 前排序/分配 | medium | `fixed` | validation/page detection 后直接 `Unimplemented` |
| Windows file `kDontNeed` 错用内容可失效的 `DiscardVirtualMemory` | high | `fixed` | file-backed 全 access mode 明确 `Unimplemented` |
| Windows empty flush 升级为 whole-file flush | medium | `fixed` | validated empty span 直接 OK |
| SharedMapping header/alignment/access/name/capacity 合同不自然 | high | `fixed` | headerless explicit size/access + subprocess/three-OS tests |
| `HugeMapping` 把 THP hint 暗示为 hard huge-page | medium | `fixed` | `LargeMapping` taxonomy；requested/accepted/applied 分开 |
| deterministic “中途第 N 次 munmap 失败”无法由公共 OS 合同安全制造 | evidence gap | `accepted` | implementation 每次成功立即更新 survivor，首个失败停止并返回；无 test-only LD_PRELOAD/fault-injection 产品 seam |

红队最终 material remaining finding 为 0。最后一项是证据机制限制，不是已证实产品缺陷：正常、重复、
invalid、unsupported、teardown 与成功 partial state 都有直接测试；为制造系统调用第 N 次失败而扩建公共
instrument 会违反本轮停止规则。

### Aggregate gates

| Gate | Verdict | 依据 |
|---|---|---|
| scope / architecture | `PASS` | 六族 COMPLETE/ROUTED；唯一 OS authority，owner policy 不下沉 |
| correctness | `PASS` | 三平台、subprocess、holes/ranges/OOM/failure、sanitizers |
| delivery | `PASS` | source/static/shared selected Mem 与 direct Sched consumer |
| usability/docs | `PASS` | natural owner/access/range/result；OS 陷阱与借用失效写入 guide |
| red team | `PASS` | 全部 material finding fixed/disproved；remaining medium+ 0 |
| engineering performance | `PASS` | qualified 4/4 5% NI；连续路线零 fragment allocation；复杂度达标 |

## Verification verdict

七项 FIX 全部完成，未处置 `MUST=0`、blocker=0。候选 LOC 由不可变 tree 复算：全库 454,713；自实现
含测试 117,630；自实现不含测试 74,519；Mem-VM compiled implementation 1,318。`CANDIDATE -> VERIFIED`
前 registry snapshot 为
`C:\public\project-hyle\_validation\mem-v6-mem-vm-v2\module-status-before-verified.json`，SHA-256
`FDE8CBEAB0EF0BB63714E0C037EABF357879EC7E150445E27BEB25A45F2773DA`。

## 接受限制与重开条件

接受限制：

- Windows whole-view primitive 没有安全 partial unmap，`release_range(s)` 返回 `Unimplemented`；
  file-backed `kDontNeed` 也不调用可能丢内容的 anonymous/private primitive；
- SharedMapping 的 payload size/readiness/synchronization 是外部 IPC protocol；Darwin/Windows native
  capacity 可按页取整，不能由无 header API 恢复精确 logical size；
- 任何已借出的 raw pointer/span/view 在 owner release/reset/move 后由调用者负责停止使用；checked
  accessor 防止新建跨 hole view，但不追踪既有 borrow；
- public OS contract 不能稳定注入“第 N 次 munmap 失败”，该分支由逐 syscall state-update mechanism
  proof 覆盖；不为测试加入 product fault-injection seam；
- P1 来自 W1-WSL guest 的 Linux ext4 journey，不是 native-Linux/Formal 证书；Windows progressive
  residency、remote NUMA、RSS/page-fault 和端到端 load-to-ready 不由该行外推。

只有以下新事实重开 Mem-VM：

1. 真实 loader/scheduler/IPC consumer 复现 mapping、hole、access、flush、lock、shared publication、
   guarded-stack 或 large-placement correctness/lifetime/易用性缺陷；
2. supported OS 增加安全同合同 partial-unmap/file-discard primitive，或现有 syscall 合同改变；
3. public API/layout、normal metadata/allocation/syscall、Platform/Error/Mem 依赖或交付形状改变；
4. 同合同正常 model-loader workload 证明 material regression，或成熟 peer 证明更优复杂度档；
5. async/direct IO、persistent memory 或 device mapping 获得具名基础 consumer，且现有 owner route 无法组合。

新 benchmark、symbol 数量、命名偏好、单次点值、地址/执行顺序翻转或无新事实复查不构成重开。

## Closure decision

本轮裁决 `CLOSED`：七项冻结 FIX 完成，六族能力全部 `COMPLETE/ROUTED`，全部 aggregate gate `PASS`，
独立红队 remaining material finding 为 0。候选保持
`9604bf4a33f13bd5f84ef78fedce55f305ac9d50` /
`57adb69f185cf02d42ab7f220ebeccbb91170b1b`；terminal 文档不改变产品 identity。

同候选 LOC：全库 454,713；自实现含测试 117,630；自实现不含测试 74,519；Mem-VM compiled
implementation 1,318。raw performance archive 与 hashes 已列在 verification manifest；terminal end
commit 由随后 registry 记录完整 ID。
