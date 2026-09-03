# Mem v3：VM/storage owner 定向迁移

<!-- MODULE CLOSURE RECORD: HISTORICAL -->
<!-- MODULE CLOSURE RECEIPT: CLOSED -->

## Reopen evidence（`REOPENED`）

- component / owner / round / kind：`mem` / Mem / `mem-v3-vm-factorization` / `REOPEN`；
- baseline：`f077a34c73fba624cbcfaedd2b5cf8dd31655847`；前轮 `mem-v2`；
- trigger：`MEM-VM-FACTORIZATION-2026-09-03`；
- 具体新事实：Platform v3 已接受的因子图要求同步 fs 与 host facts 分开，但当前 Mem public mapping
  在 layout 中直接持有两个公开 Platform storage owner；这既是 Mem public dependency/layout，也是真实
  source/static/shared delivery shape 变化，命中既有 reopen trigger；
- 当前授权仅登记 trigger。完整 capability trace、API/owner freeze、有限 FIX、预算和验证矩阵必须在
  `SCOPED/AUDITED` 后才能修改产品。

## Closure card（`SCOPED`）

- component / owner / round / kind：`mem` / Mem / `mem-v3-vm-factorization` / `REOPEN`；baseline 与
  trigger 同上；
- 真实 consumer：model weight mapping/prefault、large NUMA arena、Sched guarded fiber stack、named
  shared-memory transport；北极星为锁定 llama.cpp/ggml model loader；
- 范围：只迁移 mapping/residency/huge/guarded storage 的 owner、public layout/source/header/target，补齐
  同步 populate 旅程，并复签被触及的 Mem product delivery；
- 非目标：allocator/heap/arena/pool/resource/buffer/planner 算法或 public shape、partial-hole mapping、
  async/direct IO、backend policy、Compute/Graph product、Native FFI/C API、stable ABI、公共仪器；
- 预期图：保留现有 `mem` ID 作为其余 Mem factor；新增 `mem_vm -> PUBLIC core,error,mem,platform,
  platform_fs`，普通 `hyle::mem` façade 同时选择 `mem` 与 `mem_vm`；
- tier / cost：`P1`，但只续签受影响的 model-mapping normal journey 与现有 Mem P1 regression；owner
  wrapper 不增加 heap/refcount/virtual dispatch/copy，mapping setup O(1)，populate O(pages)；
- 总调查/实现预算三小时，性能预算三十分钟；每个 family 最多一次机制诊断、两次产品修正；
- machines：W1-WIN correctness/P1，W1-WSL 与 M1-MAC correctness/sanitizer/delivery；instrument
  `closure-perf-v1`，5% non-inferiority；
- peers：Folly MemoryMapping 与 latest llama.cpp mapping journey；OS mapping ABI 是 correctness authority；
- gates：range/EOF/offset/overflow/mode/flush/advice/populate/lock/move/reset/system failure、anonymous/NUMA/
  shared/guarded paths、three-platform sanitizer、exact factor、source/static/shared、direct Mem/Sched consumers、
  docs/FFI scan/red team；
- 停止条件：只执行本卡和后续审计冻结 FIX；若无法建立唯一 owner 或同交付无额外 cost，进入
  `BLOCKED`，不能保留两套 public storage。

## 定向能力与 surface 审计（`AUDITED`）

审计复用了 mem-v2 的完整 allocator/pool/resource/buffer/planner 账本，只重放所有 mapping、huge、
fiber-stack public entries 和真实 consumers。最新 llama.cpp/ggml 增量需要同步 prefault 与可独立释放的
model fragments；前者属于通用 mapping residency，后者不应破坏连续 `bytes()` owner，可由现有
`open/map(offset,length)` 创建多个独立 mapping 表达。因此没有理由建立 holey view 或每页 metadata。

| ID | class | frozen shape / ownership | cost target | action / verification |
|---|---|---|---|---|
| `MEM3-VM01` | `MUST` | `mem::VirtualMapping/FileMapping/SharedMapping/HugeMapping/FiberStack` 是唯一 public storage owners；删除公开 Platform MappedFile/SharedMemoryRegion | inline native state，无 pimpl/refcount/heap/额外 virtual call | `FIX` native-state migration、layout/lifetime/range tests |
| `MEM3-VM02` | `MUST` | `FileMapping::populate(offset,length)` 同步触页；backing file 在 view lifetime 内不得 truncate | O(covered pages)，无 per-page metadata；不是 will-need hint | `FIX` range/empty/subrange/real page-fault journey |
| `MEM3-VM03` | `MUST` | fragment 通过多个 offset/length mapping owners；`DontNeed` 只释放 residency，不制造 hole | 每个 fragment O(1) owner；不添加 hidden segment table | `KEEP/ROUTE` loader policy，guide 说明 |
| `MEM3-VM04` | `MUST` | `mem_vm` 精确拥有 mapping/huge/guarded headers/source；Mem facade 聚合 `mem + mem_vm` | graph acyclic，旧普通 Mem consumer 不多取 fs/thread | `FIX` manifest、tests、direct Sched/IO edge、三交付 |
| `MEM3-VM05` | `MUST` | 受影响 mapping normal journey 与未触及 Mem-v2 P1 rows 回归 | 5% non-inferiority；setup/touch/reset、bytes 全计费 | `FIX` qualification/replay；无同合同则限于结构结论 |
| `MEM3-VM06` | `MUST` | architecture/user/performance/sample/FFI 清洁度跟随唯一 owner | 无 runtime cost | `FIX` docs/scans/red team/LOC |

上述六票是全部 FIX。API freeze 为 `FROZEN`（2026-09-03）：path 或 owned FileHandle 映射、显式
advise/populate/lock、named shared create/open、anonymous NUMA mapping 与 guarded stack 的自然 journeys
和公开头自含均已通过。失败继续使用共享 Status/Result，控制操作外部同步，byte access 的 data race
仍由 caller 负责。没有 C/Native FFI、stable C++ ABI 或兼容 alias；P1 不得反向改变 owner/layout。

六问当前结论：mimalloc 是未触及 Mem factor 的既有 direct product dependency，`mem_vm` 自身只用
OS/Platform mechanisms；推理 mapping 能力在 VM02 落地后完整，partial-hole 被自然分段 owner 路由；
性能已在 API freeze 后进入候选取证；旧 FFI 扫描干净；新 factor 的 source/static/shared 尚待
candidate；C++ taxonomy 保持 mem-v2，用户不需要知道 factor 名。`AUDITED` 只授权以上六票。

## 候选前实现检查点

- VM01--VM04/VM06 已完成，VM05 进入候选取证；Platform storage owners 与旧 convenience 已删除，
  Mem 类型直接持有 native state，`FileMapping::populate` 是唯一新增操作；
- W1-WIN 精确 mem_vm 根下 Mem `223/223`、VM `60/60`；GCC 14 `-Werror` 同为 `223/223` 与 `60/60`，
  两个 factor 的公开头自含通过；
- 该检查点不续签未触及 Mem-v2 families 的新性能冠军结论；当时 M1-MAC、sanitizer、source/static/
  shared、direct consumers、P1 与红队尚未裁决，后续进度以紧接的 Candidate 节为准。

## Candidate（`CANDIDATE`）

- product commit/tree：`e9b9a43d4782c60d84badc02b1a41fbe21207a00` /
  `c6429316279aaac1e0f5d7a7a71872a60cf967b6`；
- VM01--VM06、唯一 storage owner、公开 layout/API 和 mapping/residency normal workloads 已冻结；其余
  Mem-v2 allocator/pool/resource/buffer/planner 算法不是本 candidate 的产品改动；
- W1-WIN、GCC 14 `-Werror`、ASan/UBSan、Mem/VM 公开头自含、Windows exact mem_vm factor 与 Mem
  source/relocated shared/static journeys 已通过；
- M1-MAC、direct consumers、正式 P1、FFI/symbol scan 与最终红队仍待裁决；产品变更必须退回
  `AUDITED`。

首个共同 candidate 因 Platform host 的 Apple `getentropy` 声明缺失退回 `AUDITED`；Mem API、layout
与算法没有发现缺陷，也没有新增 FIX，修复依赖后重放同一验证。

## 第二候选（`CANDIDATE`）

- product commit/tree：`b1a526d6fdb05b6bb42c4cb041788959e74511de` /
  `f808df6fcbd0b5549777e0c6939240a3feab9512`；
- M1-MAC 已重新构建并通过 `mem` / `mem_vm` 的公开头自含与实现测试，同时共同依赖的三个 Platform
  factor 也通过；
- Mem API、layout、算法、mapping/residency workloads 与 VM01--VM06 均未改变。仍待 sanitizer、
  direct consumers、正式 P1、FFI/symbol scan 与最终红队裁决。

## 最终验证矩阵（`VERIFIED`）

验证绑定 immutable product candidate
`b1a526d6fdb05b6bb42c4cb041788959e74511de` / tree
`f808df6fcbd0b5549777e0c6939240a3feab9512`。候选后的合同、文档与测试夹具提交没有修改 Mem/Mem VM
公开头、source、factor graph、API、layout 或算法。

| Gate / journey | Verdict | Evidence / limit |
|---|---|---|
| W1-WIN correctness / headers | `PASS` | 当前 exact roots 的 Mem 与 Mem VM 实现/公开头 4/4；既有 VM/range/lifetime/populate 与未触及 Mem-v2 tests 通过 |
| W1-WSL correctness / sanitizers | `PASS` | 五组件 Release 10/10；ASan 与 UBSan 各 5/5；TSan 在 `setarch x86_64 -R` 下五个实现 binary 均通过且无 race report |
| M1-MAC correctness / sanitizer | `PASS` | AppleClang 21 arm64 Release 10/10、ASan 10/10；依赖的 host entropy 声明修复已包含于最终候选 |
| product delivery | `PASS` | W1-WIN/M1-MAC Mem source embedding、relocated shared、relocated static sample 全部运行通过；普通用户只链接 `hyle::mem` |
| exact factor / direct consumer | `PASS` | `mem_vm` 精确 factor、header owner 与 product 聚合通过；Sched 2/2、IO 11/11；Container 独立 product header/implementation 2/2，未被错误拉入 VM/thread factor |
| engineering performance | `PASS` | [Mem VM P1](../../evidence/performance/benchmarks-mem-vm.md)：mapping 4/4、residency 8/8 均 `PARITY` 且 5% NI `PASS`；manifest 当前为 `QUALIFIED/EXACT/CURRENT` |
| usability / docs / FFI | `PASS` | architecture/user/performance/sample 只给 Mem 一个 storage owner；无旧 Platform alias、C schema/bridge/option/managed sample 或公开 Native FFI export |

### 最终红队

- Platform 公开 storage owner 与旧 convenience 已从产品树清除；历史 benchmark contract 只作历史事实，
  不恢复双 owner；
- `FileMapping::populate` 是显式 O(pages) 同步操作，普通 map 不偷偷 prefault；fragment 用多个 O(1)
  owner，不为 benchmark 建 hole table 或每页常态 metadata；
- `mem_vm` 直接实现只用 OS 与 Platform mechanisms；mimalloc 仍只属于未触及的 base Mem allocator 路径，
  版本/许可/离线锁由 build/package owner 管理；
- source/static/shared 与直接消费者确认 factor 名没有泄露为用户必选项，失败仍统一走 Error；
- candidate 后产品 boundary diff 为空，P1 没有反向改变 API/layout/常态 metadata；
- public/install/export 扫描没有旧 C API 或 Native FFI，private mimalloc `extern "C"` 边界不是 HYLE FFI。

没有未处置的 VM `MUST`、correctness/security/ownership/delivery finding 或 material P1 regression。

## Owner 六问（`VERIFIED`）

1. **第三方实现：**base Mem 直接使用锁定 mimalloc；Mem VM 自身没有第三方实现，只经 Platform/OS
   mechanisms。Folly/Boost/llama.cpp 是锁定 peer/consumer，不链接入 VM 产品。
2. **能力完整度：**本轮 mapping/residency/huge/shared/guarded storage、同步 populate 与 fragment route
   均 `COMPLETE`；allocator/pool/resource/buffer/planner 保持 Mem-v2 已冻结能力，async/direct/backend
   policy 有明确 owner。
3. **性能：**受影响的完整 mapping 与 progressive residency 正常旅程在同语义强 peer 下通过 5% NI；
   wrapper 无 pimpl/refcount/heap/virtual dispatch，setup O(1)、populate/lock/advice O(covered pages/range)。
   不借此重签未触及 Mem-v2 families 或宣称整个 Mem 无条件 SOTA。
4. **旧 C ABI / Native FFI：**公开 surface 为 `CLEAN/N/A`；private mimalloc C linkage 是依赖实现 seam，
   不是 HYLE 外语运行时合同。
5. **构建与导入：**`mem`/`mem_vm` 精确 factor、`hyle::mem` façade、三平台、source/static/shared、
   relocation 与 Sched/IO/Container direct slices 均通过；中央 manifest 保证无环和真实 public edges。
6. **C++ API / 易用性 / 文档：**用户只面对 `hyle::mem` taxonomy；move-only owners、显式
   advise/populate/lock、path/FileHandle mapping、shared create/open、NUMA/huge/guarded journeys 与 lifetime/
   external synchronization/truncate 陷阱已在 guide 中按场景说明。

VM01--VM06 全部 `COMPLETE`，scope、architecture、correctness、delivery、usability/docs、red-team、
engineering-performance 七门全部 `PASS`。`CANDIDATE -> VERIFIED` 使用与 Platform 共用的前态 registry
snapshot：`C:\public\project-hyle\_validation\platform-v3\module-status-before-verified.json`，SHA-256
`94928BBC6B3FA987F64DDCC748CB8B3C5E15E775FC8322FEE8D734924350FC8D`。当前只提升为
`VERIFIED`；terminal receipt 与 LOC 在下一步单独登记。

## Closure decision 与 terminal receipt

本轮 terminal verdict 为 **`CLOSED`**。VM01--VM06、唯一 storage ownership 与所有适用工程门已
`COMPLETE/PASS`；没有重开或偷偷改写 Mem-v2 未触及的 allocator/pool/resource/buffer/planner 产品
合同。Formal 保持 `PENDING`。

已知限制：

- mapping owner 要求 backing file 在 view lifetime 内不被不安全 truncate；byte access 的 data race、
  control-operation sequencing 与 shared-name 权限仍由调用者协调；
- `populate` 是显式同步 O(pages) 触页，residency/lock 受 OS quota 与调度影响；ordinary map 不隐式
  prefault，fragment 由多个 owner 表达而非一个 holey mapping；
- huge page、NUMA 与 guarded stack 受 OS/hardware/privilege 能力约束；单节点测试不证明跨 socket
  remote/local 带宽；
- base Mem 仍直接依赖锁定 mimalloc，升级若影响算法、ABI/codegen、license 或交付必须定向重开；
  Mem VM 自身不引入该依赖；
- 当前 P1 只证明 W1-WIN 冻结 mapping 与 progressive-residency loader journeys 的 5% NI，不续签所有
  Mem-v2 family、不比较 async/direct I/O，也不声明端到端 token latency 或 universal SOTA。

只有下列新事实可以重开 Mem/Mem VM 的受影响部分：

1. 具名真实 consumer 复现 mapping/residency/shared/huge/NUMA/guarded storage 的 correctness、security、
   lifetime、OOM/failure 或 C++ 易用性缺陷；
2. 支持 OS/compiler 击穿 offset/range/EOF/move/reset/populate/lock/flush/advice/guard 合同；
3. public API/layout、唯一 owner、factor dependency、正常成本或 source/static/shared delivery materially
   改变，或 mimalloc/Platform 依赖改变实际产品语义；
4. mature same-contract peer 证明更优复杂度档，或真实 loader/scheduler profile 复现 material regression；
5. async/direct/backend policy 或 sparse/holey storage 被正式准入，并证明当前多个自然 owner 无法表达
   共同 Mem-owned journey。

新 API 数、单次页缓存点值、命名偏好、LOC、未改变产品输入的下载位置或无新事实复检不构成重开。

Terminal receipt：

- product candidate/tree：`b1a526d6fdb05b6bb42c4cb041788959e74511de` /
  `f808df6fcbd0b5549777e0c6939240a3feab9512`；final verification evidence commit 为 `10fbb9ef`，
  P1 harness source 为 `28987e31`；
- immutable candidate 的四项 LOC 由
  `python scripts/loc-report.py --module mem --commit b1a526d6fdb05b6bb42c4cb041788959e74511de --self-check --json`
  重算通过：全库 `434,612`、自实现含测试 `110,971`、自实现不含测试 `70,261`、Mem conventional
  boundary `7,392`；该数统一计入 `include/hyle/mem` 与 `src/mem`；
- `VERIFIED -> CLOSED` 前态 snapshot 位于
  `C:\public\project-hyle\_validation\platform-v3\module-status-before-closed.json`，SHA-256
  `BD58D0F5CECAF1067330E4731B2F67CC2F090B609F9AAFD41B1AFCB1A3131B30`；
- end receipt commit：`865413af2255cc58061943b556efe7d98d39a7ac`；registry anchor commit：
  `9c18057bc9e895558594f2b7f61cfeaafefc6548`，后者是首个真实保存本轮 `CLOSED` state/candidate、
  terminal fields 与人读投影的中央快照；
- anchor 校准前态 snapshot 位于
  `C:\public\project-hyle\_validation\platform-v3\module-status-before-anchor.json`，SHA-256
  `7946752F902F797CFA17A9C6C5A9E698827AFD71F656D47CC1AEE9EC8F3B435D`；本次 evidence-only
  校准不改变产品、状态或 verdict。
