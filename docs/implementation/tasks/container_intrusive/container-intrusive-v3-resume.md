# container_intrusive v3 收口恢复记录

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 本页是当前 `RESUME` 轮的施工记录。权威状态是
> [`module-status.json`](../../module-status.json)；v2 的 `HISTORICAL` 记录只提供旧候选和阻塞事实，
> 不授权本轮修改，也不产生当前结论。

## Closure card（`VERIFIED`）

- component / owner / round ID：`container_intrusive` / container owner / `container-intrusive-v3`
- task kind：`RESUME`
- 前一 round：`container-intrusive-v2`
- 前一 blocker：冻结候选没有绑定 `closure-perf-v1` 的 W1-WIN P1 engineering `PASS`，最终独立
  red-team 也没有签署 candidate identity。
- 前一 resume condition：在冻结的 5% margin 下生成一份 candidate-bound W1-WIN P1 manifest，随后在
  不迎合 benchmark 改产品形状的前提下取得最终 identity/red-team verdict。
- baseline commit：`a5c74e2be31d258c67e3d6244c8a24d3abeb21c0`
- 进入原因与 demand/defect ID：`INTR-RESUME-PERF-01`（旧工程性能 blocker）、
  `INTR-CURRENT-DRIFT-01`（依赖升级和 build/package v2 后重新绑定当前产品/交付 identity）、
  `INTR-CONSUMER-REVIEW-01`（按真实消费者与成熟库形状复查能力、API 和 negative space）。
- 真实 consumer task / normal workload：`time::TimerEntry` 的显式 safe retirement；task/timer/
  completion/object-pool 中 stable-address、caller-owned node 的 FIFO drain、已知节点取消、遍历、整链/
  range handoff 和 owner/node retirement。
- 范围：公开 auto/safe member-hook 双向链表的能力、C++ API、ownership/lifetime/concurrency/failure
  合同、算法复杂度与常态 layout；`core` 直接依赖、`time` 直接消费者和独立 component 的 source/
  static/shared 交付；现有统一性能合同及用户/架构/证据文档。
- 非目标：完整 owning/concurrent `container`、time 算法、sched/io 私有 link 状态机；node reclamation、
  内部同步、offset pointer、完整 Boost.Intrusive 策略/算法矩阵；C API、Native FFI；Formal release
  certification。
- 直接依赖 / 直接消费者：`PUBLIC core` / `PUBLIC time`；完整 `container` 不聚合本组件。
- tier 与理由：`P1`；公开容器 primitive 的 mutation/traversal/retirement 会在 runtime 高频执行。
- material margin / resource budget：同合同 Boost peer 的每个 required workload 采用冻结的 5%
  material/non-inferiority margin；64-bit auto/safe hook 和 list 各不超过两个 pointer；热操作零分配、
  无锁、无 syscall，已知节点 mutation/range relink 保持 O(1)，遍历/clear/consume 保持 O(n)。
- 总调查预算 / 性能预算：人工审计与归因最多 120 分钟；机器采样只执行 contract 固定阶段，不以墙钟
  截断必需样本。性能最多一轮 `probe`、一轮 fresh n=15 `qualify-platform`；只有统一仪器停止规则允许
  时自动进行全新 n=31。
- 最多复现、机制诊断、产品修改次数：一次正常运行；合同内 order/placement strata；一次 codegen/
  profile 机制诊断；至多一个有机制与消费双证据的产品假设及重跑。不得发明额外 benchmark 姿势。
- machine profile：`W1-WIN` 是 P1 engineering 裁决机；`W1-WSL`、`M1-MAC` 只承担适用 correctness/
  portability 复核，不池化性能样本。
- instrument profile：P1 使用 `closure-perf-v1`；correctness/delivery 使用本轮冻结的风险矩阵和现有
  component/package 合同。
- strong peers：Boost.Intrusive commit `df8eeb90846648464e3c8e378e312f0cec772e83`（同合同能力、API、
  算法与 P1 peer，local source `C:/public/project-hyle/boost`）；Folly commit
  `423fd7f8d97fd7bf52fc4c986dfd20320c4de8f9`（消费者形状和能力查漏，local source
  `C:/public/project-hyle/folly`）。每项 capability 最多使用这两个预选 peer。
- 无同合同 peer 时的预算及禁止声明：不适用；Boost 提供 auto/safe member-hook 同合同 peer。Folly
  只用于形状查漏，不把不可对齐入口计入胜负。
- correctness gates：normal、empty、range/same-list boundary、move/swap/destroy/lifetime、callback
  reentrancy contract、random differential、Debug misuse、ASan/UBSan、supported compiler ABI/codegen、
  public-header self-containment 和 `time` 代表消费者。
- delivery gates：当前 component graph/direct target edge；source standalone；relocated static/shared
  SDK consumer；header-only 组件不得制造 binary artifact；不存在 Native FFI surface。
- performance gates：`bench/contracts/container-intrusive-p1-v1.json` 的全部 required comparable
  workloads/strata/controls；contract、peer、candidate、build 和 artifact identity 必须绑定。
- 预期输出：有限 capability/gap ledger 与 `FIX` 集；当前 candidate commit/tree；correctness、delivery、
  docs/usability、red-team 与 P1 manifest；四项 LOC、限制和 reopen triggers。
- 停止条件 / 可接受限制：所有适用工程门通过则 `CLOSED`；仪器/机器/功效仍不能产生 P1 `PASS` 则写
  一个可点名 blocker 后 `BLOCKED`。可接受限制是 64-bit、stable-address、外部同步、caller-owned
  direct-member hook，以及编译器受支持的 pointer-to-data-member ABI；没有真实独立需求时不扩成完整
  intrusive framework。

## Capability / gap ledger

| ID | requirement class | demand kind | exact source | normal shape | owner | ownership/failure/concurrency | complexity/resource target | verdict | severity | verification / negative space |
|---|---|---|---|---|---|---|---|---|---|---|
| INTR-LIFE-01 | capability | direct consumer + peer parity | `time::TimerEntry`；Boost.Intrusive auto/safe member hook；Folly `IntrusiveList`/`SafeIntrusiveList` | auto-unlink 用于 unknown-owner cancellation；safe-link 用于 owner-lock 下显式 retirement | public | caller owns storage；stable address；外部同步；safe 违反析构前摘链合同即程序错误 | 两 pointer/hook；mutation O(1)；零分配 | `KEEP` | P0 | normal/empty/lifetime/Debug misuse；不增加 owner metadata |
| INTR-LINK-01 | capability | timer/completion handoff | `TimerQueue`、scheduler due drain；Boost range splice/clear-and-dispose | push/pop/remove/iterator、单项/整链/range splice、append、`consume_all`、遍历、list move/swap | public | callback 合同逐入口表达；list 不拥有或销毁 node | known-node/range relink O(1)；drain/traversal O(n) | `KEEP` | P0 | deterministic + random differential、codegen；不扩成 owning list |
| INTR-ABI-01 | API/implementation | peer defect | Boost `parent_from_member` 不要求 standard-layout；Folly `TimedMutex::MutexWaiter` 是 public hook + private payload | direct member hook 可位于 non-standard-layout complete object；只要所选 PMD representation 可解码 | public | live complete object、stable address；未知/复杂 PMD ABI 编译期 fail closed | node recovery O(1)，无常态 metadata | `FIX` | P0 | 删除 blanket `is_standard_layout<T>`；三编译器 mixed-access node test |
| INTR-AUTO-SWAP-01 | capability/API | real peer consumer | Boost hook `swap_nodes`；Folly movable `ChannelCallbackHandleHolder` | auto list specialization 提供 typed static O(1) `swap_nodes(T*,T*)`，交换同一 selected member contract 的 membership/链上位置；不改变默认 move 语义 | public | self、linked/unlinked、相邻、同链/异链均定义；类型系统阻止 ready/timeout 等不同 member hook 误换；只改 topology，业务 owner/count/state 镜像须由调用方在同一锁/事务更新；被交换 node 的 iterator 失效 | O(1)，零分配 | `FIX` | P1 | table + random differential；hook/safe list 不获得 public untyped 入口 |
| INTR-SAFE-SWAP-01 | negative space | lifecycle boundary | safe hook 只能经 owner list/state machine 改 membership | 不提供 hook-level safe swap；未来真实 movable-safe consumer 可设计 list-level `replace` | public future | 不允许绕过 owner state machine | 若重开必须 O(1) 且无 owner tax | `REJECT` hook / `DEFER` list replace | P1 | concept negative test；本轮无 consumer |
| INTR-SIZE-01 | negative space | peer scan | Folly counted/uncounted list；Boost constant-time-size policy | 不提供 linear `.size()`；冷路径用 `ranges::distance`；counted safe list 等独立 consumer | public future | auto unlink 与 hidden count 冲突；range splice 需 scan 或 caller count | 不让所有 list 支付 word/write；不伪造 `sized_range` | `REJECT` linear / `DEFER` counted | P2 | time/sched 当前计数都不是通用 list membership count |
| INTR-API-01 | usability | current consumers | time/sched 持有 node pointer；nullable front/pop 与 `try_*` | mutation 保持 pointer API；不复制一套 reference overload | public | 普通入口前置条件；`try_*` 只处理 null/already-linked | 无额外分支或 metadata | `KEEP` | P1 | user guide decision table；不把 `try_*` 伪装成 owner validator |
| INTR-CORR-01 | correctness | red-team gap | randomized differential 目前只在两条 list 间移动 | 增加同链 range splice 的 empty、boundary、prefix/suffix、前移/后移表格和有限随机差分 | tests | 无效的 inside-range position 仍是前置条件违反 | 不改产品复杂度 | `FIX` | P0 | deterministic + differential tests；sanitizer |
| INTR-DELIVERY-01 | delivery/docs | direct target audit | component graph：`time -> container_intrusive`；完整 `container` 没有直接 edge 且不安装该 header | direct consumer 必须选择/链接 `hyle::container_intrusive`；只有 `time` 自动传播 | build/docs | 不能依赖 `container -> mem -> platform -> time` 的偶然传递闭包 | exact selected closure，无 binary artifact | `FIX` | P0 | source/static/shared relocated component + time consumer smoke |
| INTR-USE-01 | usability/docs | real C++ type shape | `TimerEntryListAccess` friend adapter；public-hook peer nodes | pointer-to-member alias 形成处必须能访问 hook；公开 hook 最简单，私有 hook 用具名 friend adapter | public docs | adapter 只暴露 hook selector，不改变 storage ownership | compile-time only | `FIX` | guide sample/note + mixed-access compile test |
| INTR-PERF-01 | performance evidence | real consumer replay | `TimerQueue::inspect`、scheduler due list 使用 `consume_all`；Boost `clear_and_dispose` | 4096 stable-address preallocated batch replay；HYLE/peer 都使用各自正常批量 drain API | evidence | 不把合成 component replay 标成 actual consumer benchmark | 5% P1 non-inferiority；统一 8 strata/controls | `FIX` | P1 | workload tier 改为 `replay`；FIFO drain 改为 `consume_all` vs `clear_and_dispose`；legacy diagnostic parser/示例同步；current evidence page 与 fresh W1 qualification |
| INTR-POLICY-01 | negative space | peer breadth scan | Boost base/function/offset hooks、完整 list algorithms | 保持窄 member-hook linking primitive；高级 intrusive framework 直接用 Boost | external/public future | reclamation、allocator、内部同步归 owning/concurrent layer | 不因 peer API 数量或 LOC 扩面 | `REJECT` | P2 | 只有具名 consumer/更优复杂度才重开 |
| INTR-FFI-01 | delivery boundary | Native FFI policy | template + pointer-to-member + caller-owned C++ lifetime | 仅 C++ source/static/shared；Native FFI=`N/A` | public | 不导出 raw hook/list layout | header-only | `KEEP` | P0 | component smoke；没有 C/FFI symbol |
| TIME-BACKLOG-01 | adjacent module | semantic consumer scan | time 内旧 `TimerEntryAccess`/`CommonTimeoutQueueAccess` 与 C-adapter 注释 | 在 time 自身 round 删除死 seam/过时注释 | time | 不改变本轮 intrusive candidate | N/A | `DEFER` | P2 | 登记但禁止跨 owner 顺手修改 |

- 公开入口反向追踪：`IntrusiveListHook`/`SafeIntrusiveListHook`、两种 list 的全部 mutation、iterator、
  callback 与 lifetime 入口均落入 INTR-LIFE/LINK/AUTO-SWAP/API；没有无法归属的公开入口。
- 语法消费者 / 语义重复扫描：唯一直接通用 consumer 是 `time`；sched waiter、IOCP、cancellation hlist
  与 pool free-list 的 storage/state-machine 合同不同，保留特化，不建立虚假依赖。
- public/internal/higher-layer 裁决：本组件只拥有通用 caller-owned 双向 member-hook primitive；deadline
  prefix、wake chain、reclamation、同步、allocator 和 owning algorithms 归上层。
- 本轮冻结 `FIX` 集：INTR-ABI-01、INTR-AUTO-SWAP-01、INTR-CORR-01、INTR-DELIVERY-01、
  INTR-USE-01、INTR-PERF-01。只有 correctness/security/当前构建阻断可进入本轮；其他发现登记 backlog。
- 产品修改只有前两项；其余是测试、交付说明或证据合同修正。总计仍受 closure card 的一次机制归因和
  一轮 fresh qualification 约束。
- 相邻模块 backlog：TIME-BACKLOG-01 留给 time round，不在本轮修改。

## Candidate

当前产品候选固定为：

- commit：`b5816b810d536f2041520e4b370c1e17d3ac7a4a`
- tree：`3bf043c40dbb5e4b49aba563f23f71b3c2d7caf3`

候选只实现冻结 ledger：auto list 的 membership transfer 现在只能通过
`IntrusiveList<T, Hook>::swap_nodes(T*, T*)` 进入，raw hook surgery 保持 private，因此同一对象的
ready/timeout 等不同 member contract 不能互换；safe list 不提供该入口。操作只交换 hook topology，
不替调用方更新 owner/count/state/deadline 等业务镜像；这些镜像和 move-assignment policy 必须由消费方
在同一外部锁/事务中维护。该路径保持 O(1)、零分配、零常态 metadata。

direct member hook 不再用 `standard_layout<T>` 粗暴排除 mixed-access 类型；实现只接受能够由当前
64-bit 编译器 ABI 安全解码的 pointer-to-data-member representation，对未知/复杂表示编译期
fail closed。合同仍要求 live complete object、direct member 与 stable address。

测试补齐同链 range-splice 边界、确定性 swap topology 和 4000-step 固定种子 owning-model differential；
交付说明明确独立组件由直接消费者选择/链接，`time` 的 PUBLIC edge 自动传播，而完整 `container`
不聚合该 target。benchmark 只校正为冻结的真实 replay 形状：HYLE `consume_all` 对 Boost
`clear_and_dispose`；产品 API/layout 没有由 benchmark 反推。

候选绑定验证：Windows MSVC Debug/Release 与 clang-cl Release focused gate、WSL GCC 14 Release +
ASan/UBSan、macOS AppleClang Debug + WERROR 均通过；intrusive 与直接 `time` 消费者的 source、relocated
static/shared SDK smoke 均通过。header-only intrusive SDK 为 13 个文件、0 个 binary artifact；parser、
contract、docs 与 closure checker 通过。上述验证均绑定本段 commit/tree，性能 instrument identity
另由最终 manifest 绑定，不能混作 product identity。

第一次实现尝试 `e172d24df9cd3afd324f6b700d222c8e9fca27b9` 没有进入 registry candidate：最终红队发现
public untyped `IntrusiveListHook::swap_nodes` 能把同一 node 的 ready/timeout 等不同 member contract
互换，链仍完整但后续 `node_of` 使用错误 offset；同时测试未兑现 ledger 的 randomized swap differential。
因此该 identity 被拒绝，随后按 INTR-AUTO-SWAP-01/CORR-01 改为上述 typed list surface。它已经启动的
W1 n=31 在发现产品 blocker 后终止，不能作为当前轮性能证据，也不消耗 replacement candidate 的
唯一 fresh qualification。

## Verification manifest

所有记录均绑定 candidate `b5816b810d536f2041520e4b370c1e17d3ac7a4a` / tree
`3bf043c40dbb5e4b49aba563f23f71b3c2d7caf3`；不得继承 v2 点值、dirty 结果或被拒绝的 `e172d24`。

| correctness risk | verdict | candidate-bound evidence |
|---|---|---|
| normal/empty、auto/safe lifetime、move/list swap、typed node swap | PASS | Windows MSVC Debug/Release、clang-cl Release focused suite 各 5/5；单元测试覆盖 self/unlinked/singleton/adjacent/same-list/cross-list |
| range 与 same-list boundary | PASS | deterministic prefix/suffix/empty/boundary 表格 + fixed-seed differential |
| wrong member contract | PASS | public raw hook swap 不可调用；公开 typed static API 同时绑定 `T` 与 member hook；safe negative concept test |
| random topology/state | PASS | 4000-step owning-model differential 覆盖两条 list、linked/unlinked、same/cross-list swap 与移动 |
| mixed-access/non-standard-layout member recovery | PASS | MSVC、clang-cl、GCC 14、AppleClang 的 mixed-access live complete-object test；未知 PMD representation 编译期 fail closed |
| callback/lifetime reentrancy | PASS | `consume_all` 在 callback 前摘当前 node；callback `noexcept`；owner teardown 与 node retirement 的 opaque phase observation + actual-object codegen gate |
| misuse diagnostics | PASS | Debug safe linked destruction/reinsert 前置条件 fail-fast；Release 正常 safe destruction 无常态检查路径 |
| sanitizer | PASS | WSL GCC 14 ASan 4/4、UBSan 4/4；instrumented build 不运行无意义的 disassembly gate |
| header/ABI/codegen | PASS | public-header self-containment；MSVC/clang-cl/GCC/AppleClang；64-bit/two-pointer layout 与 supported PMD ABI gate |
| concurrency | NOT_APPLICABLE | primitive 明确外部同步、无内部 atomic/lock；同步和 reclamation 属 owner module |
| allocation/failure | NOT_APPLICABLE | caller-owned、零分配、无返回失败；无效 owner/range/lifetime 是调用方前置条件违反 |

平台结果：

- Windows：MSVC Debug/Release 与 clang-cl Release focused gate 各 5/5；benchmark primary/placement
  build、23 个 intrusive evidence parser case、`bench-stat` parser、contract/schema/docs/closure checks 通过。
- WSL：GCC/G++ 14.2.0 Release + WERROR 5/5；ASan 4/4；UBSan 4/4；scratch
  `/root/hyle-scratch/container-intrusive-b5816b8-wsl-20260827-0355`。
- macOS：Mac16,12 arm64、AppleClang 21.0.0.21000101、Debug + WERROR focused 5/5，unit 33/33；
  focused log SHA-256 `57b78e77233b45c230e291ef14e36f3aea4d170b688cd86d0e47f4108b6453cc`。

交付结果：

| closure | source | relocated shared | relocated static | artifact contract |
|---|---|---|---|---|
| `core,container_intrusive` | PASS | PASS | PASS | 13-file SDK，0 binary；direct consumer 选择/链接 target |
| `core,container_intrusive,time` | PASS | PASS | PASS | `time` PUBLIC edge 传播；真实 `TimerEntry` consumer 编译/运行 |

Windows scratch 为
`C:\hyle-scratch\container-intrusive-v3-b5816b8-win-delivery` 与
`C:\hyle-scratch\container-intrusive-v3-b5816b8-win-time-delivery`。macOS source/shared/static logs 的
SHA-256 分别为 intrusive
`b15b6ad1bcf8321748263122f89ada78699e249b94f78f4d736e8c2c7a687723` /
`8fc07d16c91fd9950b530b980af3aeeed0b581851efd1995ece319a6edc8bae8` /
`060e9c55a748b1b06dd5ed0b9f626c6f962dbe050aa6ff9cce89e34b9f531780`，time
`c67380f2901ef70de3efba0fe2958b148ba70765f613823bda885dce7667b4de` /
`69a5d06edbe942a261c93c33e41e89f9beca5645787d7d78982d123a97112521` /
`f8e5bf93dc704cbbb582c052432a22783aeb339cc130e8390dd2a5069936e52d`。

P1 工程性能：

- contract SHA-256：`5a0f4fb64811616a472988fa6480ff2b61cd5a9593e88cfaa522e345f4e0f2a8`；
- manifest：`C:\hyle-scratch\container-intrusive-v3\perf-qualify-b5816b8-output\manifest.json`，
  SHA-256 `3f081670022973a1b07ea6b4c04ab93b8d8fcbad61feef8bc6c584f30a10bd70`；
- W1-WIN、fresh MSVC x64 Release/static、portable ISA、LTO off；Boost commit/tree 与 candidate identity
  均 clean/stable；artifact/build validator 均 PASS；
- A/A、5%/10% known delta、timer floor、placement controls 全 PASS；n=15 只按停止规则因功效扩成
  fresh n=31，两阶段不合并；
- 9 个 required replay workload × 8 strata 全部 comparable，5% non-inferiority 9/9 PASS；六项 combined
  exact PARITY，三项 INCONCLUSIVE，无 required NI fail；summary/observations SHA-256 分别为
  `f7c0b557fe81009c286f0f97b56d7b1dab7d274bd546efe08fd46d18e9db6fb4` /
  `a6d1af6b796f586d89aded4afd89d792b29e66769bd15daa7728758cc55480d0`；
- `python scripts/hyle-perf.py verify --contract bench/contracts/container-intrusive-p1-v1.json
  --manifest <path>` 返回 `qualification=QUALIFIED`、`paths_verified=true`。详细表和证据保留边界见
  [performance evidence](../../evidence/performance/benchmarks-container-intrusive.md)。

易用性/文档：architecture、user guide、component/package guide、性能页与本 task 已同步；决策表明确
何时选 auto/safe、pointer mutation、typed swap、callback、owner 镜像、iterator invalidation、私有 hook
friend adapter、缺少 `.size()`/safe swap 的理由及 Native FFI=`N/A`。`check-docs.py` 和
`check-module-closure.py` 通过。

## 独立红队

能力/consumer/peer 审计由两路只读 reviewer 独立完成。被拒绝的第一候选没有被“修文档”放行；替换候选
`b5816b8` 的最终独立红队 verdict 为 `PASS`：确认 raw hook swap 私有化、typed API、random differential、
owner/state 责任、v2/v3 证据隔离和 benchmark/contract/parser labels 全部一致；未发现新的产品、测试、
target/build、文档或性能合同 regression，也未发现 ledger 外扩面或 benchmark shaping。

性能 manifest 另由未参与采样/实现的 reviewer 对 candidate/peer/contract/host/build/artifact identity、
controls、validators、9×8 verdict、hash 与 n=15→31 停止规则做只读复核，verdict 为 `PASS`：n=15
controls 全 PASS，仅 auto/safe cancel 的 combined NI 为 `UNRESOLVED`，其余七项 PASS 且没有 FAIL，
所以完整 fresh n=31 合法；最终 72/72 strata NI PASS，2,232 observations 的 run key 完整唯一；473 个
证据引用零缺失、零 hash mismatch。最接近边界的 CI lower 为 `0.956264795`，仍高于 5% NI 阈值
`0.952380952`。该签署只证明 P1 engineering，不签 Formal 或“全面快于 Boost”。

## Closure decision

当前为 `VERIFIED`，七项 aggregate gate 均为 `PASS`，没有 terminal receipt。下一步只记录 closure
限制、reopen triggers、immutable candidate LOC 与 receipt；不得再改产品或证据合同。
