# Timer queue v1：Time 产品的 node/data-structure 因子

<!-- MODULE CLOSURE RECORD: HISTORICAL -->
<!-- MODULE CLOSURE RECEIPT: CLOSED -->

## Closure card（`SCOPED`）

- component / owner / round / kind：`timer_queue` / Time / `timer-queue-v1-factor` / `NEW`；
- baseline：`f342f8d92cebaae9f4870fe05e78afe6c6ca1391`；进入原因：已接受
  `B-F06 SPLIT_PRIME` 把 caller-owned timer node/data structure 与 host clock artifact 分开；
- 真实 consumer / normal workload：`src/sched/scheduler.cc` 每 worker 一个 queue，stable scattered nodes，
  mixed deadlines，大量 arm/cancel，park 前 advance 并 drain callback/data；IO `TimedSlot` 嵌入
  `TimerEntry` 参与 I/O-vs-timeout 仲裁；
- 范围：`TimerEntry/TimerList`、exact-ns radix `TimerQueue`、one-duration
  `CommonTimeoutQueue`；将现有公开 header/test/bench 所有权迁到独立 header-only factor；
  用已冻结 Time provider 补齐不改 layout 的 clock-composition overload；
- 非目标：callback 执行、owning callable、fiber/thread wait、cross-thread command、periodic policy、
  cancellation propagation、total-order guarantee、为 benchmark 增加 index/heap metadata、Sched/IO 状态机重写、
  C/Native FFI、稳定 C++ ABI、仪器改造与 Formal certification；
- 直接依赖：`core`、`time`、`container_intrusive` 全是 public header/type 边，无第三方
  产品实现；直接消费者：Sched implementation/public forward-type route、IO public `TimedSlot`，
  以及 consumer-facing `hyle::time` facade；
- tier：`P1`；normal primary 继续是当前 scheduler N=64，45/64 cancel，64 个 advance tick，
  19 个 delivery/callback observation 的完整 lifecycle replay，对照 Seastar `timer_set`；
- 语义档位：arbitrary mixed deadline 用 `TimerQueue`；一个单调 owner 上的单一 duration
  用 `CommonTimeoutQueue`；需要全序/indexed update 时组合 Container heap；执行/wait/cross-thread
  使用 Sched/IO service；
- 常态成本：queue/node 零分配、零 lock/atomic/syscall；`TimerEntry` 只保留 hook/deadline/
  payload/callback/exact owner；radix arm/cancel/next O(1)，advance O(touched buckets + delivered/
  rebucketed)；common arm/cancel/next O(1)，advance O(delivered)；provider overload 不增对象 metadata；
- margin / budget：W1-WIN `closure-perf-v1` 对同合同 Seastar 5% non-inferiority；总调查/
  取证预算 2 小时/性能 45 分钟；异常后最多一次复现/反序、一次 codegen/profile、
  一个产品修改和一次重跑；
- machines / instruments：`W1-WIN`、`W1-WSL`、`M1-MAC`；engineering
  `closure-perf-v1`，Formal `PENDING/formal-release-v1`；
- strong peers：Seastar `991cf9a908482a3f11ca2fc34a1b3970a06d8622`
  `timer-set.hh/timer.hh`（arbitrary deadline）；libevent
  `d82464a277d0f42703702c4dfd9af6af38595a83` `event.c common_timeout`（single duration）；
- provenance：Scheduler/IO 为 `REAL_CONSUMER`，libevent serving shape 为
  `PROVISIONAL_ARCHETYPE`，旧 synthetic microbench 只作 `DIAGNOSTIC_SYNTHETIC`；
- correctness gates：normal、empty/zero/past、full 64-bit/radix boundary/overflow、wrong owner/invalid order、
  reschedule/cancel/cancel-all、queue-first/entry-first/list lifetime、million-operation oracle，ASan/UBSan；OOM
  对 caller-owned zero-allocation factor `NOT_APPLICABLE`，concurrent operation 因 single-owner 合同 `NOT_APPLICABLE`；
- delivery gates：factor header self-containment/macro composition，`hyle::time` product source/static/shared，
  developer `HYLE_DEV_FACTORS=timer_queue`，Sched/IO direct consumer，三平台，无伪 binary/FFI 反证；
- 预期输出：独立 header-only target/test/bench owner，真实直接边，不变的用户
  `hyle::time` facade，provider 组合，重签 current scheduler P1 证据与同步文档；
- 停止条件：完成审计冻结的有限 `FIX` 后裁决；不研究亚纳秒单操作排名，
  正常 consumer gate 无资格或未越过则 `BLOCKED`，不修公共仪器。

`SCOPED` 只冻结任务合同；在 capability/public-surface 反向追踪和有限 `FIX` 集完成前，
不授权产品修改。

## Capability-family coverage map（`AUDITED`）

| Family | journeys / evidence | required variants / lifecycle | current route | coverage | missing edge / negative space |
|---|---|---|---|---|---|
| caller-owned node/list | Sched sleep、IO timed slot、event loop | stable address、exact owner、payload/callback hand-off、delivery list、entry/queue-first teardown | `TimerEntry/TimerList` | `COMPLETE` | callback 只是借用的 consumer dispatch metadata，queue 不执行 |
| mixed arbitrary deadline | per-worker timer service，Seastar timer-set | arm/reschedule/cancel/advance/next/cancel-all，past/full-range、high cancel | radix `TimerQueue` | `COMPLETE` | cross-bucket total order 不承诺；需要时组合 sort/heap |
| one-duration timeout | server idle timeout，libevent common timeout | fixed duration、nondecreasing arm、transactional reject、FIFO、cancel/drain/teardown | `CommonTimeoutQueue` | `COMPLETE` | duration router 由 IO/server 拥有 |
| clock composition | deterministic timer tests/runtime | system/manual/ref 采样，批次只读一次 now | 只接受 explicit raw/typed now，convenience 固定 system | `PARTIAL` | 缺已冻结 `MonotonicClockSource` overload |
| execution/wait/cancel policy | Sched/IO runtime | callback thread、wake exactly once、cross-thread command、periodic/budget propagation | Sched/IO | `ROUTED` | 不给 node/queue 增 lock、waiter 或 owning callable |
| factor/delivery | host-only、timer-only、Time product users | independent header owner/test/bench，source/static/shared product closure | 当前混在 compiled `time` | `MISSING` | 缺 target/header/test/bench/direct-edge/facade 拆分 |

### Capability / gap ledger

| ID | class / source | normal shape / owner | ownership/failure/concurrency | complexity/resource | coverage | action / severity | verification / negative space |
|---|---|---|---|---|---|---|---|
| `TQ1-NODE-01` | MUST / Sched+IO | stable `TimerEntry` + detached `TimerList` / timer_queue | caller owns; one source; wrong owner/lifetime fail-fast; callback borrowed, never invoked here | fixed per-node metadata; zero allocation/atomic/lock | `COMPLETE` | `KEEP` | lifecycle/death/consumer tests; callback execution excluded |
| `TQ1-RADIX-01` | MUST / Sched+Seastar | exact 65-bucket queue / timer_queue | single owner; reschedule/cancel idempotence; monotonic advance | O(1) arm/cancel/next; output-sensitive advance | `COMPLETE` | `KEEP` | million oracle + boundary suite + P1 replay |
| `TQ1-COMMON-01` | GENERAL / libevent+serving | one-duration ordered intrusive queue / timer_queue | single owner; rejected reschedule preserves arm | O(1) arm/cancel/next; O(fired) advance | `COMPLETE` | `KEEP` | million oracle/order/lifetime; no production-speed claim |
| `TQ1-CLOCK-01` | MUST / Time injection | provider overloads for construct/schedule/advance / timer_queue | provider borrowed for one call; same monotonic domain | template inline; no object/layout cost | `PARTIAL` | `FIX` / blocker | manual/system/ref compile+runtime tests |
| `TQ1-FACTOR-01` | MUST / accepted graph | header-only `hyle_timer_queue`; `hyle::time` facade aggregates / build | public core/time/intrusive; actual Sched/IO edges | no binary; host-only closure drops intrusive | `MISSING` | `FIX` / blocker | selected factor/product, header ownership, direct-edge contract |
| `TQ1-PERF-01` | MUST / current Scheduler | N=64 scheduler lifecycle vs Seastar / timer_queue | identical stable nodes/events/drain/accounting | 5% W1-WIN P1 | `PARTIAL` | `FIX` / blocker | renew current scheduler provenance and peer identity; qualified replay |
| `TQ1-POLICY-01` | GENERAL / runtime | callback/wait/periodic/cross-thread / Sched+IO | owning runtime owns lifetime/race/failure | no policy cost in data factor | `ROUTED` | `DEFER` | negative dependency/API scan |

### Public-surface 反向追踪与有限 `FIX`

- `timer_entry.h`：`TimerEntry::{scheduled,deadline_ns,data,set_data,fire_callback,set_fire_callback}`、
  `TimerList`；全部有 Sched/IO 的 storage/delivery 用途。callback pointer 不是 executor：只由 Sched drain
  解释，queue 代码从不 invoke；在 Sched 未拥有自己 node wrapper 前删除它只会增加
  container-of/side-table 成本，没有语义或复杂度收益；
- `timer_queue.h`：三种构造、raw/typed schedule/schedule-after/cancel/advance/next/cancel-all/size/now；
  每个入口映射 arbitrary-deadline lifecycle，没有 benchmark-only API；
- `common_timeout_queue.h`：duration 构造、schedule/try variants/cancel/advance/next/cancel-all/size；
  `try_schedule` 是真实 recoverable ordering boundary，不是快慢档；
- 直接边：`timer_queue -> PUBLIC core,time,container_intrusive`；Sched 实现依赖 queue，
  IO 公开 header 实体化 entry，故 IO 必须是直接 public edge；`hyle::time` 产品继续聚合 host + queue；
- 成熟 peer 差异：Seastar `insert()` 返回是否刷新 minimum；HYLE 在 same-owner 进入 park 前统一
  读 `next_deadline()`，而 cancel 刻意允许 conservative early wake 以保 O(1)。当前没有需要该 bool
  才能成立的 consumer，不复制 peer 形状；
- 本轮冻结 `FIX` 只有：
  1. 新建 header-only target/test/bench owner，修正 target graph、product facade 和 Sched/IO 实际边；
  2. 为 queue construct/schedule-after/advance 与 common schedule-now 添加已冻结 provider overload；
  3. 分开 host clock 与 timer correctness/header tests，保持 product package sample 同时覆盖两个 factor；
  4. 将 timer bench/contract 所有权迁到本 factor，绑定当前 scheduler SHA-256
     `9fc1b77d51b70b701962418ca5b3003244b9828ac676fd5e9e33a49716842ba3` 与当前
     Seastar，不改仪器/正常 workload；
  5. 更新 architecture/user/performance/task，扫描旧 component 名和伪 FFI/binary 表述。

## API / usability freeze

- 状态：`FROZEN`；现有 node/queue/common 公开名、layout 和语义保持，用户仍 include
  `<hyle/time/...>` 并 link `hyle::time`；只有 developer factor ID 变为 `timer_queue`；
- provider 新形状：`TimerQueue(source)`、`schedule_after(entry, timeout, source)`、
  `advance(source, out)`；`CommonTimeoutQueue::schedule(entry, source)` 与
  `try_schedule_now(entry, source)`。source 必须满足 `MonotonicClockSource`，只在一次调用期间借用；
- 批次路径仍推荐显式只采一次 `now`；provider convenience 不存入 queue，所以不会为
  之后的 arm/advance 每次偷付 dispatch，也没有 source lifetime member；
- wrong owner、linked delivery node、倒退 advance 继续 fail-fast；Common 显式顺序违反继续用
  transactional `try_schedule=false`；这两种 failure 不合并；
- 不因 P1 结果加入 fast/light queue、第二套 node metadata 或 peer-only minimum bool；
- user guide 继续按 mixed/common/total-order/executor 四个任务选型，新增 manual/ref 示例和
  “queue 不保存 clock”陷阱。

## Owner 六问（`AUDITED` 直答）

1. **第三方：**产品路径无第三方；直接 HYLE 依赖是 Core、Time、Container-intrusive。
   Seastar/libevent 仅是绑定 peer，本地 faithful port 仅在 benchmark TU，不进入产品。
2. **推理能力：**llama.cpp/ggml 不自带这一底层 public data structure，但 HYLE Sched/IO 是
   构建推理 runtime 的真实消费者。arbitrary/common/total-order/execution 四档均为
   `COMPLETE` 或完整 `ROUTED`；当前未处置 `MUST` 仅 `TQ1-CLOCK/FACTOR/PERF` 冻结 FIX。
3. **性能：**`P1/PLANNED`；算法复杂度已与 Seastar/libevent 同档，但必须在 current
   scheduler provenance、当前 candidate、八 strata 与 5% margin 下重签；此前不声称 SOTA。
4. **C ABI/Native FFI：**`N/A/CLEAN`；全部是 C++ header/type/template，无 C schema/bridge/export/sample/
   option/package metadata；header-only factor 不制造伪 DLL。
5. **三交付：**当前 `MISSING`；独立 target 尚未存在。完成后 factor 本身 header-only，
   而 `hyle::time` product 必须在 source/static/shared 都同时提供 host compiled artifact 和 queue headers。
6. **C++ API/文档：**现有 mixed/common 选择、ownership、failure、single-owner、teardown 已自然；
   provider 尚需手写 `.now()` 且构建形状尚错误，所以当前 `FAIL`。按冻结 overload 和
   factor/user/architecture/performance 文档修复后再裁决。

## Candidate

- product commit / tree：`381eb4b195973ed35ed23ca63c05f9605a132f9b` /
  `a5d48a259161c1a60e19450e42b4a6eac9a1a712`；
- product/API：`TimerQueue` 的 provider construct/schedule-after/advance 与 `CommonTimeoutQueue` 的
  provider schedule/try-schedule-now 已落地；provider 只在单次调用期间借用，queue/node layout
  零增长，现有 raw/typed 明确-now 路线不变；
- factor/delivery：公开 header 由新 header-only `timer_queue` factor 拥有，直接边为
  `core,time,container_intrusive`；Sched 和 IO 已声明实际直接边，consumer-facing
  `hyle::time` 继续聚合 host artifact 和 timer headers，单选 Container 不会伪造不完整 facade；
- correctness/resources：新 provider tests 与原有 owner/lifecycle/radix/common-timeout 测试在 W1-WIN
  MSVC 19.50 Release 共 4/4 通过；node/queue 仍为单 owner、零 allocation/lock/atomic/syscall，
  arm/cancel/next 仍 O(1)，advance 仍按 touched/delivered/rebucketed 计费；
- P1 contract：当前 scheduler lifecycle contract SHA-256 为
  `48e636cb9e59d578b1148ef0af091cb04b6daeb38097581b6e315b9f1eb760fe`，已绑定当前
  Scheduler consumer 与 Seastar `991cf9a908482a3f11ca2fc34a1b3970a06d8622`；未用旧 bundle
  冒充新 candidate 证据；
- pending：W1-WIN 合格 P1 bundle、W1-WSL/M1-MAC correctness/sanitizer、最终 source/static/shared
  product journey、docs/red-team/aggregate gates。`CommonTimeoutQueue` 没有当前 HYLE 产品消费者，
  本轮只签 correctness/complexity 而不伪造 P1 性能声明。

`AUDITED -> CANDIDATE` 的前态 registry 快照与 Time 共用
`C:\hyle-scratch\time-v3-audited-before-candidate.json`，SHA-256
`53B2350D98372F57D5DAB398BC24F02408EB26B96FF80170B008E1A78CD4F75A`。candidate 后若改变公开合同、
产品实现、factor graph 或冻结 workload，必须退回 `AUDITED`。

## Verification manifest

验证绑定与 Time 共用的 immutable product candidate
`381eb4b195973ed35ed23ca63c05f9605a132f9b` / tree
`a5d48a259161c1a60e19450e42b4a6eac9a1a712`。candidate 后 `109cdfc4c` 只清理 Sched/IO test
target 的重复 `hyle_time` archive link，timer_queue header/layout/algorithm/component graph 零变化。

| Gate / risk | Verdict | Evidence / limit |
|---|---|---|
| owner/lifecycle/boundary/oracle | `PASS` | 三平台 timer_queue focused suite；wrong-owner/linked-delivery/decreasing advance death、reschedule/cancel/cancel-all、queue-first/entry-first、full 64-bit/radix boundary 和 million-operation oracle |
| provider composition | `PASS` | system/manual/ref 的 construct/schedule-after/advance 与 common schedule-now tests；provider 只借用一次调用，object layout 零增长 |
| sanitizer/concurrency | `PASS / N/A` | W1-WSL ASan/UBSan/TSan 各 2/2；queue operation 为明示 single-owner，不伪造 concurrent-success test |
| factor/delivery/direct consumers | `PASS` | 中央 component contract，三平台 `hyle::time` 三交付，三平台 Sched/IO/timeout 各3/3；header-only factor 不制造 DLL |
| P1 engineering performance | `PASS` | `timer-queue-consumer-v1` 对 Seastar 的8/8 strata 全部 `WIN` 且 5% NI `PASS`；deep verifier `QUALIFIED/EXACT/CURRENT` |
| common-timeout performance | `NOT_APPLICABLE` | 有 libevent 成熟形状与 correctness/complexity 证据，但无当前 HYLE 产品 consumer，不伪造 P1 排名 |
| fake C ABI/binary/stale owner | `PASS / N/A` | 负向扫描无 C/FFI，Windows Time DLL 只有 12 个 host-clock C++ decorated export，timer_queue 零 binary symbol |
| docs/usability | `PASS` | guide 按 mixed/common/total-order/executor 选型，明示 stable-address/single-owner/delivery-drain/provider-not-stored 陷阱；性能页与当前 contract 同步 |

P1 retained bundle 位于
`C:\hyle-scratch\time-v3-timer-perf-evidence\manifest.json`；contract/manifest/summary SHA-256 分别是
`48e636cb9e59d578b1148ef0af091cb04b6daeb38097581b6e315b9f1eb760fe`、
`B7D23B40DB8EE3D5B77CB0CD6EC7B6F5F5810F4BBA609A1B92BF605FBCE0FB90`、
`2C2D986359FCB43725850EEBCF9F352D11465D7B4D9CDE3ADD1D22EDC33AF124`。八个 strata 的 median
advantage ratio 为 1.1293--1.1898，99.26% CI lower 最低 1.0911，全部在冻结 5%
margin 外胜出。

## Independent red team

- callback pointer 仍只是 Sched 借用的 dispatch metadata，queue 实现零 invoke；未偷添 executor/
  owning callable/waiter/lock；
- cancel minimum 后的 conservative early wake 是保持 cancel O(1) 的明示取舍，不会漏发；无真实
  consumer 需要 Seastar-style minimum-changed bool；
- common-timeout 的 transactional reject 与 fail-fast wrong-owner 仍是两种不同 failure contract，没有为统一
  API 破坏可恢复 ordering boundary；
- 独立 factor 的 owned header、test、bench、public deps 和 product facade 均由中央合同验证，单选
  host Time 不获得 intrusive node，普通用户链 `hyle::time` 仍得到完整产品；
- 性能 WIN 在八个顺序/放置/分配 strata 中一致，不是单姿势或亚纳秒翻转；没有
  为此结果增加第二套 node metadata 或 fast/light API。

红队未发现未处置 `MUST`、correctness/security/ownership/delivery 缺陷或 material
consumer regression。

`CANDIDATE -> VERIFIED` 的前态 registry 快照与 Time 共用
`C:\hyle-scratch\time-v3\transitions\module-status-before-verified.json`，SHA-256
`A6FDF69D1C1E5FBF0E068ACD118B8F821CF5F5D3D472CDF25B26D55063348116`。

## Closure decision

本轮 `CLOSED`。node/radix/common/provider 能力、factor/dependency、correctness/sanitizer、
三交付与 Sched/IO 直接消费、C++ usability/docs、red team 和 P1 engineering performance
七门全部 `PASS`。callback/wait/periodic/cross-thread 是已路由的 Sched/IO owning policy，不是
timer data factor 遗漏。

可接受限制是：queue/node 是 caller-owned single-owner 且不执行 callback；cancel minimum
允许一次 conservative early wake，跨 radix bucket 不承诺全序；CommonTimeoutQueue 没有当前
产品性能 consumer，所以只签 correctness/complexity；P1 WIN 只属 W1-WIN 冻结 Scheduler
N=64 合同；header-only factor 无 binary/FFI 或稳定 C++ ABI 承诺。

终态 end/registry anchor 是 VERIFIED commit
`e8fb222bf975f76e45e777e5db3a7854db1d23e0`；product candidate 与 Time 共用
`381eb4b195973ed35ed23ca63c05f9605a132f9b` / tree
`a5d48a259161c1a60e19450e42b4a6eac9a1a712`。`VERIFIED -> CLOSED` 的前态 registry
快照是 `C:\hyle-scratch\time-v3\transitions\module-status-before-closed.json`，SHA-256
`3AEE61B58E39C9A8F51C5B0022A52E791DB76C9D4AFC3D960478CB203F06F412`。

四项 LOC 从 immutable candidate 执行：

```powershell
python scripts/loc-report.py --module timer_queue `
  --module-path include/hyle/time/common_timeout_queue.h `
  --module-path include/hyle/time/timer_entry.h `
  --module-path include/hyle/time/timer_queue.h `
  --module-path src/timer_queue `
  --commit 381eb4b195973ed35ed23ca63c05f9605a132f9b --self-check --json
```

`self_check=pass`：

1. 全库代码：420,468；
2. 自实现包含测试：101,897；
3. 自实现不包含测试：63,430；
4. timer_queue 实现：438。

Time host 1,192 + timer_queue 438 与共享目录聚合值 1,630 完全一致。
