# Mem reclamation v1：显式 domain 的 reader-safe 回收

<!-- MODULE CLOSURE RECEIPT: BLOCKED -->

## Closure card（CANDIDATE）

- component / product / round：`mem_reclaim` / Mem / `mem-reclaim-v1`；
- task kind：`NEW`；baseline commit：`31b86dd809ec61cd1fb7c42ed986e2697d20cb91`；
- trigger：全库能力因子化的 `MEM-A-32`。`ConcurrentNodeHashMap` 为避免 reader/table publication race
  把所有旧 bucket table 保留到析构，Obs 又私有实现 snapshot reader epoch；这两处说明 HYLE 缺少唯一的
  reader-safe lifetime owner；
- real consumers：read-mostly registry/config snapshot；并发 map 扩容后的旧 table；Obs sink snapshot。
  这些 consumer 都先发布 immutable 或 externally synchronized state，再在短 read region 后延迟回收；
- scope：显式 `ReclamationDomain`、move-only read guard、零分配 intrusive retire record、typed pointer
  protection helper、nonblocking collect、blocking grace-period synchronization、externally-quiescent drain、
  pending/reclaimed observation，以及当前直接消费者的适用迁移；
- non-goals：intrusive membership、容器/日志策略、lock-free 容器、Sched executor、自动为每次 retire 分配、
  完整 hazard-pointer family、GPU fence retirement、C/Native FFI、稳定 C++ ABI、Formal 或 universal SOTA；
- dependencies：`core` only。回收 callback 在触发回收的线程同步执行；factor 不依赖 Mem allocator、Platform、
  Sched 或 Container，避免内存生命周期底座反向成环；
- direct consumers：候选为 `container` 与 `obs`；只有迁移后仍保持其既有并发/回调合同且通过自己的直接
  测试，才建立真实 edge；不能为了证明本 factor 有用而强迫热路径付费；
- performance tier：`P1`。read guard 是潜在 registry/container 热路径；retire/collect/synchronize 是冷写侧；
- material margin：同合同 warmed read-side route 对锁定 Folly RCU 不劣于 5%；若 peer 无法在 HYLE 独立
  package 中形成可资格化 same-contract 构建，则以固定 API 的原子/分配/锁结构、HYLE direct-consumer
  holdout 和明确 `INCONCLUSIVE` 停止，不换 workload 追点值；
- resource budget：guard acquire/release 不分配、不锁、不 syscall；普通不超过 64 个同时参与线程时，warmed
  route 是一次统一 TLS lease 查询、owner-local epoch store、验证 load 和 release store，不做共享 reader RMW；
  超额线程走固定 fallback stripe，嵌套同 domain guard 只更新本线程 depth；retire 不分配，常态一次短 queue
  lock；record metadata 只由 opt-in retire owner 支付；无 reader 时 collect 为 O(R)，有 reader 时 nonblocking 返回；
- investigation / modification budget：一个 API audit、一个实现、一次 correctness 修复；P1 最多一次
  qualification、一次机制诊断、一次产品重跑；禁止因亚纳秒、顺序或地址翻转改变 public layout；
- machines：`W1-WIN` correctness/P1，`W1-WSL` 与 `M1-MAC` correctness/sanitizer/交付；instrument：
  `closure-perf-v1`；
- strong peers：Folly `176c088840724715b2c3c6211b2b347611676002` 的 RCU/hazptr 与 concurrent-map
  consumption；oneTBB `b03a694ee0fe4b188bc77daa8a567551e3e3ebfc` 只作 concurrent-container lifetime
  反向检查；llama.cpp `b96806d96061049a5b574269b049bf6241d63d46` 与 ggml
  `d4716378882593333721eb33f153144b6885caf2` 用于确认推理 runtime 的实际并发 publication/retirement
  需求，不按名字复制 API；
- required gates：scope、architecture、correctness/sanitizer、source/static/shared C++ delivery、直接
  consumer、guide/误用诊断、red-team、P1 engineering；Native FFI `NOT_APPLICABLE`；
- stop：有限 FIX 全部完成才进入 candidate。若正确实现必须依赖未完成的 Platform/Mem split、同合同 P1
  无法资格化或迁移会让现有正常 consumer material 回退，则 `BLOCKED` 或保留显式 opt-in，不扩写通用
  并发框架。

## 初始能力分类

| family | current HYLE | normal consumer need | preliminary verdict |
|---|---|---|---|
| read-side lifetime guard | Obs 私有、Container 无通用 owner | 短 guard 后安全读已发布 pointer | `MUST / FIX` |
| intrusive deferred callback | 各模块私有保留/计数 | 不为每次 retire 分配，支持 allocator-aware owner callback | `MUST / FIX` |
| grace-period synchronize | 私有 drain 或保留到析构 | 等待调用开始前已存在 readers | `MUST / FIX` |
| nonblocking collection + pressure | 缺失 | writer 可尝试回收且观察 pending | `MUST / FIX` |
| default process domain | 缺失 | 普通小型 publication 与共享容器无需各自携带大型 reader registry | `MUST / FIX` |
| hazard pointer slots/cohorts | 缺失 | stalled reader 独立回收单对象 | `MAY / DEFER`，当前 consumer 未证明必须 |
| async callback executor | 缺失 | 将析构移到指定线程 | `ROUTE` to Sched/owner composition |

## Public surface 与 consumer 反向追踪（AUDITED）

- 当前 `include/hyle/mem/**` 没有 reclamation 类型；`container_intrusive` 只拥有 membership，并明确要求
  外部 owner 负责 retirement；
- `ConcurrentNodeHashMap` 的 accessor 在 bucket lock 内保护 node，但 reader 在 `current_` load 与 bucket
  lock 之间仍可能落后于 rehash。当前通过把旧 `Table` 几何级数地保留到 map 析构消除 UAF，空间上界小于
  当前 table，但长期 map 永远不能主动回收旧 bucket arrays；
- Obs 的 `snapshot_state` 只保护两个固定 snapshot 的复制/复用，随后还有 slot-active 和 callback self-remove
  协议。它不是任意对象 retirement；本轮 `REJECT` 强制迁移，避免通用 API 吸收日志特例；
- llama.cpp/ggml 当前没有公开通用 RCU/hazptr owner，因此它们只证明推理 runtime 存在并发 publication，
  不单独构成新增 API 的依据。真实直接消费由 HYLE Container 提供。

## Peer borrow / reject

| peer observation | decision |
|---|---|
| Folly `rcu_domain` 以 read guard、retire、synchronize 与 barrier 组成普通路线 | `BORROW` 语义分层和显式 domain；callback 在 grace period 后运行 |
| Folly 默认 domain 让多个轻量 owner 共享 reader registry | `BORROW` 进程生命周期 convenience，但指南优先显式 domain 隔离长 reader；不承诺跨 DSO/plugin 单例 |
| Folly read path 通过 thread-cached counters 避免一个全局 reader RMW | `BORROW PRINCIPLE`；HYLE 用进程映像内统一 TLS lease 给 live thread 分配独占 cache-line slot，线程退出归还；超过 64 个同时参与线程才走固定 fallback stripe，不引入 Folly ThreadLocal/Executor 依赖 |
| Folly free-function retire 会分配 callback node，hazptr 还提供 holder/cohort/executor 全家族 | `REJECT FOR V1`；HYLE 当前 consumer 可嵌入 record，不能让每次 retire 隐藏分配或把 Sched 拉到底层 |
| oneTBB concurrent containers 封装自己的 storage lifetime，没有独立 reclamation 产品 API | `BORROW` consumer-facing container 应隐藏机制；`REJECT` 以 oneTBB 缺少公开 API 证明 HYLE 不需要唯一内部 owner |
| 当前 Obs 双 snapshot + self-removal | `KEEP SPECIALIZED`；这是 fixed-generation handoff，不是通用任意对象回收 |

## API / usability freeze

公开头固定为 `<hyle/mem/reclamation.h>`，命名空间为 `hyle::mem`：

```cpp
using ReclaimCallback = void (*)(void*) noexcept;

class RetireRecord;       // one outstanding retire; reusable after callback
class ReclamationGuard;   // move-only; protects pointer loads in one domain
class ReclamationDomain;  // non-copyable, non-movable owner

ReclamationDomain& default_reclamation_domain() noexcept;

auto guard = domain.read();
State* state = guard.protect(published); // acquire load from std::atomic<State*>
domain.retire(record, state, reclaim_callback);
domain.try_reclaim();     // never waits for a reader; may run callbacks here
domain.synchronize();     // waits for pre-existing readers; runs eligible callbacks
domain.drain();           // external quiescence; empties the retire queue
```

`ReclamationDomain` constructor接受 retire threshold；`0` 表示只显式 collect，默认值用于在持续 writer churn
下周期性发起 nonblocking grace period。`pending()`、`reclaimed()` 与 threshold query 是 observation，不改变
回收顺序。返回的计数都只描述调用时快照；callbacks 没有 FIFO 保证。

### Ownership / lifetime / failure / concurrency

- domain 长寿于全部 guard 和 pending record；显式 domain 析构要求外部停止新 reader/retire，并执行 drain；
  process-default domain 不析构；
- guard 只保护**在 guard 建立后**通过 `protect()` acquire-load 得到的 pointer；把裸 load 放在 guard 前是
  contract violation。guard 可 move、可嵌套，不可跨 domain 保护，也必须在进入 read region 的同一线程离开；
- record 与 callback state 长寿于 callback；同一 record pending 时再次 retire、析构 pending record、空
  callback 是 fail-fast contract violation；在本 domain read guard 内调用 blocking synchronize/drain 会
  自等待，属于必须由调用者排除的 lifetime violation；
- retire/try/synchronize 可由多个 writer 调用；同一 domain 同时只有一个 collector 执行 callbacks。
  callback 在 collector 调用线程同步、`noexcept`、无顺序保证；可再次 retire；同 domain 的递归
  `try_reclaim` 是无操作，递归 `synchronize/drain` fail-fast；
- API 无 OOM 返回：normal route 不分配。需要 owning callback storage 的用户把 record 嵌入 owner 或由自己的
  allocator 管理；HYLE 不伪装一个可能分配的 `noexcept` convenience。

### 冻结成本结构

- reader：O(1)，无 allocation/lock/syscall；普通 live-thread 数内使用进程映像统一租赁的独占 slot，warmed
  路线一次 TLS route 查询、一次 enter store、一次 epoch validation load、一次 exit store，嵌套只改本线程
  depth；第 65 个同时参与线程起退化到固定 stripe 的 enter/exit RMW，极少数 epoch flip 重试；
- retire：O(1) queue insertion 和 opt-in record metadata；达到 threshold 时只发起一次 bounded nonblocking
  collect attempt；
- try_reclaim：有未完成 grace period 时扫描固定 stripe 数，reader 未清空即 O(S) 返回；成功时 O(S+R)；
- synchronize：O(S+R+等待 pre-existing readers)，epoch flip 后的新 reader 不延长本次等待；
- drain：要求外部不再 retire，O(S+R+grace periods)。S 是编译期固定 stripe 数，R 是本次可回收 record 数。

## 冻结 FIX ledger

| ID | action | verification |
|---|---|---|
| `MR-01` | 实现 header/compiled cold path、两代 epoch、固定分片 reader counters、intrusive records 与默认 domain | boundary、nested/move、double-retire/death、delayed reader、多 writer、recursive retire、exact callback count |
| `MR-02` | 让 `ConcurrentNodeHashMap` 用 guard 覆盖 table load→bucket-lock handoff，并在 rehash 后 retire 旧 table；析构完成自己的 grace period | forced interleaving、rehash/read stress、allocator balance、TSan/ASan；若 5% normal read gate失败则停止而非静默默认变慢 |
| `MR-03` | 中央图加入真实 Container direct edge；Mem product façade 只在 factor 自身 candidate 后选择它 | exact-factor、Mem/Container source、static、shared、relocated header/target/symbol journey |
| `MR-04` | 增加按“什么时候用 domain/什么时候继续用 lock/shared_ptr/什么时候需要 hazard pointer”组织的 user guide；更新 architecture/roadmap | compile-only normal recipe、误用诊断、docs checker、red team |
| `MR-05` | 资格化 read-mostly pointer publication 与 Container rehash holdout；只对冻结 API 做一次 P1 裁决 | locked contract、order/allocation strata、5% non-inferiority、结构资源审计 |

以上五项是本轮全部 FIX。Hazard slots/cohorts、executor、GPU fence、自动 owning node 和 Obs 重写不准在实现中
顺手加入；新事实需要退回 `AUDITED`。本轮现在允许写产品代码，但 P1 计时仍须等 API/correctness candidate。

## Candidate freeze

- candidate commit：`66671c39197765623f581dd872f4e626958880db`；tree：
  `dc17f2b62e30c72b020b868b4ee3961157a1603b`；首个冻结点
  `fc656855eae5e984b98c396817578c266e2a4491` 在 W1-WSL/GCC WERROR 暴露测试 fixture 的
  `-Wmissing-field-initializers` 后被替代，产品算法未改变；
- 已实现 `MR-01` 的 public header、compiled cold path、默认/显式 domain、TLS 独占 reader slot、固定
  fallback stripes、intrusive retire record、grace-period collection 与 misuse fail-fast；
- `W1-WIN` exact-factor shared/static WERROR 构建、header isolation、unit/death/stress tests 已通过；candidate
  之后只允许执行冻结卡已有验证或修复可复现 correctness 缺陷；
- `MR-02`/`MR-03` 尚未完成：对 CLOSED Container 的试迁移已撤销，不能把未关闭 factor 偷渡进既有产品；
  Mem façade 也未暴露本 factor；
- Folly peer 在本机尚不能形成资格化 same-contract 构建；P1、三平台 sanitizer/交付、直接 consumer 和
  user guide gate 当时仍未裁决，因此该 candidate freeze 本身不是 terminal verdict。

## Verification manifest

验证全部绑定 immutable candidate `66671c39197765623f581dd872f4e626958880db` / tree
`dc17f2b62e30c72b020b868b4ee3961157a1603b`。首个 candidate 只暴露了 GCC WERROR 测试夹具告警；
`66671c39` 仅把聚合指定初始化改为显式默认构造和字段赋值，产品 header/source 与算法不变。

| risk / journey | verdict | evidence |
|---|---|---|
| normal、empty、threshold、nested/move | `PASS` | Windows、WSL、Mac 的 exact-factor unit/header 2/2；显式 collect、自动 threshold、嵌套 guard 与 move/reset 均覆盖 |
| lifetime / invalid use | `PASS` | double retire、pending record/domain teardown、live reader teardown death tests；callback recursive retire 与 collector re-entry 测试 |
| concurrency / stress / fallback | `PASS` | 8 readers/512 records exact-once；72 simultaneous readers 覆盖 exclusive slots 之外的 fixed fallback |
| OOM / overflow | `NOT_APPLICABLE` | read/retire/collect 路线不分配；固定 slot/stripe 无容量分配失败；depth 与非法 lifetime 为 fail-fast invariant |
| W1-WIN | `PASS` | MSVC 19.50 Release WERROR；developer-factor shared 2/2、static 2/2 |
| W1-WSL | `PASS` | GCC 14.2 Release shared 2/2；static ASan、UBSan、TSan 各 2/2；TSan 由 `setarch -R` 真运行 |
| M1-MAC | `PASS` | AppleClang 21 arm64 Release shared/static 各 2/2；static ASan 2/2 |
| exact factor build | `PASS` | `HYLE_DEV_FACTORS=mem_reclaim` 只解析 `core,mem_reclaim`，三平台 WERROR；公开头 self-contained |
| product source/static/shared journey | `INCONCLUSIVE` | factor 尚未进入 Mem façade，且 maintainer factor 按合同不可 install/embed；未制造虚构 SDK PASS |
| direct consumer | `INCONCLUSIVE` | 临时 Container 迁移已完全撤销；Container 为 CLOSED，必须由独立 reopen/resume card 授权后才能改变 |
| usability / user guide | `INCONCLUSIVE` | header 与本卡已冻结 ownership/concurrency/misuse；未准入产品的能力不写入当前用户指南 |
| P1 engineering | `INCONCLUSIVE` | Folly 官方 CMake 路线因本机缺完整 Boost Context/Filesystem/ProgramOptions/Regex 及后续依赖而不能形成 qualified same-contract peer；没有直接 consumer holdout 可以替代 |

### 性能调查边界

- candidate 的 warmed read route 保持 O(1)、allocation-free、lock-free：普通 live-thread 数内为 TLS route、
  owner-local slot stores 与 epoch validation；第 65 个同时参与线程起才使用 fixed fallback RMW；
- immutable snapshot 的一次非资格化 natural-route 诊断约为 HYLE `12.04–12.09 ns/read`、
  `std::atomic<std::shared_ptr>` `13.79–13.92 ns/read`、unsafe raw atomic floor `1.28–1.29 ns/read`。
  它只说明成本阶梯，不是 Folly same-contract verdict；
- 临时 Container 试迁移的单次读数在 single/8-reader/mixed 之间既有改善也有 material 回退，且随后已
  从 product tree 撤销；这些数字没有 immutable consumer candidate、重复采样或合格 peer，不进入 gate；
- 不手抄一个“Folly-shaped”局部实现冒充 Folly，也不因 peer 构建缺依赖改换 benchmark 或放宽 5% margin。

## Red-team result

30 分钟边界内，候选前/验证中发现并修复四项真实 correctness/portability 问题：domain 在 pending=0
但仍有 reader 时析构、callback recursive retire 进入 collector、grace-period 对 late reader 的错误稳定零
断言、永久 reader-slot owner 导致线程 churn 耗尽；另修复 GCC WERROR 测试夹具。最终 candidate 的
三平台测试与 sanitizers 均通过。未发现新的 API ownership、复杂度档、常态 per-record metadata、
allocation/lock/syscall 或依赖环问题。红队不能消除缺失的 qualified peer 和 direct-consumer 事实。

## Closure decision

本轮从 `CANDIDATE` 终止为 **`BLOCKED`**。scope、architecture、correctness 与 red-team 为 `PASS`；
delivery、usability/docs 和 P1 engineering 为 `INCONCLUSIVE`。这不是产品算法已证失败：阻塞原因是
`MR-P1-PEER-CONSUMER-01`——冻结的 P1 合同既没有可构建的 Folly same-contract peer，也没有经 owner
授权的真实产品 consumer；因此不能签 non-inferiority，也不能把开发 factor 偷渡进 Mem façade。

- blocker owner：Mem/Container capability owner 与 performance evidence owner；
- resume condition：另建 `RESUME` round 从 `SCOPED` 开始，同时具备（1）可在锁定环境中按官方合同
  构建的 Folly RCU peer；（2）Container 的独立 reopen card 或另一个已准入真实 consumer，允许冻结
  migration/holdout；随后在 `closure-perf-v1`、W1-WIN、5% margin 下完成 same-contract P1，且
  source/static/shared 产品 journey 与“什么时候使用”指南均通过。只补依赖、重跑现有自然路线、手写
  peer、放宽 margin 或擅自改 CLOSED Container 均不满足恢复条件；
- known limitations：factor 当前只供顶层 maintainer exact-factor 构建，不属于 Mem 产品 façade/SDK；
  显式 domain 约含 64 个 cache-line slot，适合共享或确有隔离需求的 owner；guard 必须同线程退出；
  不提供 hazard-pointer cohort、异步 callback executor、GPU fence retirement、Native FFI 或稳定 C++ ABI；
- reopen/resume triggers：上述有限恢复条件；或真实 consumer 证明 correctness/security/lifetime/API 缺陷、
  需要 hazard/executor 等已 deferred family、依赖/平台改变语义，或强 peer 证明更优复杂度档。

四项 LOC 由 candidate 运行
`python scripts/loc-report.py --module mem_reclaim --module-path include/hyle/mem/reclamation.h --module-path src/mem_reclaim --commit 66671c39197765623f581dd872f4e626958880db --self-check --json`，
self-check `PASS`：全库代码 425,583；自实现包含测试 104,614；自实现不包含测试 65,136；
`mem_reclaim` 实现 457。
