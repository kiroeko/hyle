# Mem reclamation v1：显式 domain 的 reader-safe 回收

<!-- MODULE CLOSURE RECORD: ACTIVE -->

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

- candidate commit：`fc656855eae5e984b98c396817578c266e2a4491`；tree：
  `3686af7b1abed59b30ef518f9bc55a4308959a2e`；
- 已实现 `MR-01` 的 public header、compiled cold path、默认/显式 domain、TLS 独占 reader slot、固定
  fallback stripes、intrusive retire record、grace-period collection 与 misuse fail-fast；
- `W1-WIN` exact-factor shared/static WERROR 构建、header isolation、unit/death/stress tests 已通过；candidate
  之后只允许执行冻结卡已有验证或修复可复现 correctness 缺陷；
- `MR-02`/`MR-03` 尚未完成：对 CLOSED Container 的试迁移已撤销，不能把未关闭 factor 偷渡进既有产品；
  Mem façade 也未暴露本 factor；
- Folly peer 在本机尚不能形成资格化 same-contract 构建；P1、三平台 sanitizer/交付、直接 consumer 和
  user guide gate 仍未裁决，因此本记录还不是 terminal receipt。
