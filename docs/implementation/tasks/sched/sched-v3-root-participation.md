# Sched v3 根参与批执行恢复记录

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 本页是 `sched-v3` 的 `RESUME` 施工记录。权威状态是
> [`module-status.json`](../../module-status.json)；已经终止的
> [`sched-v2`](sched-v2-batch-structured.md) 只保存候选、验证和失败事实，不在本轮回写。

## Closure card（`SCOPED`）

- component / owner / round ID：`sched` / Sched execution owner / `sched-v3-root-participation`；
  task kind：`RESUME`。
- 前一 round / blocker：`sched-v2-batch-structured` / `SCHED-V2-PERF-ONETBB-01`。最终候选在资格有效、
  同 consumer 合同的 W1-WIN replay 上没有通过 oneTBB 的 5% non-inferiority gate；peer-first advantage
  为 `0.916`，99.26% CI 为 `[0.777, 0.937]`。
- 前一 resume condition：出现一个有复杂度依据、能在不改变已准入 consumer 合同的前提下消除 material
  gap，并且可以用有限 proof plan 验证的 execution/partitioning mechanism。
- 恢复事实：正式中位数折算后，HYLE 相对 oneTBB 每个 16-round replay 多约 `75.7--104.8 us`，即每个
  batch 固定多约 `4.7--6.6 us`；Marl 的 enqueue/wait 路线与 HYLE 基本持平。oneTBB
  `simple_partitioner` 对 grain=1 一直拆到不可分 range，双方仍执行 128 个用户 callback；共同的
  `tile_digest` 是逐步依赖的 rotate/multiply recurrence，没有跨元素 SIMD 路线。当前最强机制假设是：
  oneTBB 已进入 arena 的 root task 直接完成一份 range，而 HYLE root Fiber 先投递全部 runner、挂起、
  再被 terminal Event 唤醒，支付了额外批级 handoff。
- baseline commit：`3a2bbca61038f5343729bf2dad66596bffc687b3`；产品输入仍是 v2 候选
  `22ec2382589656d1e45ff5a0665e384aba97dc4c`，本状态不把它冒充 v3 candidate。
- 范围：审计并冻结“同一个 Scheduler 上已经运行的 root Fiber 参与其同步 `bulk_execute`”机制；必要时
  增加仅供 Sched 实现使用的 current-owner identity seam；保持既有 shared first-error/cancel、partial
  admission、固定 public grain、caller-owned lifetime、O(N) work、O(worker) runner 与 O(1) batch state。
- 非目标：外部调用线程参与计算；跨 Scheduler Fiber 借用错误 worker；公开 partitioner/auto-grain；
  修改 benchmark tile、fanout、顺序、CPU 或 peer；Graph/Compute/IO 迁移；重写完整 scheduler、Event、
  Fiber 或 work-stealing；为结果删除错误/取消语义；Native FFI/C API；P2/Formal 声明。
- 真实 consumer / normal workload：在 Scheduler Fiber 内同步 fan-out 一批独立、non-suspending、
  caller-owned 输出 tile；固定为 v2 的 16×128×1024 replay，四个 pinned provider workers，外部线程只
  enqueue/wait。
- 直接依赖 / 消费者：产品依赖保持 v2 的 Core/Error/Time/Platform/Mem/Util-core 与 private Container
  queue；IO/Compute/Graph 只运行既有代表 smoke，不取得模块收口 verdict。
- tier / margin / machine / instrument：`P1` / 5% / `W1-WIN` / `closure-perf-v1`；`W1-WSL` 与
  `M1-MAC` 只做 correctness、sanitizer、portable codegen 和 delivery，不池化性能样本。
- strong peer：oneTBB `a8216b95202f3c55f9749870f4f8cfa4df0b32e3` 的 natural
  `task_arena + parallel_for(blocked_range, simple_partitioner)`；Marl 只作机制诊断，不以它抵消 oneTBB
  hard fail。
- 总预算：机制归因与 audit 最多 60 分钟；至多一次非裁决 diagnostic、一个产品机制修改、一次
  v2 normal workload 的 fresh n=15 qualification（只有统一停止规则允许才 n=31）。不得发明新的产品
  benchmark 姿势或因结果继续调 reservation/tile。
- correctness 风险：same-Scheduler identity、单 worker、零/一 runner、partial admission、first error、
  mid-cancel、root/runner terminal race、跨 Scheduler 调用、shutdown 与 stack-borrow quiescence。
- delivery 风险：产品若改变则重跑 Windows/WSL/macOS 的 source、relocated static/shared Sched component
  journey、public headers、symbol/旧 FFI 清洁度和既有直接 consumer slice。
- 停止条件：`AUDITED` 前必须把当前假设裁决为一个有限 `FIX` 或否决并重新 `BLOCKED`；实施后只允许
  一个 candidate 和一次正式重跑。所有适用门和 oneTBB 5% NI 通过才能 `CLOSED`；稳定 material FAIL
  必须以新的具名 blocker 终止，不得继续换姿势。

## SCOPED audit ledger

| ID | question | evidence required before `AUDITED` | current |
|---|---|---|---|
| `SCHED-V3-EQUIV-01` | oneTBB 是否减少了用户 callback 数或使用 SIMD | peer source、callback 边界与共同 body codegen | `PARTIAL`：source 已排除 range 合并与算法级 SIMD，待形成审计签名 |
| `SCHED-V3-COST-01` | 差距是每 item 还是每 batch 固定成本 | v2 immutable summary 的 throughput→wall-time 分解；至多一个 diagnostic | `PARTIAL`：现有中位数支持 4.7--6.6 us/batch |
| `SCHED-V3-ROOT-01` | root participation 是否保持同合同并消除一次 handoff | same-owner 判定、状态图、复杂度与失败/取消/lifetime proof | `PLANNED` |
| `SCHED-V3-FIX-01` | 应冻结的唯一产品修改 | 上述三项完成后形成有限 capability/gap ledger | `PLANNED` |

本页在 `SCOPED` 阶段不授权产品修改。先完成上述 audit，进入 `AUDITED` 并冻结 `FIX` 集后才可施工。

