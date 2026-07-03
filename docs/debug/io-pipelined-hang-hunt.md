# io pipelined P0 Hang Hunt（>ring 深度在途 op 永不收割 —— 已定位并修复）

> 调查日志（已结案）。指令（用户）：**hang 和 crash 没区别，高性能库不能有这种结果，一查到底、不限手段**。
> 关联：[`../performance/benchmarks-io.md`](../performance/benchmarks-io.md)（pipelined 结果 + 修法摘要）、[`../architecture/io-design.md`](../architecture/io-design.md)。
> 结论先行：**worker 空闲时唯一的 reap 是 `reap(timeout=0)`，它在空 CQ 上只做 submit-only `io_uring_enter`（不带 `GETEVENTS`）；而 io_uring 的完成 task-work 只在本线程带 `GETEVENTS` 进内核那一下才跑** → run-queue 抽干 + 几百 op 在途时 completion 永不 surface → 死等。修：worker park 边界有在途 op 时改阻塞在 GETEVENTS reap。commit `e95cdd1`。

## 0. 现象

- **负载**：`bench/io/echo_bench.cc` 的 pipelined 变体 —— **512 条独立 echo 流共享一个 worker，每流 16 round-trip**（512 » io_uring ring 深度 256）。
- **现象**：进程 100% 挂死，`drain()` 永不返回。CPU 一核 100%（worker 空转），非死锁（无线程互等）而是**活锁 + 完成丢失**。
- **反差**：同文件的 ping-pong echo（64 连接 × 64 round-trip）**从不挂**。这个反差是最重要的线索（见 §4）。
- **平台**：**仅 Linux io_uring**。Windows IOCP 跑同负载不挂（IOCP 完成经端口投递，与本线程是否 enter 内核无关）。→ 复现 + 验证只能在 WSL 原生 ext4。

## 1. 二分定位：边界正好是 ring 深度

| # | 配置 | 结果 |
|---|---|---|
| B1 | pipelined 64 流 | 净 |
| B2 | pipelined 256 流（= ring 深度）| 净 |
| B3 | pipelined 260 流 | **挂** |
| B4 | pipelined 300 / 512 流 | **挂** |

**边界 = 256 = `init(entries=256)` 的 ring 深度。** 这把嫌疑锁定到"在途 op 数超过 ring 深度"这一类机制，而非某个具体 op 的正确性。

## 2. 误诊：以为是 CQ 溢出（诚实记录，方向错了一次）

**假说（错）**：在途 op > CQ 深度 → CQ 满 → 后来的 completion 被 io_uring 丢弃 → 那些 fiber 永不唤醒。

**据此做的事**：加了 `IORING_SQ_CQ_OVERFLOW` 恢复循环 —— reap 后检查 `sq_flags & CQ_OVERFLOW`，置位就 `GETEVENTS` enter 把内核 backlog flush 进 CQ（commit `dca5205`）。

**为什么错**：加诊断计数器（`stat_overflow`，commit `1352f8f`）后**实测 `overflow=0`** —— 溢出位从未置起。io_uring 有 `IORING_FEAT_NODROP`：CQ 满时 completion 进内核 backlog **不丢**，可我这里根本没到溢出，backlog 是空的。**恢复循环修的是一个不存在的问题。**

> **教训**：别凭"听起来像"的机制下手。"在途 > ring 深度"直觉上像溢出，但**直觉不是证据**。是加了计数器拿到 `overflow=0` 才把这条路排除。修好真根因后这段恢复循环 + `sq_flags` 字段已整体撤除（commit `e95cdd1`）——它是对错误假说的冗余防御，留着还会重复计在途数。

## 3. 加诊断手段（用户明确要求"调试和日志的定位手段"）

眼睛看代码看不出来，上计数器 + 实时 trace（全部 env-gated，零常态开销）：

- `HYLE_IO_BENCH_STATS`（析构时打）：`enters / sqes / avg_batch / reaped / overflow / diff(sqes-reaped)`。
- `HYLE_IO_TRACE`（每 20 万次 reap 调用打一次）：running `reaps / reaped / sqes / overflow / pending` —— **专为"卡住时看 worker 在干嘛"**，不靠进程崩了才在析构打。

**挂死时的 trace 铁证**：
```
reaped=1202（冻结不动）  sqes=1802（冻结不动）  overflow=0  pending=0
reaps=500... 1000... 2000...（只有这个在涨）
```
读法：**submit 了 1802 个 SQE，只收割回 1202 个** —— 差 600 个 completion 卡在内核永不 surface；`reaps` 疯涨 = worker 在空转 `reap(0)` 每次返 0；`overflow=0` = 不是溢出；`pending=0` = 没有攒着没提交的 SQE。**600 个 op 已进内核、内核却没把完成给回来。**

## 4. 决定性实验：阻塞 reap probe

**假说**：worker 空闲路径从不带 `GETEVENTS` 进内核，所以内核的完成 task-work 没机会跑。

**probe**：把 worker drain 路径那次 `poll(...,timeout=0)` 临时改成 `timeout=100us`（即让它带 `GETEVENTS` 阻塞一下）。

| # | 配置 | 结果 | 判读 |
|---|---|---|---|
| E1 | 原样（drain reap timeout=0）| 300 流**挂** | 复现 |
| E2 | drain reap 改 timeout=100us（带 GETEVENTS）| 300 流**跑完，pipelined 表打印出来** | **根因确认** |

一行 `0 → 100us` 让挂死变通过 = **病根就是"从没做过一次 GETEVENTS enter"**，与在途数、溢出、fiber 都无关。

## 5. 根因（io_uring GETEVENTS 语义，务必内化）

io_uring 的**完成侧 task-work（把内核完成的 op 写进 CQ）不是后台自动跑的，而是搭本线程进入内核的便车** —— 具体说，只有带 `IORING_ENTER_GETEVENTS` 的 `io_uring_enter` 才驱动它。

批量优化落地时（commit `c2a3756`/`f39241b`）把 reap 挪到"run-queue 抽干"处、worker 空闲走 futex parker（`kIoPollNs` 1ms 上限重 poll）。问题出在两点叠加：

1. **worker 空闲阻塞在 futex parker**（`park_timeout`），parker 是纯用户态 futex，**根本不进 io_uring 内核**。
2. **它唯一的 reap 是 drain 路径的 `reap(timeout=0)`**，而 `reap(0)` 遇空 CQ 只做 **submit-only enter（`min_complete=0`，不带 GETEVENTS）** 或干脆不 enter。

于是一旦 run-queue 抽干、几百 op 在途：没有任何一处带 GETEVENTS 进内核 → 完成 task-work 不跑 → CQ 一直空 → `reap(0)` 一直返 0 → worker 空转、`drain()` 死等 `outstanding != 0`。

**为什么 ping-pong 侥幸不挂**：socketpair 的 send 在 submit 的那次 enter 内联就完成了，CQ 稳态非空，纯用户态收割即可 —— **它从来不需要 GETEVENTS**。所以 ping-pong 全绿彻底掩盖了这个结构性 hang。这也是为什么"能跑的负载"不能证明机制对。

## 6. 修法（正确性要求，非延迟优化）

worker 的 park 边界改为：**io hook 注册且有在途 op 时，把阻塞交给 proactor 一次 GETEVENTS `io_uring_enter`（上限 `kIoPollNs` 1ms），而非 futex park。**

- hook（`IoContext::poll`）返回 **是否有在途 op**（`Proactor::in_flight()`）：
  - **有** → 在环里带 GETEVENTS 阻塞收割，返回后 `continue` 重查队列（完成会 re-inject fiber）。
  - **无** → 落到 futex park，**可被跨线程 `unpark()` 立即打断**（阻塞 reap 不能被 unpark 打断）→ 保住"无 io 在途时"的即时跨线程唤醒。
- `Proactor::in_flight()` 三后端各自实现：io_uring 计 submit-not-completed（`submit_one` ++，reap 每个**终结** completion --，`kMore` multishot 中间态不减）；IOCP 查在途 OVERLAPPED 链非空；epoll 查 `pending`/`posted` 非空。
- `Scheduler::set_io_poll` 的 hook 签名从 `void` 改 `bool`（多返一个"有无在途"给调度器决策）。
- 1ms 上限的意义：即使 io 在途，一个钉在本核的 fiber 被外核 push + `unpark_specific` 时，本 worker 最多 1ms 后从阻塞 reap 超时返回、循环顶重查 `find_task` 看到它。**消这 1ms 需在环里注册 eventfd 即时打断阻塞 reap = 纯延迟优化，不影响正确性，留作后续。**

## 7. 验证

| 面 | 结果 |
|---|---|
| pipelined 512 流（原挂死负载）| **跑完**：1w 328K → 2w 1.02M → 4w 1.63M → 8w 1.71M echo/s（5.2x 扩展）|
| ping-pong 64×64 不回归 | 1w **798K**（修前批量后是 728K），4w ~2M/s |
| Windows IOCP | sched 44/44 + io 8/8 + net 3/3 + tcp 1/1 |
| WSL io_uring | io 11/11 + net 3/3 + tcp 1/1 |
| WSL TSan（`setarch -R`）| net 3/3 + tcp 1/1 + sched 44/44 + io 11/11，零 race |

## 8. 教训

1. **"能跑的负载"绿 ≠ 机制对。** ping-pong 从不需要 GETEVENTS，所以它全绿却掩盖了 hang。**优化/机制落地后必须配一个真能压到该机制的 benchmark**（这里是 pipelined 高扇出）才能把潜藏 P0 逼出来。用户原话："做好之后你应该弄一个符合那种负载的 benchmark。"
2. **hang == crash == P0，绝不给"登记不修"的选项**（高性能库尤其）。我一度想把它当"已知限制"登记 —— 被用户驳回，对的。
3. **别凭直觉下手，上诊断计数器。** CQ-overflow 那条弯路就是凭"听起来像溢出"动的手；`overflow=0` 这个数据才把它排除。眼睛看代码看不出收割节奏问题，`reaped` 冻结在 1202 而 `sqes=1802` 一目了然。
4. **io_uring 完成侧不是自动的**：task-work 搭 GETEVENTS enter 的便车。任何"空闲就 park 在别的原语上"的设计，都必须保证在途 op 存在时仍会带 GETEVENTS 进内核 —— 这是 proactor 集成进任意调度器循环的通用铁律，不止本例。

## 9. 相关 commit（时间序）

- `c2a3756`/`f39241b` — 批量优化：reap 挪到 run-queue 抽干（**引入本 hang 的改动**，但批量本身是对的）。
- `b399e27` — 加 pipelined bench（**揪出 hang 的负载**）。
- `dca5205`/`1352f8f`/`d0e6aa7` — 误诊 CQ-overflow + 加诊断计数器/trace（弯路 + 定位工具）。
- `e95cdd1` — **真修**：park 边界阻塞 GETEVENTS reap + `Proactor::in_flight()` 三后端 + 撤 CQ-overflow 恢复循环 + 清诊断脚手架。
- `e21e652` — 回填 performance/io-design/milestones 文档。
