# platform-01：thread + park（地址等待原语）

| 字段 | 值 |
|---|---|
| **阶段** | P1 |
| **模块** | platform/ |
| **Tier** | 2 |
| **LOC 估** | 2-3k |
| **骨架?** | ✅ |
| **状态** | 进行中（设计对齐中） |

## 前置依赖

- 硬前置：`error-01`（park 超时/失败走 `Status`，thread spawn 失败返回 `Result`）。
- 决策依据：`sched-design.md` §2.2（park 在 loop 末位）、§2.3（spin-then-park + unpark 接力 + 防惊群）、§4 收尾（"platform/ 提供 park 原语 futex/WaitOnAddress，是 sched 的 T2 仆从"）；`engineering-charter.md` §3（贴硬件、cache-line、per-core）。

## 目的（为什么做这一步）

sched 的 worker 空闲要能**睡**（不空转烧 CPU）、有活要能被**精准唤醒**（不惊群）。这要两块平台能力：① 把 worker 跑在可命名、可绑核的线程上；② 一个"等在某地址、被唤醒"的低层原语（futex 等价）。不做这步，sched 的 spin-then-park（§2.3）没有落点，只能 busy-loop。

## 在架构中的位置

`platform/` 是 T2 基质，被 `sched/` 直接依赖（worker 线程 + park）。本步是 walking skeleton 的第三块（`00 → error-01 → platform-01 → mem-01 → container-01 → sched-01`），为 sched-01 的"单 worker 单队列跑一 fiber"提供线程底座与挂起/唤醒。`time-01` 之后会给 park 加 deadline 变体（本步**不做**超时）。

## 关键设计决策（动手前定，附理由）

### D1：两层结构——平台原语 `wait_on_address` + 可移植 `Parker`
- **底层（平台特定，唯一要写三遍的东西）**：`wait_on_address(addr, expected, ...)` / `wake_one(addr)` / `wake_all(addr)`。这是 futex 的最小公约数，三平台几乎 1:1 映射。
- **上层（可移植，只写一遍）**：`Parker`/`Unparker`——每 worker 一个，内含 `atomic<uint32_t> state_`（EMPTY/PARKED/NOTIFIED 三态机），park/unpark 搭在 `wait_on_address` 上。**spurious wakeup 在这层用 `while(state != NOTIFIED)` 重检吸收**，sched 代码拿到的是干净语义。
- **WHY**：sched §2.3 的 unpark 接力是"唤醒特定 worker"，正是 Parker 模型（Rust std / tokio `parking` 同款）。把三态机和 spurious 重检封在 Parker，平台层只需提供 futex 原语——三平台差异面积最小。

### D2：三后端映射（仅 Windows 本机可验证）
| 平台 | `wait_on_address` 后端 | 链接 | 本机验证 |
|---|---|---|---|
| **Windows** | `WaitOnAddress` / `WakeByAddressSingle` / `WakeByAddressAll`（Win8+，API set）| `Synchronization.lib` | ✅ **真编真测** |
| **Linux** | `futex(2)` `FUTEX_WAIT_PRIVATE`/`FUTEX_WAKE_PRIVATE`（直接 syscall，不依赖 glibc 2.30 封装）| — | ⚠️ **未本机验证** |
| **macOS** | `os_sync_wait_on_address`（公开，**14.4+** 地板）| — | ⚠️ **未本机验证** |

- **macOS < 14.4 的 fallback 标 `待调研`**：`__ulock` 是私有 API（App Store 拒），`dispatch_semaphore` 需改 Parker 为信号量模型（偏离地址等待）。**不在本步发明未经测试的双路径**——按"硬问题标待调研、不写半成品"原则挂起，等真有 macOS 旧机需求再定。
- **诚实边界**：Linux/macOS 路径我会写全、走查、`static_assert` 能查的查，但**绝不声称"测过"**。CI 矩阵或换机补验前，它们是"已写未验"。

### D3：`Thread` 抽象——std::thread 的真实增量
- `spawn(fn, ThreadOpts{name, stack_size, affinity})` + `join()`。std::thread **没有**命名、栈大小、绑核——而 sched §3.3 的 per-core pinning 刚需。
- **affinity 尽力而为**：Windows `SetThreadAffinityMask` / Linux `pthread_setaffinity_np` 硬绑；macOS **无硬绑**（只有 QoS/affinity hint），尽力而为 + 文档标明。
- 附 `hardware_concurrency()`、`current_thread_id()` 小工具。

## 具体事务（做什么）

- [ ] `include/hyle/platform/thread.h`：`Thread` 类（spawn/join/joinable/name）、`ThreadOpts`、`hardware_concurrency()`、`this_thread::get_id()`/`yield()`。
- [ ] `include/hyle/platform/wait.h`：`detail::wait_on_address/wake_one/wake_all`（低层，可标 detail）；`Parker`/`Unparker`（公开）。
- [ ] `src/platform/thread_win.cc` / `thread_posix.cc`：spawn/join/name/affinity 三平台实现（`#if HYLE_OS_*` 分派）。
- [ ] `src/platform/wait_win.cc` / `wait_linux.cc` / `wait_mac.cc`：`wait_on_address` 三后端。
- [ ] `src/platform/parker.cc`：可移植 Parker 三态机（搭在 wait_on_address 上，只一份）。
- [ ] **不做**：超时/deadline park（→ time-01/sched-04）；virtual_memory/dynlib（→ platform-02）；path/fs/env（→ platform-03）；NUMA 感知绑核（→ mem-06）；优先级/调度类。

## 交付物

- thread.h + wait.h + 平台实现 + Parker + 测试。
- 最小演示：起 N 线程各自 `park()`，主线程 `unpark` 逐个唤醒，全部 join 干净退出。

## 验收标准

- **编译**：Windows 真编过（MSVC 19.50）；Linux/macOS 源码走查 + 能静态查的 `static_assert`，标"未本机验证"。
- **测试**（Tier 2，但 park 是并发原语 → 按 T0 严要求对待）：
  - Parker 三态机：park→unpark 唤醒、unpark-before-park（NOTIFIED 不丢）、重复 unpark 幂等。
  - **spurious wakeup 注入**：伪造 wake 不满足条件时 park 自旋重睡（不假返回）。
  - Thread：spawn/join、name 设置、affinity 设置不崩（值正确性 best-effort）。
  - 压力：M 线程 park / 主线程随机 unpark N 轮，无丢唤醒、无死锁（**Windows 上**跑；TSan 待 Linux CI）。
- **硬指标**：`unpark` 唤醒延迟基准（park→wake→返回往返），对标裸 `WaitOnAddress`/futex 不显著退化（Parker 封装开销 < 一次原子 CAS 量级）。
- **正确性闸门**：TSan 必过——但 TSan 在 Linux/Clang，**本步只能 Windows 验，TSan 标 CI 待补**（这是诚实缺口，记在案）。

## 参考

- `sched-design.md` §2.2 / §2.3 / §4。
- Windows `WaitOnAddress` / `WakeByAddressSingle`（MSDN synchapi.h）。
- Linux `futex(2)` man；`FUTEX_*_PRIVATE`。
- macOS `os_sync_wait_on_address`（os/os_sync_wait_on_address.h，14.4+）。
- Rust std `sys::pal::*::futex`、tokio `loom`/`parking` 的 Parker 三态机（思路参考，非照搬）。
