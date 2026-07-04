# when_any loser UAF Hunt（栈上 GroupSignal 生命周期竞态 —— 红队揪出、TSan 反向证实、已修）

> 调查日志（已结案）。触发：sched 支柱系统性对抗审查（"值得排除红队对抗审查/对比 SOTA/分析平台行为，确保正确性、查怪异数据根因、找复杂度 SOTA 机会"）。
> 关联：[`join.h`](../../include/hyle/sched/join.h)（修复点）、[`../../tests/sched/join_test.cc`](../../tests/sched/join_test.cc)（复现测试 `WhenAnyLoserCompletesConcurrently`）。commit `4f6c823`。
> 结论先行：**`when_any` 的 `GroupSignal` 是等待方栈上对象；`JoinState::complete_` 原本在解锁自己的 `mu` 之后才调 `group->on_complete()`。一个 loser 读走 `group` 指针、解锁、被抢占；其间 winner 触发 group、等待方醒来对每个 state 调 `clear_group()` 并返回——栈上 `GroupSignal` 析构——loser 再 `g->on_complete()` 踩已释放内存。** 修：把 `on_complete` 移进 `state.mu` 临界区内，`clear_group` 也取该锁，于是持锁跨 `on_complete` 逼等待方的 `clear_group` 阻塞到 group 用完，栈对象保活。

## 0. 为什么这篇值得独立成文：这是 TSan 抓不到的那一类

TSan 是数据竞争探测器——它抓"两线程无同步地访问同一内存"。但这个 bug 的**触发窗口本身依赖一个逻辑顺序错误**：一个指针在它指向的栈对象析构后才被解引用。若没有一个测试**恰好制造出那个时序**，TSan 根本没有可观察的冲突访问去报。sched 此前 101 个测试 + 多轮 TSan 全绿，恰恰因为**没有测试触发那个窗口**（见 §4）。所以本篇记录的方法论是：**逻辑竞态要靠"读代码找时序 + 写出制造该时序的测试 + 用 TSan 反向证实"三步，而不是指望既有测试套自己撞上。**

## 1. 现象：红队审查读出的一个可疑时序

不是崩溃报告触发的——是系统性红队审查 `join.h` 时，盯住 `JoinState<T>::complete_` 的这个结构（修复前）：

```cpp
void complete_(Status s, T* val) noexcept {
  GroupSignal* g;
  int gi;
  mu.lock();
  status = std::move(s);
  if (status.ok() && val != nullptr) new (storage) T(std::move(*val));
  done = true;
  g = group;          // 读出 group 指针
  gi = group_idx;
  mu.unlock();        // ← 解锁
  if (g != nullptr) g->on_complete(gi);  // ← 锁外解引用 g
  cv.notify_all();
}
```

`GroupSignal` 是 `when_any` 的**栈对象**（`join.h` 的 `when_any`：`detail::GroupSignal g; return detail::group_wait(g, 1, ...)`）。`group_wait` 一返回，`g` 随栈帧销毁。可疑点：`complete_` 在 `mu.unlock()` 之后、`g->on_complete()` 之前，若被抢占，`g` 可能已经死了。

## 2. 时序核验：窗口真的存在（when_any，两任务 A/B 在不同 worker）

| 步 | 线程 | 动作 |
|---|---|---|
| 1 | worker-B（loser B）| `complete_`：`mu.lock()` → 读 `g=group`(非空) → `gi` → `mu.unlock()`。**B 已持 g，尚未调 on_complete，被延迟。** |
| 2 | worker-A（winner A）| `complete_` → `on_complete` → `--remaining==0` → `fired=true` → `notify_all()` |
| 3 | 等待方 fiber | 从 `cv.wait` 醒 → 读 winner → 对 A、B 各调 `clear_group()`。**B.clear_group 把 B.group 置 null，但 B 早在步1读走了 g，置空无效。** |
| 4 | 等待方 fiber | `group_wait` 返回 → `when_any` 返回 → **栈上 `g` 析构** |
| 5 | worker-B | 恢复执行 `g->on_complete(gi)` → 在**已析构**的 g 上 `mu.lock()` → **UAF** |

`clear_group` 的注释原本自信地说"uninstall BEFORE g dies (critical: a not-yet-fired loser would otherwise call on_complete on a dead stack object)"——**但它只能防"还没读 g 的 loser"，防不住"已经读走 g、正卡在 unlock 之后"的 loser。** 这是与 SOTA 的实质偏差：folly 的 WhenN 用堆上 `shared_ptr<Context>`，引用计数让 Context 活到最后一个在途回调 drop 之后；HYLE 把 GroupSignal 放栈上、仅靠 `clear_group` 兜底，恰好漏掉这段窗口。

## 3. 写出制造该时序的复现测试

关键是让 **loser 真的走 signal-group 路径**（`g = group` 非空 → 锁外 `on_complete`）。`when_any` 两个任务都要**能完成**且**近乎同时**，并重复多轮 + 多 worker 放大窗口：

```cpp
TEST(Join, WhenAnyLoserCompletesConcurrently) {
  Allocator al = tsan_visible_allocator();   // 见 §3.1
  Scheduler sched(8, al, 1024);
  constexpr int kRounds = 2000;
  for (int i = 0; i < kRounds; ++i) {
    auto a = spawn_task(sched, al, add_coro(al, i, 1));
    auto b = spawn_task(sched, al, add_coro(al, i, 2));
    int w = when_any(a, b);        // 一个赢，另一个仍完成 = loser 走 signal 路径
    EXPECT_TRUE(w == 0 || w == 1);
    EXPECT_EQ(*a.join(), i + 1);   // join 两个，确认结果没被竞态破坏
    EXPECT_EQ(*b.join(), i + 2);
  }
  sched.shutdown();
}
```

### 3.1 为什么用 tsan_visible_allocator 而非默认 mimalloc

默认 `system_allocator()` 是 mimalloc，走自己的 mmap 堆，**绕过 TSan 的 malloc 拦截器** → TSan 看不到 free→reuse 的 happens-before 边，会在复用地址上**误报**（KI-2 同族的工具交互问题）。但这里 GroupSignal 是**栈**对象，UAF 发生在栈 slot 复用上——用 TSan 可见的 `std::aligned_alloc` 背书 JoinState，让 TSan 能看见相关 state 的分配/释放边，保持竞态闸门有意义。

### 3.2 为什么现有测试漏掉了它

既有的 `WhenAnyReturnsAWinner` 用一个 **parked**（永久阻塞、永不完成）的 loser：

```cpp
auto slow = spawn_task(...wait_then(&box, 99));  // 阻塞在 box，永不 complete
auto fast = spawn_task(...add_coro(7, 7));
int w = when_any(slow, fast);   // slow 从不进 complete_ 的 signal 路径
```

loser 从不调 `complete_` → 从不读 `group` → **恰好避开了 §2 的整个窗口**。这解释了 bug 为何在全绿的测试套下潜伏：**没有任何测试让一个 loser 真正完成。**

## 4. TSan 反向验证：证明测试有效 + bug 真实（不是纸上推演）

双向跑（WSL，`setarch -R`，RelWithDebInfo + `-DHYLE_SANITIZER=tsan`）：

| 代码状态 | 跑 `WhenAnyLoserCompletesConcurrently` | 结果 |
|---|---|---|
| **修复前**（锁外 on_complete，即 bug 代码）| setarch -R TSan | **报 4 个 data race**，直指 `join.h:144 clear_group()`（等待方读 `group` 字段 vs loser 访问竞争）|
| **修复后**（on_complete 移入锁内）| setarch -R TSan | **干净，0 warning，PASSED** |

修复前那 4 个 race 精确落在 `clear_group` 读 `group` 字段处——正是 §2 步3 与步1/5 竞争的那个字段。这就把"推演出的时序窗口"钉成了"工具可见的真实竞态"。**先用测试触发、再用 TSan 反向证实，是逻辑竞态区别于纸上分析的关键一步。**

## 5. 修复：on_complete 移入 state.mu 临界区（选项 B）

```cpp
void complete_(Status s, T* val) noexcept {
  mu.lock();
  status = std::move(s);
  if (status.ok() && val != nullptr) new (storage) T(std::move(*val));
  done = true;
  // 持 mu 调 on_complete：clear_group 也取 THIS state 的 mu，故持锁跨 on_complete
  // 逼等待方的 clear_group 阻塞到我们用完 g，栈对象保活。
  if (group != nullptr) group->on_complete(group_idx);
  mu.unlock();
  cv.notify_all();
}
```

**锁序论证（无死锁）**：`complete_` 走 `state.mu → g.mu`（on_complete 内取 g.mu）。反向不存在：`group_wait` 的 `clear_group`/`install_group` **只取 state.mu**（不嵌套 g.mu），`on_complete` **只取 g.mu**。无环 → 无死锁。等待方 `clear_group` 拿 loser 的 state.mu 时会短暂阻塞到 loser 的 `complete_` 释放，之后照常拆除——group 在此期间保活。T 与 void 两个特化都改。

## 6. 选项 A vs 选项 B（本次选 B，A 的对比另见性能文档）

- **选项 B（已采用）**：on_complete 移入锁内。零新字段、零分配、锁序单向易证。代价：临界区含一次 `on_complete`（几个原子操作，非阻塞）。
- **选项 A（folly 式）**：GroupSignal 改**堆分配 + 引用计数**，每个 install 加 ref，on_complete 后 release，`group_wait` 结束时 release 自己那份——最后一个在途 signaler drop 时才析构。更"彻底"（不依赖锁序保活），但引入堆分配 + 每次完成的 refcount 原子。A 是否更快是一个经验问题，用 `join_bench` 实测对比（见 [`../performance/benchmarks-sched.md`](../performance/benchmarks-sched.md) 的 when_any 行）。

## 教训

1. **TSan 全绿 ≠ 无并发 bug**：它只在有测试制造出冲突时序时才有可观察对象。逻辑竞态要靠"读代码找时序 → 写测试制造它 → TSan 反向证实"。
2. **既有测试的"方便选择"会系统性掩盖窗口**：`WhenAnyReturnsAWinner` 用 parked loser 让测试好写，恰好绕开了 signal 路径。写测试时要问"我这个 setup 是不是躲开了要测的那条路径"。
3. **栈上共享对象要证明它活到最后一个访问者**：GroupSignal 放栈上省一次分配，但共享给可能延迟的 signaler 时，必须有机制（锁保活或 refcount）保证它不早死。folly 用 refcount，HYLE 用锁序——两者都行，裸靠"析构前 uninstall"不行。
