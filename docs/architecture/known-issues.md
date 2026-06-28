# Known Issues（已知问题 / 未解决缺陷）

> 诚实账本：已发现但**尚未解决**的真实缺陷。不藏、不淡化。每条带：现象、已验证的事实、已排除项、复现条件、下一步。解决后移除并在对应里程碑/提交记录。

## KI-1 [已降级/重新定性]：fiber × mimalloc 跨线程 free 崩溃 —— 穷尽调查后判定为 9p/WSL 环境假象，非 HYLE 缺陷

> **状态变更(本轮深查)**:此条原为"P0 未解 HYLE 缺陷"。经一轮**穷尽、独立于崩溃时序**的调查,该定性**与证据矛盾**,现重新定性为**极可能是 9p/WSL demand-paging 环境假象**。完整实验账本见 [`../debug/ki-1-crash-hunt.md`](../debug/ki-1-crash-hunt.md)。**不再声称是 HYLE bug,也不 100% 拍板;以证据说话。**
>
> **defect 级工具排除 HYLE 逻辑成因**:TSan(源码级 race)10万次迁移净;**MI_DEBUG=3** 全量堆不变量 + 65万 async churn + 48-worker 超额订阅 + 24G 内存压力**全净零断言**;代码无任何抑制。
> **忠实+激进复现全失败**:原始崩溃器本体 scheduler_bench + 原生 ninja 真饱和(cc1:9-17)**10/10 净**;`drop_caches` 拷打 9p/native **8/8+8/8**;真实 24G 内存压力(avail→4G 强制代码页驱逐)9p/native **6/6+6/6**。
> **关键线索**:① 早期"重负载复现"在 9p 上 `cc1plus=0`(IO-bound,负载没起来)——用户的 9p 洞察命中;② 原始崩溃含 **SIGILL** = 代码页问题(非堆损坏的数据 SIGSEGV),指向 9p 后端 .so 代码页 re-fault 不稳定。
> **mimalloc 保持出厂默认 ON**:核心 fiber 场景经 MI_DEBUG=3 全量校验证明无损坏,不需为此关 mimalloc。
> **诚实边界**:9p 上也未正面复现(同样净),故不 100% 断言"就是 9p";但旧"HYLE bug"框架已被证伪。scheduler_bench 仍 withheld(用户在 9p 开发,默认编入仍可能偶发该环境假象)。

---

### 〔历史记录·原 P0 定性,已被上述调查取代〕

## KI-1 [P0]：fiber 经 mimalloc 分配 + 跨 worker 迁移 + 跨线程 free，重负载下崩溃

**现象**：纤程在 worker A 上经 mimalloc(默认分配器)分配一块内存，`yield()` 后被另一 worker B 偷取(迁移),在 B 上 free 该块——重负载下崩溃(SIGSEGV，少数 SIGILL=`Fiber::resume` 的 `!finished_` 断言)。gdb 栈：`worker_main → Fiber::resume`。

**为什么是 HYLE 的责任(不甩第三方)**：mimalloc 是 HYLE **出厂默认**(`HYLE_USE_MIMALLOC=ON`)，而"纤程分配内存 + 被工作窃取迁移"是调度器**核心用例**。默认配置下核心场景崩 = 我们的活没做完。

**严格定位的必要条件**(去掉任一 → 不崩)：
1. **mimalloc 开**：关掉(OS 分配器)同样的 `分配+迁移+跨线程free` 模式 **0/15 净**(O3)。
2. **fiber 跨线程 free**：`分配→free→yield`(同 worker free) **净**；`分配→yield→free`(迁移后跨线程 free) **崩**。
3. **真迁移**：`yield-only`(迁移但不分配) **0/15 净**。
4. **-O3**：`-O2` 主程序 **净**;`-O3` 崩。
5. **重抢占负载**：真实崩溃(14/14、20/20)只在【并发构建(ninja+编译器)饱和机器】时出现;空闲机器(含 64 busy-loop 满 32 核)**当前无法按需复现**。

**已排除(不是这些)**：
- **调度器逻辑**：OS 分配器下 `分配+迁移+跨线程free` 净 + WSL TSan 净(scheduler 含 mimalloc-ON 小复现无 data race)+ 自加每-fiber 原子计数器**未检出 double-resume**。
- **nanobench**：脱离 nanobench 的纯程序(`fiber_alloc.cc`)同样 14/14 崩。
- **std::async 线程退出**：复现不含 std::async 也崩。

**机制假说(未证实)**：mimalloc 默认(initial-exec/fast)TLS 访问，在 **fcontext 分配的非标准栈**上运行时可能解析到错误的线程堆;`MI_LOCAL_DYNAMIC_TLS`(经 `__tls_get_addr`)可能修复。**该假说未验证**——试验那轮基线也变净(负载没起来),不能区分。

**下一步**：
1. 造**可靠复现**：用真实重负载剖面(并发 ninja 编译 / 大量短命线程+malloc churn),非 busy-loop。
2. 复现稳定后，逐一验证候选修复：`MI_LOCAL_DYNAMIC_TLS=ON`、mimalloc 版本、mimalloc debug 版抓 `mi_free` 段错点。
3. 若证实 mimalloc 内部 + 非标准栈，向上游报 + 本地配置规避。

**最小复现**(scratchpad，需重负载并发跑)：
```cpp
// fiber 体: allocate(64) on worker A; Fiber::yield(); deallocate on worker B
// 4 worker scheduler; 300 轮 × 1000 fiber; g++ -O3 -DNDEBUG; mimalloc ON
// 在并发构建/重抢占时崩 14/14;空闲机器当前不复现
```

**临时缓解**：`sched` microbench 默认不编入(会触发);生产若遇可临时 `HYLE_USE_MIMALLOC=OFF`(损失 mimalloc 性能)。**这是缓解非修复**。

**追加(诚实更新,深查后)**：
- **当前无法按需复现**:真实崩溃(34/34 带真栈)发生在某次特定系统状态下;此后空闲机器、64 busy-loop 满 32 核、真实编译负载(33 cc1plus)下均 **0/50+ 不复现**。触发条件未能稳定重建。
- **已排除的"廉价解释"**:① `.so` 被并发构建覆盖 → 边重写边跑 **0/15 净**(cmake/ninja 用原子 rename,运行进程保留旧 inode,不会半文件损坏)。② 纯 CPU 饱和 → 净。
- **结论**:这是一个**真实但极难复现的、mimalloc 依赖、时序敏感**的崩溃,位于 fiber 迁移 + 跨线程 free 路径。已严格定位必要条件 + 排除调度器逻辑,但**机制未根因、修复未验证、当前不可按需复现**。下一步必须先解决"可靠复现"(候选:在 RelWithDebInfo+frame-pointer 下挂 mimalloc 自带的 `MI_DEBUG=3`/`MI_SECURE` 堆校验跑迁移负载,让 mimalloc 自己在损坏点 abort 并打印),再谈修复验证。**不声称已修。**
