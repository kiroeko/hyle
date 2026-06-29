# Known Issues（已知问题 / 未解决缺陷）

> 诚实账本：已发现但**尚未解决**的真实缺陷。不藏、不淡化。每条带：现象、已验证的事实、已排除项、复现条件、下一步。解决后移除并在对应里程碑/提交记录。

## KI-1 [已降级/重新定性]：fiber × mimalloc 跨线程 free 崩溃 —— 穷尽调查后判定为 9p/WSL 环境假象，非 HYLE 缺陷

> **状态变更(本轮深查)**:此条原为"P0 未解 HYLE 缺陷"。经一轮**穷尽、独立于崩溃时序**的调查,该定性**与证据矛盾**,现重新定性为**极可能是 9p/WSL demand-paging 环境假象**。完整实验账本(必要条件、已排除项、机制假说、复现尝试)见 [`../debug/ki-1-crash-hunt.md`](../debug/ki-1-crash-hunt.md)。**不再声称是 HYLE bug,也不 100% 拍板;以证据说话。**
>
> **defect 级工具排除 HYLE 逻辑成因**:TSan(源码级 race)10万次迁移净;**MI_DEBUG=3** 全量堆不变量 + 65万 async churn + 48-worker 超额订阅 + 24G 内存压力**全净零断言**;代码无任何抑制。
> **忠实+激进复现全失败**:原始崩溃器本体 scheduler_bench + 原生 ninja 真饱和(cc1:9-17)**10/10 净**;`drop_caches` 拷打 9p/native **8/8+8/8**;真实 24G 内存压力(avail→4G 强制代码页驱逐)9p/native **6/6+6/6**。
> **关键线索**:① 早期"重负载复现"在 9p 上 `cc1plus=0`(IO-bound,负载没起来)——用户的 9p 洞察命中;② 原始崩溃含 **SIGILL** = 代码页问题(非堆损坏的数据 SIGSEGV),指向 9p 后端 .so 代码页 re-fault 不稳定。
> **mimalloc 保持出厂默认 ON**:核心 fiber 场景经 MI_DEBUG=3 全量校验证明无损坏,不需为此关 mimalloc。
> **⚠️ 勿与 sched-05 的 TSan×fcontext SEGV 混淆(不同 bug)**:sched-05 修过一个 `__tsan_switch_to_fiber` 缺失导致的 **TSan 构建内** SEGV(libtsan 影子栈跟踪,release 零影响、已修)。KI-1 是 **release -O3 + mimalloc TLS-on-fcontext栈** 的运行时崩溃,**与 TSan 无关**——两者机制不同,别因都是"WSL fiber 崩"而当成一回事。本会话 conditional-notify 大增 fiber 迁移 + 原生 ext4 重压(NoLostWakeup 100×/marl bench 多轮)**未复现 KI-1**,与"9p 假象"一致(非证实)。
> **诚实边界**:9p 上也未正面复现(同样净),故不 100% 断言"就是 9p";但旧"HYLE bug"框架已被证伪。scheduler_bench **已编入默认 build**(提交 2702df2 解封)——若该环境假象再现,直接正面查,不藏。
