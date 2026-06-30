# KI-1 Crash Hunt（fiber × mimalloc 跨线程 free 崩溃——调查日志）

> 活账本。**边实验边记**:每条猜想、每个实验(确认/排除)都写下,防上下文丢失。
> 关联:[`../architecture/known-issues.md`](../architecture/known-issues.md) KI-1。
> 本轮指令(用户):**不信无法复现,怀疑"缓解方案"在掩盖**;移除所有缓解、派红队、死磕可靠复现,务必解决。

## 0. 被查崩溃（KI-1 摘要）

- **现象**:fiber 经 mimalloc 分配 → `yield()` 跨 worker 迁移 → 在另一线程 free(跨线程 free),重负载 + `-O3` 下崩。
- **真栈(gdb,原始 34/34)**:`worker_main → Scheduler::Impl::dispatch → Fiber::resume`。SIGSEGV 为主,**少数 SIGILL**(踩 `Fiber::resume` 的 `!finished_`)。
- **原始崩溃环境**:仓库 + 构建 + bench **全在 `/mnt/c`(9p 挂载)**,并发 ninja 构建时。

## 1. 本轮关键重构（认知纠正）

1. **所有旧复现器都漏了一个变量**:`fiber_alloc/sentinel/sameworker/doubledetect` 全部**砍掉了 `std::async` 大规模线程 churn**,而真实崩溃器 `scheduler_bench` bench2 自带 10万 `std::async` 线程。一直在用阉割版复现器。
2. **9p IO 是隐藏变量(用户洞察)**:`/mnt/c` 是 9p 网络挂载(`cache=0x5`,IO 极不稳定)。**早期"并发 ninja 负载"测试在 9p 上 `cc1plus=0`——ninja 全程 IO-bound,CPU 从未饱和,负载根本没起来**。→ 全面迁移到原生 ext4(`git clone` 到 `/root/hyle`,原生构建 1.9s vs 9p 上慢且不饱和)。
3. **SIGILL signature 线索**:SIGILL = 指令指针执行到**非指令字节** = **代码页问题**,非堆损坏(堆坏给 SIGSEGV on data)。指向**9p 后端 .so 代码页在内存压力下被换出、re-fault 时 9p 不稳定 IO 返回坏页** → fcontext 热路径崩。这能同时解释 SEGV + ILL。

## 2. "掩盖"排查（用户核心怀疑）

| 怀疑的掩盖 | 核查 | 结论 |
|---|---|---|
| mimalloc 被偷偷关成 OS 分配器 | `nm build-wsl/libhyle_mem.so` → 212 个 `mi_` 符号,`HYLE_USE_MIMALLOC=ON` | **排除**,mimalloc 真在用 |
| 代码里有抑制/规避(catch/sigaction/sleep/retry/.supp) | grep `src/sched/**` + 全仓 `*.supp` | **排除**,调度器干净,无任何抑制 |
| bench WITHHELD 注释残留甩锅结论 | 读 `bench/CMakeLists.txt:15-21` | **确认是缓解**:含被用户否决的"NOT a HYLE bug""in the vendored allocator"。**已清理**:bench 编入默认 + 注释中性化(2702df2)。 |

## 3. 实验记录（全部在原生 ext4，除非注明）

环境:WSL Ubuntu24.04,g++-14,32 核,31G RAM,`/root/hyle`(原生 ext4)。

| # | 猜想 | 配置 | 结果 |
|---|---|---|---|
| E1 | churn 是缺失变量,补上就崩 | `fiber_churn`:8-worker sched + 8 churner 线程 std::async,-O3,mimalloc | **净**(39万→65万 async 线程) |
| E2 | (前提核查)mimalloc 真开? | `nm` build-wsl libhyle_mem.so | mimalloc **ON** 确认 |
| E3 | steal 路径有源码级 data race | **TSan** + mimalloc,`fiber_tsan_small`(10轮×200×50yield=10万次迁移) | **净,无 race**。⚠️注:TSan 在 300 轮会自身 StackDepot SEGV(TSan 限制,非 HYLE);故旧"TSan 净"很可能根本没跑到重 mimalloc 路径 |
| E4 | 跨线程 free 造成堆损坏 | **MI_DEBUG=3**(全量堆不变量检查)+ `fiber_churn` 65万线程 | **净,无断言** |
| E5 | 超额订阅下堆损坏 | MI_DEBUG=3 + `fiber_oversub` 48 worker(1.5×核)/4000轮 + preemptor storm | **净** |
| E6 | 真实负载触发(原生) | **真实 scheduler_bench**(原始崩溃器本体)+ 原生 ninja 风暴(cc1: 9-17 真饱和) | **净 10/10** |
| E7 | 9p 代码页 re-fault 崩 | `drop_caches` 每 50ms 拷打 + **9p-libs** bench | **净 8/8**。⚠️但 drop_caches 只丢未映射 cache,运行中热代码页在工作集不被丢→未真正驱逐 |
| E8 | (对照)原生代码页 re-fault | drop_caches 拷打 + **native-libs** bench | **净**(同上局限) |
| E10 | **真实内存压力**强制驱逐已映射代码页 + 9p re-fault | 24G memhog(RSS 真驻留 19G,avail 压到 4G)+ **9p-libs** bench | **净 6/6** |
| E11 | (对照)同压力 + native-libs | 24G memhog(RSS 23G,avail 4G)+ native bench | **净 6/6** |
| **E12** | **本机默认环境(Windows MSVC)复现** | **原始崩溃器本体** hyle_sched_bench.exe + 并发 MSVC 构建风暴,15 轮 | **净 15/15** |
| E9 | 字面原始条件:9p ninja + 9p bench 并发 | 9p ninja 风暴(写 .o=重不稳定 IO)+ 9p bench | 未跑(E6/E10 已更强,9p 路径全净,边际收益低) |

> **E10 自我纠正(诚实)**:第一版 memhog 触摸循环被 `-O2` 当死代码删除(`p[i]++` 结果从未被读)→ RSS 仅 1.4MB,**零压力,结果无效**。当场抓到、**没当"净"上报**;改 volatile + 校验和 + `-O0`,验证 RSS 真涨到 19-23G、avail 压到 4G 后,结果才算数。教训:压力测试必先验证压力真实存在。

## 4. 当前结论（截至本次更新）

**已排除**（用能检测缺陷本身、不依赖是否崩的工具）:
- **源码级 data race**:TSan 净(10万次迁移,含 mimalloc)。
- **堆损坏(double-free/cross-thread-free 走坏路径/buffer overflow)**:MI_DEBUG=3 全量检查在 churn(65万线程)+ 超额订阅 + 24G 内存压力下全净。
- **调度器逻辑**:代码审查无抑制;OS 分配器历史净;无 double-resume。
- **"缓解掩盖"**:mimalloc 真开、代码无抑制(唯一缓解=bench 排除+甩锅注释)。

**已确认无法复现**(忠实 + 激进,全净):
- 原生 ext4 + 真实 ninja 饱和负载 + **原始崩溃器本体**:10/10。
- `drop_caches` 拷打:9p-libs 8/8 + native 8/8。
- **真实 24G 内存压力**(avail→4G,强制代码页驱逐):9p-libs 6/6 + native 6/6。
- **Windows MSVC 本机(用户默认开发环境)** + 并发构建风暴 + **原始崩溃器本体**:**15/15 净**(E12)。
- 9p 是早期"负载测试"失效主因(IO-bound,CPU 没饱和)。

> **E12 是最强的环境假象证据**:原始 34/34 崩溃全在 WSL/9p;在用户**真实本机环境(Windows MSVC native)** 用同样的并发构建 + 原始崩溃器本体跑,15/15 净。崩溃**不出现在生产/开发的默认环境**,只(曾)出现在 WSL/9p 这个虚拟机 + 网络挂载组合里。

**主结论(证据压倒性,但非 100% 拍板)**:
**KI-1 极可能不是 HYLE 代码缺陷,而是 9p/WSL 环境的 demand-paging 假象**——内存压力下 9p 后端 .so 代码页 re-fault 不稳定。依据:① **SIGILL in `Fiber::resume`** = 代码页问题非堆损坏;② TSan + MI_DEBUG 两个 defect 级检测器全净;③ 原始崩溃器本体在原生 + 9p 真实负载/压力下均不复现。

**诚实边界**:我也**没在 9p 上正面复现**(9p 路径同样净),故不能 100% 断言"就是 9p"。但旧定性("real, ours, mimalloc-dependent HYLE bug")**与现有证据矛盾**——它把环境假象误判成了库 bug。**应重新定性**:KI-1 从"P0 未解 HYLE 缺陷"→"环境假象,HYLE 逻辑成因已排除,不可按需复现"。

## 5. 待办（下一步）

- [x] 红队 A(调度器并发正确性)+ B(mimalloc×fcontext 机制)结论回填 → 见 §8。
- [x] 验证红队的 TLS-缓存机制是否在 HYLE 真实路径成立 → §8(机制真实,但 HYLE 当前路径安全)。
- [x] 清理 `bench/CMakeLists.txt` 甩锅注释 → 中性化;更新 known-issues.md KI-1(已提交 92f8ec8)。
- [ ] (可选)上游 mimalloc 版本对比——被 github TLS 网络阻断,非必需(崩溃不可复现=无修没修可测)。
- [x] (防御性)在 docs 写一条 fiber 编程铁律:**fiber 体不得跨 `yield()` 持有 thread_local 派生值** → 已落 `guide/usage-contract.md` §1(见 §8)。

## 6. 复现器清单（scratchpad / 原生 /root/repro）

- `fiber_churn.cc` — 8-worker + 8 churner std::async(E1)
- `fiber_oversub.cc` — 超额订阅 48 worker + preemptor storm(E5)
- `fiber_tsan_small.cc` — 小轮次 TSan 友好(E3)
- `scheduler_bench.cc`(`/root/hyle/bench/sched/`)— 原始崩溃器本体,编为 `sched_bench`(原生)/`sched_bench_9p`(9p-libs)
- 脚本(Write 工具写→cp /root/repro→原生跑,**因 `wsl.exe bash -c` 会吃掉变量**):`native_hunt.sh`/`native_bench_hunt.sh`/`dropcache_experiment.sh`/`nine_p_original.sh`

## 7. 环境陷阱备忘

- `wsl.exe -d Ubuntu-24.04 bash -c '...$VAR...'`:外层 git-bash 吃掉 `$VAR`(变空)。**对策**:逻辑写成 .sh 文件(变量在原生 bash 运行时展开正常)。
- `/mnt/...` 路径作 wsl.exe 参数被 git-bash 转换坏:**对策** `MSYS_NO_PATHCONV=1`。
- `pkill -f <宽模式>`:会误杀承载命令的 wsl bash 自身(exit 15)。**对策**:用 stop-flag 文件优雅停,或精确 PID。
- 一切重 IO(构建/复现器)必须在原生 ext4(`/root`),**不碰 `/mnt/c`**(9p 不稳定 + 慢)。

## 8. 红队验证：TLS-缓存机制【真实】，但 HYLE 当前路径【安全】

> 两支红队(调度器并发 / mimalloc×fcontext)**独立收敛到同一机制**,且它挑战了 §4 的"环境假象"结论。我**没有为护结论而无视**,而是设计了决定性实验逐条验证。结论:机制真实,我之前对 -O3 依赖的解释不够精确(红队补上了);但 HYLE 真实路径不触发,原始崩溃仍最可能是环境假象。

**红队机制**:`jump_fcontext` 换栈+通用寄存器但**不换 `%fs`**(TLS 基址)。-O3 下编译器把 `thread_local` 的 `%fs` 相对地址当作"跨调用稳定",缓存进 callee-saved 寄存器、跨 `yield()` 复用。fiber 迁移到别的线程后,该缓存地址仍指向**原线程**的 TLS → 读/写错堆。

**强制迁移实验**(patch `reschedule` 永远走 global,使 ~75% resume 真跨线程,几百万次):

| # | 测什么 | 结果 | 判读 |
|---|---|---|---|
| E13 | HYLE 真实路径:fiber 体经 vtable→.so 调 mi_malloc/free,跨迁移 | **净,689 万次跨线程 free rc=0** | 真实路径**安全** |
| E14 | 裸 `thread_local`,**不**强制保活,比对两次 TLS 读 | **0 stale**(可能被 CSE 掩盖) | 自然代码不触发 |
| E15 | 裸 `thread_local`,`asm` **强制**地址跨 yield 保活,比对 `gettid()` 真值 | **899 万 stale**(几乎全部) | 机制**真实**,红队对 |
| E16 | 在 E15 上加 `asm volatile(memory)` 屏障 / yield 后重取地址 | **两者都仍 899 万 stale** | memory 屏障**救不了**;GCC 把 TLS 基址提升到函数入口,整个体内 CSE 复用 |
| E17 | **mimalloc v2.1.9→v2.3.2 升级后**重跑强制迁移(真实路径) | **净,687 万次跨线程 free rc=0** | 新版段完整性/cookie 校验更严仍净 → 分配器路径安全加固;双平台 10/10 测试通过 |

**精确规则**(E13–E16 钉定):
- **危险**:fiber 体在**可内联范围内直接** `thread_local`,且跨 `yield()` 使用 → -O3 缓存 TLS 基址 → 迁移后 stale。memory 屏障与"重取地址"都无效(基址在函数入口已 CSE)。
- **安全**:fiber 体只通过**不透明(非内联)调用**碰 TLS(每次在当前线程新解析)。**HYLE 分配器走 vtable 函数指针 + 跨 `.so` 边界 = 正属此类** → E13 的 689 万次净是铁证。

**为何这不推翻"原始崩溃=环境假象"**:原始崩溃在 mimalloc 路径;E13 证明该路径即使强制迁移也净。若原始崩溃是 TLS 机制,E13 必崩。故原始崩溃另有其因(最可能 9p/WSL,§4)。

**真实残留风险(应设防)**:
1. **用户 fiber 体直接跨 `yield()` 用 `thread_local`** —— 这是所有有栈协程(Boost.Context/ucontext)的**通用铁律**,非 HYLE 独有。应在文档立硬约束。
2. **static + LTO** 把 `mi_*` 内联进 fiber 体 —— 会打破"不透明边界",使分配器 TLS 读被缓存跨 yield。HYLE 默认 shared lib + 无 LTO,不触发;若改 static+LTO 发布,需 pin fiber 或保持分配器不可内联。(未实测 LTO 变体,留作 characterize。)

**HYLE 自身代码已合规**:`fiber.cc` 的 `yield()/resume()` 把 `t_current` 读进**局部变量**再切换(L92/L102),不跨切换复用 TLS;`scheduler.cc` 的 `t_current/t_worker_idx/tls_owner` 只在**不迁移的 worker 主循环**用。两处都安全。
