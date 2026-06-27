# sched-01：fcontext 上下文切换 + fiber 切换证明

| 字段 | 值 |
|---|---|
| **阶段** | P4 |
| **模块** | sched/ |
| **Tier** | 0 |
| **LOC 估** | 2-3k |
| **骨架?** | ✅ |
| **状态** | ✅ 完成（MSVC 19.50 masm + **Linux/GCC14.2 GAS** 双汇编验证，fiber 12 case 全过；arm64/macOS 4 变体已 vendor 未编）|

> **cloc 留档（自写 vs vendored 分账，诚实）**：HYLE **自写 226 行** = 51 头（fiber.h+fcontext.h）+ 165 cc（fiber.cc + fiber_test）+ 10 CMake；**vendored asm 737 行**（18 文件，boost.context，**非 HYLE 编写**，单独计）。
> **自写远低于估 2-3k，非 stub**——骨架只证切换：fiber 封装 + 10 行 shim，重活在 vendored asm（不计自写）。run-queue/worker 循环/work-stealing/栈池/协程胶水全在 sched-02..08。fiber 测试**真证栈切换**（局部变量跨两次 yield 含嵌套帧存活、交替序精确、双 fiber 独立栈、yield 越界 abort），非寂寞版。
> **实现注记**：① fcontext 选型见 sched-design §1（已锁）。② Win x64 masm 三例程经 ml64 编，`/DBOOST_CONTEXT_EXPORT=EXPORT`（仅 ASM_MASM 语言，genex 隔离）使 PROC public。③ `Fiber` 的 `HYLE_SCHED_API` 放方法非类（避 mem::Allocator 成员触发 C4251，同 Parker）。④ 栈对齐用 `HYLE_CACHE_LINE_SIZE` 单一来源、最小栈 `kMinStackSize`、默认 `kDefaultStackSize`（消魔数）。⑤ 构建坑修复：MSVC 全局/警告 flag 用 `$<COMPILE_LANGUAGE:C,CXX>` 隔离不喂给 ml64，且**每 flag 单独 quoted genex**（`;` 在 `$<>` 内会被 add_compile_options 提前 list-split）。
> **诚实缺口**：仅 Win x64 真编测；其余 5 平台 asm 在树未编未验。Windows SEH 跨 yield 未测（没抛异常跨 fiber）。

> **2026-06 更新（WSL 补验，第二平台）**：本机 WSL Ubuntu 24.04 + g++-14.2 已 wire 并真验 **Linux x86_64 SysV/ELF GAS** fcontext（`.S` 走 ASM 语言，`.globl` 导出符号、无需 BOOST_CONTEXT_EXPORT）——fiber 全 12 case 通过含嵌套 yield 栈保存铁证。**两套独立汇编（Win masm + Linux GAS）均验**，fcontext 选型与封装得到双平台印证。剩 arm64×2 + macOS×2 共 4 变体未编未验（无对应机）。SEH 跨 yield 仍未测。

## 前置依赖

- 硬前置：`mem-01`（fiber 栈走 `Allocator`）、`container-01`（run-queue，本步暂不接）、`platform-01`（worker 线程，本步暂不接）。
- 决策依据：`sched-design.md` §1（fcontext 选型已锁、6 平台矩阵、栈管理、落地清单）。

## 目的（为什么做这一步）

整条调度脊的根基是"能在用户态切换有栈上下文"。fcontext 是这件事的最小原语。本步验掉**最高风险**：fcontext 汇编能在本工具链跑通，一个 fiber 能**切入→让出→恢复→结束**。切换不通，后面 work-stealing/completion 全是空中楼阁。

## 在架构中的位置

walking skeleton 第六块（终点）。本步**只证 fiber 切换**（用户拍板）：fcontext + `Fiber` 封装 + 最小固定栈。**run-queue 接入 + 单 worker 循环留 sched-02**（避免一次引入 切换+栈池+队列 三处风险）。

## 关键设计决策

### D1：fcontext 选型已锁（sched-design §1，不再议）
- 取 boost.context 三裸汇编例程 `jump`/`make`/`ontop_fcontext`，**不取 boost.fiber**。
- 自写 `fcontext.h`（`fcontext_t=void*`、`transfer_t={void*,void*}` 纯 POD + `extern "C"` 三声明）切断 boost。

### D2：vendor 全 6 平台 asm，只编 Win x64（用户拍板）
- `third_party/fcontext/asm/` 拷入 6 平台×3 例程=**18 个 asm**（从 boost commit `d2142b6`，BSL-1.0 + LICENSE + README 出处）：x86_64/arm64 × {sysv_elf_gas, sysv_macho_gas, ms_pe_masm/aapcs_pe_armasm}。
- **本机只编测 Win x64 masm 三例程**（ml64，`/DBOOST_CONTEXT_EXPORT=EXPORT` 使 PROC public 可链接）。其余 15 份在树未编，标未验证——换机/CI 再开。

### D3：Fiber 封装 + thread_local 当前 fiber
- `Fiber(alloc, stack_size, entry, arg)`：从 `mem::Allocator` 取固定栈，`make_fcontext` 在栈顶建上下文。
- `resume()`（外部调，切入/续跑）/ `Fiber::yield()`（fiber 内调，让回 resumer）/ `done()`。经 `thread_local current` 让 `yield()` 找到自己。trampoline 跑完 entry 标 finished 并跳回。
- **栈 guard page 暂不做**（留栈池里程碑）；本步纯 `Allocator` 固定栈。

## 具体事务

- [ ] `third_party/fcontext/`：18 asm + LICENSE_1_0.txt + README（出处/取舍）。
- [ ] `src/sched/fcontext.h`：`extern "C"` 三例程 + `transfer_t`。
- [ ] `include/hyle/sched/fiber.h` + `src/sched/fiber.cc`：`Fiber` 封装。
- [ ] CMake：`enable_language(ASM_MASM)`（仅 MSVC x64），sched 模块编 3 masm + fiber.cc，接 hyle_mem。
- [ ] **不做**：run-queue/worker 循环（sched-02）、work-stealing（sched-03）、guard page/栈池、ontop 的高层用法、协程胶水。

## 交付物

- third_party/fcontext + fcontext.h + fiber.h/.cc + 测试。最小演示：起一个 fiber，它跑→`yield()`→`resume()` 续→结束，主侧观察到分段执行。

## 验收标准

- **编译**：Win x64 真编（MSVC 19.50 + ml64）；其余 5 平台 asm 在树未编，标未验证。
- **测试**：fiber 切入执行、`yield` 让出后主侧续控、再 `resume` 从让出点续跑、entry 返回后 `done()` 真、计数器按 切换序 递增（证明栈被正确保存/恢复）。
- **硬指标**：一个 fiber 内 `yield` 两次、主侧 `resume` 三次，fiber 内局部变量跨 yield 保持（栈切换正确的铁证）。
- **基准（留 sched-02+）**：百万次切换吞吐对标"十亿次/秒"量级——本步不做，只证正确。

## 参考

- `sched-design.md` §1。
- boost.context（Oliver Kowalke，BSL-1.0），`libs/context/src/asm/`。
- WG21 P0876 `fiber_context`（未来替换位）。
