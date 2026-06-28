# HYLE

**HYLE 是一个 C++ 基础设施库**（编译成带 C-ABI 的共享库），给上层当地基——两个一等服务对象：**① 蜂群 agent**（v1 北极星，百万 IO-bound agent）和 **② AI 推理引擎**（高并发 serving + paged KV-cache + 批调度 + GPU kernel，HYLE 当引擎下面的 infra，不与 vLLM 正面竞争）；训练框架为较弱场景。它把四样东西做在**同一套底座**上：

1. **调度** — 一个调度器同时管 线程 / 进程 / 纤程 / 协程
2. **内存** — 分配器 / arena / cache-line 对齐 / NUMA
3. **计算** — CPU SIMD + GPU（CUDA / Vulkan），算子借 ggml
4. **IO** — 网络（proactor）+ 文件 + 跨进程共享内存

**差异点在"同一套底座"**：GPU 算完、网卡收到包、fiber 主动让出——都回到同一个调度器、唤醒同一个 fiber，而不是四个子系统各开线程池、各跑事件循环互相打架。所以 HYLE 不是"又一个 hashmap / 日志库"，而是让 计算 / 并发 / 内存 / IO 无阻抗拼在一起的那层地基。

> HYLE = HYbrid LEvels。回到 C、贴硬件、混合异构、把性能做到极致（SOTA）。是对 `slate-render` 中 clay infra 层的重构与扩大。

## 核心目的：性能做到 SOTA（不是"比 std 快"就行）

**HYLE 是一个高性能基础设施库，唯一的存在理由是把性能做到业界顶尖（SOTA）。** 这条是第一性的，凌驾于"功能完整""代码量"之上：

- **标尺是参考库，不是 std**：赢 `std::unordered_map`/`std::async` 是**最低门槛**——std 在这些点上本就不是为极致性能设计的。真正的标尺是**各领域的 SOTA 参考库**:hashmap→`absl::flat_hash_map`/F14、分配器→mimalloc、队列→moodycamel/tokio、SIMD→Highway/xsimd、调度→tokio/Seastar。**做到等同或超过它们，HYLE 才有价值；只赢 std 不算数。**
- **用数据说话,输了也报**:每个性能件必须有**对标 SOTA 的真实 benchmark**(非推算),诚实印出数字——赢在哪、输多少、差距登记为优化目标。掩盖差距 = 自欺。
- **当前实测铁证**:SwissTable 100k uint64(Linux GCC14)——vs `std::unordered_map` find 2.4x/insert 5.3x/erase 3.5x(轻松赢 std);**vs `absl::flat_hash_map`** insert **1.38x**、erase **1.29x**(超 SOTA),但 find **0.77x**(慢 absl ~25%,**已登记为优化目标**)。这就是"对标 SOTA"的意义:它逼出了 std 对标永远看不到的真实差距。

## 当前实现状态

> 诚实标尺:**已实现的都经双平台(Win MSVC + Linux GCC14)真编真测**,lock-free/并发件过 WSL ThreadSanitizer。逐里程碑进度(预测 vs 真实 LOC + clay API diff)见 [`docs/milestones/README.md`](./docs/milestones/README.md);clay 吸纳完整度见 [`clay-coverage-map.md`](./docs/architecture/clay-coverage-map.md)。性能对标见 [`performance.md`](./docs/architecture/performance.md)。

| 模块 | 已实现 | 状态 |
|---|---|---|
| **error** | Status(8B 指针标签内联) + `std::expected` + ASSIGN_OR_RAISE | ✅ |
| **obs** | 结构化日志:level 门控 + sink 注册表 + stderr 彩色 sink(非变参 vtable + std::format) | 🔶 hook 待 |
| **platform** | thread + park(futex/WaitOnAddress) + **dynlib(RAII)** + **fs/env(stat/mkdir/iterate/cwd/temp/home/getenv,buffer-fill+visitor)** | 🔶 vm 待 |
| **mem** | Allocator vtable(修 alignment) + **mimalloc backing(Win 4–12x>默认堆)** + MemoryArena(无锁) + Arena bump + ObjectPool + StackAllocator | 🔶 NUMA/大页 待 |
| **time** | 单调钟 + 墙钟 + sleep + Instant(ns) | ✅ |
| **container** | XXH3 哈希 + CArray + **SwissTable(SIMD 探测,对 std 快 2.7–6.1x;type-erased 冷核 + NEON)** + SmallVector + intrusive list + lock-free MPMC/SPSC/work-stealing 队列 + **string_ref/string_buf + 字节串算法 + path 路径操作** | ✅ |
| **util** | **Ref<T> 引用计数 + ScopeGuard/DEFER + 编译期 type_name + Registry 自注册** | ✅ |
| **sched** | fcontext 纤程切换 + per-core run-queue + work-stealing 调度器 + **Waker completion 脊(suspend/wake)** | 🔶 deadline/协程/取消 待 |
| **bench** | nanobench 设施 + SwissTable/string 对标 std | ✅ |
| **simd** | 运行期 CPUID dispatch 骨架 + vec<T,N> D-tag 抽象(照搬 Highway 契约) + **scalar/SSE4.2/AVX2 后端(对标 Highway 追平,4.6x over auto-vec)** | 🔶 NEON/AVX512/变体 dispatch/dtype 待 |
| io / compute / graph | — | 📋 规划 |

**性能验证(已落地部分)**:SwissTable vs `std::unordered_map`(100k keys,双平台):find-hit 2.7x、find-miss 2.3x、insert 6.1x、erase 4.1x。

**边界诚实**:已实现件局限于 **x86-64 Windows/Linux 真验**;ARM64 + macOS 路径多为"已写未验"(无测试机);simd 仅 scalar 后端+dispatch 骨架(各 ISA intrinsic 后端未落);io/compute/graph 三支柱尚未开工。

## 设计文档

> **想把 HYLE 当依赖用？先看 [`docs/usage-guide.md`](./docs/usage-guide.md)**——构建选项、各模块最小用法、SIMD 编译变体模型、C-ABI 边界（面向库使用者）。

完整的设计、背景、命名、源码位置、资产负债表、scope 与各支柱选型，集中在架构设计文档（唯一信源）：

**→ [`docs/architecture/README.md`](./docs/architecture/README.md)**

| 文档 | 内容 |
|---|---|
| [动机](./docs/architecture/motivation.md) | 为什么是现在/为谁做：AI-native 四压负载 + 蜂群 agent 北极星 + 比 Seastar 多异构维度 |
| [架构总入口](./docs/architecture/README.md) | 定位、模块划分、贯穿性设计原则、项目背景与命名 |
| [工程章程](./docs/architecture/engineering-charter.md) | 动手前最终依据：C++23 标准 / 错误模型 / 测试 / 边界 / 对外 FFI |
| [编码规范](./docs/architecture/coding-conventions.md) | 每一行怎么写才不破范式：std 分层 / 宏纪律 / C-ABI 范式 / clay 写法继承表 |
| [clay 资产负债表](./docs/architecture/clay-asset-liability.md) | clay 六子系统逐项盘点：继承 / 重写 / 从零造 |
| [scope 定义](./docs/architecture/hyle-scope.md) | 范围边界、7 模块划分、体量估算 |
| [io 网络设计](./docs/architecture/io-design.md) | proactor 模型 + 四后端 + 调度绑定 |
| [sched 调度器](./docs/architecture/sched-design.md) | fcontext 切换 + 无锁 work-stealing + Waker completion 脊 + 协程/fiber 统一 |
| [lifecycle 失败边界](./docs/architecture/lifecycle-design.md) | fail-fast 绝不 try-catch + 取消/背压/关闭/隔离 + 跨 C-ABI 错误 |
| [compute 异构计算](./docs/architecture/compute-design.md) | device/stream/kernel 抽象 + ggml 解耦 + shader 一等公民 |
| [simd 向量化](./docs/architecture/simd-design.md) | 精简可移植 vector + 运行时 dispatch + dtype(fp16/bf16) + 借 sleef 数学核 |
| [mem 内存](./docs/architecture/mem-design.md) | 通用 malloc 借 mimalloc + 自造专用分配器 + 修 5 bug + NUMA/大页/64B 对齐 |
| [container 容器](./docs/architecture/container-design.md) | 唯一造不借的 T0 + SwissTable SIMD 探测 + SmallVector/intrusive/lock-free（供 sched）|
| [实现路线图](./docs/architecture/roadmap.md) | 依赖图（无环）+ 6 批施工序 + 关键路径 + 验收门槛 + 决策清单 |
| [性能预期](./docs/architecture/performance.md) | vs std 的时间/空间预期（两桶分列）+ Amdahl 整体观 + benchmark 验证 |
| [参考库映射](./docs/architecture/references.md) | clone 的 20+ 参考库 → HYLE 模块，借/参考 + 开工优先级 |

## 目录结构

```
hyle/
├─ README.md              本文件（项目门面）
└─ docs/
   └─ architecture/       架构设计文档（唯一信源）
      ├─ README.md         总入口
      ├─ clay-asset-liability.md
      ├─ hyle-scope.md
      ├─ io-design.md
      └─ compute-design.md
```
