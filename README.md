# HYLE

**HYLE（HYbrid LEvels）是一个 C++ 基础设施库**，编译成带 C-ABI 的共享库，给上层当地基。把四样东西做在**同一套底座**上：

1. **调度** — 一个调度器同时管 线程 / 进程 / 纤程 / 协程
2. **内存** — 分配器 / arena / cache-line 对齐 / NUMA
3. **计算** — CPU SIMD + GPU（CUDA / Vulkan），算子借 ggml
4. **IO** — 网络（proactor + tcp/udp co_await socket；HTTP serving/client 规划中）+ 文件 + 跨进程共享内存

**差异点在"同一套底座"**：GPU 算完、网卡收到包、fiber 主动让出——都回到同一个调度器、唤醒同一个 fiber，而不是四个子系统各开线程池、各跑事件循环互相打架。所以 HYLE 不是"又一个 hashmap / 日志库"，而是让 计算 / 并发 / 内存 / IO 无阻抗拼在一起的那层地基。

**一等服务对象**：
1. **类 llama.cpp 推理引擎 + 通用加速库（v1 北极星）** — 四样全压：高并发 RESTful serving + 批调度 + paged KV-cache + GPU/CPU kernel。llama.cpp 现在只能东拼西凑（vendor cpp-httplib + ggml + 自写线程），HYLE 做它脚下那层统一底座。
2. 蜂群 agent（百万 IO-bound agent）——引擎的消费者之一，fiber 最契合但只压半边。训练框架为较弱场景。

**唯一的存在理由是把性能做到业界顶尖（SOTA）**：回到 C、贴硬件、混合异构，把性能做到极致（时间与空间复杂度冲突时优先时间）。标尺是各领域参考库（hashmap→absl/F14、分配器→mimalloc、队列→moodycamel/tokio、SIMD→Highway/xsimd、调度→tokio/Seastar），不是 std——赢 std 只是最低门槛。每个性能件都有对标 SOTA 的真实 benchmark，赢多少、输多少都诚实登记为优化目标。对标数字见 [`docs/performance/`](./docs/performance/README.md)，逐里程碑实现进度见 [`docs/milestones/README.md`](./docs/milestones/README.md)。

## 上手

**想把 HYLE 当依赖用？先看 [使用指南 `docs/guide/usage-guide.md`](./docs/guide/usage-guide.md)**——构建选项、各模块最小用法、SIMD 编译变体模型、C-ABI 边界（面向库使用者）。可运行的示例工程见最外层 [`samples/`](./samples/)。

## 设计文档

完整的设计、背景、命名、源码位置、资产负债表、scope 与各支柱选型，集中在架构设计文档（但代码本身仍是唯一信源）：

**→ [`docs/architecture/README.md`](./docs/architecture/README.md)**

### 总览与规范（写代码前必读）

| 文档 | 内容 |
|---|---|
| ⭐ [实现速查表](./docs/architecture/implementation-cheatsheet.md) | **写代码前只读这一张**：聚合每次实现都要用的硬规则——std 红线 / 失败模型 / 分层判据 / C-ABI / 里程碑 CHECKLIST / 构建环境 |
| [架构总入口](./docs/architecture/README.md) | 定位、模块划分、贯穿性设计原则、项目背景与命名 |
| [动机](./docs/architecture/motivation.md) | 为什么是现在/为谁做：AI-native 四压负载 + 类 llama.cpp 推理引擎北极星 + 比 Seastar 多异构维度 |
| [工程章程](./docs/architecture/engineering-charter.md) | 动手前最终依据：C++23 标准 / 错误模型 / 测试 / 边界 / 对外 FFI |
| [编码规范](./docs/architecture/coding-conventions.md) | 每一行怎么写才不破范式：std 分层 / 宏纪律 / C-ABI 范式 / clay 写法继承表 |

### 背景与盘点

| 文档 | 内容 |
|---|---|
| [clay 资产负债表](./docs/architecture/clay-asset-liability.md) | clay 六子系统逐项盘点：继承 / 重写 / 从零造 |
| [clay 覆盖映射](./docs/architecture/clay-coverage-map.md) | clay 每个文件 → HYLE 对应物/状态/LOC 的逐文件吸纳完整度 |
| [scope 定义](./docs/architecture/hyle-scope.md) | 范围边界、7 模块划分、体量估算 |

### 模块设计

| 文档 | 内容 |
|---|---|
| [mem 内存](./docs/architecture/mem-design.md) | 通用 malloc 借 mimalloc + 自造专用分配器 + 修 5 bug + NUMA/大页/64B 对齐 |
| [container 容器](./docs/architecture/container-design.md) | 唯一造不借的 T0 + SwissTable SIMD 探测 + SmallVector/intrusive/lock-free（供 sched）|
| [simd 向量化](./docs/architecture/simd-design.md) | 精简可移植 vector + 运行时 dispatch + dtype(fp16/bf16) + 借 Highway/xsimd 数学核(vendored) |
| [sched 调度器](./docs/architecture/sched-design.md) | fcontext 切换 + 无锁 work-stealing + Waker completion 脊 + 协程/fiber 统一 |
| [io 网络设计](./docs/architecture/io-design.md) | proactor 模型 + 四后端 + 调度绑定 |
| [compute 异构计算](./docs/architecture/compute-design.md) | device/stream/kernel 抽象 + ggml 解耦 + shader 一等公民 |
| [lifecycle 失败边界](./docs/architecture/lifecycle-design.md) | fail-fast 绝不 try-catch + 取消/背压/关闭/隔离 + 跨 C-ABI 错误 |

### 路线与验证

| 文档 | 内容 |
|---|---|
| [实现路线图](./docs/architecture/roadmap.md) | 依赖图（无环）+ 6 批施工序 + 关键路径 + 验收门槛 + 决策清单 |
| [性能对标](./docs/performance/README.md) | vs std / vs SOTA 两桶分列 + Amdahl 整体观 + benchmark 验证（独立 `docs/performance/` 目录） |
| [参考库映射](./docs/architecture/references.md) | clone 的 30+ 参考库 → HYLE 模块，借/参考 + 开工优先级 |
| [已知问题](./docs/architecture/known-issues.md) | 悬而未决的 bug / 存疑项 / 待目标硬件复验的清单 |

### 里程碑与使用指南（不在 architecture 下）

| 文档 | 内容 |
|---|---|
| [里程碑索引](./docs/milestones/README.md) | **唯一进度信源**：约 58 个里程碑，逐步计划 + 预测 vs 真实 LOC + clay API diff + 状态 |
| [使用指南](./docs/guide/usage-guide.md) | 面向库使用者：构建选项、各模块最小用法、SIMD 编译变体模型、C-ABI 边界 |
| [调用安全契约](./docs/guide/usage-contract.md) | 为 SOTA 性能保留的特定调用约束（fiber/thread_local、分配器跨线程 free）|

## 目录结构

```
hyle/
├─ README.md              本文件（项目门面：简介 + 目录）
├─ include/hyle/          公开头（core/error/obs/platform/mem/time/container/util/sched/simd/math/io）
├─ src/                   实现（.cc，按模块分目录）
├─ tests/                 GoogleTest 单元/并发/TSan 测试
├─ bench/                 nanobench 对标 SOTA（container/mem/sched/simd/io/time/platform/error + rust/ crossbeam）
├─ samples/               可运行示例工程
├─ scripts/               工具（bench-stat.py 统计法等）
├─ third_party/           vendored（mimalloc/Highway/xsimd/simdutf/nanobench…）
└─ docs/
   ├─ architecture/       架构设计文档（设计/背景/决策；代码才是唯一信源）
   ├─ milestones/         里程碑逐步计划（唯一进度信源）
   ├─ guide/              使用指南 + 调用安全契约（面向库使用者）
   └─ debug/              调试/事故追查记录
```
