# HYLE

**HYLE（HYbrid LEvels）是一个 C++ 基础设施库**：C++ 用户可消费 headers + shared、headers +
static 或按单独授权取得完整源码；跨语言 runtime 能力只在有真实任务时，通过 shared library 的
Native FFI 提供。当前 core 已验证“ABI bootstrap + 两个 product-identity symbols”这一
pre-release direct-binding 形状，但没有已发布 binding，因此没有兼容包袱；其余模块先裁决是否
需要 FFI，无任务可明确 `N/A`。
项目没有面向原生 C 程序员的独立产品 API。它把四样东西做在**同一套底座**上：

1. **调度** — 一个调度器同时管 线程 / 进程 / 纤程 / 协程
2. **内存** — 分配器 / arena / cache-line 对齐 / NUMA
3. **计算** — CPU SIMD + GPU（CUDA / Vulkan），算子借 ggml
4. **IO** — proactor 网络（TCP/UDP/TLS/HTTP/1.1+2）+ 文件 + 跨进程共享内存

**差异点在"同一套底座"**：GPU 算完、网卡收到包、fiber 主动让出——都回到同一个调度器、唤醒同一个 fiber，而不是四个子系统各开线程池、各跑事件循环互相打架。所以 HYLE 不是"又一个 hashmap / 日志库"，而是让 计算 / 并发 / 内存 / IO 无阻抗拼在一起的那层地基。

**一等服务对象**：
1. **类 llama.cpp 推理引擎 + 通用加速库（v1 北极星）** — 四样全压：高并发 RESTful serving + 批调度 + paged KV-cache + GPU/CPU kernel。llama.cpp 现在只能东拼西凑（vendor cpp-httplib + ggml + 自写线程），HYLE 做它脚下那层统一底座。
2. **CAD/BIM/CAE 工程计算管线** — 参数化依赖 DAG、并行重算、CPU/GPU kernel、海量几何/场数据
   与交互式 IO 同时压 graph/sched/compute/simd/mem。蜂群 agent 是调度/IO 的消费者场景之一，
   不是第二北极星。

**唯一的存在理由是把性能做到业界顶尖（SOTA）**：回到 C 的数据/调用约定形状、贴硬件、混合异构，把性能做到极致（时间与空间复杂度冲突时优先时间）。标尺是各领域参考库（hashmap→absl/F14、分配器→mimalloc、队列→moodycamel/tokio、SIMD→Highway/xsimd、调度→tokio/Seastar），不是 std——赢 std 只是最低门槛。每个性能件都有对标 SOTA 的真实 benchmark，赢多少、输多少都诚实登记为优化目标。对标数字见 [`docs/performance/`](./docs/performance/README.md)，逐里程碑实现进度见 [`docs/milestones/README.md`](./docs/milestones/README.md)。

HYLE 专注于**高性能计算**和性能优化，尤其是时间复杂度上，尽量覆盖计算优化的主要象限：硬件加速、算法与数据结构、系统调用优化 和 并发。为常用的需要计算加速的场景提供合适的工具。
用户需要一个加速框架，他们不希望在写逻辑的时候要自己从头实现和操作系统/硬件平台/分布式网络的交互细节；他们希望一个库，可以帮他们准备好针对五花八门硬件和各种工况的优化/调试/容器等工具。

## 上手

**想把 HYLE 当依赖用？先看 [使用指南 `docs/guide/usage-guide.md`](./docs/guide/usage-guide.md)**——构建选项、各模块最小用法、SIMD 编译变体模型、C++ source / Native FFI 边界（面向库使用者）。可运行的示例工程见最外层 [`samples/`](./samples/)。

**收到或制作 HYLE 交付物？看 [三种交付与用户接入 `docs/guide/distribution.md`](./docs/guide/distribution.md)**——
headers+shared、headers+static 与完整源码分别给出可直接复制的用户 recipe，并与具体授权范围分开处理。

## 文档

文档目录的职责、冲突优先级和更新纪律见
**[`docs/README.md`](./docs/README.md)**。设计、使用、性能和施工记录不再互相复制。

## 设计文档

完整的设计、背景、命名、源码位置、资产负债表、scope 与各支柱选型，集中在架构设计文档（但代码本身仍是唯一信源）：

**→ [`docs/architecture/README.md`](./docs/architecture/README.md)**

### 总览与规范（写代码前必读）

| 文档 | 内容 |
|---|---|
| ⭐ [实现速查表](./docs/architecture/implementation-cheatsheet.md) | **写代码前只读这一张**：聚合每次实现都要用的硬规则——std 红线 / 失败模型 / 分层判据 / Native FFI / 里程碑 CHECKLIST / 构建环境 |
| 🛠 [开发环境准备](./docs/setup/README.md) | **贡献者从零建环境**：按平台（Windows MSVC / WSL GCC14+TSan / macOS）分文件，构建·测试·性能测试三件事 + 依赖（OpenSSL/CUDA/ggml）|
| [架构总入口](./docs/architecture/README.md) | 定位、模块划分、贯穿性设计原则、项目背景与命名 |
| [动机](./docs/architecture/motivation.md) | 为什么是现在/为谁做：AI-native 四压负载 + 类 llama.cpp 推理引擎北极星 + 比 Seastar 多异构维度 |
| [工程章程](./docs/architecture/engineering-charter.md) | 动手前最终依据：C++23 标准 / 错误模型 / 测试 / 边界 / 对外 FFI |
| [编码规范](./docs/architecture/coding-conventions.md) | 每一行怎么写才不破范式：std 分层 / 宏纪律 / Native FFI 范式 / clay 写法继承表 |

### 背景与盘点

| 文档 | 内容 |
|---|---|
| [clay 资产负债表](./docs/architecture/clay-asset-liability.md) | clay 六子系统逐项盘点：继承 / 重写 / 从零造 |
| [clay 覆盖映射](./docs/architecture/clay-coverage-map.md) | clay 每个文件 → HYLE 对应物/状态/LOC 的逐文件吸纳完整度 |
| [scope 定义](./docs/architecture/hyle-scope.md) | 范围边界、完整组件/支柱划分、体量估算 |

### 模块设计

| 文档 | 内容 |
|---|---|
| [platform 跨 OS](./docs/architecture/platform-design.md) | CPU/thread/wait/VM/fs/dynlib 机制、系统/进程视图、内外职责和跨平台差异 |
| [mem 内存](./docs/architecture/mem-design.md) | host-RAM 边界、allocator/region/pool/stack/diagnostics、C++ source / Native FFI 候选与 NUMA/大页 |
| [time 时间与定时器](./docs/architecture/time-design.md) | typed clocks/deadline/sleep、exact-ns radix/common timer queues、single-owner 与 Native FFI 候选 |
| [container 容器](./docs/architecture/container-design.md) | 唯一造不借的 T0 + SwissTable SIMD 探测 + SmallVector/intrusive/lock-free（供 sched）|
| [simd 向量化](./docs/architecture/simd-design.md) | D-tag/七后端 + 自有 math/dtype/算法核 + Runtime/native/exact 双轨派发 |
| [sched 调度器](./docs/architecture/sched-design.md) | fcontext 切换 + 无锁 work-stealing + Waker completion 脊 + 协程/fiber 统一 |
| [io 网络设计](./docs/architecture/io-design.md) | proactor 模型 + 四后端 + 调度绑定 |
| [compute 异构计算](./docs/architecture/compute-design.md) | device/stream/kernel 抽象 + ggml 解耦 + shader 一等公民 |
| [lifecycle 失败边界](./docs/architecture/lifecycle-design.md) | fail-fast 绝不 try-catch + 取消/背压/关闭/隔离 + 跨 Native FFI 错误 |

### 路线与验证

| 文档 | 内容 |
|---|---|
| [实现路线图](./docs/architecture/roadmap.md) | 依赖图（无环）+ R0–R6 七波施工序 + 关键路径 + 验收门槛 + 决策清单 |
| [性能对标](./docs/performance/README.md) | vs std / vs SOTA 两桶分列 + Amdahl 整体观 + benchmark 验证（独立 `docs/performance/` 目录） |
| [参考库映射](./docs/architecture/references.md) | clone 的 30+ 参考库 → HYLE 模块，借/参考 + 开工优先级 |
| [已知问题](./docs/architecture/known-issues.md) | 悬而未决的 bug / 存疑项 / 待目标硬件复验的清单 |

### 里程碑与使用指南（不在 architecture 下）

| 文档 | 内容 |
|---|---|
| [里程碑索引](./docs/milestones/README.md) | Jira 式任务索引：逐项目的、依赖、验收、提交和当前待办；历史任务中的旧名/旧数字不是当前产品说明 |
| [使用指南](./docs/guide/usage-guide.md) | 面向库使用者：构建选项、各模块最小用法、SIMD 编译变体模型、C++ source / Native FFI 边界 |
| [调用安全契约](./docs/guide/usage-contract.md) | 为 SOTA 性能保留的特定调用约束（fiber/thread_local、分配器跨线程 free）|

## 目录结构

```
hyle/
├─ README.md              本文件（项目门面：简介 + 目录）
├─ include/hyle/          公开头（core/util-core/error/obs/math/platform/time/mem/simd/container/util/sched/io/compute/graph）
├─ src/                   实现（.cc，按模块分目录）
├─ tests/                 GoogleTest 单元/并发/TSan 测试
├─ bench/                 nanobench 对标 SOTA（container/mem/sched/simd/io/time/platform/error + rust/ crossbeam）
├─ samples/               C++ 用户示例 + C 语法 package/schema conformance fixtures
├─ scripts/               工具（bench-stat.py 统计法等）
├─ third_party/           已审查的 in-tree vendored/ported 资产；其他 pinned 依赖由 FetchContent 取得
└─ docs/
   ├─ setup/              开发环境准备（按平台：windows/linux-wsl/macos + dependencies；构建/测试/性能测试 × 三平台）
   ├─ architecture/       当前架构、边界和长期设计决策
   ├─ performance/        性能方法、原始结果、结论和待补证据
   ├─ milestones/         Jira 式任务、验收与提交记录
   ├─ guide/              使用指南 + 调用安全契约（面向库使用者）
   └─ debug/              调试/事故追查记录
```
