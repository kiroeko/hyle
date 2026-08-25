# HYLE 用户文档（docs/guide）

> 这个文件夹是**面向库使用者**的文档——把 HYLE 当依赖集成进自己工程的人看这里。
> 想读"为什么这么设计"的内幕（面向贡献者）去 [`../architecture/`](../architecture/)。
> 想看可运行的示例工程去仓库最外层 [`../../samples/`](../../samples/)。

本目录使用三个不能混用的名词：

- **C++ source API**：消费 `include/hyle/<module>/*.h`，由用户的 C++ translation unit 编译。
- **native C source helper**：需要另行编译 macro / `static inline` 或透明 pointer primitive；这是
  HYLE 内部桥接与遗留 surface 的分类，不是面向客户的 C 产品，也不是 Java/.NET/Rust/Python
  可直接绑定的 ABI。
- **native FFI 合同**：无需编译 C shim，只通过 shared library 导出的 C calling-convention
  符号、fixed-width POD、opaque handle 与 callback 直接绑定。本目录中的“direct FFI”均指这一层。

## 本目录文档

| 文档 | 给谁 / 讲什么 |
|---|---|
| [`usage-guide.md`](./usage-guide.md) | **怎么用**：先按 shared/static/source 选择消费入口，再查模块、构建选项、SIMD 变体、最小用法和 Native FFI 边界。**集成第一站。** |
| [`distribution.md`](./distribution.md) | **拿到什么就怎么用**：headers+shared、headers+static、完整源码三种用户接入方式；另含交付方制作命令、授权边界和离线依赖注入。 |
| [`core.md`](./core.md) | **core 选型入口 + C++ 自然写法**：先分清 source/header-inline 能力与“ABI bootstrap + 两个产品版本符号”的 direct FFI，再选 cache layout/prefetch、compiler contract 或 intrusive list。 |
| [`error.md`](./error.md) | **error 选型入口 + C++ 自然写法**：按运营失败/空值合同/程序错误/诊断边界决定 Status、Result、值合同或 fail-fast；direct FFI 再去 c-error 查 receiver 合同。 |
| [`mem.md`](./mem.md) | **mem 语言无关选型入口 + C++ 自然写法**：按 ownership/concurrency/capacity/lifetime 决定什么情况用什么；direct FFI 再去 c-mem 查 native FFI 合同候选。 |
| [`platform.md`](./platform.md) | **platform 语言无关选型入口 + C++ 自然写法**：按资源/生命周期选择 VM、文件、mapping、线程、等待、DynLib、CPU/NUMA；明确 OS 差异与当前拓扑边界。 |
| [`time.md`](./time.md) | **time 语言无关选型入口 + C++ 自然写法**：按 elapsed/墙钟/sleep/任意 deadline/同 duration timer 选择入口，并解释 single-owner、re-arm 与周期策略边界。 |
| [`util.md`](./util.md) | **util 选型入口 + C++ 自然写法**：ScopeGuard/Ref/Rng/SpinLock/type-name/demangle 什么情况选什么，明确 core/runtime 分层、复杂度与常见坑。 |
| [`obs.md`](./obs.md) | **obs 选型入口 + C++ 自然写法**：error/log/trace 怎么选，compile floor、exact-byte record、RAII sink 生命周期、递归与异步桥接，以及 native trace 的零成本边界。 |
| [`container.md`](./container.md) | **container 选型入口 + C++ 自然写法**：小序列/POD vector/flat map/字符串驻留/五类并发队列分别何时选择，明确失效、顺序和当前性能边界。 |
| [`c-error.md`](./c-error.md) | **error direct FFI receiver 合同（已按 managed binding 复审）**：NULL/opaque owning status、六个真实导出、caller-buffer、逐 artifact bootstrap 与 Java/.NET lifetime。 |
| [`c-platform.md`](./c-platform.md) | **platform native FFI 合同（managed binding 待复审）**：Parker/Thread/DynLib/VM/mapped_file/fs/cpu/numa 的 opaque/POD/自由函数候选面。 |
| [`c-mem.md`](./c-mem.md) | **mem native FFI 合同（managed binding 待复审）**：Status/out、opaque handle、allocator 与运行时分配入口候选面。 |
| [`c-time.md`](./c-time.md) | **time native FFI 合同（managed binding 待复审）**：Status、opaque timer/queue、同步 visitor、所有权和 shutdown 候选面。 |
| [`c-obs.md`](./c-obs.md) | **obs native FFI 合同（managed binding 待复审）**：Status/out、length-aware record/log、opaque sink、flush 与 runtime trace 候选面。 |
| [`c-util.md`](./c-util.md) | **util native FFI 合同（managed binding 待复审）**：Status/allocator、RNG POD、opaque Ref/Registry/SpinLock、demangle 与 callback 候选面。 |
| [`c-container.md`](./c-container.md) | **container native FFI 合同（managed binding 待复审）**：Status/out、字符串/路径、运行时擦除容器与 opaque concurrent handle 候选面。 |
| [`c-simd.md`](./c-simd.md) | **SIMD native FFI 合同（managed binding 待复审）**：whole-buffer 入口、低精度/排序 POD、RNG opaque handle 和 ISA 诊断候选面。 |
| [`c-sched.md`](./c-sched.md) | **sched native FFI 合同（managed binding 待复审）**：worker-stack/fiber、Waker、关闭、同步/消息/取消/blocking pool 的类型擦除候选面。 |
| [`c-io.md`](./c-io.md) | **io native FFI 合同（managed binding 待复审）**：proactor/completion、网络/文件提交、后端诊断、multishot/fixed-file 与共享内存候选面。 |
| [`io.md`](./io.md) | **IO C++ 使用指南**：per-worker proactor、TCP/UDP deadline/cancel、自动文件卸载、HTTP/HTTPS transport、DNS/signal 与专属快腿。 |
| [`c-core.md`](./c-core.md) | **core direct FFI 边界（已按 managed binding 复审）**：`hyle_ffi_abi_version` bootstrap 与两个 product-version 查询；compiler/cache/prefetch/intrusive 只是 C++ source 能力，core 无 native C source 产品面。 |
| [`simd-usage.md`](./simd-usage.md) | **SIMD 怎么选和怎么调**：普通动态派发、整库固定 ISA、固定 kernel、常用 batch 能力与调用合同，以及待 managed binding 复审的 whole-buffer Native FFI 候选面。 |
| [`usage-contract.md`](./usage-contract.md) | **怎么调才不出 UB**：为 SOTA 性能保留的调用安全红线（fiber 勿跨 `yield()` 持有 thread_local、分配器跨线程 free、CPUID 探测的 LTO 隔离）。每条带机制+安全模式+后果。**热路径集成前必读。** |

除 `c-core.md` 与 `c-error.md` 已按无 C shim 的 direct binding 口径复审外，其他 `c-*.md` 目前都是
**native FFI 合同候选**。每个模块先从真实外语任务裁决是否需要 FFI；没有任务可明确 `N/A`，
只有保留 direct FFI 时才以 Java FFM 与 .NET P/Invoke 做最低 direct-binding 复审。若模块进入
正式 Rust/Python binding 产品，再追加对应 package 的 ABI loader 复审。现有纯 C 编译
smoke 只能证明 schema 能被 C 工具链解析，不能证明
宏、`static inline`、自引用 POD 或源码 recipe 能被 managed runtime 直接绑定，也不是“C 易用性”
的产品证书。

严格收口时，`<module>.md` 先回答“什么场景选什么、C++ 怎样自然组合、哪里会踩坑”；
`c-<module>.md` 再以动态导出表为起点，只把 direct symbol、fixed-width POD、opaque handle 和
callback 记为 native FFI 能力。找不到这种通路就应明确登记为缺口或有意不提供，
不能用 native C header-inline 组合冒充 direct FFI。

## 用户文档的分工

- **guide（这里）**：正常怎么用 + 怎么调才安全 —— 使用者视角。
- **[`architecture/`](../architecture/)**：为什么这么造、各支柱选型、clay 资产负债 —— 贡献者视角。
- **[`performance/`](../performance/README.md)**：绑定 commit 的 benchmark 方法、点值与 win/loss；guide 不复制。
- **[`milestones/`](../milestones/README.md)**：施工任务和当前待办；guide 不维护就绪度镜像。
- **[`samples/`](../../samples/)**：照着抄就能跑的示例工程 —— 上手最快。
